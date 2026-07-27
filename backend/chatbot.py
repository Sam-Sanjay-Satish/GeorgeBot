#!/usr/bin/env python3
"""
GeorgeBot — v2 RAG backend.

Two retrieval paths, routed per-query by a single MiniMax-M3 call:

  - Graph path   — course/program facts (prereqs, credits, cross-listings,
                   descriptions, program requirements, outlines) come straight
                   from `graph_queries.GraphStore` (course_graph.pkl +
                   program_graph.pkl + course outlines). No vector search.
  - Vector path  — everything else hits the v2.2 Chroma DB, which splits the
                   corpus into two collections (`georgebot_v22_undergrad`,
                   `georgebot_v22_faculty`), each storing one entry per
                   reverse-HyDE question (5 per chunk). The caller-supplied
                   `audience` (undergrad / faculty / both) selects which to
                   search. A query embeds once, we pull the nearest
                   question-vectors from the selected collection(s), merge by
                   distance, then collapse by `chunk_id` back to distinct
                   parent chunks (full text + metadata).

Then MiniMax-M3 reads whichever context was assembled and writes the final,
source-cited answer (see `_call_llm`/`answer`/`answer_stream`). Single
provider — the official MiniMax API (OpenAI-compatible), not the Kesar-
proxied alias. Both route/rewrite AND the answer step run with
`thinking: "disabled"` — the answer step was switched off (from adaptive)
once the reference material moved into the system prompt and the prompt was
rewritten to own the "ignore irrelevant material / answer from own knowledge"
behavior directly (see `SYSTEM_PROMPT` / `_system_prompt_with_context`), for
~2x lower answer latency. Both set `reasoning_split: True` so any hidden
reasoning stays out of visible content.

Usage:
  python3 backend/chatbot.py --ask "your question"   # one-shot CLI (no server)
  python3 backend/api.py                              # FastAPI server (see api.py)

Env (.env): MINIMAX_SUB_KEY, VOYAGE_API_KEY
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

from banner import banner_instructor_retrieve, banner_retrieve   # live class availability
from rmp import rmp_retrieve   # RateMyProfessors ratings/reviews (student opinion)

BASE_DIR = Path(__file__).parent          # backend/

# Serving-artifact locations. Default to BASE_DIR-relative paths (local dev,
# artifacts copied into backend/); override via env for a Railway Volume mount.
#   DATA_DIR      — common base for all three at once (one Volume, one mount):
#                   $DATA_DIR/{chroma_db, vector_taxonomy.json, graph_data}.
#   CHROMA_DIR /
#   TAXONOMY_FILE — per-artifact overrides; win over DATA_DIR when set.
# graph_data lives in graph_queries.py (GRAPH_DATA_DIR / DATA_DIR, same scheme).
_DATA_DIR = Path(os.environ["DATA_DIR"]) if os.getenv("DATA_DIR") else BASE_DIR
CHROMA_DIR = Path(os.getenv("CHROMA_DIR", _DATA_DIR / "chroma_db"))
TAXONOMY_FILE = Path(os.getenv("TAXONOMY_FILE", _DATA_DIR / "vector_taxonomy.json"))
GENERAL_DEPARTMENT = "general / cross-departmental"
THINK_TAG_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"


def _suffix_prefix_len(s: str, tag: str) -> int:
    """Length of the longest *proper* suffix of `s` that is a prefix of `tag`.

    Used to decide how many trailing chars to hold back mid-stream in case a
    `<think>`/`</think>` tag is split across chunk boundaries (e.g. a chunk
    ending in "</thi" might be the start of a closing tag).
    """
    for k in range(min(len(s), len(tag) - 1), 0, -1):
        if s[-k:] == tag[:k]:
            return k
    return 0


def _iter_visible_deltas(chunks):
    """Yield only the visible answer text from a stream of raw `content` deltas,
    dropping any `<think>...</think>` span — even when the tags are split across
    chunk boundaries. Streaming equivalent of `THINK_TAG_RE.sub("", ...)`.

    `reasoning_split: True` is supposed to keep reasoning out of `content`
    entirely (it lands in a separate `reasoning_content` field we never read),
    but leaks were observed in practice (see `_call_llm`), so filter defensively.
    """
    pending = ""
    in_think = False
    for chunk in chunks:
        if not chunk:
            continue
        pending += chunk
        while True:
            if in_think:
                idx = pending.find(_THINK_CLOSE)
                if idx == -1:
                    # No close yet — discard reasoning, keep only a possible
                    # partial closing tag at the tail so we can match it later.
                    keep = _suffix_prefix_len(pending, _THINK_CLOSE)
                    pending = pending[len(pending) - keep:] if keep else ""
                    break
                pending = pending[idx + len(_THINK_CLOSE):]
                in_think = False
                continue
            idx = pending.find(_THINK_OPEN)
            if idx == -1:
                # Emit everything except a possible partial opening tag.
                keep = _suffix_prefix_len(pending, _THINK_OPEN)
                emit = pending[:len(pending) - keep] if keep else pending
                if emit:
                    yield emit
                pending = pending[len(pending) - keep:] if keep else ""
                break
            if idx > 0:
                yield pending[:idx]
            pending = pending[idx + len(_THINK_OPEN):]
            in_think = True
    # Flush any trailing text that wasn't a real (partial) tag.
    if not in_think and pending:
        yield pending


_CITED_TAG_OPEN = "<<CITED_SOURCES:"
_CITED_TAG_CLOSE = ">>"


def _split_cited_sources(chunks):
    """Yield visible answer text, stripping a trailing `<<CITED_SOURCES:
    2,3>>` marker (see SYSTEM_PROMPT's CITED SOURCES section) that reports
    which numbered `[n]` reference blocks the answer actually relied on —
    even when the marker straddles chunk boundaries (same buffer-then-decide
    idiom `_iter_visible_deltas` uses for `<think>` tags). The marker is
    expected at most once, at the very end of the response.

    This generator's return value (readable via `yield from` or manual
    `next()`/`StopIteration.value`) is the raw string between the tag
    delimiters, or None if no complete marker was found — callers should
    treat None as "unknown" and fail open (show every source) rather than
    hiding real ones over a parsing hiccup.
    """
    pending = ""
    tag_open = False
    done = False
    body: str | None = None
    for chunk in chunks:
        if done or not chunk:
            continue
        pending += chunk
        if not tag_open:
            idx = pending.find(_CITED_TAG_OPEN)
            if idx == -1:
                # Emit everything except a possible partial opening tag.
                keep = _suffix_prefix_len(pending, _CITED_TAG_OPEN)
                emit = pending[:len(pending) - keep] if keep else pending
                if emit:
                    yield emit
                pending = pending[len(pending) - keep:] if keep else ""
                continue
            if idx > 0:
                yield pending[:idx]
            pending = pending[idx:]
            tag_open = True
        close_idx = pending.find(_CITED_TAG_CLOSE, len(_CITED_TAG_OPEN))
        if close_idx != -1:
            body = pending[len(_CITED_TAG_OPEN):close_idx]
            pending = ""
            done = True
    # Anything still pending here is either ordinary text (no tag ever
    # started) or a truncated marker (tag opened but never closed, e.g. the
    # model got cut off mid-tag) — the latter is dropped rather than leaked
    # to the user as broken-looking text.
    if not tag_open and pending:
        yield pending
    return body


def _parse_cited_sources(body: str | None) -> list[int] | None:
    """Parse a `_split_cited_sources`/`_extract_cited_sources` body into a
    list of cited `[n]` numbers. `"none"` (the model relied on no numbered
    material) parses to an empty list; a missing/malformed body returns None
    ("unknown" — callers should fail open, not hide every source)."""
    if body is None:
        return None
    body = body.strip()
    if not body:
        return None
    if body.lower() == "none":
        return []
    nums = [int(n) for n in re.findall(r"\d+", body)]
    return nums or None


def _extract_cited_sources(text: str) -> tuple[str, list[int] | None]:
    """Non-streaming counterpart to `_split_cited_sources`: strip a trailing
    `<<CITED_SOURCES: ...>>` marker from a complete answer string, returning
    (clean_text, cited_numbers)."""
    idx = text.rfind(_CITED_TAG_OPEN)
    if idx == -1:
        return text, None
    close = text.find(_CITED_TAG_CLOSE, idx + len(_CITED_TAG_OPEN))
    if close == -1:
        return text, None
    body = text[idx + len(_CITED_TAG_OPEN):close]
    clean = (text[:idx] + text[close + len(_CITED_TAG_CLOSE):]).strip()
    return clean, _parse_cited_sources(body)


def _filter_cited_sources(sources: list[dict], cited: list[int] | None) -> list[dict]:
    """Keep only the sources the model reported actually relying on (see
    `_parse_cited_sources`). `cited=None` means no valid marker was found —
    fail open and return every source rather than hiding real ones."""
    if cited is None:
        return sources
    cited_set = set(cited)
    return [s for s in sources if s["n"] in cited_set]


# v2.2 splits the corpus into two Chroma collections in one DB. The user picks
# which to search per request (undergrad / faculty / both) — see `audience`.
COLLECTION_NAMES = {
    "undergrad": "georgebot_v22_undergrad",
    "faculty": "georgebot_v22_faculty",
}
VALID_AUDIENCES = ("undergrad", "faculty", "both")
DEFAULT_AUDIENCE = "both"
VOYAGE_MODEL = "voyage-4-large"

# Single provider — official MiniMax API (OpenAI-compatible), not the
# Kesar-proxied alias. See repo memory: minimax-official-api.
MINIMAX_BASE_URL = "https://api.minimax.io/v1"
MINIMAX_MODEL = "MiniMax-M3"
LLM_MAX_TOKENS = 4000    # route/rewrite budget
ANSWER_MAX_TOKENS = 1500  # answer budget
LLM_MAX_RETRIES = 2      # retry on finish_reason == "length" (truncated mid-answer)

# Retrieval tuning
QUESTION_K = 40        # question-vectors pulled from Chroma before collapsing
N_CONTEXT = 4           # distinct chunks handed to the answer model
MAX_CHUNK_DISTANCE = 0.75  # cosine distance cutoff (voyage-4-large) — chunks
                            # worse than this are dropped rather than backfilled
                            # just to hit N_CONTEXT, so a genuinely off-topic
                            # question can end up with empty context instead of
                            # forcing in unrelated chunks (calibrated 2026-07-14:
                            # on-topic queries in this corpus cluster <=0.65,
                            # clearly-unrelated queries land >=0.82 — 0.75 splits
                            # the two with margin on both sides; note this is a
                            # semantic-distance cutoff, not a topical one, so a
                            # query that's lexically close to real corpus content
                            # but asks something the corpus doesn't answer, e.g.
                            # "what's the capital of France" landing near UVic's
                            # France exchange-program pages, can still slip under
                            # the threshold — this narrows but doesn't eliminate
                            # that class of case)
MAX_HISTORY_TURNS = 6   # trailing conversation turns kept for context

# Campus nickname glossary for the router's search_query rewrite. Interim
# mitigation for ambiguous/colloquial place names whose reverse-HyDE question
# set doesn't anchor on the bare nickname (e.g. corpus questions for the Cove
# dining hall are phrased around "dining hall"/"Cheko'nien House", not "the
# Cove" alone) — the router has no way to disambiguate a bare nickname
# otherwise and guesses. Real fix is broader reverse-HyDE question coverage
# in georgebot-pipeline; this only patches names we've actually seen fail.
CAMPUS_TERM_GLOSSARY = {
    "the cove": "The Cove dining hall, UVic's campus dining facility in "
                 "Cheko'nien House (Student Housing and Dining) — not the "
                 "UVic Cove child care centre.",
    "cl": "Community Leader — a student staff role in UVic Student Housing "
          "(residence life), similar to a residence advisor/don at other "
          "schools.",
}

# Default-mode verify-then-answer path: combined verify-then-answer call,
# thinking adaptive. Capped gated rounds before forcing a plain (no-verify)
# answer.
MAX_VERIFY_ROUNDS = 2


# ---------------------------------------------------------------------------
# Engine — loads indexes + API clients once, serves retrieval/answer calls
# ---------------------------------------------------------------------------

class GeorgeBot:
    def __init__(self) -> None:
        from dotenv import load_dotenv
        load_dotenv()

        missing = [k for k in ("MINIMAX_SUB_KEY", "VOYAGE_API_KEY") if not os.getenv(k)]
        if missing:
            print(f"ERROR: missing env var(s): {', '.join(missing)} (set them in .env)",
                  file=sys.stderr)
            sys.exit(1)

        import openai
        import chromadb
        import voyageai

        from graph_queries import GraphStore

        t0 = time.monotonic()
        print("Loading GeorgeBot...")

        # Route/rewrite + answer — official MiniMax API (OpenAI-compatible).
        self.llm = openai.OpenAI(api_key=os.getenv("MINIMAX_SUB_KEY"), base_url=MINIMAX_BASE_URL)
        self.voyage = voyageai.Client(api_key=os.getenv("VOYAGE_API_KEY"))

        if not TAXONOMY_FILE.exists():
            print(f"ERROR: taxonomy file not found at {TAXONOMY_FILE}. "
                  f"Generate it from georgebot-pipeline v2.2's taxonomy.yaml (per-"
                  f"audience topic_families + departments) as vector_taxonomy.json.",
                  file=sys.stderr)
            sys.exit(1)
        taxonomy = json.loads(TAXONOMY_FILE.read_text())
        self.departments = taxonomy["departments"]
        # topic_families are per-audience in v2.2 (undergrad and faculty have
        # disjoint vocabularies). Keep name-lists (for the router prompt) and
        # name->slug maps (for Chroma `tf_<slug>` filtering) keyed by audience.
        self.topic_family_names = {}   # audience -> [name, ...]
        self.topic_family_slugs = {}   # audience -> {name: slug}
        for aud in COLLECTION_NAMES:
            fams = taxonomy[f"topic_families_{aud}"]
            self.topic_family_names[aud] = [t["name"] for t in fams]
            self.topic_family_slugs[aud] = {t["name"]: t["slug"] for t in fams}
        # Union map for audience="both" (validating router output across both).
        self.topic_family_slugs_all = {
            name: slug for m in self.topic_family_slugs.values() for name, slug in m.items()
        }
        print(f"  Taxonomy: "
              + ", ".join(f"{len(v)} {a}" for a, v in self.topic_family_names.items())
              + f" topic families, {len(self.departments)} departments")

        if not CHROMA_DIR.exists():
            print(f"ERROR: Chroma DB not found at {CHROMA_DIR}. Copy georgebot-"
                  f"pipeline v2.2's f_embed/chroma_db/ here.", file=sys.stderr)
            sys.exit(1)
        chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        self.collections = {
            aud: chroma_client.get_collection(name) for aud, name in COLLECTION_NAMES.items()
        }
        for aud, coll in self.collections.items():
            print(f"  Chroma collection '{coll.name}' ({aud}): {coll.count():,} question-vectors")

        self.gs = GraphStore.load()
        print(f"  Course graph: {self.gs.cg.number_of_nodes():,} nodes; "
              f"Program graph: {self.gs.pg.number_of_nodes():,} nodes; "
              f"{len(self.gs.outlines):,} outlines")

        print(f"Ready in {time.monotonic() - t0:.1f}s\n")

    # -- vector retrieval -----------------------------------------------------

    def _embed_query(self, text: str) -> list[float]:
        # Docs/questions were embedded with input_type="document"; queries use "query".
        resp = self.voyage.embed([text], model=VOYAGE_MODEL, input_type="query")
        return resp.embeddings[0]

    @staticmethod
    def _audiences(audience: str) -> tuple[str, ...]:
        """Collections to search for the requested audience."""
        return ("undergrad", "faculty") if audience == "both" else (audience,)

    def _build_where(self, topic_families: list[str], department: str | None,
                      audience: str) -> dict | None:
        """Build a Chroma `where` clause from router-predicted topic families /
        department, for one audience's collection.

        Family names not in this audience's taxonomy are silently dropped —
        harmless, and exactly what makes audience="both" work: a family
        predicted from the other corpus's vocabulary just doesn't filter here.
        `department` is expanded to [department, GENERAL_DEPARTMENT] so a
        department-specific question still surfaces genuinely cross-cutting
        content (e.g. university-wide policies) alongside the department's own.
        """
        clauses = []
        slug_map = self.topic_family_slugs[audience]
        slugs = [slug_map[f] for f in topic_families if f in slug_map]
        if slugs:
            clauses.append({"$or": [{s: True} for s in slugs]} if len(slugs) > 1 else {slugs[0]: True})
        if department:
            clauses.append({"department": {"$in": [department, GENERAL_DEPARTMENT]}})
        else:
            clauses.append({"department": GENERAL_DEPARTMENT})
        if len(clauses) == 1:
            return clauses[0]
        return {"$and": clauses}

    def _query_candidates(self, collection, emb: list[float],
                           where: dict | None) -> list[tuple[float, dict]]:
        """Nearest question-vectors from ONE collection, distance-cutoff applied,
        collapsed to distinct chunks (nearest occurrence wins), returned as
        (distance, chunk) pairs nearest-first."""
        kwargs: dict = {"query_embeddings": [emb], "n_results": QUESTION_K,
                         "include": ["documents", "metadatas", "distances"]}
        if where:
            kwargs["where"] = where
        res = collection.query(**kwargs)
        docs = res["documents"][0] if res["documents"] else []
        metas = res["metadatas"][0] if res["metadatas"] else []
        dists = res["distances"][0] if res["distances"] else []
        out: list[tuple[float, dict]] = []
        seen: set[str] = set()
        for doc, meta, dist in zip(docs, metas, dists):
            if dist > MAX_CHUNK_DISTANCE:
                break  # Chroma returns hits sorted nearest-first; nothing after this is closer
            cid = meta.get("chunk_id")
            if not cid or cid in seen:
                continue
            seen.add(cid)
            out.append((dist, {"chunk_id": cid, "text": doc, "metadata": meta}))
        return out

    @staticmethod
    def _merge_collapse(candidate_lists: list[list[tuple[float, dict]]], n: int,
                         exclude: set[str]) -> list[dict]:
        """Merge per-collection candidate lists by cosine distance (valid across
        collections since they share the embedding space), collapse to n distinct
        chunks by chunk_id, skipping `exclude`."""
        merged = sorted((c for lst in candidate_lists for c in lst), key=lambda x: x[0])
        chunks: dict[str, dict] = {}
        order: list[str] = []
        for _dist, ch in merged:
            cid = ch["chunk_id"]
            if cid in chunks or cid in exclude:
                continue
            chunks[cid] = ch
            order.append(cid)
            if len(order) >= n:
                break
        return [chunks[cid] for cid in order]

    def vector_retrieve(self, query: str, audience: str = DEFAULT_AUDIENCE,
                         n: int = N_CONTEXT, topic_families: list[str] | None = None,
                         department: str | None = None) -> list[dict]:
        """
        Query the reverse-HyDE question index(es) for the requested audience,
        collapse hits to distinct chunks.

        One query embedding is run against each selected collection and the
        results are merged by distance (see `_merge_collapse`), so audience=
        "both" returns the globally-nearest n chunks across the two corpora
        rather than n-per-corpus.

        Soft filter with backfill: if topic_families/department were predicted
        by the router, run a filtered pass first so on-topic chunks rank first;
        then top up with an unfiltered pass (deduped) so a wrong or overly-narrow
        prediction never drops recall to zero — it just re-prioritizes what plain
        vector similarity already found.

        Both passes apply MAX_CHUNK_DISTANCE, so a genuinely off-topic query can
        return fewer than n chunks — even zero — rather than being padded out to
        n with irrelevant ones just to hit the count.
        """
        emb = self._embed_query(query)
        auds = self._audiences(audience)
        has_filter = bool(topic_families or department)

        results: list[dict] = []
        if has_filter:
            filtered = [
                self._query_candidates(self.collections[a], emb,
                                       self._build_where(topic_families or [], department, a))
                for a in auds
            ]
            results = self._merge_collapse(filtered, n, exclude=set())

        if len(results) < n:
            exclude = {r["chunk_id"] for r in results}
            unfiltered = [self._query_candidates(self.collections[a], emb, None) for a in auds]
            results += self._merge_collapse(unfiltered, n - len(results), exclude=exclude)
        return results

    # -- graph retrieval --------------------------------------------------------

    def _course_facts(self, code: str, want_outline: bool, completed: list[str]) -> dict | None:
        course = self.gs.get_course(code)
        if not course:
            return None
        eligibility = self.gs.get_eligibility(code)
        coreqs = self.gs.get_corequisites(code)
        facts = {
            "code": code,
            "title": course.get("title"),
            "credits": course.get("credits"),
            "hours": course.get("hours"),
            "description": course.get("description"),
            "prereq_text": course.get("prereq_text"),
            "prereq_courses": eligibility.get("courses", []),
            "non_course_requirements": eligibility.get("non_course", []),
            # Full transitive prereq closure (everything needed before this
            # course, directly or via a chain), minus the direct prereqs already
            # listed above — so it only shows the *deeper* dependencies.
            "prereq_chain": sorted(self.gs.prereq_chain(code)),
            "corequisites": coreqs.get("courses", []),
            "cross_listed": self.gs.cross_listings(code),
            "unlocks": self.gs.get_unlocks(code),
            "alternatives": self.gs.get_alternatives(code),
            "required_by_programs": self.gs.programs_requiring(code),
            "url": course.get("url"),
        }
        if want_outline:
            outline = self.gs.get_outline(code)
            if outline:
                facts["outline"] = outline
        if completed:
            facts["completed_given"] = completed
            facts["prereq_satisfied"] = self.gs.prereq_satisfied(code, completed)
        return facts

    @staticmethod
    def _rank_program_matches(query: str, matches: list[dict]) -> list[dict]:
        """Rank candidate programs by how specifically `query` points at each.

        `search_programs` already guarantees every candidate contains all of
        the query's tokens, so the differentiator isn't *whether* the query
        matches but how much *extra* each candidate carries beyond what the
        user actually typed: the candidate with the fewest surplus tokens is
        the closest fit to the user's words. e.g. for "computer science
        honours", the standalone "Computer Science / Bachelor of Science -
        Honours" adds only {bachelor, of}, while "Computer Science and
        Mathematics / Combined Honours" also drags in {and, mathematics,
        combined} — so the standalone program ranks first.

        Returns `matches` sorted best-first, each annotated with `_extra`
        (surplus-token count; lower = closer) and `_size` (tie-break: prefer
        the more specific / smaller candidate).
        """
        def toks(s: str) -> set:
            return set(re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).split())
        qt = toks(query)
        ranked = []
        for m in matches:
            ct = toks(f"{m['title']} {m['code']} {m.get('credential') or ''}")
            ranked.append({**m, "_extra": len(ct - qt), "_size": len(ct)})
        ranked.sort(key=lambda m: (m["_extra"], m["_size"]))
        return ranked

    def _program_facts(self, query: str, completed: list[str]) -> dict:
        matches = self.gs.search_programs(query)
        if not matches:
            return {"query": query, "matches": []}
        auto_alternatives = None
        if len(matches) > 1:
            ranked = self._rank_program_matches(query, matches)
            best, runner_up = ranked[0], ranked[1]
            # Auto-pick only when the query clearly favours one program. A
            # genuine tie — e.g. a bare "computer science", where Major /
            # Honours / Minor are equally close — still asks the user rather
            # than silently guessing a wrong-but-plausible requirements list.
            if (best["_extra"], best["_size"]) == (runner_up["_extra"], runner_up["_size"]):
                return {"query": query, "matches": matches, "ambiguous": True}
            m = best
            # Surface the close-but-not-chosen options so the model can say
            # "assuming you mean X" and let the user correct course.
            auto_alternatives = [
                {"title": r["title"], "code": r["code"], "credential": r.get("credential")}
                for r in ranked[1:]
            ]
        else:
            m = matches[0]
        prog = self.gs.get_program(m["pid"])
        groups = self.gs.program_requirement_groups(m["pid"])
        specs = self.gs.program_specializations(m["pid"])
        result = {
            "query": query,
            "matches": matches,
            "auto_selected": auto_alternatives is not None,
            "alternatives": auto_alternatives or [],
            "program": {
                "code": prog.get("code"),
                "title": prog.get("title"),
                "credential": prog.get("credential"),
                "total_units": prog.get("program_total_units"),
                "description": prog.get("description"),
                "url": prog.get("url"),
            },
            "requirement_groups": [
                {"label": g["label"], "tree": g["tree"]} for g in groups
            ],
            "specializations": [s["title"] for s in specs],
            # Flat list of every course referenced anywhere in the program
            # (groups + specializations) — quick "does this program require X".
            "all_courses": self.gs.program_courses(m["pid"]),
        }
        if completed:
            result["completed_given"] = completed
            result["requirements_remaining"] = self.gs.requirements_remaining(m["pid"], completed)
        return result

    def graph_retrieve(self, course_codes: list[str], program_query: str | None,
                        want_outline: bool, completed_courses: list[str]) -> dict:
        courses = []
        for code in course_codes:
            f = self._course_facts(code, want_outline, completed_courses)
            if f:
                courses.append(f)
        program = self._program_facts(program_query, completed_courses) if program_query else None
        return {"courses": courses, "program": program}

    @staticmethod
    def _graph_context_text(graph_facts: dict) -> str:
        blocks = []
        i = 0
        for c in graph_facts.get("courses", []):
            i += 1
            lines = [
                f"[{i}] source=kuali course={c['code']}",
                f"URL: {c.get('url', 'n/a')}",
                f"{c['code']} — {c.get('title', '')} ({c.get('credits', '?')} units, hours {c.get('hours', '?')})",
                f"Description: {c.get('description', '')}",
            ]
            if c.get("prereq_text"):
                lines.append(f"Prerequisite text (calendar): {c['prereq_text']}")
            if c.get("prereq_courses"):
                lines.append(f"Prerequisite course codes: {', '.join(c['prereq_courses'])}")
            # Deeper (transitive) prereqs beyond the direct ones already listed.
            deeper = [x for x in c.get("prereq_chain", []) if x not in set(c.get("prereq_courses", []))]
            if deeper:
                lines.append(f"Full prerequisite chain also requires (indirectly): {', '.join(deeper)}")
            if c.get("non_course_requirements"):
                nc = "; ".join(r.get("description", "") for r in c["non_course_requirements"])
                lines.append(f"Other requirements: {nc}")
            if c.get("corequisites"):
                co = ", ".join(x["code"] for x in c["corequisites"])
                lines.append(f"Corequisites: {co}")
            if c.get("cross_listed"):
                lines.append(f"Cross-listed with: {', '.join(c['cross_listed'])}")
            if c.get("unlocks"):
                lines.append(f"Courses that require {c['code']} as a prerequisite: {', '.join(c['unlocks'])}")
            if c.get("alternatives"):
                lines.append(f"Alternative/equivalent courses — interchangeable with {c['code']} "
                              f"wherever it's used as a prerequisite option: {', '.join(c['alternatives'])}")
            if c.get("required_by_programs"):
                lines.append(f"Required by programs: {', '.join(c['required_by_programs'])}")
            if c.get("prereq_satisfied") is not None:
                ps = c["prereq_satisfied"]
                lines.append(f"Given completed courses [{', '.join(c['completed_given'])}]: "
                              f"prerequisites satisfied = {ps['satisfied']}")
                if ps.get("missing"):
                    missing_desc = []
                    for m in ps["missing"]:
                        if "one_of" in m:
                            opts = ", ".join(o["code"] for o in m["options"])
                            missing_desc.append(f"{m['one_of']} of: {opts}")
                        else:
                            missing_desc.append(m["code"])
                    lines.append(f"Missing course requirements: {'; '.join(missing_desc)}")
                if ps.get("unknown_requirements"):
                    unk = "; ".join(u.get("description", "") for u in ps["unknown_requirements"])
                    lines.append(f"Non-course requirements (cannot verify from completed list): {unk}")
            if c.get("outline"):
                o = c["outline"]
                term = o.get("term", "unknown term")
                lines.append(
                    f"HISTORICAL course outline (term {term}, source=heat):\n"
                    f"{o.get('text', '')[:4000]}"
                )
            blocks.append("\n".join(lines))

        program = graph_facts.get("program")
        if program:
            i += 1
            if program.get("ambiguous"):
                names = "; ".join(f"{m['title']} ({m['code']}, {m.get('credential', '')})"
                                   for m in program["matches"])
                blocks.append(
                    f"[{i}] source=kuali program_search=\"{program['query']}\"\n"
                    f"Multiple matching programs found — ask the user to clarify which one: {names}"
                )
            elif program.get("program"):
                p = program["program"]
                lines = [
                    f"[{i}] source=kuali program={p.get('code')}",
                    f"URL: {p.get('url', 'n/a')}",
                    f"{p.get('title', '')} — {p.get('credential', '')} ({p.get('total_units', '?')} units)",
                    f"Description: {p.get('description', '')}",
                ]
                if program.get("auto_selected"):
                    alts = "; ".join(f"{a['title']} ({a.get('credential', '')})"
                                      for a in program.get("alternatives", []))
                    lines.append(
                        f"NOTE: the user's request \"{program['query']}\" matched several "
                        f"programs; this is the closest one and was auto-selected. Briefly "
                        f"state which program you're assuming, and mention they can ask about "
                        f"another if this isn't it. Other close matches: {alts}"
                    )
                if program.get("requirements_remaining") is not None:
                    lines.append(f"Given completed courses [{', '.join(program['completed_given'])}], "
                                  f"outstanding requirements by group:")
                    for grp in program["requirements_remaining"]:
                        note = " (plus non-enumerable requirements not listed here)" if grp["has_non_course_reqs"] else ""
                        lines.append(f"  - {grp['label'] or '(unlabeled)'}: "
                                      f"{', '.join(grp['remaining']) or '(none)'}{note}")
                else:
                    for g in program.get("requirement_groups", []):
                        lines.append(f"Requirement group \"{g['label']}\": {json.dumps(g['tree'])[:1500]}")
                if program.get("specializations"):
                    lines.append(f"Specializations: {', '.join(program['specializations'])}")
                if program.get("all_courses"):
                    ac = program["all_courses"]
                    shown = ", ".join(ac[:60])
                    more = f" (+{len(ac) - 60} more)" if len(ac) > 60 else ""
                    lines.append(f"All courses referenced by this program (may include "
                                  f"options, not all required): {shown}{more}")
                blocks.append("\n".join(lines))
            elif not program.get("matches"):
                blocks.append(
                    f"[{i}] source=kuali program_search=\"{program['query']}\"\n"
                    f"No matching program found in the calendar."
                )
        return "\n\n".join(blocks)

    @staticmethod
    def _banner_section_line(s: dict) -> str:
        """One section rendered as a single line: seats, waitlist, schedule (days/time),
        instructor, and delivery mode. Shared by the availability and instructor render
        paths. Location (building/room) and campus are intentionally omitted — UVic's
        feed has no room, and campus reads misleadingly as one."""
        def _hhmm(t: str | None) -> str:
            return f"{t[:2]}:{t[2:]}" if t and len(t) == 4 else "?"

        seat = (f"{s['seats_available']} of {s['max_enrollment']} seats open "
                f"({s['enrollment']} enrolled)")
        parts = [f"  {s['section']} ({s.get('schedule_type', '')}): {seat} — "
                 f"{'OPEN' if s.get('open') else 'FULL'}."]
        if s.get("wait_capacity"):
            parts.append(f"Waitlist {s.get('wait_count', '?')}/{s['wait_capacity']} "
                         f"({s.get('wait_available', '?')} open).")
        m = s.get("meeting")
        if m and (m.get("days") or m.get("begin")):
            when = " ".join(m.get("days") or []) or "days TBA"
            parts.append(f"Meets {when} {_hhmm(m.get('begin'))}-{_hhmm(m.get('end'))}.")
        profs = ", ".join(f"{p['name']}" + (f" ({p['email']})" if p.get("email") else "")
                          for p in s.get("instructors", []) if p.get("name"))
        parts.append(f"Instructor: {profs or 'TBA'}.")
        # Delivery mode only (online / in-person); campus is deliberately left out of
        # the context (it's not a room and reads misleadingly as a location).
        if s.get("delivery"):
            parts.append(s["delivery"] + ".")
        return " ".join(parts)

    @classmethod
    def _banner_context_text(cls, banner_facts: dict, offset: int) -> tuple[str, int]:
        """Render live Banner data into numbered `[n]` blocks tagged source=banner,
        starting at `[offset+1]`. Returns (text, n_blocks). Mirrors
        `_graph_context_text`'s numbering so graph, banner, and vector blocks share one
        continuous `[n]` sequence.

        Two shapes (`kind`): "availability" -> one block per course (live seats);
        "instructor" -> a single block listing everything a professor teaches (or an
        ambiguity/no-match note the answer model acts on)."""
        term_label = banner_facts.get("term_label", "")

        if banner_facts.get("kind") == "instructor":
            i = offset + 1
            query = banner_facts.get("instructor_query", "")
            instr = banner_facts.get("instructor")
            candidates = banner_facts.get("candidates", [])
            courses = banner_facts.get("courses", [])
            if not instr and candidates:
                body = (f"Multiple instructors match \"{query}\" for {term_label}: "
                        f"{'; '.join(candidates)}. Ask the user which one they mean before "
                        f"answering.")
            elif not courses:
                body = (f"No classes found taught by an instructor matching \"{query}\" "
                        f"in {term_label}. Tell the user that and suggest checking the "
                        f"name/term, without inventing courses.")
            else:
                lines = [f"LIVE ({term_label}): {instr} is teaching the following "
                         f"(current seat counts, subject to change):"]
                for course in courses:
                    lines.append(f" {course['code']}:")
                    lines.extend(cls._banner_section_line(s) for s in course["sections"])
                body = "\n".join(lines)
            header = f"[{i}] source=banner instructor=\"{instr or query}\" term={term_label}"
            return f"{header}\n{body}", 1

        blocks = []
        i = offset
        for course in banner_facts.get("courses", []):
            i += 1
            code = course["code"]
            lines = [
                f"[{i}] source=banner course={code} term={term_label}",
                f"LIVE class availability for {code} ({term_label}) — current seat "
                f"counts pulled from UVic registration, subject to change:",
            ]
            lines.extend(cls._banner_section_line(s) for s in course.get("sections", []))
            blocks.append("\n".join(lines))
        return "\n\n".join(blocks), i - offset

    @staticmethod
    def _rmp_context_text(rmp_facts: dict, offset: int) -> tuple[str, int]:
        """Render RateMyProfessors data into numbered `[n]` blocks tagged source=rmp,
        starting at `[offset+1]`. Returns (text, n_blocks). One block per professor,
        continuing the same `[n]` sequence as graph/banner/vector.

        Each block is subjective student opinion (rating/difficulty/would-take-again
        + a few recent reviews), an ambiguous-name note (ask the user to pick), or a
        no-match note (say there's nothing on RMP — don't invent ratings)."""
        blocks = []
        i = offset
        for p in rmp_facts.get("professors", []):
            i += 1
            query = p.get("query", "")
            if not p.get("matched"):
                name_for_header = query
                if p.get("candidates"):
                    body = (f"Multiple professors on RateMyProfessors match "
                            f"\"{query}\": {'; '.join(p['candidates'])}. Ask the user "
                            f"which one they mean before giving a rating.")
                else:
                    body = (f"No RateMyProfessors listing found for a professor "
                            f"matching \"{query}\". Tell the user there's no RMP data "
                            f"for them (suggest checking the spelling) — do not invent "
                            f"a rating.")
            elif not p.get("num_ratings"):
                name_for_header = p.get("name") or query
                body = (f"{name_for_header} is on RateMyProfessors but has no ratings "
                        f"yet, so there's no student feedback to summarize.")
            else:
                name_for_header = p.get("name") or query
                wta = (f"{p['would_take_again']}% would take again"
                       if p.get("would_take_again") is not None else "would-take-again n/a")
                lines = [
                    f"STUDENT OPINION (RateMyProfessors, self-selected reviews — NOT "
                    f"official UVic data): {name_for_header}"
                    + (f", {p['department']}" if p.get("department") else ""),
                    f"Overall rating {p['rating']}/5, difficulty {p['difficulty']}/5, "
                    f"{wta}, based on {p['num_ratings']} student ratings.",
                ]
                if p.get("reviews"):
                    lines.append("Recent reviews:")
                    for r in p["reviews"]:
                        meta = []
                        if r.get("course"):
                            meta.append(r["course"])
                        if r.get("date"):
                            meta.append(r["date"])
                        if r.get("quality") is not None:
                            meta.append(f"quality {r['quality']}/5")
                        if r.get("difficulty") is not None:
                            meta.append(f"difficulty {r['difficulty']}/5")
                        tag = f" [{', '.join(r['tags'])}]" if r.get("tags") else ""
                        head = f" - ({', '.join(meta)}){tag}" if meta or tag else " -"
                        comment = (r.get("comment") or "").strip()
                        lines.append(f"{head} {comment}".rstrip())
                body = "\n".join(lines)
            header = f"[{i}] source=rmp instructor=\"{name_for_header}\""
            blocks.append(f"{header}\n{body}")
        return "\n\n".join(blocks), i - offset

    # -- LLM steps ----------------------------------------------------------

    def _call_llm(self, messages: list[dict], system: str | None = None,
                   max_tokens: int = LLM_MAX_TOKENS, thinking: str = "disabled") -> str | None:
        """
        Call MiniMax-M3 and return the visible text, or None on failure.

        `reasoning_split: True` is supposed to keep `message.content` clean of
        raw `<think>...</think>` text regardless of the `thinking` setting
        (see memory: minimax-official-api), but this isn't 100% reliable in
        practice (observed leaks even with the flag set) — strip any
        `<think>` block defensively regardless. If the response gets cut off
        mid-answer (finish_reason == "length") discard and retry rather than
        return a truncated answer.
        """
        full_messages = ([{"role": "system", "content": system}] if system else []) + messages
        kwargs: dict = {
            "model": MINIMAX_MODEL,
            "max_tokens": max_tokens,
            "messages": full_messages,
            "extra_body": {"reasoning_split": True, "thinking": {"type": thinking}},
        }
        for _attempt in range(LLM_MAX_RETRIES + 1):
            resp = self.llm.chat.completions.create(**kwargs)
            choice = resp.choices[0]
            content = choice.message.content
            if content is None:
                continue
            if choice.finish_reason == "length":
                continue
            return THINK_TAG_RE.sub("", content).strip()
        return None

    def rewrite_and_route(self, question: str, history: list[dict],
                          audience: str = DEFAULT_AUDIENCE) -> dict:
        """
        One minimax-m3 call: rewrite the question into a standalone search query
        AND classify whether it needs structured course/program facts (graph
        route) or general document retrieval (vector route).

        `audience` selects which topic-family vocabulary the router chooses from
        (undergrad / faculty / both) — it must match the collection(s) that
        vector_retrieve will search, or the predicted families won't filter.
        """
        # Topic-family vocabulary the router may choose from for this audience.
        if audience == "both":
            family_names = self.topic_family_names["undergrad"] + self.topic_family_names["faculty"]
            valid_families = set(self.topic_family_slugs_all)
        else:
            family_names = self.topic_family_names[audience]
            valid_families = set(self.topic_family_slugs[audience])

        convo = ""
        if history:
            lines = []
            for turn in history[-MAX_HISTORY_TURNS:]:
                role = "User" if turn.get("role") == "user" else "Assistant"
                lines.append(f"{role}: {turn.get('content', '')}")
            convo = "Conversation so far:\n" + "\n".join(lines) + "\n\n"

        # Whole-word match (not substring) — a short acronym like "cl" would
        # otherwise false-positive inside "class", "declare", "include", etc.
        question_lower = question.lower()
        glossary_hits = {
            term: meaning for term, meaning in CAMPUS_TERM_GLOSSARY.items()
            if re.search(rf"\b{re.escape(term)}\b", question_lower)
        }
        glossary_block = ""
        if glossary_hits:
            glossary_lines = "\n".join(f'  - "{t}": {m}' for t, m in glossary_hits.items())
            glossary_block = (
                f"\nCampus term glossary (use this to resolve ambiguous/"
                f"colloquial names mentioned in the question — incorporate the "
                f"real meaning into search_query instead of guessing):\n"
                f"{glossary_lines}\n"
            )

        prompt = (
            f"{convo}Today's date is {time.strftime('%Y-%m-%d')}.\n"
            f"You are the query router for a University of Victoria (UVic) "
            f"chatbot. Given the latest user question, produce a JSON object (and "
            f"nothing else) with these fields:\n"
            f"{glossary_block}\n"
            f'  "search_query": a single standalone search query for a document '
            f"knowledge base — resolve pronouns/references using the conversation, "
            f"keep course codes/program names/proper nouns.\n"
            f'  "course_codes": array of UVic course codes explicitly named or '
            f'clearly implied (e.g. ["CSC225", "MATH122"]), normalized with no '
            f"space (e.g. \"CSC 225\" -> \"CSC225\"). Empty array if none.\n"
            f'  "program_query": a short program name/title to search for (e.g. '
            f'"computer science honours") if the question is about degree/program '
            f"requirements, otherwise null.\n"
            f'  "wants_outline": true if the user is asking about a specific '
            f"past/current course's syllabus, grading scheme, schedule, or "
            f"instructor for a term (course outline content), else false.\n"
            f'  "wants_availability": true if the user is asking about LIVE class '
            f"availability for a specific course — open seats, whether a course/"
            f"section is full, waitlist space, section meeting times/days/room, or "
            f"who is teaching it this term. Else false.\n"
            f'  "term_season": if the question names or implies a specific academic '
            f'term, one of "spring" (Jan-Apr), "summer" (May-Aug), or "fall" '
            f"(Sep-Dec); otherwise null. Resolve relative references like \"next "
            f'spring\" using today\'s date above.\n'
            f'  "term_year": the 4-digit year of that term (e.g. 2026) when '
            f"determinable, otherwise null.\n"
            f'  "instructor_query": if the user is asking WHICH courses or sections a '
            f"specific professor/instructor teaches (a person-centric question, not "
            f"about one named course), the instructor's name as written (e.g. "
            f'"Yong", "Quinton Yong"); otherwise null. Leave course_codes empty when '
            f"this is set — the lookup is by instructor, not course.\n"
            f'  "professor_query": if the user is asking whether a specific named '
            f"professor is any GOOD — their teaching quality, rating, reviews, how "
            f"hard/easy they are, whether to take them — that professor's name as "
            f'written (e.g. "Yong", "Quinton Yong"); otherwise null. This is about '
            f"opinion/quality of a named person, distinct from instructor_query "
            f"(which courses they teach) and from wants_availability (who teaches a "
            f"section). Leave null if no specific professor is named.\n"
            f'  "wants_rating": true if the user wants teaching-quality/rating/review '
            f"information about whoever is involved — e.g. \"is X a good prof\", "
            f'"are they good", "who teaches CSC 225 and are they any good", "should I '
            f'take this section". This is the signal to consult RateMyProfessors; set '
            f"it true whenever professor_query is set, and ALSO when the quality "
            f"question is about a course's instructor(s) rather than a named person. "
            f"Else false.\n"
            f'  "completed_courses": array of UVic course codes the user has '
            f"stated (anywhere in the conversation, not just the latest message) "
            f"that they have already taken/completed/passed, normalized the same "
            f"way as course_codes. Only include courses explicitly described as "
            f"done — do not include a course just because it was discussed or "
            f"asked about. Empty array if none mentioned.\n"
            f'  "topic_families": array of 1-3 entries copied VERBATIM from this '
            f"exact list, whichever best describe what the question is about "
            f"(empty array if none fit well — don't force a bad match):\n"
            f"{json.dumps(family_names)}\n"
            f'  "department": copied VERBATIM from this exact list, the single '
            f"department/faculty the question is clearly about (e.g. the user "
            f"named a program, or asked something specific to one department's "
            f"process) — null if the question is general/cross-departmental or "
            f"you're not confident which department applies:\n"
            f"{json.dumps(self.departments)}\n"
            f"Populate course_codes whenever the question is actually ABOUT a "
            f"specific course — either its structured catalog facts (prerequisites, "
            f"credits, cross-listings, description, degree/program requirements, "
            f"outline content) OR its live availability/sections (seats, waitlist, "
            f"meeting times, instructor — i.e. whenever wants_availability is true). "
            f"program_query is for degree/program requirements only. For questions "
            f"about services, policies, deadlines, or general info, leave "
            f"course_codes empty and program_query null even if a course code is "
            f"mentioned only in passing. completed_courses should be populated whenever "
            f"the user has mentioned finished coursework, even on a general-info "
            f"question, so downstream prerequisite/requirement checks can use it.\n\n"
            f"Latest question: {question}"
        )
        try:
            out = self._call_llm([{"role": "user", "content": prompt}])
            if out is None:
                raise RuntimeError("router LLM returned no content")
            out = out.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            data = json.loads(out)
            dept = data.get("department") or None
            if dept not in self.departments:
                dept = None
            season = (data.get("term_season") or "").strip().lower() or None
            if season not in ("spring", "summer", "fall"):
                season = None
            try:
                year = int(data["term_year"]) if data.get("term_year") else None
            except (TypeError, ValueError):
                year = None
            return {
                "search_query": data.get("search_query") or question,
                "completed_courses": [c.replace(" ", "").upper() for c in (data.get("completed_courses") or [])],
                "course_codes": [c.replace(" ", "").upper() for c in (data.get("course_codes") or [])],
                "program_query": data.get("program_query") or None,
                "wants_outline": bool(data.get("wants_outline")),
                "wants_availability": bool(data.get("wants_availability")),
                "instructor_query": (data.get("instructor_query") or None),
                "professor_query": (data.get("professor_query") or None),
                "wants_rating": bool(data.get("wants_rating")),
                "term_season": season,
                "term_year": year,
                "topic_families": [f for f in (data.get("topic_families") or []) if f in valid_families],
                "department": dept,
            }
        except Exception as e:
            print(f"  [rewrite_and_route] failed, falling back to vector-only: {e}", file=sys.stderr)
            return {"search_query": question, "course_codes": [], "program_query": None,
                    "wants_outline": False, "wants_availability": False, "instructor_query": None,
                    "professor_query": None, "wants_rating": False,
                    "term_season": None, "term_year": None, "completed_courses": [],
                    "topic_families": [], "department": None}

    @staticmethod
    def _build_context(chunks: list[dict], graph_text: str, offset: int) -> str:
        blocks = []
        for i, ch in enumerate(chunks, offset + 1):
            meta = ch.get("metadata", {})
            tags = []
            if meta.get("source"):
                tags.append(f"source={meta['source']}")
            if meta.get("document_type"):
                tags.append(f"type={meta['document_type']}")
            if meta.get("department"):
                tags.append(f"department={meta['department']}")
            if meta.get("topic_families"):
                tags.append(f"topics={meta['topic_families']}")
            header = f"[{i}] {' | '.join(tags)}\nURL: {meta.get('origin', 'n/a')}"
            blocks.append(f"{header}\n{ch.get('text', '')}")
        text = "\n\n".join(blocks)
        if graph_text:
            text = f"{graph_text}\n\n{text}" if text else graph_text
        return text

    SYSTEM_PROMPT = (
        "You are GeorgeBot, a helpful assistant that answers questions about the "
        "University of Victoria (UVic) for students and staff.\n\n"
        "USING THE REFERENCE MATERIAL\n"
        "Alongside the user's question you are given additional reference "
        "material supplied by the SYSTEM — not by the user. The user did not "
        "write, choose, attach, or see this material, and is not aware it "
        "exists. Never speak as though the user provided it, and never refer to "
        "'the reference material', 'the provided information', or 'the context' "
        "when talking to the user.\n"
        "This material is gathered automatically and is NOT guaranteed to be "
        "relevant — some of it may be off-topic or only loosely related to what "
        "was actually asked. Read it, keep only the parts that genuinely help "
        "answer the question, and silently ignore everything else. If none of it "
        "is relevant, ignore all of it and answer from your own knowledge as a "
        "knowledgeable UVic assistant.\n"
        "Never tell the user that reference material is missing, irrelevant, or "
        "insufficient, and never describe what it happens to cover instead of "
        "answering — that reads as a broken non-answer. Just answer the question "
        "directly and helpfully. If there is a specific UVic fact you cannot "
        "confirm, point the user to where they can get it (e.g. 'check your "
        "course syllabus or ask your department directly') rather than saying "
        "you don't have it.\n\n"
        "ACCURACY\n"
        "- Ground specific facts (a status like 'no longer offered', a course "
        "sequence, a number, a date, a policy detail) in what the reference "
        "material actually states — do not infer, extrapolate, or reconstruct "
        "them from partial or messy data. Piecing together a plausible-sounding "
        "narrative from material that doesn't clearly state it is fabrication; "
        "state only what is explicit instead.\n"
        "- Some material comes from PDF program-worksheets or curriculum grids "
        "(course-by-term tables) that lose their row/column structure and read "
        "as a flattened run-on list of course codes and terms. Treat these as "
        "unreliable for anything beyond 'this course appears somewhere in this "
        "program's plan' — do not infer sequencing, prerequisites, term-by-term "
        "order, or offering status from a jumbled table you can't confidently "
        "parse. When unsure whether you are reading such a table correctly, say "
        "so rather than presenting a guess as fact.\n"
        "- Material tagged source=kuali is the official course/program catalog — "
        "treat it as the authoritative, current source for prerequisites, "
        "credits, and program requirements.\n"
        "- Material tagged source=banner is LIVE registration data for a specific "
        "term (named in the block) — current seat counts, waitlist space, section "
        "meeting days/times, and instructors. Use it for 'is it full / how many "
        "seats / when does it meet / who teaches it' questions. (Room numbers are not "
        "available — don't state or guess one.) Always name the "
        "term, cite section codes (e.g. A01), and call out when a section is full "
        "or waitlist-only. These numbers change constantly, so present them as "
        "current-as-of-now, not a guarantee. (Contrast: source=kuali is the static "
        "catalog; HISTORICAL outlines are past terms.) A source=banner block may "
        "instead list everything a named instructor teaches that term, or say the "
        "name was ambiguous (ask the user which person they mean) or matched no "
        "classes (tell them so — don't invent courses).\n"
        "- Material tagged source=rmp is third-party STUDENT OPINION from "
        "RateMyProfessors — subjective, self-selected reviews, NOT official UVic "
        "data and not a measure of fact. Use it only for 'is this professor any "
        "good / what are they like / are they hard' questions. Attribute it "
        "explicitly (e.g. 'students on RateMyProfessors rate…'), always give the "
        "overall rating together with how many ratings it's based on (a score from "
        "very few reviews is weak evidence — say so), and present it as opinion, "
        "never as an official or guaranteed judgement. You may summarize the "
        "themes of the written reviews, but don't present one student's take as "
        "the consensus. If a name was ambiguous, ask which professor they mean; if "
        "there's no RMP listing, say so plainly and never fabricate a rating.\n"
        "- Course outlines tagged HISTORICAL are past-term snapshots only. When "
        "you rely on one, always name the specific term, and never present "
        "historical grading weights, instructors, or schedules as current — for "
        "current course details, direct the user to Brightspace or the "
        "department.\n"
        "- When the material states 'Given completed courses [...]: prerequisites "
        "satisfied = ...' or 'outstanding requirements by group', that "
        "evaluation was computed programmatically from the courses the user said "
        "they've taken — trust it over your own tally, but flag any listed "
        "non-course requirements (year standing, GPA, permission, etc.) since "
        "those can't be auto-verified from a course list.\n"
        "- If a program search returned multiple candidates, ask the user to "
        "clarify which one they mean rather than guessing.\n\n"
        "STYLE\n"
        "- Be concise and direct. Use plain text (no LaTeX).\n\n"
        "CITED SOURCES\n"
        "After you finish writing your answer, on a new line by itself, "
        "report which numbered reference blocks (the [n] tags in the "
        "reference material above) you actually relied on to write it. "
        "Format: `<<CITED_SOURCES: 2,3>>` listing only the numbers you "
        "used, in any order — not every number that was offered to you, "
        "just the ones you actually leaned on. If you answered from your "
        "own knowledge without relying on any numbered material (including "
        "when all of it was irrelevant), write `<<CITED_SOURCES: none>>` "
        "instead. This must be the very last thing in your response, "
        "exactly once, with nothing after it, and never mentioned or "
        "explained anywhere in the visible answer. (This does not apply "
        "when you are writing a NEED_MORE JSON response instead of an "
        "answer — that response has no CITED_SOURCES tag.)"
    )

    # Appended to SYSTEM_PROMPT for default mode's combined verify-then-answer
    # call (see `answer_verified_stream`). Kept narrow by design — NEED_MORE
    # should be rare, reserved for a genuine, narrowly-scoped gap.
    VERIFY_ANSWER_ADDENDUM = (
        "\n\nRESPONSE FORMAT\n"
        "This call has one extra requirement on top of everything above. "
        "Before writing anything else, decide whether the reference material "
        "above is enough to answer the question well:\n"
        "- If it is — the common case, including when the material is "
        "irrelevant and you should answer from your own knowledge per the "
        "rules above — start your response with exactly the line "
        "`<<SUFFICIENT>>` followed by a newline, then write the answer "
        "normally.\n"
        "- ONLY if there is a concrete, identifiable gap that better "
        "retrieval could plausibly fix — the question names a specific "
        "term/date you have no data for, or the search query clearly "
        "doesn't match what was actually asked — start your response with "
        "exactly the line `<<NEED_MORE>>` followed by a newline, then a "
        "single JSON object and nothing else: {\"reason\": \"<one short "
        "sentence>\", \"search_query\": \"<a better search query>\" or "
        "null, \"term_season\": \"spring\"|\"summer\"|\"fall\" or null, "
        "\"term_year\": <4-digit year> or null}. Only set search_query if a "
        "different phrasing would plausibly find better material; only set "
        "term_season/term_year if the question needs a specific term you "
        "don't have data for.\n"
        "- Do NOT use NEED_MORE just because the material could be more "
        "thorough, more detailed, or cover more related topics — that is "
        "not what this is for, and it should be rare. Reserve it strictly "
        "for a clear, nameable retrieval miss, never for general "
        "thoroughness."
    )

    def _quick_mode_system_prompt(self) -> str:
        """SYSTEM_PROMPT + a sparing, at-most-once nudge to try quick mode
        off when there's a genuine, identifiable gap. Quick mode has no
        verification step (that's what `answer_verified_stream` is for in
        default mode), so this is the only signal the user gets that
        something might be missing."""
        return self.SYSTEM_PROMPT + (
            "\n\nQUICK MODE\n"
            "This is a fast, single-pass answer with no verification step. "
            "If, and only if, you notice a concrete, identifiable gap in "
            "what you were able to answer — the question needs live/current "
            "data you weren't given, or names something the reference "
            "material clearly doesn't cover — you may add ONE brief closing "
            "sentence noting that and suggesting the user try again with "
            "quick mode off for a more thorough answer. Do this sparingly: "
            "never as a routine hedge, never more than once, and never when "
            "you were able to answer fully."
        )

    def _system_prompt_with_context(self, context: str, base_prompt: str | None = None) -> str:
        """Build the system message: instructions + the retrieved reference
        material, framed as system-supplied (NOT part of the user turn) so the
        model doesn't treat it as something the user typed or attached.

        `base_prompt` overrides the default `SYSTEM_PROMPT` — used by
        `_quick_mode_system_prompt` to supply a mode-specific behavioral
        contract while reusing the same reference-material framing below."""
        base = base_prompt if base_prompt is not None else self.SYSTEM_PROMPT
        context = (context or "").strip()
        if context:
            ref = (
                "\n\n=== BEGIN SYSTEM-SUPPLIED REFERENCE MATERIAL ===\n"
                "(The user did not provide or see the following. It was gathered "
                "automatically and may include irrelevant items — use only what "
                "helps, ignore the rest; if none is relevant, answer from your "
                "own knowledge.)\n\n"
                f"{context}\n"
                "=== END SYSTEM-SUPPLIED REFERENCE MATERIAL ==="
            )
        else:
            ref = (
                "\n\n=== SYSTEM-SUPPLIED REFERENCE MATERIAL ===\n"
                "None was gathered for this question. Answer from your own "
                "knowledge as a knowledgeable UVic assistant.\n"
                "=== END SYSTEM-SUPPLIED REFERENCE MATERIAL ==="
            )
        return base + ref

    def _answer_messages(self, question: str, history: list[dict]) -> list[dict]:
        """Conversation turns + the user's question only — the reference
        material lives in the system prompt (see `_system_prompt_with_context`),
        never in the user turn."""
        messages: list[dict] = []
        for turn in history[-MAX_HISTORY_TURNS:]:
            role = turn.get("role")
            if role in ("user", "assistant") and turn.get("content"):
                messages.append({"role": role, "content": turn["content"]})
        messages.append({"role": "user", "content": question})
        return messages

    def answer(self, question: str, context: str, history: list[dict],
               system_prompt: str | None = None) -> tuple[str, list[int] | None]:
        """MiniMax-M3, thinking disabled — reference material is supplied via the
        system prompt (`_system_prompt_with_context`), not the user turn.

        `system_prompt` overrides the default behavioral contract (e.g. a
        mode-specific answer prompt).

        Returns (answer_text, cited_source_numbers) — see
        `_extract_cited_sources`/`_parse_cited_sources` for what the second
        element means. Callers should pass it to `_filter_cited_sources`
        before showing sources to the user."""
        messages = self._answer_messages(question, history)
        text = self._call_llm(messages,
                               system=self._system_prompt_with_context(context, system_prompt),
                               max_tokens=ANSWER_MAX_TOKENS, thinking="disabled")
        text = text or "Sorry, I couldn't generate an answer right now — please try again."
        return _extract_cited_sources(text)

    def _stream_answer_raw(self, messages: list[dict], system: str, thinking: str):
        """Low-level streamed MiniMax-M3 call, shared by `answer_stream` and
        `answer_verified_stream`. Builds the full message list, streams, and
        pipes raw `content` deltas through `_iter_visible_deltas` (drops any
        `<think>...</think>` span, even split across chunk boundaries —
        regardless of `thinking`, since `reasoning_split` isn't 100% reliable,
        see `_call_llm`) and then `_split_cited_sources` (strips a trailing
        `<<CITED_SOURCES: ...>>` marker the same way). Yields raw visible
        text deltas with no leading-whitespace trimming or empty-stream
        fallback; callers own that.

        Returns (via this generator's return value — retrievable through
        `yield from` or manual `next()`/`StopIteration.value`) the parsed
        cited-source numbers (list[int] | None; see `_parse_cited_sources`)."""
        full_messages = [{"role": "system", "content": system}] + messages
        stream = self.llm.chat.completions.create(
            model=MINIMAX_MODEL,
            max_tokens=ANSWER_MAX_TOKENS,
            messages=full_messages,
            extra_body={"reasoning_split": True, "thinking": {"type": thinking}},
            stream=True,
        )

        def _content_deltas():
            for event in stream:
                delta = event.choices[0].delta
                text = getattr(delta, "content", None)
                if text:
                    yield text

        visible = _iter_visible_deltas(_content_deltas())
        body = yield from _split_cited_sources(visible)
        return _parse_cited_sources(body)

    def answer_stream(self, question: str, context: str, history: list[dict],
                      system_prompt: str | None = None):
        """Streamed MiniMax-M3 answer, thinking disabled — yields
        `("token", text)` token-by-token, then a final `("cited",
        numbers_or_None)` once (see `_stream_answer_raw`/
        `_parse_cited_sources`) — callers should pass that to
        `_filter_cited_sources` before showing sources to the user.

        The reference material rides in the system prompt (not the user turn),
        so the model treats it as system-supplied. This lets the answer stream
        to the client incrementally instead of landing as one buffered chunk.
        Leading whitespace on the first visible piece is trimmed to match the
        non-streaming `answer()` behavior.
        """
        messages = self._answer_messages(question, history)
        system = self._system_prompt_with_context(context, system_prompt)
        gen = self._stream_answer_raw(messages, system, thinking="disabled")
        emitted = False
        cited: list[int] | None = None
        while True:
            try:
                piece = next(gen)
            except StopIteration as stop:
                cited = stop.value
                break
            if not emitted:
                piece = piece.lstrip()
                if not piece:
                    continue
            emitted = True
            yield ("token", piece)
        if not emitted:
            yield ("token", "Sorry, I couldn't generate an answer right now — please try again.")
        yield ("cited", cited)

    def answer_verified_stream(self, question: str, context: str, history: list[dict]):
        """Combined verify-then-answer call for default mode's answering path.

        Streamed, `thinking: "adaptive"`. The model prefixes its response with
        `<<SUFFICIENT>>` or `<<NEED_MORE>>` (see `VERIFY_ANSWER_ADDENDUM`) before writing the
        answer or a correction request, so one call serves as both judge and
        answerer in the common case. The header is peeled off by buffering up
        to the first newline — the same buffer-then-decide idiom
        `_iter_visible_deltas` already uses for `<think>` tags.

        Yields:
          ("token", text)   -- repeatedly, for a SUFFICIENT verdict; caller
                                forwards these live as the answer.
          ("cited", numbers_or_None) -- once, after the last "token", for a
                                SUFFICIENT verdict (see `_parse_cited_sources`);
                                caller should pass it to `_filter_cited_sources`
                                before showing sources to the user.
          ("need_more", {"reason", "search_query", "term_season", "term_year"})
                                -- once, for a NEED_MORE verdict. The stream is
                                fully drained here to parse the JSON — nothing
                                else is yielded, so nothing reaches the user.
        """
        messages = self._answer_messages(question, history)
        system = self._system_prompt_with_context(
            context, self.SYSTEM_PROMPT + self.VERIFY_ANSWER_ADDENDUM)
        raw = self._stream_answer_raw(messages, system, thinking="adaptive")

        header, rest, found_newline = "", "", False
        for piece in raw:
            header += piece
            if "\n" in header:
                header, _, rest = header.partition("\n")
                found_newline = True
                break
        header = header.strip()

        if found_newline and header == "<<NEED_MORE>>":
            body = rest + "".join(raw)
            body = body.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            try:
                data = json.loads(body)
            except (json.JSONDecodeError, TypeError):
                data = {}
            try:
                term_year = int(data["term_year"]) if data.get("term_year") else None
            except (TypeError, ValueError):
                term_year = None
            yield ("need_more", {
                "reason": data.get("reason") or "",
                "search_query": data.get("search_query") or None,
                "term_season": (data.get("term_season") or "").strip().lower() or None,
                "term_year": term_year,
            })
            return

        # SUFFICIENT, or an unrecognized/missing header -- fail open and
        # answer with everything buffered so far rather than losing a real
        # response over a malformed marker.
        if found_newline and header == "<<SUFFICIENT>>":
            first = rest
        elif found_newline:
            first = header + ("\n" + rest if rest else "")
        else:
            first = header
        emitted = False
        cited: list[int] | None = None
        pending_first: str | None = first
        while True:
            if pending_first is not None:
                piece, pending_first = pending_first, None
            else:
                try:
                    piece = next(raw)
                except StopIteration as stop:
                    cited = stop.value
                    break
            if not piece:
                continue
            if not emitted:
                piece = piece.lstrip()
                if not piece:
                    continue
            emitted = True
            yield ("token", piece)
        if not emitted:
            yield ("token", "Sorry, I couldn't generate an answer right now — please try again.")
        yield ("cited", cited)

    # -- source formatting --------------------------------------------------

    @staticmethod
    def format_sources(chunks: list[dict], graph_facts: dict | None,
                       banner_facts: dict | None = None,
                       rmp_facts: dict | None = None) -> list[dict]:
        """Deduplicate retrieved chunks + graph facts + live Banner facts + RMP
        ratings into a clean source list. Order matches the context numbering:
        graph, then banner, then rmp, then vector chunks."""
        seen: set[str] = set()
        sources = []
        n = 0

        if graph_facts:
            for c in graph_facts.get("courses", []):
                n += 1
                url = c.get("url", "")
                key = url or f"course:{c['code']}"
                if key in seen:
                    continue
                seen.add(key)
                sources.append({
                    "n": n,
                    "url": url,
                    "source": "kuali",
                    "title": f"{c['code']} — {c.get('title', '')}",
                    "course_code": c["code"],
                    "term": None,
                    "historical": False,
                })
                if c.get("outline"):
                    n += 1
                    o = c["outline"]
                    sources.append({
                        "n": n,
                        "url": o.get("url") or url,
                        "source": "heat",
                        "title": f"{c['code']} outline",
                        "course_code": c["code"],
                        "term": o.get("term"),
                        "historical": True,
                    })
            program = graph_facts.get("program")
            if program and program.get("program"):
                n += 1
                p = program["program"]
                url = p.get("url", "")
                key = url or f"program:{p.get('code')}"
                if key not in seen:
                    seen.add(key)
                    sources.append({
                        "n": n,
                        "url": url,
                        "source": "kuali",
                        "title": p.get("title", ""),
                        "course_code": None,
                        "term": None,
                        "historical": False,
                    })

        if banner_facts:
            term_label = banner_facts.get("term_label", "")
            instr = banner_facts.get("instructor")  # set only on the instructor path
            for course in banner_facts.get("courses", []):
                n += 1
                code = course["code"]
                sources.append({
                    "n": n,
                    # Banner's class-search UI (no stable per-course deep link); the
                    # data itself is served inline, this is just a "look it up" pointer.
                    "url": "https://banner.uvic.ca/StudentRegistrationSsb/ssb/classSearch/classSearch",
                    "source": "banner",
                    "title": f"{code} — {'taught by ' + instr if instr else 'live availability'}",
                    "course_code": code,
                    "term": term_label or None,
                    "historical": False,
                })

        if rmp_facts:
            for p in rmp_facts.get("professors", []):
                # Only matched professors get a clickable source; ambiguous/no-match
                # blocks are prompts to the model, not citable pages.
                if not p.get("matched"):
                    continue
                n += 1
                legacy = p.get("legacy_id")
                url = f"https://www.ratemyprofessors.com/professor/{legacy}" if legacy else ""
                sources.append({
                    "n": n,
                    "url": url,
                    "source": "rmp",
                    "title": f"{p.get('name', '')} — RateMyProfessors",
                    "course_code": None,
                    "term": None,
                    "historical": False,
                })

        for ch in chunks:
            n += 1
            meta = ch.get("metadata", {})
            url = meta.get("origin", "")
            key = url or ch.get("chunk_id", str(n))
            if key in seen:
                continue
            seen.add(key)
            sources.append({
                "n": n,
                "url": url,
                "source": meta.get("source", ""),
                "title": meta.get("title") or "",
                "document_type": meta.get("document_type") or None,
                "department": meta.get("department") or None,
                "historical": False,
            })
        return sources

    # -- orchestration ------------------------------------------------------

    def retrieve_with_route(self, route: dict,
                            audience: str = DEFAULT_AUDIENCE) -> tuple[list[dict], dict, dict, dict]:
        """Run graph/banner/rmp/vector retrieval for an already-computed route.
        Returns (chunks, graph_facts, banner_facts, rmp_facts).

        Split out of `retrieve()` so callers that already have a route (the
        planner call was made separately, e.g. to emit a status event first)
        don't need to re-run it or re-implement this gating inline.

        `audience` (undergrad / faculty / both) selects which collection(s) the
        vector path searches. Graph facts are corpus-independent (the course/
        program graph is shared), so audience doesn't affect them.

        Banner (live availability) is a gated live-data step: it fires only when the
        router named a course AND flagged an availability/section question — parallel
        to how graph retrieval fires on a course code. It's best-effort (returns {} on
        any failure), so audience/vector answers are never blocked by Banner being down.

        RMP (RateMyProfessors ratings) is a second gated live step that runs AFTER
        Banner so it can reuse Banner's instructor names: it fires on an explicit
        professor_query, or on a course-quality question (wants_rating) once Banner
        has resolved who teaches the course. Also best-effort.
        """
        graph_facts = {}
        if route["course_codes"] or route["program_query"]:
            graph_facts = self.graph_retrieve(
                route["course_codes"], route["program_query"], route["wants_outline"],
                route["completed_courses"],
            )
        banner_facts = {}
        if route["course_codes"] and route["wants_availability"]:
            banner_facts = banner_retrieve(
                route["course_codes"], route["term_season"], route["term_year"],
            )
        elif route["instructor_query"]:
            banner_facts = banner_instructor_retrieve(
                route["instructor_query"], route["term_season"], route["term_year"],
            )
        rmp_facts = self._rmp_retrieve_for(route, banner_facts)
        chunks = self.vector_retrieve(
            route["search_query"], audience=audience,
            topic_families=route["topic_families"], department=route["department"],
        )
        return chunks, graph_facts, banner_facts, rmp_facts

    def retrieve(self, question: str, history: list[dict],
                 audience: str = DEFAULT_AUDIENCE) -> tuple[dict, list[dict], dict, dict, dict]:
        """Route the question, run retrieval, and return
        (route, chunks, graph_facts, banner_facts, rmp_facts)."""
        route = self.rewrite_and_route(question, history, audience)
        chunks, graph_facts, banner_facts, rmp_facts = self.retrieve_with_route(route, audience)
        return route, chunks, graph_facts, banner_facts, rmp_facts

    @staticmethod
    def _instructor_names_from_banner(banner_facts: dict) -> list[str]:
        """Distinct instructor names Banner resolved, in first-seen order — the
        input to the course-case RMP chain. Covers both Banner shapes."""
        names: list[str] = []
        seen: set[str] = set()
        def _add(name: str | None) -> None:
            if name and name not in seen:
                seen.add(name)
                names.append(name)
        _add(banner_facts.get("instructor"))       # instructor-lookup shape
        for course in banner_facts.get("courses", []):
            for section in course.get("sections", []):
                for instr in section.get("instructors", []):
                    _add(instr.get("name"))
        return names

    @classmethod
    def _rmp_retrieve_for(cls, route: dict, banner_facts: dict) -> dict:
        """Decide the RMP names and fetch. Explicit professor_query wins; otherwise a
        course-quality question (wants_rating) uses whoever Banner just resolved."""
        if route.get("professor_query"):
            names = [route["professor_query"]]
        elif route.get("wants_rating") and banner_facts:
            names = cls._instructor_names_from_banner(banner_facts)
        else:
            names = []
        return rmp_retrieve(names) if names else {}

    @staticmethod
    def _n_graph_blocks(graph_facts: dict) -> int:
        return len(graph_facts.get("courses", [])) + (1 if graph_facts.get("program") else 0)

    def _assemble_context(self, chunks: list[dict], graph_facts: dict,
                          banner_facts: dict, rmp_facts: dict | None = None) -> tuple[str, int]:
        """Build the full numbered context (graph -> banner -> rmp -> vector) and
        return (context, n_non_vector_blocks). Shared by `ask()` and api.py's
        stream path."""
        graph_text = self._graph_context_text(graph_facts) if graph_facts else ""
        n_graph = self._n_graph_blocks(graph_facts)
        banner_text, n_banner = ("", 0)
        if banner_facts:
            banner_text, n_banner = self._banner_context_text(banner_facts, n_graph)
        rmp_text, n_rmp = ("", 0)
        if rmp_facts:
            rmp_text, n_rmp = self._rmp_context_text(rmp_facts, n_graph + n_banner)
        n_prefix = n_graph + n_banner + n_rmp
        prefix = "\n\n".join(t for t in (graph_text, banner_text, rmp_text) if t)
        context = self._build_context(chunks, prefix, n_prefix)
        return context, n_prefix

    def ask(self, question: str, history: list[dict] | None = None,
            audience: str = DEFAULT_AUDIENCE) -> dict:
        history = history or []
        route, chunks, graph_facts, banner_facts, rmp_facts = self.retrieve(
            question, history, audience)
        context, n_prefix_blocks = self._assemble_context(
            chunks, graph_facts, banner_facts, rmp_facts)
        answer, cited = self.answer(question, context, history)
        sources = _filter_cited_sources(
            self.format_sources(chunks, graph_facts, banner_facts, rmp_facts), cited)
        return {
            "answer": answer,
            "sources": sources,
            "search_query": route["search_query"],
            "n_chunks": len(sources),
        }


# ---------------------------------------------------------------------------
# Entry point (CLI smoke test only — see api.py for the HTTP server)
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="GeorgeBot RAG backend — CLI smoke test")
    parser.add_argument("--ask", required=True, help="one-shot CLI question")
    parser.add_argument("--audience", default=DEFAULT_AUDIENCE, choices=VALID_AUDIENCES,
                         help=f"corpus to search (default {DEFAULT_AUDIENCE})")
    parser.add_argument("--mode", default="quick", choices=("quick", "default"),
                         help="quick (default): flat retrieve->answer, 2 calls. "
                              "default: retrieve->answer->self-verify, with one "
                              "targeted re-fetch if the model flags a gap.")
    args = parser.parse_args()

    bot = GeorgeBot()
    if args.mode == "quick":
        result = bot.ask(args.ask, audience=args.audience)
    else:
        # Mirrors /api/chat's non-streaming "default" dispatch (api.py) —
        # imported lazily so plain `--mode quick` runs never need fastapi.
        from api import _default_verified_events, _drain_events

        route = bot.rewrite_and_route(args.ask, [], args.audience)
        events = _default_verified_events(bot, args.ask, [], args.audience, route)
        result = _drain_events(events)
        result.setdefault("search_query", route["search_query"])

    print(f"\nSearch query: {result.get('search_query', '')}")
    print(f"Chunks used:  {result.get('n_chunks', len(result.get('sources', [])))}\n")
    print("Answer:")
    print(result.get("answer") or result.get("error", "(no answer)"))
    print("\nSources:")
    for s in result.get("sources", []):
        tags = " ".join(t for t in [s.get("source", ""), s.get("course_code") or "",
                                     s.get("term") or "",
                                     "HISTORICAL" if s.get("historical") else ""] if t)
        print(f"  [{s.get('n', '?')}] {s.get('url', '')}  ({tags})")


if __name__ == "__main__":
    main()
