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

# v2.2 splits the corpus into two Chroma collections in one DB. The user picks
# which to search per request (undergrad / faculty / both) — see `audience`.
COLLECTION_NAMES = {
    "undergrad": "georgebot_v22_undergrad",
    "faculty": "georgebot_v22_faculty",
}
VALID_AUDIENCES = ("undergrad", "faculty", "both")
DEFAULT_AUDIENCE = "undergrad"
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

    def _program_facts(self, query: str, completed: list[str]) -> dict:
        matches = self.gs.search_programs(query)
        if not matches:
            return {"query": query, "matches": []}
        if len(matches) > 1:
            return {"query": query, "matches": matches, "ambiguous": True}
        m = matches[0]
        prog = self.gs.get_program(m["pid"])
        groups = self.gs.program_requirement_groups(m["pid"])
        specs = self.gs.program_specializations(m["pid"])
        result = {
            "query": query,
            "matches": matches,
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

        prompt = (
            f"{convo}You are the query router for a University of Victoria (UVic) "
            f"chatbot. Given the latest user question, produce a JSON object (and "
            f"nothing else) with these fields:\n\n"
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
            f"{json.dumps(self.departments)}\n\n"
            f"Use course_codes/program_query (graph route) only for structured "
            f"catalog facts: prerequisites, credits, cross-listings, course "
            f"descriptions, degree/program requirements, or outline content. For "
            f"anything else (services, policies, deadlines, general info) leave "
            f"course_codes empty and program_query null even if a course code is "
            f"mentioned in passing. completed_courses should be populated whenever "
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
            return {
                "search_query": data.get("search_query") or question,
                "completed_courses": [c.replace(" ", "").upper() for c in (data.get("completed_courses") or [])],
                "course_codes": [c.replace(" ", "").upper() for c in (data.get("course_codes") or [])],
                "program_query": data.get("program_query") or None,
                "wants_outline": bool(data.get("wants_outline")),
                "topic_families": [f for f in (data.get("topic_families") or []) if f in valid_families],
                "department": dept,
            }
        except Exception as e:
            print(f"  [rewrite_and_route] failed, falling back to vector-only: {e}", file=sys.stderr)
            return {"search_query": question, "course_codes": [], "program_query": None,
                    "wants_outline": False, "completed_courses": [], "topic_families": [], "department": None}

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
        "- Be concise and direct. Use plain text (no LaTeX)."
    )

    def _system_prompt_with_context(self, context: str) -> str:
        """Build the system message: instructions + the retrieved reference
        material, framed as system-supplied (NOT part of the user turn) so the
        model doesn't treat it as something the user typed or attached."""
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
        return self.SYSTEM_PROMPT + ref

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

    def answer(self, question: str, context: str, history: list[dict]) -> str:
        """MiniMax-M3, thinking disabled — reference material is supplied via the
        system prompt (`_system_prompt_with_context`), not the user turn."""
        messages = self._answer_messages(question, history)
        text = self._call_llm(messages, system=self._system_prompt_with_context(context),
                               max_tokens=ANSWER_MAX_TOKENS, thinking="disabled")
        return text or "Sorry, I couldn't generate an answer right now — please try again."

    def answer_stream(self, question: str, context: str, history: list[dict]):
        """Streamed MiniMax-M3 answer, thinking disabled — yielded token-by-token.

        The reference material rides in the system prompt (not the user turn),
        so the model treats it as system-supplied. Raw `content` deltas are
        still piped through `_iter_visible_deltas` as a safety net, which drops
        any `<think>...</think>` span even when a tag straddles chunk boundaries
        (`reasoning_split` isn't 100% reliable — see `_call_llm`). This lets the
        answer stream to the client incrementally instead of landing as one
        buffered chunk. Leading whitespace on the first visible piece is trimmed
        to match the non-streaming `answer()` behavior.
        """
        messages = self._answer_messages(question, history)
        full_messages = (
            [{"role": "system", "content": self._system_prompt_with_context(context)}] + messages
        )
        stream = self.llm.chat.completions.create(
            model=MINIMAX_MODEL,
            max_tokens=ANSWER_MAX_TOKENS,
            messages=full_messages,
            extra_body={"reasoning_split": True, "thinking": {"type": "disabled"}},
            stream=True,
        )

        def _content_deltas():
            for event in stream:
                delta = event.choices[0].delta
                text = getattr(delta, "content", None)
                if text:
                    yield text

        emitted = False
        for piece in _iter_visible_deltas(_content_deltas()):
            if not emitted:
                piece = piece.lstrip()
                if not piece:
                    continue
            emitted = True
            yield piece
        if not emitted:
            yield "Sorry, I couldn't generate an answer right now — please try again."

    # -- source formatting --------------------------------------------------

    @staticmethod
    def format_sources(chunks: list[dict], graph_facts: dict | None) -> list[dict]:
        """Deduplicate retrieved chunks + graph facts into a clean source list (by URL)."""
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

    def retrieve(self, question: str, history: list[dict],
                 audience: str = DEFAULT_AUDIENCE) -> tuple[dict, list[dict], dict]:
        """Route the question, run retrieval, and return (route, chunks, graph_facts).

        `audience` (undergrad / faculty / both) selects which collection(s) the
        vector path searches and which topic-family vocabulary the router uses.
        Graph facts are corpus-independent (the course/program graph is shared),
        so audience doesn't affect them.
        """
        route = self.rewrite_and_route(question, history, audience)
        graph_facts = {}
        if route["course_codes"] or route["program_query"]:
            graph_facts = self.graph_retrieve(
                route["course_codes"], route["program_query"], route["wants_outline"],
                route["completed_courses"],
            )
        chunks = self.vector_retrieve(
            route["search_query"], audience=audience,
            topic_families=route["topic_families"], department=route["department"],
        )
        return route, chunks, graph_facts

    def ask(self, question: str, history: list[dict] | None = None,
            audience: str = DEFAULT_AUDIENCE) -> dict:
        history = history or []
        route, chunks, graph_facts = self.retrieve(question, history, audience)
        graph_text = self._graph_context_text(graph_facts) if graph_facts else ""
        n_graph_blocks = len(graph_facts.get("courses", [])) + (1 if graph_facts.get("program") else 0)
        context = self._build_context(chunks, graph_text, n_graph_blocks)
        answer = self.answer(question, context, history)
        return {
            "answer": answer,
            "sources": self.format_sources(chunks, graph_facts),
            "search_query": route["search_query"],
            "n_chunks": len(chunks) + n_graph_blocks,
        }


# ---------------------------------------------------------------------------
# Entry point (CLI smoke test only — see api.py for the HTTP server)
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="GeorgeBot RAG backend — CLI smoke test")
    parser.add_argument("--ask", required=True, help="one-shot CLI question")
    parser.add_argument("--audience", default=DEFAULT_AUDIENCE, choices=VALID_AUDIENCES,
                         help=f"corpus to search (default {DEFAULT_AUDIENCE})")
    args = parser.parse_args()

    bot = GeorgeBot()
    result = bot.ask(args.ask, audience=args.audience)
    print(f"\nSearch query: {result['search_query']}")
    print(f"Chunks used:  {result['n_chunks']}\n")
    print("Answer:")
    print(result["answer"])
    print("\nSources:")
    for s in result["sources"]:
        tags = " ".join(t for t in [s.get("source", ""), s.get("course_code") or "",
                                     s.get("term") or "",
                                     "HISTORICAL" if s.get("historical") else ""] if t)
        print(f"  [{s['n']}] {s['url']}  ({tags})")


if __name__ == "__main__":
    main()
