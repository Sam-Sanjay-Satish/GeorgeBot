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
import hybrid_retrieve   # entity-routed hybrid vector retrieval (HYBRID_RETRIEVAL_ENABLED, default off)

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
N_CONTEXT = 4           # distinct chunks the FILTERED pass may contribute
N_UNFILTERED_BACKFILL = 3  # distinct chunks the UNFILTERED pass always adds on top
                            # (so up to N_CONTEXT + this reach the answer model).
                            # See `vector_retrieve`: the unfiltered pass used to run
                            # only when the filtered pass came up short, which let a
                            # single wrong `department` prediction hide the globally
                            # best chunks in the corpus (measured 2026-08-04).
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

# Gated the same way as rewrite_and_route's named_entities prompt field
# (chatbot.py, "hybrid retrieval" changes): SYSTEM_PROMPT is a class-level
# constant built once at import time, and hybrid_retrieve.HYBRID_RETRIEVAL_ENABLED
# is already resolved by then (hybrid_retrieve is imported above), so this is
# just as byte-identical-when-off as the router prompt change. Framed as a
# SOFT signal on purpose, not a trust hierarchy by retrieval method — see the
# design discussion this replaced: a chunk found by only one arm can still be
# exactly right, and a chunk multiple arms agree on can still be irrelevant,
# so the rule tells the model to keep judging content first.
_AGREEMENT_TAG_RULE = (
    "- Some vector-search chunks carry an `agreement=N/M` tag, meaning N of M "
    "independent retrieval signals used this turn (a reverse-HyDE "
    "question-similarity search, a direct chunk-text similarity search, and — "
    "only when a specific named entity was detected — a keyword search) "
    "independently surfaced that chunk. Treat higher agreement as a mild "
    "extra signal that a chunk is on-topic, never as a substitute for judging "
    "it — a chunk multiple signals agree on can still be irrelevant, and a "
    "chunk only one signal found (or with no tag at all) can still be exactly "
    "right. Judge every chunk primarily by whether its own content actually "
    "answers the question.\n"
) if hybrid_retrieve.HYBRID_RETRIEVAL_ENABLED else ""

# Router-output hard bounds (2026-08-01, audit issue 4 — outbound amplification).
# course_codes / completed_courses / the name queries come from the LLM router at
# whatever length the model emits. Each course code fans out into ~10 live Banner
# HTTP calls (handshake + searchResults + per-CRN faculty lookups), so a question
# engineered to make the router list dozens of codes would turn one inbound
# request into hundreds of outbound requests to banner.uvic.ca. Cap the lists and
# validate each code against the real UVic shape at parse time, so arbitrary
# router output never reaches banner.py's URL params. banner.py / rmp.py carry
# matching boundary caps + outbound concurrency limits of their own.
MAX_COURSE_CODES = 5        # courses per question — plenty (drives graph + Banner fan-out)
MAX_COMPLETED_COURSES = 20  # graph-only (never hits Banner), so roomier — a real course list
MAX_RMP_NAMES = 5           # professors chained from Banner into one RMP lookup
MAX_NAME_QUERY_LEN = 80     # instructor_query / professor_query length bound
# Normalized shape: "CSC225", "CSC225A", and the hyphenated Education subjects
# ("ED-D301", "ED-P420"). The hyphen branch is NOT optional polish — validated
# against course_graph.pkl, 52 of the 4015 real course codes are ED-D/ED-P, and
# a pattern without it silently drops every one of them from graph + Banner
# retrieval. Re-check against the graph before tightening this.
COURSE_CODE_RE = re.compile(r"^[A-Z]{2,4}(?:-[A-Z]{1,2})?\d{3}[A-Z]?$")


def _clean_course_codes(raw, cap: int) -> list[str]:
    """Normalize ("csc 225" -> "CSC225"), validate against COURSE_CODE_RE, dedupe
    (order-preserving), and cap a router-emitted course-code list. Non-strings and
    anything not shaped like a real UVic course code are dropped silently — a bad
    entry degrades that code to vector-only retrieval, never an error."""
    out: list[str] = []
    for c in raw or []:
        if not isinstance(c, str):
            continue
        code = c.replace(" ", "").upper()
        if COURSE_CODE_RE.fullmatch(code) and code not in out:
            out.append(code)
            if len(out) >= cap:
                break
    return out


def _clean_name_query(raw) -> str | None:
    """Length-bound a router-emitted single-name field (instructor_query /
    professor_query). None for anything empty or non-string."""
    if not isinstance(raw, str):
        return None
    return raw.strip()[:MAX_NAME_QUERY_LEN].strip() or None

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

        # None when HYBRID_RETRIEVAL_ENABLED is unset/false — zero file I/O,
        # zero extra boot time in the default state. See hybrid_retrieve.py.
        self.hybrid = hybrid_retrieve.maybe_load()

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

        Soft filter with an UNCONDITIONAL unfiltered pass: if topic_families/
        department were predicted by the router, run a filtered pass first so
        on-topic chunks rank first; then *always* add up to
        `N_UNFILTERED_BACKFILL` more from an unfiltered pass (deduped).

        ⚠️ The unfiltered pass used to run only when the filtered pass returned
        fewer than `n`, and that was a real wrong-answer bug, not just lost
        recall (measured 2026-08-04). A program spanning two departments carries
        exactly ONE `department` tag, so whichever department the router names,
        the other one's documents are filtered out — and when the filtered pass
        still returns a full `n`, the backfill never fires and they are never
        seen. Concretely: "geog and environmental studies general degree" routed
        to department="Environmental Studies", while the Geography and
        Environmental Studies Double Major worksheets are tagged
        department="Geography". Those worksheets are the three globally-nearest
        chunks in the corpus for that query (d=0.446/0.460/0.470) and were
        replaced by chunks ~0.05 worse, so the answer denied that the program
        exists. This falsified the old claim that a wrong/narrow prediction
        "degrades to no filtering benefit, not a wrong answer".

        Why the filtered budget stays at `n` rather than shrinking to make room:
        capping it lower (3+3 was measured) evicts the filtered pass's 4th chunk,
        which cost real content on 5 of 10 test queries (e.g. the tuition query
        lost "International Undergraduate Tuition Fees"). Keeping `n` filtered
        and adding `N_UNFILTERED_BACKFILL` on top makes the result a strict
        superset of the old behaviour — zero regression — at ~1.75x context.

        Both passes apply MAX_CHUNK_DISTANCE, so a genuinely off-topic query can
        return fewer chunks — even zero — rather than being padded out just to
        hit a count.
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

        # With no filter the unfiltered pass IS the retrieval, so it supplies the
        # full `n`. With a filter it always contributes N_UNFILTERED_BACKFILL on
        # top, and still tops up to `n` if the filtered pass came up short.
        want = max(N_UNFILTERED_BACKFILL, n - len(results)) if has_filter else n
        exclude = {r["chunk_id"] for r in results}
        unfiltered = [self._query_candidates(self.collections[a], emb, None) for a in auds]
        results += self._merge_collapse(unfiltered, want, exclude=exclude)
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

    # Words a student naturally attaches to a program name that are not part of
    # any program's title or credential. `search_programs` is an all-tokens-AND
    # matcher, so each of these single-handedly forces a zero match: only 7 of
    # 280 programs have a word starting with "degree" (the Law joint/double
    # degrees and the post-degree BEds), so "geography general degree" matched
    # nothing while "geography general" matched the two correct programs.
    # Stripping them is query-side only and loses nothing — none of them
    # discriminates between programs. Deliberately NOT here: "general", "major",
    # "minor", "honours", "combined", "arts", "science" (all real credential
    # words), "studies" (a real title word — Environmental Studies), and "and"
    # (a real title word AND the conjunction the split below keys on).
    _PROGRAM_FILLER = {
        "degree", "degrees", "program", "programs", "programme", "programmes",
        "the", "a", "an", "in", "of", "for", "at", "uvic",
    }
    # Credential-ish words that qualify a subject rather than name one. When a
    # query names two subjects joined by "and", a trailing qualifier applies to
    # both ("geography and environmental studies general" = the General program
    # in each), so it gets copied onto whichever part lacks one.
    _PROGRAM_QUALIFIERS = {
        "general", "major", "minor", "honours", "honors", "combined",
        "bachelor", "arts", "science", "sciences", "ba", "bsc",
        "certificate", "diploma",
    }

    @staticmethod
    def _norm_tokens(s: str) -> list[str]:
        return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).split()

    def _relaxed_program_queries(self, query: str) -> tuple[list[str], list[str]]:
        """Rescue queries to try after a strict `search_programs` came back empty.

        `search_programs` requires EVERY token to match one program node, which
        silently returns nothing in two common cases that are not "the program
        doesn't exist":

        1. the query carries a generic word no title contains ("...general
           degree" -> `_PROGRAM_FILLER`); and
        2. the query names TWO programs joined by "and". A UVic General degree
           is built from two areas and the catalog lists one program per
           subject, so "geog and environmental studies general degree" can
           never match a single node — geography matches 9 programs,
           environmental matches 2, and the intersection is empty.

        Returns (queries_that_matched, parts_that_matched_nothing). Only reached
        when the strict search already failed, so the conjunction split can't
        break a real title containing "and" ("Physical Geography and Earth and
        Ocean Sciences" matches strictly and never gets here).
        """
        words = self._norm_tokens(query)
        kept = [w for w in words if w not in self._PROGRAM_FILLER]
        if not kept:
            return [], []
        # (a) filler-only relaxation — fixes "geography general degree".
        cleaned = " ".join(kept)
        if cleaned != " ".join(words) and self.gs.search_programs(cleaned):
            return [cleaned], []
        # (b) conjunction split — fixes the two-subject case.
        raw_parts = re.split(r"\s+and\s+|[&,+/]", (query or "").lower())
        if len(raw_parts) < 2:
            return [], []
        quals = [w for w in kept if w in self._PROGRAM_QUALIFIERS]
        found, missed = [], []
        for rp in raw_parts:
            part = [w for w in self._norm_tokens(rp) if w not in self._PROGRAM_FILLER]
            if not part:
                continue
            if not any(w in self._PROGRAM_QUALIFIERS for w in part):
                part = part + quals
            cand = " ".join(part)
            (found if self.gs.search_programs(cand) else missed).append(cand)
        return found, missed

    def _program_facts(self, query: str, completed: list[str]) -> dict:
        """Resolve `query` to program facts, retrying with relaxed matching when
        the strict name match finds nothing.

        A strict hit (including a genuine ambiguity) is returned unchanged. Only
        a zero match falls through to `_relaxed_program_queries`; if that finds
        two or more programs the result carries a `parts` list — one entry per
        resolved program, each in this same single-program shape — which every
        renderer walks. See `_n_program_blocks` for the numbering contract."""
        res = self._resolve_program(query, completed)
        if res.get("program") or res.get("ambiguous"):
            return res
        found, missed = self._relaxed_program_queries(query)
        sub = [self._resolve_program(q, completed) for q in found]
        sub = [s for s in sub if s.get("program") or s.get("ambiguous")]
        if not sub:
            return res
        if len(sub) == 1:
            return {**sub[0], "relaxed_from": query, "unmatched_parts": missed}
        return {"query": query, "relaxed_from": query,
                "unmatched_parts": missed, "parts": sub}

    def _resolve_program(self, query: str, completed: list[str]) -> dict:
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

    def _course_suggestions(self, code: str, limit: int = 4) -> list[str]:
        """Catalog codes that look like the one asked for — in practice the
        letter-suffixed variants of a base code (PSYC100 -> PSYC100A, PSYC100B).

        This is not a nicety: ~225 UVic course bases exist ONLY as A/B variants
        (PSYC 100, SOCI 100, GEOG 101, LING 100, SPAN 100, BIOL 150/190, most
        HSTR and MUS courses), and students type the base form."""
        try:
            nodes = self.gs.cg.nodes
        except AttributeError:
            return []
        return sorted(
            n for n in nodes
            if isinstance(n, str) and n != code and n.startswith(code)
            and len(n) == len(code) + 1
        )[:limit]

    def graph_retrieve(self, course_codes: list[str], program_query: str | None,
                        want_outline: bool, completed_courses: list[str]) -> dict:
        courses = []
        not_found = []
        for code in course_codes:
            f = self._course_facts(code, want_outline, completed_courses)
            if f:
                courses.append(f)
            else:
                # Record the miss instead of dropping it silently. A dropped code
                # left the answer model with no idea the lookup had happened, so it
                # filled the gap from its own knowledge and stated invented catalog
                # facts with full confidence (measured: "PSYC 100" produced a
                # fabricated prereq relationship between PSYC 100A and 100B).
                not_found.append({"code": code, "suggestions": self._course_suggestions(code)})
        program = self._program_facts(program_query, completed_courses) if program_query else None
        return {"courses": courses, "not_found": not_found, "program": program}

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
            #
            # `prereq_chain` walks every prereq edge, and edges are emitted for each
            # course leaf INCLUDING the ones inside "complete 1 of" groups — so this
            # is a union of mutually exclusive alternatives, not a conjunction. The
            # old label ("also requires (indirectly)") read as a checklist and the
            # model relayed it as one: CSC 225 came out as needing MATH 100, 101,
            # 102, 109, 110, 120 AND 151, where a student needs at most two of them.
            deeper = [x for x in c.get("prereq_chain", []) if x not in set(c.get("prereq_courses", []))]
            if deeper:
                lines.append(
                    f"Courses appearing further back in {c['code']}'s prerequisite "
                    f"chain: {', '.join(deeper)}. "
                    f"IMPORTANT: this is a flat union of every course reachable "
                    f"through the chain, INCLUDING alternatives that satisfy the same "
                    f"'complete 1 of' requirement (e.g. several different calculus "
                    f"courses). A student needs only ONE course from each such group, "
                    f"NOT all of these. Do not present this list as required "
                    f"coursework, as a checklist, or as a sequence — the authoritative "
                    f"requirements are the prerequisite text above."
                )
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
                # `get_alternatives` aggregates co-membership in ANY "1 of" group
                # anywhere in the catalog, which is much weaker than equivalence —
                # it routinely returns the course's own prerequisite or the next
                # course in the sequence (CHEM101 -> CHEM102, ASTR101, GEOG103;
                # MATH100 -> MATH101; CSC115 -> CSC110, CSC226). The old
                # "interchangeable with" label asserted substitutability the data
                # does not support, and being tagged source=kuali the model treated
                # it as authoritative and passed it on.
                lines.append(
                    f"Courses that appear alongside {c['code']} in some "
                    f"\"complete 1 of\" prerequisite group somewhere in the catalog: "
                    f"{', '.join(c['alternatives'])}. "
                    f"IMPORTANT: this is derived automatically from co-occurrence and "
                    f"does NOT mean these are equivalent to {c['code']} or accepted in "
                    f"its place — the list can include {c['code']}'s own prerequisite "
                    f"or a later course in the same sequence. Never tell the student "
                    f"one of these substitutes for {c['code']}; for substitution or "
                    f"transfer-credit questions, refer them to the department or "
                    f"academic advising."
                )
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
            blocks.append("\n".join(lines))
            # The outline is its own numbered block, not a trailer on the course
            # block. `format_sources` has always emitted a separate `heat` source
            # for it, so folding it into the course block here made the context's
            # [n] sequence one short of the source list from that point on — every
            # <<CITED_SOURCES>> number after an outline resolved to the wrong
            # source. Numbering it here is what keeps the two sequences aligned
            # (and lets the model cite the outline specifically).
            if c.get("outline"):
                i += 1
                o = c["outline"]
                term = o.get("term", "unknown term")
                blocks.append(
                    f"[{i}] source=heat course={c['code']} term={term}\n"
                    f"URL: {o.get('url') or c.get('url', 'n/a')}\n"
                    f"HISTORICAL course outline for {c['code']} (term {term}) — a past-term "
                    f"snapshot, not current. Name the term whenever you use it, and never "
                    f"present its grading, instructor, or schedule as this term's:\n"
                    f"{o.get('text', '')[:4000]}"
                )

        for nf in graph_facts.get("not_found", []):
            i += 1
            code = nf["code"]
            suggestions = nf.get("suggestions") or []
            if suggestions:
                tail = (
                    f"The catalog does list {', '.join(suggestions)} — at UVic this "
                    f"course is split into lettered halves, so that is almost "
                    f"certainly what the user means. Answer about those, and say "
                    f"which ones you're using."
                )
            else:
                tail = (
                    "No similarly-named course exists either. Tell the user you "
                    "couldn't find that course code and suggest they double-check it "
                    "in the UVic Academic Calendar. Do NOT guess at other course "
                    "codes they might have meant — a plausible-looking code you "
                    "made up is worse than no suggestion."
                )
            blocks.append(
                f"[{i}] source=kuali course_lookup=\"{code}\"\n"
                f"NOT FOUND: \"{code}\" is not a course code in the UVic Academic "
                f"Calendar. {tail} "
                f"Do NOT state prerequisites, credits, descriptions, or any other "
                f"catalog fact for \"{code}\" itself — there is no record to base "
                f"them on, and inventing them is the specific failure this block "
                f"exists to prevent."
            )

        program = graph_facts.get("program")
        if program:
            for one in (program.get("parts") or [program]):
                i += 1
                blocks.append(GeorgeBot._program_block(i, one, program))
        return "\n\n".join(blocks)

    @staticmethod
    def _relaxed_note(one: dict, parent: dict) -> str:
        """Explain to the model how a program block was reached when the user's
        exact wording didn't match. Only fires on the relaxed path."""
        original = parent.get("relaxed_from")
        if not original:
            return ""
        # ⚠️ Do NOT restore the earlier phrasing "the catalog lists one program
        # per subject" — it is FALSE and it caused a wrong answer. The program
        # graph has no Double Major programs at all (0 of 280; see the known-issues
        # entry in CLAUDE.md), so "Geography and Environmental Studies Double Major
        # (BA)" is real, has a published worksheet, and is simply absent here.
        # Stating the graph's shape as a fact about UVic, under the authoritative
        # source=kuali tag, is what let the model deny the program existed.
        note = (f" NOTE: the user asked about \"{original}\", which matches no program "
                f"in this catalog index by name, so it was resolved by subject as "
                f"\"{one.get('query')}\". This index is known to be INCOMPLETE for "
                f"programs that span two subjects (double majors in particular are "
                f"missing from it entirely), so treat the block below as related "
                f"background, NOT as proof of what UVic offers. If any other "
                f"reference material here describes a combined, double-major or "
                f"joint program covering what the user asked about, PREFER IT over "
                f"this block and answer from it.")
        # A multi-part split means the user named SEVERAL subjects in one
        # request — the parts are the halves of one question, not a menu. The
        # first wording here ("answer about this program and make clear which
        # one it is") made the model lead with "there isn't a single program
        # called that" and then ask the user to choose ONE, which is the
        # original wrong answer in a softer form: someone naming two subjects
        # for a General degree wants BOTH, and each part is already a real
        # program. It also volunteered an invented "Combined Major such as
        # Geography and Environmental Studies" to bridge the two, so the
        # anti-speculation clause is load-bearing too.
        n_parts = len(parent.get("parts") or [])
        if n_parts > 1:
            note += (f" The request names {n_parts} subjects and each was resolved to its "
                     f"own single-subject program, numbered separately here. If no other "
                     f"material describes a real combined/double-major program covering "
                     f"them, treat these as COMPONENTS OF ONE REQUEST, not alternatives: "
                     f"cover every one of them rather than asking the user to pick between "
                     f"them. Never state or imply that UVic has no program combining these "
                     f"subjects — this index cannot support that claim. Only ask a "
                     f"clarifying question for a part that is itself ambiguous (e.g. BA vs "
                     f"BSc), and ask it about that part alone. Do NOT invent or speculate "
                     f"about a combined/joint program not evidenced in the material.")
        else:
            note += " Otherwise answer about this program and make clear which one it is."
        missed = parent.get("unmatched_parts") or []
        if missed:
            note += (f" Part of the request matched nothing by name and is NOT covered "
                     f"below: {'; '.join(missed)} — do not claim those don't exist; ask "
                     f"the user for the exact calendar name if they need them.")
        return note

    @staticmethod
    def _program_block(i: int, program: dict, parent: dict | None = None) -> str:
        """Render ONE numbered program block. Called once per entry in a relaxed
        multi-program `parts` list, or once for the single-program result."""
        parent = parent if parent is not None else program
        if program.get("ambiguous"):
            names = "; ".join(f"{m['title']} ({m['code']}, {m.get('credential', '')})"
                               for m in program["matches"])
            return (
                f"[{i}] source=kuali program_search=\"{program['query']}\"\n"
                f"Multiple matching programs found — ask the user to clarify which one: {names}"
                f"{GeorgeBot._relaxed_note(program, parent)}"
            )
        if program.get("program"):
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
            relaxed = GeorgeBot._relaxed_note(program, parent)
            if relaxed:
                lines.append(relaxed.strip())
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
            return "\n".join(lines)
        # No match. This block used to read only "No matching program found in the
        # calendar." — a bare negative under source=kuali, which the answer rules
        # call the authoritative catalog. The model read it as proof of
        # non-existence and told a student the degree they were enrolled in
        # doesn't exist at UVic ("geog and environmental studies general degree",
        # which the catalog lists as two separate General programs). The lookup is
        # a NAME match over program titles only, so an empty result carries no
        # information about whether the program exists — say so explicitly, the
        # same way the course NOT FOUND block does.
        return (
            f"[{i}] source=kuali program_search=\"{program['query']}\"\n"
            f"No catalog program matches that NAME. This is a name-matching result "
            f"ONLY: it means nothing in the catalog is titled that way. It is NOT "
            f"evidence that the program does not exist, is not offered, or is not "
            f"in the calendar. Do NOT tell the user the program/degree/combination "
            f"doesn't exist or isn't offered at UVic, do NOT cite this block as "
            f"support for such a claim, and do NOT describe this lookup or its "
            f"result to the user. Students routinely name a program differently "
            f"from the calendar, and this index is known to be INCOMPLETE for "
            f"programs spanning two subjects (double majors are missing from it "
            f"entirely) — so a combination the user names may well exist and be "
            f"absent here. If any other reference material describes the program, "
            f"answer from that. Otherwise ask the user for the exact program name "
            f"as it appears in the academic calendar (or which individual subjects "
            f"they mean), and answer whatever else the question asks."
        )

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
        # Status must come from the SEAT COUNT, not from Banner's `openSection`.
        # They are independent fields and disagree constantly: `openSection` means
        # "the department has this section open for registration", NOT "it has free
        # seats". CSC 320 A01 (Fall 2026) is seats=0 / openSection=True, as is most
        # of CSC 110 — all of which rendered as "0 of 125 seats open — OPEN", which
        # is precisely the wrong word for "are there seats?". `openSection=False`
        # still means closed outright (CSC 320 T02), so all three signals matter.
        seats_left = s.get("seats_available")
        wait_left = s.get("wait_available") or 0
        if not s.get("open"):
            status = "CLOSED — not open for registration"
        elif isinstance(seats_left, int) and seats_left > 0:
            status = "SEATS AVAILABLE"
        elif wait_left > 0:
            status = "FULL — no seats left, but waitlist space remains"
        else:
            status = "FULL — no seats and no waitlist space"
        parts = [f"  {s['section']} ({s.get('schedule_type', '')}): {seat} — {status}."]
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

        # Set when a season/year hint named a term Banner no longer serves live data
        # for (a past "View Only" term); the data below is the current/upcoming term
        # instead, and the model must not pass it off as the term that was named.
        asked_for = banner_facts.get("requested_term_label")
        substitution = ""
        if asked_for:
            substitution = (
                f"\nNOTE: {asked_for} was named for this lookup, but it is a past term "
                f"with no live registration data, so these are {term_label} numbers "
                f"instead. Say which term these figures are for. If the user genuinely "
                f"meant {asked_for}, tell them seat counts for a finished term aren't "
                f"available and are not meaningful."
            )

        blocks = []
        i = offset
        for course in banner_facts.get("courses", []):
            i += 1
            code = course["code"]
            lines = [
                f"[{i}] source=banner course={code} term={term_label}{substitution}",
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

        # Gated behind the flag so the prompt sent to MiniMax is BYTE-IDENTICAL
        # to today's whenever hybrid retrieval is disabled — no risk to the
        # existing, well-tested router behavior in the default (off) state.
        named_entities_field = ""
        if hybrid_retrieve.HYBRID_RETRIEVAL_ENABLED:
            named_entities_field = (
                f'  "named_entities": array of genuinely distinctive proper nouns in '
                f"search_query worth a targeted keyword search — the name of a "
                f'specific facility/building/organization/club/service (e.g. "CARSA", '
                f'"UVSS", "SUB"), a person\'s name, or a specific named external '
                f'institution being compared/referenced (e.g. "SFU", "UBC"). Do NOT '
                f"include: generic academic/grading jargon even if capitalized (GPA, "
                f"WE, N, CGPA, SAP); course-code subject prefixes (CSC, ENGR, MATH, "
                f'SENG, PSYC, STAT, ECE, PHYS); "UVic"/"University of Victoria" '
                f"itself; generic program/department names (Computer Science, "
                f"Engineering, Statistics) unless the question is fundamentally an "
                f"identity lookup for that one unit and nothing else distinguishes "
                f"it; generic role words alone (professor, student, staff) without "
                f"an attached proper name. Empty array if none qualify.\n"
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
            f'  "needs_retrieval": false ONLY when the turn is pure conversational '
            f"filler that no UVic reference material could inform — a greeting "
            f'("hi", "whats up"), thanks, goodbye, an acknowledgement ("ok", "got '
            f'it"), or small talk about you the assistant. TRUE for everything '
            f"else, including vague, off-topic, or poorly-worded questions — when "
            f"in doubt, true.\n"
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
            f"{named_entities_field}"
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
                # Default TRUE on a missing/garbled field: skipping retrieval on a
                # real question is a far worse failure than retrieving for a
                # greeting, so only an explicit `false` turns it off.
                "needs_retrieval": data.get("needs_retrieval") is not False,
                "completed_courses": _clean_course_codes(data.get("completed_courses"), MAX_COMPLETED_COURSES),
                "course_codes": _clean_course_codes(data.get("course_codes"), MAX_COURSE_CODES),
                "program_query": data.get("program_query") or None,
                "wants_outline": bool(data.get("wants_outline")),
                "wants_availability": bool(data.get("wants_availability")),
                "instructor_query": _clean_name_query(data.get("instructor_query")),
                "professor_query": _clean_name_query(data.get("professor_query")),
                "wants_rating": bool(data.get("wants_rating")),
                "term_season": season,
                "term_year": year,
                "topic_families": [f for f in (data.get("topic_families") or []) if f in valid_families],
                "department": dept,
                # Only populated when HYBRID_RETRIEVAL_ENABLED (the field isn't in
                # the prompt otherwise, so the model has nothing to return here).
                "named_entities": [e for e in (data.get("named_entities") or []) if isinstance(e, str)],
            }
        except Exception as e:
            print(f"  [rewrite_and_route] failed, falling back to vector-only: {e}", file=sys.stderr)
            # needs_retrieval TRUE here on purpose: a router failure must degrade to
            # vector-only, never to "no retrieval at all".
            return {"search_query": question, "needs_retrieval": True,
                    "course_codes": [], "program_query": None,
                    "wants_outline": False, "wants_availability": False, "instructor_query": None,
                    "professor_query": None, "wants_rating": False,
                    "term_season": None, "term_year": None, "completed_courses": [],
                    "topic_families": [], "department": None,
                    # Present in both branches so downstream code can safely use
                    # route.get("named_entities", []) OR route["named_entities"] --
                    # but .get() is still the documented convention (see
                    # hybrid_retrieve.retrieve()) in case a field is ever added
                    # here without remembering both branches again.
                    "named_entities": []}

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
            if meta.get("agreement"):
                # Only present on hybrid-retrieval chunks (HYBRID_RETRIEVAL_ENABLED) --
                # "N of M retrieval arms independently surfaced this chunk". See the
                # AGREEMENT rule in _ANSWER_RULES_HEAD for how the model should read it.
                tags.append(f"agreement={meta['agreement']}")
            header = f"[{i}] {' | '.join(tags)}\nURL: {meta.get('origin', 'n/a')}"
            blocks.append(f"{header}\n{ch.get('text', '')}")
        text = "\n\n".join(blocks)
        if graph_text:
            text = f"{graph_text}\n\n{text}" if text else graph_text
        return text

    # The answer-model contract is assembled from three pieces so the two modes
    # can differ in their style/scope section ONLY. Quick mode must be exactly
    # as accurate and as grounded as default mode — it is narrower in what it
    # volunteers, never looser about facts — so every accuracy rule lives in
    # the shared head and applies to both by construction. Don't fork these
    # blocks; edit the head and both modes move together.
    _ANSWER_RULES_HEAD = (
        "You are GeorgeBot, a helpful assistant that answers questions about the "
        "University of Victoria (UVic) for students and staff.\n\n"
        "WHO YOU CURRENTLY SERVE\n"
        "You cover two audiences: UNDERGRADUATE students, and FACULTY/STAFF (HR, "
        "governance, research administration). The user chooses which of those to "
        "search. There is NO GRADUATE-STUDIES COVERAGE YET — graduate program "
        "requirements, graduate admissions, funding and awards, supervision, "
        "candidacy, and thesis/dissertation regulations are all out of scope for "
        "now.\n"
        "This matters because graduate regulations at UVic genuinely differ from "
        "undergraduate ones. Everything you retrieve is undergraduate- or "
        "staff-oriented, so NEVER present an undergraduate regulation, deadline, "
        "fee, GPA or academic-standing rule, or program requirement as though it "
        "applies to a graduate student. That is a wrong answer, not a close "
        "approximation.\n"
        "When the question is clearly about graduate study — or the user says "
        "they are a grad, master's, or PhD student — say plainly that graduate "
        "content isn't supported yet, answer only the parts that genuinely apply "
        "to everyone on campus, and point them to the Faculty of Graduate Studies "
        "or their supervisor / graduate adviser for the rest. Say it as a current "
        "gap ('not supported yet'), not as a refusal, and never guess at a "
        "graduate rule to fill it.\n"
        "Do not over-correct: plenty of questions are audience-independent "
        "(library, parking, transit, counselling, IT, recreation, campus "
        "services). If a graduate student asks one of those, just answer it "
        "normally — there is no need to mention the limitation at all.\n\n"
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
        "- ARITHMETIC: you may calculate only when EVERY input is either stated in "
        "the material or given by the user. Never fill a missing input from your "
        "own general knowledge and then present the result as a fact — the answer "
        "inherits the invented number and looks sourced. This bites hardest on "
        "cost: UVic charges tuition PER UNIT of course weight, and UVic uses a "
        "UNIT system, not a US-style credit-hour system. Do not assume a course is "
        "3 credits, do not assume a full-time load is 15 credits or 5 courses a "
        "semester, and never multiply a per-unit rate by a course load you were "
        "not given. If someone asks what tuition costs and no load is stated, give "
        "the per-unit rate, say plainly that the total depends on how many units "
        "they take (plus their program and ancillary fees), and point them to "
        "UVic's tuition estimator — do NOT manufacture a 'typical term' or "
        "'typical year' total. If the user does give you a load, you may compute "
        "it, but show the arithmetic (units x rate) so they can check it.\n"
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
        f"{_AGREEMENT_TAG_RULE}"
        "- Course outlines tagged HISTORICAL are past-term snapshots only. When "
        "you rely on one, always name the specific term, and never present "
        "historical grading weights, instructors, or schedules as current — for "
        "current course details, direct the user to Brightspace or the "
        "department.\n"
        "- DATES: today's date is given to you below. Everything except "
        "source=banner blocks is a static snapshot of UVic pages captured at an "
        "unknown earlier time, so any date in it may already have passed. Before "
        "you call something 'current', 'this term', 'upcoming', 'the next one', "
        "or 'now', check it against today's date. If it has already passed, do "
        "NOT present it as still ahead — give the fact together with the date it "
        "applied to, and tell the user to confirm the current one. Keep the "
        "specifics when you do that: '$25 as of August 2025 — worth confirming "
        "the current amount' is far more useful than dropping the number and "
        "saying only that a fee exists. Never manufacture precision you weren't "
        "given, either ('mid-March' stays 'mid-March', it does not become 'March "
        "15'). This covers event and info-session dates, application and "
        "registration deadlines, published fees and rates, and terms of office or "
        "appointments with an end date (an appointment whose stated end date is in "
        "the past must not be described as who currently holds the role). For "
        "something that recurs on the same rough schedule every year, you may "
        "describe the usual pattern, but name the cycle you took it from instead "
        "of implying it is this year's confirmed date. Flagging an elapsed date is "
        "a CORRECTNESS requirement, not a caveat, a tip, or an optional extra — it "
        "applies in full no matter how short the answer is supposed to be, and any "
        "instruction below to be brief or to add nothing beyond what was asked "
        "never licenses dropping it. Stating a passed date in the present tense is "
        "simply a wrong answer.\n"
        "- When the material states 'Given completed courses [...]: prerequisites "
        "satisfied = ...' or 'outstanding requirements by group', that "
        "evaluation was computed programmatically from the courses the user said "
        "they've taken — trust it over your own tally, but flag any listed "
        "non-course requirements (year standing, GPA, permission, etc.) since "
        "those can't be auto-verified from a course list.\n"
        "- If a program search returned multiple candidates, ask the user to "
        "clarify which one they mean rather than guessing.\n"
        "- A program search that reports no NAME match is not evidence that the "
        "program does not exist — it only means nothing in the catalog is titled "
        "that way. Never tell a user that a program, degree, major, minor, or "
        "combination of subjects isn't offered at UVic, doesn't exist, or isn't "
        "in the calendar on that basis, and never invent corroboration for such "
        "a claim from other material. Students often name a program differently "
        "from the calendar, and a degree spanning two subjects is listed as one "
        "program per subject. Ask for the exact calendar name instead.\n\n"
    )

    _CITED_SOURCES_RULES = (
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

    _DEFAULT_STYLE = (
        "STYLE\n"
        "- Be concise and direct. Use plain text (no LaTeX).\n\n"
    )

    # Quick mode's style/scope section. The goal is a *smaller* answer, not a
    # weaker one: it constrains what the model volunteers, never what it has
    # to get right. The carve-out in the last bullet matters — the ACCURACY
    # rules in the shared head mandate certain qualifications (naming a
    # Banner/HISTORICAL term, RMP attribution + sample size), and without an
    # explicit exemption a hard "nothing extra" rule reads as license to drop
    # them.
    _QUICK_STYLE = (
        "SCOPE AND STYLE\n"
        "- Answer exactly what was asked, and nothing else. This is the most "
        "important rule about how you write.\n"
        "- No preamble, no restating or rephrasing the question, no summary or "
        "closing line, no sign-off, no offer of further help.\n"
        "- Do not volunteer related, adjacent, or background information the "
        "user did not ask for — no unsolicited tips, next steps, caveats, "
        "alternatives, related courses or programs, deadlines, or 'you may "
        "also want to know' additions. If they asked one narrow thing, answer "
        "that one narrow thing.\n"
        "- Length follows the question: if one sentence fully answers it, "
        "write one sentence and stop. Never pad an answer to look more "
        "complete.\n"
        "- No LaTeX ever. Markdown is allowed but must earn its place: use it "
        "only when it makes the answer genuinely easier to read, and not when "
        "the answer would read just as well without it. A short list of "
        "parallel items — course sections, requirement options, steps in a "
        "procedure — is worth a list; two or three sentences of explanation "
        "are not. If you find yourself bolding a phrase mid-sentence, "
        "bolding every course code, or adding a heading above a single "
        "paragraph, drop it. Default to plain prose and reach for structure "
        "only when the content is actually structured.\n"
        "- Keep any list flat and light: one short line per item, no headings "
        "above it, no nested sub-bullets, no bolded lead-ins on every line.\n"
        "- This is about volume, not rigour. Everything the ACCURACY rules "
        "above require you to state — naming the term for live or historical "
        "data, section codes, attributing RateMyProfessors as student opinion "
        "with its sample size, flagging that a date you found has already "
        "passed and naming the cycle it came from, asking for clarification "
        "when a program search was ambiguous — still applies in full. Those "
        "are part of the answer, not extras.\n\n"
    )

    SYSTEM_PROMPT = _ANSWER_RULES_HEAD + _DEFAULT_STYLE + _CITED_SOURCES_RULES

    # Quick mode's full contract: same accuracy head, tighter scope, same
    # cited-sources machinery. Passed as `system_prompt` to `answer()` /
    # `answer_stream()`, which hand it to `_system_prompt_with_context`.
    QUICK_SYSTEM_PROMPT = _ANSWER_RULES_HEAD + _QUICK_STYLE + _CITED_SOURCES_RULES

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
        """Quick mode's answer contract — see `QUICK_SYSTEM_PROMPT`.

        This used to be SYSTEM_PROMPT plus a nudge inviting the model to add a
        closing sentence suggesting the user retry with quick mode off when it
        spotted a gap. That nudge is gone: it was the one thing quick mode was
        licensed to append beyond the answer itself, which is exactly what the
        tightened scope rules now forbid. If a gap signal is wanted again,
        surface it in the UI rather than in the answer text."""
        return self.QUICK_SYSTEM_PROMPT

    def _system_prompt_with_context(self, context: str, base_prompt: str | None = None) -> str:
        """Build the system message: instructions + the retrieved reference
        material, framed as system-supplied (NOT part of the user turn) so the
        model doesn't treat it as something the user typed or attached.

        `base_prompt` overrides the default `SYSTEM_PROMPT` — used by
        `_quick_mode_system_prompt` to supply a mode-specific behavioral
        contract while reusing the same reference-material framing below."""
        base = base_prompt if base_prompt is not None else self.SYSTEM_PROMPT
        context = (context or "").strip()
        # Today's date, for the DATES rule in `_ANSWER_RULES_HEAD`. It has to be
        # injected here rather than written into the prompt constants because
        # those are class-level and would freeze at import time — a long-running
        # container would then answer with the date it booted on.
        #
        # Deliberately OUTSIDE the reference-material delimiters: everything
        # inside them is framed as "gathered automatically, may be irrelevant,
        # ignore what doesn't help", which is exactly the wrong framing for the
        # one fact the model must never ignore.
        #
        # The router prompt has had today's date for a while; the answer step did
        # not, so it had no "now" to measure a retrieved date against and read
        # every date it found as still upcoming — an acting appointment whose
        # stated term ended in June was reported as who currently holds the role,
        # and January info sessions were described as "their next ones".
        today = f"\n\nTODAY'S DATE: {time.strftime('%Y-%m-%d')}"
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
        return base + today + ref

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
        graph, then banner, then rmp, then vector chunks.

        `n` is the block's index in the assembled context, NOT its position in this
        list: every numbered block advances `n`, but only *citable* blocks add an
        entry — a not-found note, a program-ambiguity note, and an unmatched-name
        note are instructions to the model, not pages a user can open.

        This walk must stay in lockstep with `_graph_context_text` /
        `_banner_context_text` / `_rmp_context_text` / `_build_context`, which assign
        the numbers. `_filter_cited_sources` resolves the model's <<CITED_SOURCES>>
        numbers against `n`, so any drift here silently attaches the wrong sources
        to an answer (this function used to skip the numbers for those non-citable
        blocks, which shifted every later source by one)."""
        seen: set[str] = set()
        sources = []
        n = 0

        if graph_facts:
            for c in graph_facts.get("courses", []):
                n += 1
                url = c.get("url", "")
                key = url or f"course:{c['code']}"
                if key not in seen:
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
            for _nf in graph_facts.get("not_found", []):
                n += 1          # numbered in the context; nothing to link to
            program = graph_facts.get("program")
            if program:
                # One block per resolved program (a relaxed two-subject match
                # renders both); ambiguity / no-match notes still take a number
                # and contribute no source. Count via `_n_program_blocks`.
                for one in (program.get("parts") or [program]):
                    n += 1      # numbered even when ambiguous / no match
                    if not one.get("program"):
                        continue
                    p = one["program"]
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
            # Banner's class-search UI (no stable per-course deep link); the data
            # itself is served inline, this is just a "look it up" pointer.
            banner_url = "https://banner.uvic.ca/StudentRegistrationSsb/ssb/classSearch/classSearch"
            if banner_facts.get("kind") == "instructor":
                # The instructor shape is ONE block covering every course found (or
                # an ambiguity/no-match note) — not one per course.
                n += 1
                instr = banner_facts.get("instructor")
                if instr and banner_facts.get("courses"):
                    sources.append({
                        "n": n,
                        "url": banner_url,
                        "source": "banner",
                        "title": f"Courses taught by {instr}",
                        "course_code": None,
                        "term": term_label or None,
                        "historical": False,
                    })
            else:
                for course in banner_facts.get("courses", []):
                    n += 1
                    code = course["code"]
                    sources.append({
                        "n": n,
                        "url": banner_url,
                        "source": "banner",
                        "title": f"{code} — live availability",
                        "course_code": code,
                        "term": term_label or None,
                        "historical": False,
                    })

        if rmp_facts:
            for p in rmp_facts.get("professors", []):
                n += 1
                # Only matched professors get a clickable source; ambiguous/no-match
                # blocks are prompts to the model, not citable pages — but they still
                # occupy a number in the context.
                if not p.get("matched"):
                    continue
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
        # `needs_retrieval=False` is the router calling this turn pure conversational
        # filler (a greeting, "thanks", "ok"). Skipping the vector pass is the fix for
        # a real user-visible bug: "whats up" was rewritten to search_query="greeting",
        # which still matched 4 chunks under MAX_CHUNK_DISTANCE (Air Quality, IT
        # Support, Cathinones, Fluorofentanyl) — the lexical-overlap case documented
        # in "what NOT to over-fix" #2. Those then reached the user because the answer
        # model omits its <<CITED_SOURCES>> marker ~33% of the time on chit-chat and
        # `_filter_cited_sources` deliberately fails open. Not retrieving at all fixes
        # it at the source, without touching that fail-open policy for real questions.
        # Gated on the vector path ONLY: if the router somehow set this false while
        # naming a course, the graph/banner/rmp facts above still stand.
        chunks = []
        if route.get("needs_retrieval", True):
            if self.hybrid is not None:
                chunks = self.hybrid_vector_retrieve(route, audience)
            else:
                chunks = self.vector_retrieve(
                    route["search_query"], audience=audience,
                    topic_families=route["topic_families"], department=route["department"],
                )
        return chunks, graph_facts, banner_facts, rmp_facts

    def hybrid_vector_retrieve(self, route: dict, audience: str = DEFAULT_AUDIENCE) -> list[dict]:
        """Entity-routed hybrid retrieval (dense-on-questions + dense-on-chunks,
        plus sparse search scoped to a named entity when the router found one).
        Only called when `self.hybrid` is loaded (HYBRID_RETRIEVAL_ENABLED=1);
        see hybrid_retrieve.py for the full implementation."""
        return hybrid_retrieve.retrieve(self, route, audience, self.hybrid)

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
            # Bounded: Banner can resolve many distinct instructors across sections;
            # don't let that chain into unbounded RMP lookups (audit issue 4).
            names = cls._instructor_names_from_banner(banner_facts)[:MAX_RMP_NAMES]
        else:
            names = []
        return rmp_retrieve(names) if names else {}

    @staticmethod
    def _n_graph_blocks(graph_facts: dict) -> int:
        """How many numbered blocks `_graph_context_text` emits — the offset the
        banner/rmp renderers continue from. Must stay in lockstep with BOTH
        `_graph_context_text` (which assigns the numbers) and `format_sources`
        (which the cited numbers are resolved against); an outline, a not-found
        course, and a program block each take a number."""
        courses = graph_facts.get("courses", [])
        return (
            len(courses)
            + sum(1 for c in courses if c.get("outline"))
            + len(graph_facts.get("not_found", []))
            + GeorgeBot._n_program_blocks(graph_facts.get("program"))
        )

    @staticmethod
    def _n_program_blocks(program: dict | None) -> int:
        """How many numbered blocks the program result emits. A relaxed
        multi-program result renders one block per resolved program, so this is
        the single place all three walkers derive that count from — keeping
        `_graph_context_text`, `_n_graph_blocks` and `format_sources` from
        drifting the way they did before the 2026-08-04 numbering fix."""
        if not program:
            return 0
        return len(program["parts"]) if program.get("parts") else 1

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
        # `ask()` is the quick-mode path (non-streaming /api/chat and the CLI
        # smoke test), so it takes quick mode's contract — it was silently
        # using the default one, which is why CLI answers read longer than the
        # streaming ones the frontend shows.
        answer, cited = self.answer(question, context, history,
                                    system_prompt=self.QUICK_SYSTEM_PROMPT)
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
