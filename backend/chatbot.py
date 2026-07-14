#!/usr/bin/env python3
"""
GeorgeBot — v2 RAG backend.

Two retrieval paths, routed per-query by a single MiniMax-M3 call:

  - Graph path   — course/program facts (prereqs, credits, cross-listings,
                   descriptions, program requirements, outlines) come straight
                   from `graph_queries.GraphStore` (course_graph.pkl +
                   program_graph.pkl + course outlines). No vector search.
  - Vector path  — everything else hits the v2 Chroma collection
                   (`georgebot_v2`), which stores one entry per reverse-HyDE
                   question (5 per chunk). A query embeds once, we pull the
                   nearest question-vectors, then collapse by `chunk_id` back
                   to distinct parent chunks (full text + metadata).

Then MiniMax-M3 reads whichever context was assembled and writes the final,
source-cited answer (see `_call_llm`/`answer`/`answer_stream`). Single
provider — the official MiniMax API (OpenAI-compatible), not the Kesar-
proxied alias. Route/rewrite runs with `thinking: "disabled"` (mechanical
classification, ~90x faster per the measured MiniMax finding); the answer
step runs with `thinking: "adaptive"` (real judgment payoff from reasoning
over multi-chunk context). Both set `reasoning_split: True` so hidden
reasoning never leaks into visible content.

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
CHROMA_DIR = BASE_DIR / "chroma_db"
TAXONOMY_FILE = BASE_DIR / "vector_taxonomy.json"
GENERAL_DEPARTMENT = "general / cross-departmental"
THINK_TAG_RE = re.compile(r"<think>.*?</think>", re.DOTALL)

COLLECTION_NAME = "georgebot_v2"
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
                  f"Run data-pipeline/v2/final-databases/embed_questions.py and copy "
                  f"output/taxonomy.json here as vector_taxonomy.json.", file=sys.stderr)
            sys.exit(1)
        taxonomy = json.loads(TAXONOMY_FILE.read_text())
        self.topic_family_names = [t["name"] for t in taxonomy["topic_families"]]
        self.topic_family_slugs = {t["name"]: t["slug"] for t in taxonomy["topic_families"]}
        self.departments = taxonomy["departments"]
        print(f"  Taxonomy: {len(self.topic_family_names)} topic families, {len(self.departments)} departments")

        if not CHROMA_DIR.exists():
            print(f"ERROR: Chroma DB not found at {CHROMA_DIR}. "
                  f"Run data-pipeline/v2/final-databases/embed_questions.py and copy "
                  f"output/chroma_db/ here.", file=sys.stderr)
            sys.exit(1)
        chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        self.collection = chroma_client.get_collection(COLLECTION_NAME)
        print(f"  Chroma collection '{COLLECTION_NAME}': {self.collection.count():,} question-vectors")

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

    def _build_where(self, topic_families: list[str], department: str | None) -> dict | None:
        """Build a Chroma `where` clause from router-predicted topic families / department.

        Unknown family names (not in the taxonomy) are silently dropped —
        harmless, since an unrecognized field just doesn't exist to filter on.
        `department` is expanded to [department, GENERAL_DEPARTMENT] so a
        department-specific question still surfaces genuinely cross-cutting
        content (e.g. university-wide policies) alongside the department's own.
        """
        clauses = []
        slugs = [self.topic_family_slugs[f] for f in topic_families if f in self.topic_family_slugs]
        if slugs:
            clauses.append({"$or": [{s: True} for s in slugs]} if len(slugs) > 1 else {slugs[0]: True})
        if department:
            clauses.append({"department": {"$in": [department, GENERAL_DEPARTMENT]}})
        else:
            clauses.append({"department": GENERAL_DEPARTMENT})
        if len(clauses) == 1:
            return clauses[0]
        return {"$and": clauses}

    def _query_chunks(self, emb: list[float], n: int, where: dict | None,
                       exclude: set[str]) -> tuple[dict[str, dict], list[str]]:
        kwargs: dict = {"query_embeddings": [emb], "n_results": QUESTION_K,
                         "include": ["documents", "metadatas", "distances"]}
        if where:
            kwargs["where"] = where
        res = self.collection.query(**kwargs)
        chunks: dict[str, dict] = {}
        order: list[str] = []
        ids = res["ids"][0] if res["ids"] else []
        docs = res["documents"][0] if res["documents"] else []
        metas = res["metadatas"][0] if res["metadatas"] else []
        dists = res["distances"][0] if res["distances"] else []
        for _qid, doc, meta, dist in zip(ids, docs, metas, dists):
            if dist > MAX_CHUNK_DISTANCE:
                break  # Chroma returns hits sorted nearest-first; nothing after this is closer
            cid = meta.get("chunk_id")
            if not cid or cid in chunks or cid in exclude:
                continue
            chunks[cid] = {"chunk_id": cid, "text": doc, "metadata": meta}
            order.append(cid)
            if len(order) >= n:
                break
        return chunks, order

    def vector_retrieve(self, query: str, n: int = N_CONTEXT,
                         topic_families: list[str] | None = None,
                         department: str | None = None) -> list[dict]:
        """
        Query the reverse-HyDE question index, collapse hits to distinct chunks.

        Soft filter with backfill: if topic_families/department were predicted
        by the router, run a filtered pass first so on-topic chunks rank
        first; then top up with an unfiltered pass (deduped) so a wrong or
        overly-narrow prediction never drops recall to zero, it just
        re-prioritizes what's already found by plain vector similarity.

        Both passes apply MAX_CHUNK_DISTANCE, so a genuinely off-topic query
        can return fewer than n chunks — even zero — rather than being padded
        out to n with irrelevant ones just to hit the count.
        """
        emb = self._embed_query(query)
        where = self._build_where(topic_families or [], department) if (topic_families or department) else None

        chunks: dict[str, dict] = {}
        order: list[str] = []
        if where:
            chunks, order = self._query_chunks(emb, n, where, exclude=set())
        if len(order) < n:
            more_chunks, more_order = self._query_chunks(emb, n - len(order), None, exclude=set(order))
            chunks.update(more_chunks)
            order.extend(more_order)
        return [chunks[cid] for cid in order]

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
                meta = o.get("metadata", {})
                term = meta.get("term_code", "unknown term")
                lines.append(
                    f"HISTORICAL course outline (term {term}, source={meta.get('source', 'unknown')}):\n"
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

    def rewrite_and_route(self, question: str, history: list[dict]) -> dict:
        """
        One minimax-m3 call: rewrite the question into a standalone search query
        AND classify whether it needs structured course/program facts (graph
        route) or general document retrieval (vector route).
        """
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
            f"{json.dumps(self.topic_family_names)}\n"
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
                "topic_families": [f for f in (data.get("topic_families") or []) if f in self.topic_family_slugs],
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
            if meta.get("source_file"):
                tags.append(f"source={meta['source_file']}")
            if meta.get("document_type"):
                tags.append(f"type={meta['document_type']}")
            if meta.get("department"):
                tags.append(f"department={meta['department']}")
            if meta.get("topic_families"):
                tags.append(f"topics={meta['topic_families']}")
            header = f"[{i}] {' | '.join(tags)}\nURL: {meta.get('url', 'n/a')}"
            blocks.append(f"{header}\n{ch.get('text', '')}")
        text = "\n\n".join(blocks)
        if graph_text:
            text = f"{graph_text}\n\n{text}" if text else graph_text
        return text

    SYSTEM_PROMPT = (
        "You are GeorgeBot, a helpful assistant that answers questions about the "
        "University of Victoria (UVic) for students and staff. Answer ONLY using the "
        "numbered context chunks provided in the user message.\n\n"
        "Rules:\n"
        "- Ground every claim in the provided chunks — every specific fact (a "
        "status like 'no longer offered', a course sequence, a number, a date, "
        "a policy detail) must be something the chunk text actually states, not "
        "something you infer, extrapolate, or reconstruct from partial/messy "
        "data. If you're piecing together a narrative from data that doesn't "
        "clearly state it, that's fabrication even if it sounds plausible — stop "
        "and state only what's explicit instead.\n"
        "- Some chunks are PDF program-worksheets or curriculum grids (course-"
        "by-term tables) that lose their row/column structure in extraction and "
        "read as a flattened run-on list of course codes and terms. Treat these "
        "as unreliable for anything beyond 'this course appears somewhere in "
        "this program's plan' — do not infer sequencing, prerequisites, "
        "term-by-term order, or offering status from a jumbled table you can't "
        "confidently parse. When in doubt whether you're reading a garbled "
        "table correctly, say so explicitly rather than presenting a guess as fact.\n"
        "- The chunks are retrieved by similarity search, not guaranteed relevance —"
        " not every numbered chunk is necessarily accurate or on-topic for this "
        "specific question. Use each chunk's tags (source/type/department/topics) "
        "and content to judge fit before relying on it; silently ignore chunks that "
        "don't actually address the question rather than forcing them into the "
        "answer or citing them just because they were retrieved.\n"
        "- If the chunks don't cover the specific thing asked, don't say so and "
        "don't describe what the chunks do cover instead — that reads as a "
        "broken non-answer. Just answer the question directly and helpfully, the "
        "way you would with no context at all, and if there's a specific UVic "
        "fact you can't confirm, point the user to where they can get it (e.g. "
        "'check your course syllabus or ask your professor directly for office "
        "hours') instead of explaining that you don't have it.\n"
        "- Course outlines marked HISTORICAL are past-term snapshots only. When you "
        "cite one, always name the specific term. Never present historical grading "
        "weights, instructors, or schedules as current — for current course "
        "details, direct the user to Brightspace or the department.\n"
        "- source=kuali chunks are the official course/program catalog — treat them "
        "as the authoritative, current source for prerequisites, credits, and "
        "program requirements.\n"
        "- If a program search returned multiple candidates, ask the user to "
        "clarify which one they mean rather than guessing.\n"
        "- When a chunk states 'Given completed courses [...]: prerequisites "
        "satisfied = ...' or 'outstanding requirements by group', that evaluation "
        "was computed programmatically from the courses the user said they've "
        "taken — trust it over your own tally, but flag any listed non-course "
        "requirements (year standing, GPA, permission, etc.) since those can't be "
        "auto-verified from a course list.\n"
        "- Be concise and direct. Use plain text (no LaTeX)."
    )

    def _answer_messages(self, question: str, context: str, history: list[dict]) -> list[dict]:
        context = context or "(no relevant context found)"
        messages: list[dict] = []
        for turn in history[-MAX_HISTORY_TURNS:]:
            role = turn.get("role")
            if role in ("user", "assistant") and turn.get("content"):
                messages.append({"role": role, "content": turn["content"]})
        messages.append({
            "role": "user",
            "content": f"Context chunks:\n\n{context}\n\n---\n\nQuestion: {question}",
        })
        return messages

    def answer(self, question: str, context: str, history: list[dict]) -> str:
        """MiniMax-M3, thinking adaptive (route/rewrite stays disabled — mechanical)."""
        messages = self._answer_messages(question, context, history)
        text = self._call_llm(messages, system=self.SYSTEM_PROMPT, max_tokens=ANSWER_MAX_TOKENS,
                               thinking="adaptive")
        return text or "Sorry, I couldn't generate an answer right now — please try again."

    def answer_stream(self, question: str, context: str, history: list[dict]):
        """Streamed MiniMax-M3 answer, thinking adaptive.

        Buffered in full, not replayed token-by-token: a `<think>` block can
        span multiple stream chunks, and `reasoning_split` isn't 100%
        reliable at keeping it out of `content` (see `_call_llm`), so it's
        not safe to forward raw deltas — strip after the full response lands.
        """
        messages = self._answer_messages(question, context, history)
        full_messages = [{"role": "system", "content": self.SYSTEM_PROMPT}] + messages
        stream = self.llm.chat.completions.create(
            model=MINIMAX_MODEL,
            max_tokens=ANSWER_MAX_TOKENS,
            messages=full_messages,
            extra_body={"reasoning_split": True, "thinking": {"type": "adaptive"}},
            stream=True,
        )
        parts: list[str] = []
        for event in stream:
            delta = event.choices[0].delta
            text = getattr(delta, "content", None)
            if text:
                parts.append(text)
        full_text = THINK_TAG_RE.sub("", "".join(parts)).strip()
        if full_text:
            yield full_text
        else:
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
                    meta = c["outline"].get("metadata", {})
                    sources.append({
                        "n": n,
                        "url": url,
                        "source": "heat" if meta.get("source") == "heat" else "kuali",
                        "title": f"{c['code']} outline",
                        "course_code": c["code"],
                        "term": meta.get("term_code"),
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
            url = meta.get("url", "")
            key = url or ch.get("chunk_id", str(n))
            if key in seen:
                continue
            seen.add(key)
            sources.append({
                "n": n,
                "url": url,
                "source": meta.get("source_file", ""),
                "title": meta.get("title") or "",
                "document_type": meta.get("document_type") or None,
                "department": meta.get("department") or None,
                "historical": False,
            })
        return sources

    # -- orchestration ------------------------------------------------------

    def retrieve(self, question: str, history: list[dict]) -> tuple[dict, list[dict], dict]:
        """Route the question, run retrieval, and return (route, chunks, graph_facts)."""
        route = self.rewrite_and_route(question, history)
        graph_facts = {}
        if route["course_codes"] or route["program_query"]:
            graph_facts = self.graph_retrieve(
                route["course_codes"], route["program_query"], route["wants_outline"],
                route["completed_courses"],
            )
        chunks = self.vector_retrieve(
            route["search_query"], topic_families=route["topic_families"], department=route["department"],
        )
        return route, chunks, graph_facts

    def ask(self, question: str, history: list[dict] | None = None) -> dict:
        history = history or []
        route, chunks, graph_facts = self.retrieve(question, history)
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
    args = parser.parse_args()

    bot = GeorgeBot()
    result = bot.ask(args.ask)
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
