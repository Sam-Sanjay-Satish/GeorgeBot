# GeorgeBot — Production Webapp (this repo)

This is the **public, production-deploy repo** for GeorgeBot's website: the
RAG backend (FastAPI) and React frontend. It's deployed on **Railway**.

**Companion repo:** `georgebot-pipeline` (private, sibling repo — was this
project's `data-pipeline/`) is where the Chroma vector DB, BM25/graph
artifacts, crawling, chunking, and embedding all happen. This repo only
*serves* what that one produces — it has no crawling/embedding code and
never will. If you're asked to tune retrieval quality by changing what's
*in* the index (chunking, taxonomy, embeddings), that's the wrong repo —
say so and point back to `georgebot-pipeline`. If you're asked to tune
*how retrieval behaves at serve time* (N_CONTEXT, distance cutoffs, the
system prompt, routing logic), that's here.

**History note:** this repo was split off from a monorepo
(`georgebot-pipeline`, formerly just `GeorgeBot`) on 2026-07-14 via a plain
`cp -R` of its `webapp/` folder — no shared git history. The path prefix
`webapp/` from that old repo's docs no longer applies; everything below is
relative to *this* repo's root.

---

## TL;DR Architecture

```
user question + chat history
   │
   ├─ rewrite_and_route()  → MiniMax-M3 (official API, thinking DISABLED)
   │                          (ONE call: standalone search query +
   │                          course_codes[] + program_query + wants_outline
   │                          + topic_families[] + department)
   │
   ├─ graph_retrieve()   (only if course_codes / program_query present)
   │     └─ GraphStore (course_graph.pkl + program_graph.pkl + course outlines)
   │        → prereqs, credits, cross-listings, descriptions, program
   │          requirements, outline text. No vector search.
   │
   ├─ vector_retrieve(query)
   │     └─ Voyage query embedding → Chroma georgebot_v2 top-40 question-vectors
   │        → distance-filtered (MAX_CHUNK_DISTANCE=0.75) → collapse by
   │          chunk_id → up to N_CONTEXT=4 distinct chunks (full text)
   │
   └─ answer()  → MiniMax-M3 (official API, thinking ADAPTIVE)
                  → cited answer (graph facts + vector chunks numbered together)
                  + format_sources()
```

Everything lives in **one engine class**, `GeorgeBot`, in `backend/chatbot.py`.
`backend/api.py` is a thin FastAPI wrapper around it (also used by the CLI
`--ask` smoke test). Tuning `chatbot.py` tunes everything.

**No BM25.** The index is dense-only (reverse-HyDE question embeddings).

**Single LLM provider — the official MiniMax API**, `MiniMax-M3` via
`https://api.minimax.io/v1` (OpenAI-compatible SDK, `MINIMAX_SUB_KEY`).
Replaced an earlier Kesar-router/Chinese-API-answer split — there is **no
cross-provider fallback** anymore. Every call sets
`extra_body={"reasoning_split": True, "thinking": {"type": ...}}`:

- **Route/rewrite: `thinking: "disabled"`** — mechanical classification,
  ~90x faster per a measured MiniMax finding, and doesn't need reasoning.
- **Answer: `thinking: "adaptive"`** — deliberately *not* disabled here.
  Disabling it was tried first and broke a specific product requirement
  (below); adaptive fixed it, at a real latency cost (11s–60s+ observed,
  occasional multi-minute stalls under back-to-back calls — see
  "Known issues" below).
- `reasoning_split` is supposed to keep `<think>...</think>` text out of
  visible `content` regardless of the `thinking` setting, but **isn't
  100% reliable** — a leak was observed and reproduced even with the flag
  set. `_call_llm` and `answer_stream` both defensively regex-strip any
  `<think>...</think>` block from the final text as a safety net. The
  streaming path (`answer_stream`) buffers the **full** response before
  yielding anything (can't safely forward raw deltas — a `<think>` block
  can span multiple stream chunks), so `/api/chat/stream`'s `token` event
  currently arrives as one big chunk, not word-by-word.

**Prompt/retrieval behavior — what NOT to over-fix.** Two real product
requirements drove recent changes, and the fixes matter more than the
prompt wording used to get there — don't re-litigate from scratch without
reading this first:

1. *"Don't expose retrieval internals — no 'I don't have that information
   in the provided context' type non-answers."* Tried multiple prompt-only
   fixes (banned-phrase lists, worked examples, "hard rule" framing) —
   none were fully reliable; MiniMax-M3 with `thinking: "disabled"` on the
   answer step reliably ignored even a literal banned-word list. Switching
   the answer step to `thinking: "adaptive"` fixed it far more effectively
   than any further prompt engineering. Current `SYSTEM_PROMPT` rule for
   this is deliberately a single plain paragraph, **not** a heavily
   enforced one — that's intentional (explicit user direction: don't
   over-enforce in the prompt; let the model + thinking setting do the
   work).
2. *"If retrieved chunks are irrelevant, don't force them into context —
   answer from general knowledge instead."* This turned out to be a
   **retrieval-layer** problem, not a prompt problem: distance alone
   doesn't cleanly separate "genuinely irrelevant" from "on-topic" in this
   corpus (a completely off-topic query can still land a *closer* cosine
   distance than a real, answerable UVic query, because of incidental
   lexical overlap — e.g. "capital of France" scores closer than "CSC 225
   prerequisites" against this index). `MAX_CHUNK_DISTANCE = 0.75` (in
   `vector_retrieve`/`_query_chunks`) was calibrated against real queries
   and narrows this a lot (drops clearly-unrelated chunks to zero) but
   does **not** eliminate the lexical-overlap edge case — know that going
   in, don't assume a "chunk irrelevance" bug report means the prompt is
   broken again; check `n_chunks` and the actual retrieved chunk titles
   first.

---

## Layout

```
.
├── CLAUDE.md
└── backend/
│   ├── chatbot.py             # GeorgeBot engine — routing + retrieval + LLM pipeline (tune here)
│   ├── graph_queries.py       # GraphStore — copy of georgebot-pipeline's course-graph/graph_queries.py
│   ├── api.py                 # FastAPI server (thin wrapper over GeorgeBot); Railway PORT/HOST-aware
│   ├── requirements.txt       # serving deps only (openai, voyageai, chromadb, networkx, fastapi, uvicorn)
│   ├── chroma_db/             # Chroma vector DB, collection georgebot_v2 (gitignored — see Data below)
│   ├── vector_taxonomy.json   # topic_families/department controlled vocabulary (gitignored)
│   └── graph_data/            # course_graph.pkl, program_graph.pkl, course_outlines_final.json (gitignored)
└── frontend/                  # React + Vite + TS chat UI (wired to the API)
    └── src/
        ├── App.tsx            # chat state; calls the backend, renders messages + sources
        ├── lib/api.ts         # askGeorge() → POST /api/chat; maps API sources → UI Source
        ├── types.ts           # Message, Source
        └── components/        # MessageBubble, SourcePanel, SourceBadge, ChatInput, ...
```

`backend/graph_queries.py` is a **copy**, not a symlink, of
`georgebot-pipeline`'s `data-pipeline/v2/course-graph/graph_queries.py`,
repointed at `./graph_data/` instead of `./output/`. If the pipeline repo's
version changes, re-copy and re-patch the paths here (or diff first — they
drift if edited independently).

---

## Data: how the serving artifacts get here (Railway Volumes, not git)

`chroma_db/`, `graph_data/`, and `vector_taxonomy.json` are gitignored —
they're too large for a normal git push and don't belong in this repo's
history. They're produced by `georgebot-pipeline` and get onto this app
via a **Railway Volume**, not a git commit:

1. Rerun/re-embed happens in `georgebot-pipeline`.
2. Copy the fresh `chroma_db/`, `graph_data/`, `vector_taxonomy.json` into
   this repo's `backend/` locally, for dev/testing — this refreshes your
   *local* copy only.
3. **Separately**, push the same data into the Railway Volume mounted on
   the backend service in production — copying locally does **not**
   propagate to Railway by itself. (As of this writing the exact
   volume-mount path and any code change to point `chatbot.py`'s
   `CHROMA_DIR`/`TAXONOMY_FILE`/`graph_data` paths at that mount — they're
   currently hardcoded relative to `backend/` via `BASE_DIR = Path(__file__).parent`
   — is still **pending**; don't assume it's wired up without checking.)

## Deployment (Railway)

- Two services from this one repo, each scoped via Railway's **Root
  Directory** setting: one → `backend`, one → `frontend`.
- Backend service needs a **Volume** for the three artifacts above (mount
  path TBD — see "Data" section; this requires a `chatbot.py` path change
  that hasn't landed yet).
- Backend env vars: `MINIMAX_SUB_KEY`, `VOYAGE_API_KEY`. Railway also
  injects `PORT` automatically — `api.py`'s `main()` already reads
  `$PORT`/`$HOST` (default `0.0.0.0`) so no Railway-side config needed for
  that part; this was a real gap fixed on 2026-07-14 (originally hardcoded
  `127.0.0.1:5001`, which would not have bound correctly on Railway).
- Frontend service: set `VITE_API_BASE` to the backend's Railway URL
  (defaults to `http://127.0.0.1:5001` otherwise, which is dev-only).
- Rebuild scoping: Railway's **Watch Paths** can restrict each service to
  rebuild only on changes under its own `backend/`/`frontend/` — set this
  up to avoid frontend rebuilds on backend-only commits and vice versa.
- Repo is intentionally **public** (companion pipeline repo is private) —
  this was a deliberate choice: monorepo-vs-split reasoning landed on
  "split by visibility, not by deploy mechanics," since Railway's
  Root-Directory/Watch-Paths features already solve the deploy-noise
  problem within one repo. Don't re-merge these repos without revisiting
  that reasoning.

---

## Environment

- **Machine (dev):** MacBook Air (darwin-arm64); **Python:** 3.14.3 (`python3`)
- **Venv:** not yet set up in this repo standalone — currently tested by
  reusing `georgebot-pipeline`'s `venv/`. Worth creating this repo's own
  venv + lockfile before it's truly independent of the pipeline repo.
- **.env** (this repo's root, gitignored): `MINIMAX_SUB_KEY`, `VOYAGE_API_KEY`
  only — this repo does **not** use `KESAR_API_KEY`/`CHINESE_API_KEY` (that
  provider split was retired; see the "Single LLM provider" section above).
  `chatbot.py` calls `load_dotenv()` with no path; it walks up from the
  cwd, so run from repo root (or anywhere under it).

### Run it

```bash
source <path-to-venv>/bin/activate

# One-shot CLI question (no server) — fastest retrieval/answer smoke test
python3 backend/chatbot.py --ask "What are the prerequisites for CSC 225?"

# FastAPI server (default 0.0.0.0:5001, or $PORT if set)
python3 backend/api.py
python3 backend/api.py --port 8000

# Frontend dev server (talks to http://127.0.0.1:5001 by default)
cd frontend && npm run dev
```

### HTTP API (FastAPI, CORS open — fine for local/staging, revisit before wide prod traffic)

| Method | Path | Body | Returns |
|---|---|---|---|
| GET  | `/health` | — | `{status, chunks}` |
| POST | `/api/chat` | `{question, history?}` | `{answer, sources[], search_query, n_chunks}` (JSON) |
| POST | `/api/chat/stream` | `{question, history?}` | `text/event-stream` (SSE: `sources`, `token`, `done`, `error`) |

`history` is a list of `{role: "user"|"assistant", content: str}`.

```bash
curl -s -X POST http://127.0.0.1:5001/api/chat -H 'Content-Type: application/json' \
  -d '{"question":"How do I apply for co-op?"}'
```

---

## Retrieval Pipeline — Full Detail (tune here)

### 1. Query rewrite + route — `MiniMax-M3` (thinking disabled)
`rewrite_and_route(question, history)` is **one** call that returns JSON:
`search_query`, `course_codes` (normalized, no space — `"CSC 225"` →
`"CSC225"`), `program_query`, `wants_outline`, `completed_courses`,
`topic_families` (0-3, copied verbatim from the taxonomy), `department`
(copied verbatim, or null). `course_codes`/`program_query` are only meant
to be populated for **structured catalog facts** (prereqs, credits,
cross-listings, descriptions, program requirements, outline content) — a
course code mentioned in passing should leave both empty. `topic_families`/
`department` must match `vector_taxonomy.json` exactly (passed into the
prompt in full) — anything not in that list is dropped on parse. On any
failure (bad JSON, API error), falls back to vector-only with everything
else empty/null/False. Note: MiniMax-M3 is **not fully deterministic even
at temperature 0** — the exact rewritten `search_query` text varies run to
run, which can shift retrieval results slightly between identical-looking
requests.

### 2. Graph retrieval — `GraphStore` (no LLM, no vector search)
Only runs when `course_codes` or `program_query` is non-empty.
- **Per course code:** `_course_facts()` pulls title/credits/hours/description,
  `get_eligibility()`, `get_corequisites()`, `cross_listings()`,
  `get_unlocks()`, `get_alternatives()`, `programs_requiring()`, and — if
  `wants_outline` — `get_outline()` (truncated to 4000 chars, tagged
  HISTORICAL with its term).
- **Program query:** `_program_facts()` calls `search_programs()` (fuzzy,
  token-based, can return multiple candidates). If ambiguous, the context
  block lists all candidates and the system prompt tells the model to ask
  the user to disambiguate. If exactly one match, pulls `get_program()` +
  `program_requirement_groups()` + `program_specializations()`.
- `_graph_context_text()` renders this into numbered `[n]` blocks tagged
  `source=kuali`, same numbering scheme as vector chunks.

### 3. Vector retrieval — Voyage + Chroma (reverse-HyDE), with distance cutoff
- **Model:** `voyage-4-large`. Queries embed with `input_type="query"`; the
  index was built with `input_type="document"` (asymmetric — must stay
  matched; don't "simplify" this).
- **Store:** Chroma `PersistentClient`, collection **`georgebot_v2`**,
  cosine similarity, ~22,470 entries (one per reverse-HyDE question, 5 per
  chunk, ~4,494 chunks). The *question* is embedded; the *full parent
  chunk text* is the entry's `document`.
- `vector_retrieve(query)` pulls **`QUESTION_K = 40`** nearest
  question-vectors, drops any with cosine distance **> `MAX_CHUNK_DISTANCE`
  (0.75)**, then collapses the rest by `metadata.chunk_id` in rank order
  until **`N_CONTEXT = 4`** distinct chunks are collected. Both the
  distance cutoff and the collapse happen inside `_query_chunks` — Chroma
  returns hits nearest-first, so the loop breaks as soon as it sees a
  distance past the cutoff (everything after is worse).
- **This means a query can legitimately return fewer than N_CONTEXT chunks
  — including zero** — instead of always being padded out with irrelevant
  ones. That's intentional (see "what NOT to over-fix" point 2 above), but
  double-check `n_chunks` in the response before assuming a thin answer is
  a retrieval bug rather than a genuinely sparse/off-topic match.
- **`N_CONTEXT=4` was tuned empirically (8 → 3 → 4).** N=1 missed context
  often even with filtering; N=2-3 still missed content split across
  multiple chunks (e.g. "what a minor is" vs. "how to declare it" as
  separate chunks). N=4 covered the empirical test batch. Noise at N=4 is
  tolerated by design (off-topic chunks can still slip in even under the
  0.75 cutoff) — the answer model is told to silently judge fit per-chunk
  via its tags rather than treat every retrieved chunk as relevant.
- Runs on *every* question, graph route or not — graph facts are additive.

**Metadata filtering (`department` + `topic_families`) — soft filter with
backfill**, on top of the distance cutoff. The router predicts up to 3
`topic_families` and one `department` from the controlled vocabulary in
`vector_taxonomy.json`. `vector_retrieve()` runs a filtered pass first
(`department $in [predicted, "general / cross-departmental"]` AND/OR
`tf_<slug>` booleans), then backfills with an unfiltered pass (deduped)
only if the filtered pass returned fewer than `N_CONTEXT`. Both passes
respect `MAX_CHUNK_DISTANCE`. A wrong/narrow prediction degrades to
"no filtering benefit," not "wrong answer."

**`department` is reliable; `topic_families` accuracy is bounded by
taxonomy quality** — see `georgebot-pipeline`'s
`data-pipeline/v2/final-databases/CLAUDE.md` for the `TOPIC_FAMILY_ALIASES`
dedup history (46 raw families → 31) if you see a filtering-related miss
after new corpus content lands.

### 4. Fetch + answer — `MiniMax-M3` (thinking adaptive)
- `_build_context()` numbers graph blocks first, then vector chunks
  (`source=webpage|document`, `type=<document_type>`,
  `department=<department>`, `topics=<pipe-joined topic_families>`) + URL
  + full chunk text.
- `answer()`/`answer_stream()` call MiniMax-M3 with `thinking: "adaptive"`,
  `max_tokens=1500`, `SYSTEM_PROMPT`, and the last `MAX_HISTORY_TURNS`
  conversation turns. Both defensively strip any leaked `<think>` block
  (see "Single LLM provider" above) before returning.

### Tunable constants (top of `chatbot.py`)

| Const | Value | Effect |
|---|---|---|
| `VOYAGE_MODEL` | `voyage-4-large` | embedding model (query + doc must match index) |
| `MINIMAX_MODEL` | `MiniMax-M3` (official API) | query rewrite + route classifier, and final answer |
| `QUESTION_K` | 40 | question-vectors pulled from Chroma before collapsing |
| `MAX_CHUNK_DISTANCE` | 0.75 | cosine distance cutoff — chunks worse than this are dropped, not backfilled |
| `N_CONTEXT` | 4 | max distinct chunks handed to the answer model (graph blocks additive, uncapped; can be fewer if the cutoff filters hard) |
| `LLM_MAX_TOKENS` | 4000 | route/rewrite budget |
| `ANSWER_MAX_TOKENS` | 1500 | answer budget |
| `MAX_HISTORY_TURNS` | 6 | trailing turns kept for rewrite + answer |

### System prompt (answer model) — behavioral contract
- Answer **only** from the numbered context chunks; cite chunk numbers inline (`[1]`, `[2][5]`).
- **Anti-fabrication rule for garbled table data (real incident, root-caused).**
  A user reported the bot confidently claiming "CSC 116 is no longer offered"
  plus a fabricated course-sequencing narrative, and — worse — when asked to
  explain itself, hallucinated a false self-diagnosis ("no context was
  provided at all"), disproven by directly reproducing the turn. Root cause:
  a program-worksheet PDF (course-by-term curriculum grid) that loses its
  row/column structure in extraction and reads as a flattened run-on list.
  The prompt now explicitly names this failure mode: don't infer
  sequencing/status/order from a chunk that looks like a flattened table,
  say so explicitly when a chunk is too garbled to read confidently. v2.2's
  improved table extraction (pipeline repo) should reduce how often this
  chunk shape exists at all — this prompt rule is the interim mitigation.
- Chunks are retrieved by similarity search, not guaranteed relevance — use
  each chunk's tags + content to judge fit; silently drop chunks that don't
  address the question rather than forcing them in.
- **If the chunks don't cover what's asked, don't say so and don't describe
  what they do cover instead** — answer directly and helpfully as you would
  with no context, and for any specific UVic fact you can't confirm, point
  the user to where to get it (e.g. "check your syllabus or ask your
  professor") instead of explaining you don't have it. (See "what NOT to
  over-fix" above — this is deliberately a single plain rule, not a
  heavily-enforced one; the `thinking: adaptive` setting on the answer step
  does more of the actual work here than the wording does.)
- Course outlines marked **HISTORICAL** are past-term snapshots; always
  name the term when citing one; never present historical grading/
  instructors/schedules as current.
- `source=kuali` chunks are the authoritative, current catalog — trust
  them over web pages for prereqs/credits/program requirements.
- If a program search returned multiple candidates, ask the user to
  clarify rather than guessing.
- Concise, plain text (no LaTeX).

---

## Known issues / open items

- **Answer-step latency with `thinking: adaptive`**: observed 11s–60s+ per
  call, and a full stall past 90s when a few calls landed close together —
  most likely MiniMax's account-level Token Plan rate limit (opaque,
  token-throughput-based, not a flat request cap) rather than a hang on
  GeorgeBot's side. Not yet mitigated. If this becomes a live-traffic
  problem, options to explore: a dedicated/higher Token Plan tier,
  request queuing/backoff, or revisiting whether adaptive thinking is
  worth the latency for this specific product requirement.
- **Railway Volume wiring is not done yet** — `chatbot.py`'s
  `CHROMA_DIR`/`TAXONOMY_FILE`/graph-data paths are hardcoded relative to
  `backend/` (`BASE_DIR = Path(__file__).parent`). Needs an env-var
  override (or similar) before a Volume mount actually gets picked up in
  production.
- **This repo has no venv/lockfile of its own yet** — currently dev-tested
  by borrowing `georgebot-pipeline`'s venv. Fine short-term since both
  repos share a machine during this transition, but worth fixing before
  this repo is truly standalone.
- **Frontend**: `/api/chat/stream` is implemented server-side but not
  wired into the UI (`App.tsx` uses non-streaming `/api/chat`). Login is a
  mock shell (no real OAuth). `mockData.ts` exists but is unused.

---

## Frontend (chat UI)

- React + Vite + TypeScript, Tailwind v4 + shadcn/ui.
- `App.tsx` holds the message list; on send it appends a user msg + a
  loading assistant msg, calls `askGeorge(text, priorMessages)`, then
  fills the answer + sources (or an error string if unreachable).
- `lib/api.ts` posts to `${VITE_API_BASE ?? 'http://127.0.0.1:5001'}/api/chat`,
  builds `history` from prior non-loading messages, maps API sources to
  the UI `Source` type.

### Vector chunk metadata schema (what you can filter / rank / display on)
Every Chroma entry (one per question) carries, in `metadata`:
```json
{
  "chunk_id": "5ebf5c9d3c87278c#0",
  "doc_id": "5ebf5c9d3c87278c",
  "url": "https://...",
  "title": "...",
  "document_type": "campus / facilities / services",
  "department": "general / cross-departmental",
  "source_file": "webpage | document",
  "topic_families": "pipe | joined | string",
  "tf_advising_program_planning": true,
  "tf_...": "one boolean per family this chunk belongs to",
  "question_index": 0,
  "question": "..."
}
```
`department` and `tf_<slug>` booleans are the actual filter keys.
`topic_families` (pipe-joined) is display-only. `chunk_id` is what
`vector_retrieve()` collapses on.

`format_sources()` returns `source` values `webpage | document | kuali |
heat` (plus occasional `calendar`). UI badge enum: `webpage → uvic_html`,
`document → uvic_docs`, `heat → heat`, `kuali → kuali`.

---

## Pointers

- Everything about *how the data was built* (crawling, chunking, embedding,
  taxonomy, course graph) — `georgebot-pipeline` (private, sibling repo),
  see its root `CLAUDE.md` and `data-pipeline/v2/CLAUDE.md`.
- This repo is read-only with respect to that data at serve time.
