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
user question + chat history + audience (undergrad | faculty | both)
   │
   ├─ rewrite_and_route(…, audience)  → MiniMax-M3 (official API, thinking DISABLED)
   │                          (ONE call: standalone search query +
   │                          course_codes[] + program_query + wants_outline
   │                          + wants_availability + term_season/term_year
   │                          + topic_families[] + department; the topic_family
   │                          vocabulary offered depends on audience)
   │
   ├─ graph_retrieve()   (only if course_codes / program_query present)
   │     └─ GraphStore (course_graph.pkl + program_graph.pkl + HEAT outlines)
   │        → prereqs, credits, cross-listings, descriptions, program
   │          requirements, outline text. No vector search. Audience-independent.
   │
   ├─ banner_retrieve()  (only if course_codes AND wants_availability — LIVE data)
   │     └─ banner.py — UVic Banner 9 registration JSON (banner.uvic.ca), NOT the
   │        static index: live seats/waitlist, per-section schedule/room, instructor,
   │        delivery/campus, for the resolved term. In-process TTL cache. Best-effort
   │        (returns {} on any failure). Audience-independent. See BANNER_API.md.
   │
   ├─ vector_retrieve(query, audience)
   │     └─ Voyage query embedding (ONE) → queried against the selected
   │        collection(s) — georgebot_v22_undergrad and/or _faculty — top-40
   │        question-vectors each → distance-filtered (MAX_CHUNK_DISTANCE=0.75)
   │        → merged by distance across collections → collapse by chunk_id →
   │          up to N_CONTEXT=4 distinct chunks (full text)
   │
   └─ answer()  → MiniMax-M3 (official API, thinking DISABLED)
                  → answer from graph + banner + vector blocks (supplied via the
                    SYSTEM prompt, numbered together) + format_sources()
```

**Audience (v2.2).** The corpus is split into two Chroma collections —
`georgebot_v22_undergrad` and `georgebot_v22_faculty` — with disjoint
topic-family vocabularies (student services vs. HR/governance/research-admin).
The **user** picks the scope per question via a frontend toggle
(undergrad / faculty / both); there is no LLM audience-guessing. `audience`
flows from the request → `rewrite_and_route` (which topic-family list to offer)
and `vector_retrieve` (which collection(s) to search). `both` embeds once and
distance-merges results across the two collections (globally-nearest N_CONTEXT,
not N per corpus). Both collections share the embedding space (`voyage-4-large`,
1024-d), which is what makes cross-collection distance-merge valid. The
course/program graph is shared, so audience does not affect graph facts.

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
- **Answer: `thinking: "disabled"`** (as of 2026-07-14 — was `"adaptive"`).
  Adaptive was originally adopted because a `disabled` answer step leaked
  retrieval internals (see "what NOT to over-fix" below); it was switched
  back to `disabled` once two things landed together: (a) the reference
  material moved out of the *user* turn and into the **SYSTEM** prompt (via
  `_system_prompt_with_context`), and (b) `SYSTEM_PROMPT` was rewritten to
  own the "ignore irrelevant material / if nothing's relevant answer from
  your own knowledge / never announce missing context" behavior directly.
  A live 8-question benchmark confirmed the rewrite holds up on `disabled`
  (0/8 leaked internals) at ~2x lower answer latency (adaptive ~7.8s avg
  vs disabled ~4.8s). **If you see retrieval-internals leakage return,
  fix the prompt — do not silently flip back to adaptive without re-reading
  that section.**
- `reasoning_split` keeps `<think>...</think>` text out of visible `content`
  (reasoning lands in a separate `reasoning_content` field we never read).
  With the answer step now on `thinking: "disabled"` there should be no
  reasoning at all, but `_call_llm` (regex) and `answer_stream`
  (`_iter_visible_deltas`, a streaming tag-stripper) both still strip any
  `<think>` block defensively. `answer_stream` now forwards visible deltas
  **incrementally** (token-by-token), so `/api/chat/stream`'s `token` event
  streams progressively — the frontend (`askGeorgeStream`) is wired to it.

**Prompt/retrieval behavior — what NOT to over-fix.** Two real product
requirements drove recent changes, and the fixes matter more than the
prompt wording used to get there — don't re-litigate from scratch without
reading this first:

1. *"Don't expose retrieval internals — no 'I don't have that information
   in the provided context' type non-answers."* History (still worth
   knowing): early prompt-only fixes (banned-phrase lists, worked examples,
   "hard rule" framing) were unreliable with a `disabled` answer step, so
   the answer step was moved to `thinking: "adaptive"`, which fixed it.
   **That is no longer how it's solved.** As of 2026-07-14 the answer step
   is back on `disabled`, and the fix now lives in the prompt + message
   structure: the retrieved material is supplied via the **SYSTEM** prompt
   (not the user turn) so the model knows *the user didn't write or see it*,
   and `SYSTEM_PROMPT` explicitly instructs it to ignore irrelevant
   material, answer from its own knowledge when nothing is relevant, and
   never announce missing/insufficient context. This is now a **deliberately
   enforced** prompt rule (reversing the earlier "keep it a single plain
   paragraph, don't over-enforce" guidance — that assumed adaptive thinking
   was carrying the load; it no longer is). Verified across an 8-question
   live batch, including the two cases that used to leak (`parking cost`,
   `clubs`) and an off-topic query that must answer from own knowledge.
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
│   ├── banner.py              # live class availability (UVic Banner 9 JSON): session + TTL cache + term resolution (BANNER_API.md)
│   ├── graph_queries.py       # GraphStore — copy of georgebot-pipeline's course-graph/graph_queries.py
│   ├── api.py                 # FastAPI server (thin wrapper over GeorgeBot); Railway PORT/HOST-aware
│   ├── Dockerfile             # backend container build (Railway); multi-stage python:3.14-slim, CMD python api.py
│   ├── .dockerignore          # excludes volume artifacts (chroma_db/graph_data/taxonomy), .env, pycache
│   ├── requirements.txt       # serving deps only (openai, voyageai, chromadb, networkx, fastapi, uvicorn)
│   ├── chroma_db/             # Chroma vector DB, collections georgebot_v22_undergrad + _faculty (gitignored — see Data below)
│   ├── vector_taxonomy.json   # per-audience topic_families + shared department vocab, generated from pipeline taxonomy.yaml (gitignored)
│   └── graph_data/            # course_graph.pkl, program_graph.pkl, heat_outlines.json (gitignored)
└── frontend/                  # React + Vite + TS chat UI (wired to the API)
    └── src/
        ├── App.tsx            # chat state; calls the backend, renders messages + sources
        ├── lib/api.ts         # askGeorgeStream() → SSE /api/chat/stream (askGeorge/non-stream also here); maps + cleanTitle sources
        ├── types.ts           # Message, Source
        └── components/        # MessageBubble, SourcePanel, SourceBadge, ChatInput, ...
```

`backend/graph_queries.py` is a **copy**, not a symlink, of
`georgebot-pipeline`'s `course_graph/graph_queries.py`, patched two ways:
(1) paths use this repo's env-var scheme (`GRAPH_DATA_DIR`/`DATA_DIR`/
`./graph_data`) instead of the pipeline's `config.*`; (2) the outline loader
reads v2.2's **`heat_outlines.json`** — a flat `{code: {course, term, url,
text}}` dict, HEAT-only (eng/CS courses; others have no outline) — instead of
the old list-shaped `course_outlines_final.json`. The **accessor API is
identical** to the pipeline's (all 19 `GraphStore` methods). If the pipeline
version changes, re-copy and re-apply those two patches (diff first — they
drift if edited independently). The outline record shape is also read directly
in `chatbot.py` (`_graph_context_text`, `format_sources`) — `term`/`text`/
`url`, no nested `metadata`.

---

## Data: how the serving artifacts get here (Railway Volumes, not git)

`chroma_db/`, `graph_data/`, and `vector_taxonomy.json` are gitignored —
they're too large for a normal git push and don't belong in this repo's
history. They're produced by `georgebot-pipeline` and get onto this app
via a **Railway Volume**, not a git commit:

1. Rerun/re-embed happens in `georgebot-pipeline` (v2.2).
2. Copy the fresh artifacts into this repo's `backend/` locally, for
   dev/testing (refreshes your *local* copy only):
   - `chroma_db/` ← pipeline `json_db/data/f_embed/chroma_db/` (both
     collections live in one DB).
   - `graph_data/course_graph.pkl` + `program_graph.pkl` ←
     `course_graph/output/`; `graph_data/heat_outlines.json` ←
     `acquisition/heat/output/heat_outlines.json`.
   - `vector_taxonomy.json` is **generated**, not copied: run a small
     script over the pipeline's `taxonomy.yaml` that derives `tf_<slug>`
     keys and emits `{departments, topic_families_undergrad,
     topic_families_faculty}` (each family list as `{slug, name}`). Slug
     rule must match `f_embed/embed.py`. Verify generated slugs equal the
     `tf_*` keys actually present in each collection before trusting the
     filter.
3. **Separately**, push the same data into the Railway Volume mounted on
   the backend service in production — copying locally does **not**
   propagate to Railway by itself.

**How the Volume actually got seeded (done 2026-07-15).** A Railway Volume
is only reachable from *inside* the running container — there's no upload-to-
volume file API — so the seed goes over `railway ssh`:
1. Build a clean local tar of the three artifacts. **Set `COPYFILE_DISABLE=1`
   (or `--no-mac-metadata`)** — macOS `tar` otherwise injects `._*`
   AppleDouble members that `--exclude '._*'` does NOT catch (they're
   synthesized at archive time), which is how the old junk got on the Volume.
   Checksum it (`shasum -a 256`).
2. Stream it in: `cat v22.tar | railway ssh -- "cat > /data/_upload.tar"`.
   `railway ssh` command mode preserves binary stdin (verified with a
   checksum round-trip), so no base64 needed. ~1.2 GB took ~9 min (~4.5 MB/s).
3. Verify the container-side `sha256sum` matches, extract to a **staging dir**
   (`/data/_new`) — non-destructive, the live service keeps serving the old
   files — sanity-check (chroma `list_collections()` counts, taxonomy keys),
   then atomically swap (`rm -rf` old + `mv` staged into place) and delete the
   tarball. Removing the old chroma while the service runs is safe (Linux
   deleted-but-open inodes); the new data loads on the **next restart**.
4. **Code and data must be deployed together.** The old single-collection
   code crashes on the v2.2 data (`get_collection("georgebot_v2")` → not
   found) and vice-versa. Deploy the matching code (git push → Railway
   rebuild) as part of the same migration; don't restart the old deployment
   against new data.

**Current production Volume state:** `georgebot-volume`, mounted at **`/data`**
on the backend service, `DATA_DIR=/data` set. Holds v2.2 (`chroma_db/` two
collections, `graph_data/{course_graph.pkl, program_graph.pkl,
heat_outlines.json}`, `vector_taxonomy.json`), ~1.3 GB of a 5 GB volume.

**Pointing the code at the Volume (env vars — this landed 2026-07-14).**
The artifact paths are no longer hardcoded. They resolve, in priority order:

- `CHROMA_DIR` / `TAXONOMY_FILE` / `GRAPH_DATA_DIR` — per-artifact absolute
  overrides; each wins over everything below when set.
- `DATA_DIR` — common base for all three at once (the one-Volume case):
  `$DATA_DIR/chroma_db`, `$DATA_DIR/vector_taxonomy.json`,
  `$DATA_DIR/graph_data`.
- **Neither set** → the original `BASE_DIR`-relative defaults under
  `backend/` (local dev, unchanged).

`CHROMA_DIR`/`TAXONOMY_FILE` resolve in `chatbot.py`; the graph-data dir
resolves in `graph_queries.py` (same scheme — it computes its own paths
because `chatbot.py` calls `GraphStore.load()` with no args). **For
Railway: mount one Volume and set `DATA_DIR=<mount-path>`**, then drop the
three artifacts under it — no per-deploy code change.

## Deployment (Railway backend + Vercel frontend)

- **Split hosting:** the **backend** runs on **Railway** (Root Directory
  `backend`, at `georgebot-production.up.railway.app`); the **frontend** is
  deployed on **Vercel** (Root Directory `frontend`, served at
  `georgebot.org` / `www.georgebot.org`). Both come from this one repo.
- **Backend build: `backend/Dockerfile`** (multi-stage `python:3.14-slim`;
  final `CMD python api.py`, which reads `$PORT` and binds `0.0.0.0`).
  Because a Dockerfile is present, Railway builds the backend service with
  it instead of Nixpacks — the service's Root Directory **must** be
  `backend` so the build context and the Dockerfile's `COPY . .` resolve.
  The image deliberately does **not** bake in `chroma_db/`/`graph_data/`/
  `vector_taxonomy.json` (they're gitignored and `.dockerignore`d) — those
  arrive at runtime via the Volume below.
- **Frontend build: Vercel** (Vite/framework auto-detection, Root Directory
  `frontend`) — no Dockerfile. Vercel auto-deploys on push to `main`.
- Backend service has a **Volume** (`georgebot-volume`) mounted at **`/data`**
  holding the three artifacts, with `DATA_DIR=/data` set. Live as of
  2026-07-15 — see "Data" section for how it was seeded and how to re-seed.
- Backend env vars: `MINIMAX_SUB_KEY`, `VOYAGE_API_KEY`. Railway also
  injects `PORT` automatically — `api.py`'s `main()` already reads
  `$PORT`/`$HOST` (default `0.0.0.0`) so no Railway-side config needed for
  that part; this was a real gap fixed on 2026-07-14 (originally hardcoded
  `127.0.0.1:5001`, which would not have bound correctly on Railway).
- Frontend (Vercel): set `VITE_API_BASE` to the backend's Railway URL
  (`https://georgebot-production.up.railway.app`); it defaults to
  `http://127.0.0.1:5001` otherwise, which is dev-only. The backend's
  `CORS_ALLOW_ORIGINS` must include the Vercel domain(s) — currently
  `https://georgebot.org,https://www.georgebot.org`.
- Rebuild scoping: Railway's **Watch Paths** restrict the backend service to
  rebuild only on changes under `backend/`; Vercel's ignored-build-step /
  root-directory settings similarly keep the frontend from rebuilding on
  backend-only commits. Set both so a backend-only push doesn't rebuild the
  frontend and vice-versa.
- Repo is intentionally **public** (companion pipeline repo is private) —
  this was a deliberate choice: monorepo-vs-split reasoning landed on
  "split by visibility, not by deploy mechanics," since per-service
  Root-Directory + watch/ignore settings (Railway for the backend, Vercel
  for the frontend) already solve the deploy-noise problem within one repo.
  Don't re-merge these repos without revisiting that reasoning.

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
| POST | `/api/chat` | `{question, history?, audience?}` | `{answer, sources[], search_query, n_chunks}` (JSON) |
| POST | `/api/chat/stream` | `{question, history?, audience?}` | `text/event-stream` (SSE: `status`, `sources`, `token`, `done`, `error`) |

`audience` is `"undergrad"` (default) `| "faculty" | "both"`; unrecognized
values fall back to the default. The `status` SSE event fires before `token`s
with a short, templated phase line (e.g. `"Looking up CSC 225…"`, `"Reading
through sources…"`) derived from the route — no extra LLM call.

`history` is a list of `{role: "user"|"assistant", content: str}`.

```bash
curl -s -X POST http://127.0.0.1:5001/api/chat -H 'Content-Type: application/json' \
  -d '{"question":"How do I apply for co-op?"}'
```

---

## Retrieval Pipeline — Full Detail (tune here)

### 1. Query rewrite + route — `MiniMax-M3` (thinking disabled)
`rewrite_and_route(question, history, audience)` is **one** call that returns
JSON: `search_query`, `course_codes` (normalized, no space — `"CSC 225"` →
`"CSC225"`), `program_query`, `wants_outline`, `wants_availability`,
`term_season` (`spring|summer|fall|null`), `term_year` (int|null),
`completed_courses`, `topic_families` (0-3, copied verbatim from the taxonomy),
`department` (copied verbatim, or null). `course_codes` is populated for
**structured catalog facts** (prereqs, credits, cross-listings, descriptions,
program requirements, outline content) **OR live availability** (when
`wants_availability` — seats/sections/schedule/instructor); `program_query` is
program-requirements only. A course mentioned merely in passing leaves both
empty. `term_season`/`term_year` carry a user-named term to `banner_retrieve`
(today's date is injected so relative phrases like "next spring" resolve). The
`topic_families`
list shown in the prompt is **audience-dependent** (undergrad list, faculty
list, or the union for `both`); `department` is one shared list. Both must
match `vector_taxonomy.json` exactly (passed into the prompt in full) —
anything not in the valid set for that audience is dropped on parse. On any
failure (bad JSON, API error), falls back to vector-only with everything
else empty/null/False (including `wants_availability=False`, so a router failure
also disables the Banner path). Note: MiniMax-M3 is **not fully deterministic even
at temperature 0** — the exact rewritten `search_query` text varies run to
run, which can shift retrieval results slightly between identical-looking
requests.

### 2. Graph retrieval — `GraphStore` (no LLM, no vector search)
Only runs when `course_codes` or `program_query` is non-empty.
- **Per course code:** `_course_facts()` pulls title/credits/hours/description,
  `get_eligibility()`, `get_corequisites()`, `cross_listings()`,
  `get_unlocks()`, `get_alternatives()`, `programs_requiring()`,
  `prereq_chain()` (full transitive prereq closure; rendered as the *deeper*
  deps beyond the direct prereqs), and — if `wants_outline` — `get_outline()`
  (truncated to 4000 chars, tagged HISTORICAL with its term).
- **Program query:** `_program_facts()` calls `search_programs()` (fuzzy,
  token-based, can return multiple candidates). If ambiguous, the context
  block lists all candidates and the system prompt tells the model to ask
  the user to disambiguate. If exactly one match, pulls `get_program()` +
  `program_requirement_groups()` + `program_specializations()` +
  `program_courses()` (flat list of every course the program references,
  capped at 60 in the rendered block).
- These accessors run as a deterministic **bulk fetch** — the router names a
  course/program, the code pulls the full fact set; the LLM does not pick
  accessors. Two `GraphStore` methods remain intentionally unused:
  `get_prereqs` (superseded by `get_eligibility`'s option-group view) and
  `course_with_programs` (a convenience wrapper that only re-bundles accessors
  already called here) — wiring either in would duplicate facts already shown.
- `_graph_context_text()` renders this into numbered `[n]` blocks tagged
  `source=kuali`, same numbering scheme as vector chunks.

### 2b. Live availability — `banner.py` (UVic Banner 9, no LLM, no vector search)
Gated live-data step, parallel to graph retrieval. Fires **only when
`course_codes` is non-empty AND the router set `wants_availability`** (asked
about seats/openings/waitlist/"is X full"/section times/where-it-meets/who-
teaches-it). This is why the router prompt populates `course_codes` for
availability questions too — not just catalog facts — or the gate would never
open. See `BANNER_API.md` for the endpoint research.
- **Source of truth vs. the static index.** Chroma/graph are a static snapshot
  from a Railway Volume; Banner is *current, per-section, real-time* enrollment
  from `banner.uvic.ca` registration self-service (no auth for class search).
  Banner owns **only** live seats/waitlist, section schedule/room, instructor,
  delivery/campus — prereqs/credits/program requirements stay with the graph.
- **`banner_retrieve(course_codes, season, year)`** resolves the term (default:
  nearest current/upcoming non-"View Only" term; router-named term via
  `term_season`/`term_year` wins — today's date is injected into the router
  prompt so "next spring" resolves), then per course: `searchResults` (all
  sections) + one `getFacultyMeetingTimes` per CRN (instructor; `searchResults`
  returns `faculty: []`). Best-effort — returns `{}` on any failure so the answer
  never breaks.
- **Caching is in-process + ephemeral (NOT the Volume)** — freshness is the point.
  Module-level `requests.Session` (3-step handshake, reused, `threading.Lock`-
  guarded since the SSE generator runs in a threadpool), 120s TTL on section
  payloads (seat counts must stay honest), hours-long TTL on the term list and
  per-section instructor. See `BANNER_API.md` "Caching".
- `_banner_context_text()` (in `chatbot.py`) renders this into numbered `[n]`
  blocks tagged `source=banner`, inserted **graph → banner → vector** so all three
  share one continuous `[n]` sequence. `_assemble_context()` owns that ordering
  (used by both `ask()` and api.py's stream path).

### 3. Vector retrieval — Voyage + Chroma (reverse-HyDE), with distance cutoff
- **Model:** `voyage-4-large`. Queries embed with `input_type="query"`; the
  index was built with `input_type="document"` (asymmetric — must stay
  matched; don't "simplify" this).
- **Store:** Chroma `PersistentClient`, one DB, **two collections**
  (`georgebot_v22_undergrad` ~42,585 vectors, `georgebot_v22_faculty`
  ~20,350), cosine similarity. One entry per reverse-HyDE question (5 per
  chunk); the *question* is embedded, the *full parent chunk text* is the
  entry's `document`. `vector_retrieve(query, audience, …)` opens both at
  load, embeds the query once, and queries whichever the audience selects;
  for `both` it merges hits from the two by distance (`_merge_collapse`)
  before collapsing. See the Audience note in the TL;DR.
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

### 4. Fetch + answer — `MiniMax-M3` (thinking disabled)
- `_build_context()` numbers graph blocks first, then vector chunks
  (`source=webpage|document`, `type=<document_type>`,
  `department=<department>`, `topics=<pipe-joined topic_families>`) + URL
  + full chunk text.
- **The assembled context is injected into the SYSTEM message, not the user
  turn** — `_system_prompt_with_context()` wraps it in explicit
  "SYSTEM-SUPPLIED REFERENCE MATERIAL (the user didn't write or see this)"
  delimiters, and `_answer_messages()` puts only the question in the user
  turn. This is what stops the model treating the material as user-provided.
- `answer()`/`answer_stream()` call MiniMax-M3 with `thinking: "disabled"`,
  `max_tokens=1500`, the context-augmented system prompt, and the last
  `MAX_HISTORY_TURNS` conversation turns. Both defensively strip any leaked
  `<think>` block (see "Single LLM provider" above); `answer_stream`
  forwards visible deltas incrementally (token-by-token).

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
- **Reference material is framed as SYSTEM-supplied, and the word "chunk"
  never appears in the prompt** (explicit user direction). The prompt tells
  the model the user didn't write or see the material, that some of it may
  be irrelevant (ignore it), and that if none is relevant it should answer
  from its own knowledge — never announcing missing/insufficient context.
  Runs on `thinking: "disabled"`, so the prompt carries this behavior itself
  (see "what NOT to over-fix" #1).
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
- **If the material doesn't cover what's asked, don't say so and don't
  describe what it does cover instead** — answer directly and helpfully as
  you would with no context, and for any specific UVic fact you can't
  confirm, point the user to where to get it (e.g. "check your syllabus or
  ask your professor") instead of explaining you don't have it. (See "what
  NOT to over-fix" #1 above — as of 2026-07-14 this is a deliberately
  enforced prompt rule + SYSTEM-message framing, not a single plain
  paragraph leaning on adaptive thinking.)
- Course outlines marked **HISTORICAL** are past-term snapshots; always
  name the term when citing one; never present historical grading/
  instructors/schedules as current.
- `source=kuali` chunks are the authoritative, current catalog — trust
  them over web pages for prereqs/credits/program requirements.
- `source=banner` blocks are **live** registration data for the term named in
  the block (seats, waitlist, section times/rooms, instructor) — use them for
  "is it full / seats left / when does it meet / who teaches it". Always name the
  term, cite section codes (A01), flag full/waitlist-only sections, and present
  counts as current-as-of-now (they change constantly), not a guarantee.
- If a program search returned multiple candidates, ask the user to
  clarify rather than guessing.
- Concise, plain text (no LaTeX).

---

## Known issues / open items

- **MiniMax account-level Token Plan rate limit (429)**: opaque,
  token-throughput-based (not a flat request cap). Hit reliably under
  back-to-back calls (e.g. batch testing) — surfaces as `rate_limit_error
  (2062)` and, in `rewrite_and_route`, falls back to vector-only. The answer
  step now runs `thinking: "disabled"` (~4.8s avg vs ~7.8s on adaptive), so
  per-call latency is lower, but the throughput ceiling is unchanged. If it
  becomes a live-traffic problem: a dedicated/higher Token Plan tier,
  pay-as-you-go, or request queuing/backoff.
- **Railway Volume — DONE (2026-07-15).** `georgebot-volume` at `/data`,
  `DATA_DIR=/data`, seeded with the v2.2 artifacts and live in production
  (`/health` → `chunks: 62935`). Re-seeding procedure is in "Data" above.
- **Cold start after a redeploy — first few requests are slow, then fast.**
  Expected, not a bug: a fresh container has none of the ~1.2 GB `chroma_db`
  in the OS page cache, and the Volume is network-attached storage, so early
  queries read the HNSW index + SQLite off disk; once the working set is
  cached in RAM (and the Voyage/MiniMax keep-alive connections are
  established, and each collection's HNSW graph is loaded on its first query)
  it hits steady-state speed. Self-resolving. Optional fix if it ever
  matters: a boot warm-up that fires a throwaway retrieval per collection
  before serving real traffic (not implemented).
- **This repo has no venv/lockfile of its own yet** — currently dev-tested
  by borrowing `georgebot-pipeline`'s venv. Fine short-term since both
  repos share a machine during this transition, but worth fixing before
  this repo is truly standalone.
- **Frontend**: `App.tsx` uses the streaming endpoint (`askGeorgeStream` →
  `/api/chat/stream`); tokens now arrive incrementally and a client-side
  typewriter (`REVEAL_CHARS_PER_TICK`/`REVEAL_TICK_MS`) paces the reveal.
  Sources are buffered and shown only after the reveal completes. Login is a
  mock shell (no real OAuth). `mockData.ts` exists but is unused.

---

## Frontend (chat UI)

- React + Vite + TypeScript, Tailwind v4 + shadcn/ui.
- `App.tsx` holds the message list; on send it appends a user msg + a
  loading assistant msg, then calls `askGeorgeStream(text, priorMessages,
  audience, handlers)` (streaming). Tokens land in a queue and a client-side
  typewriter (`REVEAL_CHARS_PER_TICK=3` / `REVEAL_TICK_MS=16`) paces the reveal
  so bursty deltas read smoothly. **Sources are buffered and attached only when
  the reveal finishes** (`finish()`), so they appear with the completed
  answer, not mid-stream. `askGeorge()` (non-streaming `/api/chat`) still
  exists in `lib/api.ts` but isn't used by the UI.
- **Audience toggle** (`components/AudienceToggle.tsx`): a segmented
  Undergrad / Faculty / Both control rendered above `ChatInput`; the selection
  (`Audience` in `types.ts`) lives in `App` state, defaults to `undergrad`, and
  is passed into every `askGeorgeStream` call as the request `audience`. This is
  the only audience signal — the backend does not guess it.
- **Status line** (`MessageBubble.tsx`): while the assistant message is
  `loading`, a small muted line (from the `status` SSE event, stored on
  `Message.status`) shows the pre-answer phase before the dots; it's cleared
  when the first `token` arrives.
- `lib/api.ts`: `askGeorgeStream(question, priorMessages, audience, handlers)`
  POSTs to `${VITE_API_BASE ?? 'http://127.0.0.1:5001'}/api/chat/stream` and
  hand-parses the SSE frames (`status`/`sources`/`token`/`done`/`error`); builds
  `history` from prior non-loading messages; maps API sources to the UI `Source`
  type. `cleanTitle()` trims the raw page-`<title>` breadcrumb tail (drops
  faculty/university/`Home` segments, keeps the two most specific parts) so
  source labels read cleanly.
- **Theme / dark mode** (`components/ThemeToggle.tsx`): a sun/moon button in
  the header (left of the account menu) toggles the `.dark` class on
  `document.documentElement` (the CSS variant in `index.css`), persisted to
  `localStorage` and initialized from saved choice or `prefers-color-scheme`.
  Light-mode page background is white with slightly-darker (`--muted`) chat
  bubbles; dark-mode background is a dark grey (`oklch(0.19)`), a step below
  the card/bubble surfaces so they stay layered.
- **Branding**: browser tab uses `frontend/public/favicon.svg` (a violet-
  gradient "G" mark) + `<title>GeorgeBot</title>`; the assistant message
  avatar (`MessageBubble.tsx`) and header both render a "G".

### Vector chunk metadata schema (what you can filter / rank / display on)
Every Chroma entry (one per question) carries, in `metadata`:
```json
{
  "chunk_id": "004b89e2d172a946#1",
  "doc_id": "004b89e2d172a946",
  "origin": "https://...",           // v2.2: was "url" in v2
  "title": "...",
  "document_type": "policy / regulation document",
  "department": "English",
  "source": "webpage | document | calendar",   // v2.2: was "source_file"
  "topic_families": "pipe | joined | string",
  "tf_degree_programs_requirements_curriculum": true,
  "tf_...": "one boolean per family this chunk belongs to",
  "question_index": 0,
  "question": "...",
  "src_doc_type": "layout", "src_filetype": "pdf", "src_path_prefix": "humanities"  // extra v2.2 passthrough, mostly unused at serve time
}
```
**v2.2 key renames the serving code depends on:** `url` → **`origin`**,
`source_file` → **`source`** (both read in `_build_context` / `format_sources`).
`department` and `tf_<slug>` booleans are the actual filter keys.
`topic_families` (pipe-joined) is display-only. `chunk_id` is what
`vector_retrieve()` collapses on. **The `tf_<slug>` set differs per collection**
(undergrad vs. faculty have disjoint families) — slug =
`"tf_"+re.sub(r"[^a-z0-9]+","_",name.lower())`, matching the pipeline's
`f_embed/embed.py`. `vector_taxonomy.json` carries both lists
(`topic_families_undergrad` / `topic_families_faculty`) so `_build_where`
uses the right one per audience.

`format_sources()` returns `source` values `webpage | document | kuali |
heat | banner` (plus occasional `calendar`). UI badge enum: `webpage → uvic_html`,
`document → uvic_docs`, `heat → heat`, `kuali → kuali`, `banner → banner`
(rendered as a rose "Live" badge in `SourceBadge.tsx`).

---

## Pointers

- Everything about *how the data was built* (crawling, chunking, embedding,
  taxonomy, course graph) — `georgebot-pipeline` (private, sibling repo),
  see its root `CLAUDE.md` and `data-pipeline/v2/CLAUDE.md`.
- This repo is read-only with respect to that data at serve time.
