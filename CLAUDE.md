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
   │                          + wants_availability + instructor_query
   │                          + term_season/term_year
   │                          + topic_families[] + department; the topic_family
   │                          vocabulary offered depends on audience)
   │
   ├─ graph_retrieve()   (only if course_codes / program_query present)
   │     └─ GraphStore (course_graph.pkl + program_graph.pkl + HEAT outlines)
   │        → prereqs, credits, cross-listings, descriptions, program
   │          requirements, outline text. No vector search. Audience-independent.
   │
   ├─ banner_retrieve() / banner_instructor_retrieve()  (LIVE data, gated)
   │     └─ banner.py — UVic Banner 9 registration JSON (banner.uvic.ca), NOT the
   │        static index. Two gated paths: (a) course availability, when course_codes
   │        AND wants_availability — live seats/waitlist, per-section schedule/room,
   │        instructor; (b) instructor→courses reverse lookup, when instructor_query —
   │        everything a named prof teaches that term (or an ambiguity/no-match note).
   │        In-process TTL cache. Best-effort ({} on failure). See BANNER_API.md.
   │
   ├─ rmp_retrieve(names)  (RateMyProfessors ratings, gated, runs AFTER banner)
   │     └─ rmp.py — RMP internal GraphQL (unofficial, public token), scoped to
   │        UVic. Fires on professor_query (a named prof's quality/rating) OR on a
   │        course-quality question (wants_rating) using the instructor names Banner
   │        just resolved. Returns avg rating/difficulty/would-take-again + up to 20
   │        recent reviews (or ambiguity/no-match note). Subjective STUDENT OPINION,
   │        not official data. In-process TTL cache. Best-effort ({} on failure).
   │
   ├─ vector_retrieve(query, audience)
   │     └─ Voyage query embedding (ONE) → queried against the selected
   │        collection(s) — georgebot_v22_undergrad and/or _faculty — top-40
   │        question-vectors each → distance-filtered (MAX_CHUNK_DISTANCE=0.75)
   │        → merged by distance across collections → collapse by chunk_id →
   │          up to N_CONTEXT=4 distinct chunks (full text)
   │
   └─ answer()  → MiniMax-M3 (official API, thinking DISABLED)
                  → answer from graph + banner + rmp + vector blocks (supplied via
                    the SYSTEM prompt, numbered together) + format_sources()
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

**No graduate coverage — and the answer prompt now says so (2026-08-04).**
There are exactly two collections; nothing in the corpus covers graduate program
requirements, graduate admissions, funding/awards, supervision, candidacy, or
thesis regulations. The risk this creates is not "we can't answer" — it's that a
grad question retrieves *undergraduate* regulations and the model serves them as
if they applied, which is a wrong answer rather than a partial one (a master's
student asking a drop deadline would have gotten the undergrad Oct 31 date).
`_ANSWER_RULES_HEAD` therefore opens with a **WHO YOU CURRENTLY SERVE** section
stating the two audiences, forbidding transfer of any undergraduate regulation,
deadline, fee, GPA/standing rule, or program requirement to a graduate student,
and routing grad questions to the Faculty of Graduate Studies / supervisor.
Two deliberate framings: it's phrased as a current gap ("not supported yet"),
not a refusal; and it explicitly tells the model **not** to over-correct on
audience-independent questions (library, parking, transit, counselling, IT,
recreation) — a grad student asking where the gym is should just get an answer
with no mention of the limitation. Verified 5/5 grad questions flag the gap,
2/2 neutral ones stay silent about it, undergrad answers unchanged.
If a graduate collection is ever added, this section is the first thing to
update — leaving it in place would then make the bot refuse content it has.

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
│   ├── rmp.py                 # RateMyProfessors ratings/reviews (unofficial GraphQL, UVic-scoped): TTL cache, best-effort. Student opinion, not official data
│   ├── RMP_API.md             # RMP endpoint/auth/school-id/query research + the best-effort caveat
│   ├── graph_queries.py       # GraphStore — copy of georgebot-pipeline's course-graph/graph_queries.py
│   ├── api.py                 # FastAPI server (thin wrapper over GeorgeBot); Railway PORT/HOST-aware
│   ├── querylog.py            # per-turn query log (SQLite on the Volume) — corpus-gap analysis; best-effort, never breaks an answer
│   ├── admin_page.py          # the /admin query-log viewer, one self-contained HTML string (no build step)
│   ├── warmup.py              # hourly full-pipeline warm probes + per-phase timing heartbeat (see "Warm probes")
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
`georgebot-pipeline`'s `course_graph/graph_queries.py`, patched four ways:
(1) paths use this repo's env-var scheme (`GRAPH_DATA_DIR`/`DATA_DIR`/
`./graph_data`) instead of the pipeline's `config.*`; (2) the outline loader
reads v2.2's **`heat_outlines.json`** — a flat `{code: {course, term, url,
text}}` dict, HEAT-only (eng/CS courses; others have no outline) — instead of
the old list-shaped `course_outlines_final.json`; (3) `search_programs()`
normalizes punctuation to spaces before token-matching (see the program-query
section below) — without it, a query carrying parens/hyphens (e.g. the exact
credential string the disambiguation flow displays back) matches nothing;
(4) `search_programs()` matches each query token against **whole words**
(prefix for title/credential, exact for `code`) instead of as a bare substring
of one flattened haystack string. Substring matching let a short token match
across word interiors, so `"CS"` returned **65** unrelated programs (`cs` is
inside "Economi*cs*", "Ethi*cs*", "Physi*cs*") and `"econ"` pulled in
"S*econ*dary"; prefix-matching the opaque `code` field then still surfaced
Canadian Studies (`DIPL-CSIS`) / Coastal Studies (`MNR-CSS`) while *missing*
Computer Science, hence exact-token for codes. Verified against ~26 realistic
queries: every legitimate result is unchanged (`math`→Mathematics and
`eng`→Engineering still work via prefix; `BA-ECAH`-style code search still
works), only the noise collapses. Note this is prefix matching, **not
abbreviation expansion** — `"CS"` and `"stats"` now legitimately return zero
matches (they always did for `stats`), and callers fall back to asking for the
exact program name. An alias/abbreviation map would be a taxonomy concern, so
it belongs in the pipeline repo, not here. The
**accessor API is identical** to the pipeline's (all 19 `GraphStore` methods).
Patches (3) and (4) live **only here** — deliberately not back-ported to the
pipeline (that repo is dormant for now); if the pipeline is ever revived and
re-copied, re-apply all four patches (diff first — they drift if edited
independently). The outline record shape is also read directly
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
heat_outlines.json}`, `vector_taxonomy.json`), ~1.3 GB of a 5 GB volume — plus
`query_logs.db` (see Query logging below).

⚠️ **`/data/query_logs.db` is the only NON-reproducible file on the Volume.**
Everything else can be rebuilt from `georgebot-pipeline`; the query log cannot.
A re-seed (step 3 above) must scope its `rm -rf` to the three artifact paths and
never wipe `/data` wholesale. Copy it out (`railway ssh -- "cat
/data/query_logs.db" > backup.db`) before any destructive volume work.

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
- Backend env vars: `MINIMAX_SUB_KEY`, `VOYAGE_API_KEY`, `ADMIN_TOKEN` (gates
  `/api/admin/*` — the query log; **unset means the log is unreachable over
  HTTP**, 503, never open). Railway also
  injects `PORT` automatically — `api.py`'s `main()` already reads
  `$PORT`/`$HOST` (default `0.0.0.0`) so no Railway-side config needed for
  that part; this was a real gap fixed on 2026-07-14 (originally hardcoded
  `127.0.0.1:5001`, which would not have bound correctly on Railway).
- **Chat rate limiting is two-tier** (`_check_rate_limit`, api.py): a strict
  bucket per **device** (`X-Client-Id`, a UUID the frontend keeps in
  `localStorage`) inside a loose bucket per **IP**, plus the global
  `DAILY_CHAT_CAP`. IP-only limiting was the original design and was wrong
  here — UVic campus wifi NATs every student behind one address, so the whole
  campus shared one 10/min bucket. A request with no `X-Client-Id` (curl,
  scripts) keys the strict tier on its IP, i.e. the old behaviour. The device
  ID is spoofable **by design** — it's fairness between honest clients, not
  auth; rotation falls through to the IP tier and the daily cap. See
  `issues.md` §1a before changing any of it. Effective capacity today is ~12
  concurrent turns (`MAX_INFLIGHT_CHAT`) ≈ 40 concurrent users, but
  `DAILY_CHAT_CAP=2000` binds first at ~300-600 users/day.
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
  (plus optionally `ADMIN_TOKEN` to try `/admin` locally) — this repo does
  **not** use `KESAR_API_KEY`/`CHINESE_API_KEY` (that
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
| GET  | `/admin` | — | the query-log viewer (HTML; prompts for the token) |
| GET  | `/api/admin/logs`, `/logs.csv`, `/sessions`, `/sessions/{id}`, `/stats` | — | query log, gated on `ADMIN_TOKEN` |

`audience` is `"undergrad"` (default) `| "faculty" | "both"`; unrecognized
values fall back to the default. The `status` SSE event fires before `token`s
with a short, templated phase line (e.g. `"Looking up CSC 225…"`, `"Reading
through sources…"`) derived from the route — no extra LLM call.

`history` is a list of `{role: "user"|"assistant", content: str}`.

```bash
curl -s -X POST http://127.0.0.1:5001/api/chat -H 'Content-Type: application/json' \
  -d '{"question":"How do I apply for co-op?"}'
```

### Query logging + the /admin viewer (landed 2026-07-31)

Every chat turn is recorded to a SQLite log (`backend/querylog.py`) so real
questions can be mined for **corpus gaps** — what students ask that retrieves
nothing, routes to the wrong department, or gets a generic answer. Stdlib only;
no new dependencies.

- **Location:** `$DATA_DIR/query_logs.db` (`/data/query_logs.db` in prod), same
  resolution scheme as `CHROMA_DIR`/`TAXONOMY_FILE`; `QUERY_LOG_DB` overrides the
  path, `QUERY_LOG_ENABLED=0` disables logging entirely. See the ⚠️ under
  "Current production Volume state" — it's the one file on the Volume a re-seed
  must not delete.
- **Per turn:** question, rewritten `search_query`, full route dict, final answer
  text, cited source titles/urls, audience, mode, latency, status. Retrieved
  **chunk text is deliberately not stored** — this is a usage log, not a cache.
- **Best-effort, always.** Every function swallows its own exceptions and prints
  to stderr; a broken/unwritable DB degrades to "no logs", never to a failed
  answer (same posture as `banner.py`/`rmp.py`). Verified against a read-only path.
- **Two-step write — do not "simplify" this.** `start_turn` INSERTs the row when
  the request arrives (status `started`), `update_route` attaches the route the
  moment the router returns, `finish_turn` UPDATEs with the answer at the end.
  The obvious single-write-in-a-`finally` version **silently loses every
  abandoned turn**: when the client closes the tab mid-stream, Starlette cancels
  the streaming response *without closing the sync generator*, so no
  `GeneratorExit`, no `finally`, no row. Measured on this stack, not assumed.
  Rows left `started` past `STALE_SECONDS` (10 min) are relabelled `abandoned` by
  `sweep_stale()`, which runs on each admin read.
- **Sessions are inferred, not client-supplied** (no frontend change). The
  frontend already sends the full untruncated history every turn, so
  `resolve_session()` keys a `session_chain` table on a sha256 of the
  conversation's **user** turns: empty history starts a new session, a matching
  prefix continues one. Caveat: two users whose questions match word-for-word
  from turn 1 collapse into one session — fine for gap analysis. If exact
  grouping ever matters, send a real `session_id` from `App.tsx` instead.
- **Reading it:** `GET /admin` — a self-contained HTML page (`admin_page.py`, no
  build step, no CDN, served from the backend so there's no CORS or Vercel
  involvement). It prompts once for `ADMIN_TOKEN`, keeps it in `localStorage`, and
  sends it as `X-Admin-Token`. Sessions view (transcript per conversation), Turns
  view (flat, searchable, status filter, **no-sources filter** — the primary gap
  signal), stats strip, CSV export. `_require_admin` fails **closed**: no
  `ADMIN_TOKEN` in the environment → 503 on every admin route.
- **CLI peek**, no server: `python3 backend/querylog.py 20`.
- ⚠️ **The local DB is gitignored, and it must stay that way — this repo is
  PUBLIC (fixed 2026-08-04).** With no `DATA_DIR` set, the path falls back to
  `backend/query_logs.db`, i.e. *inside the repo*, so just running the server
  locally drops a log of full question + answer text into the working tree. Two
  bugs had let that through and both are now fixed:
  - The ignore rule was written `backend/query_logs.db*   # local query log …`.
    **`.gitignore` has no trailing-comment syntax** — `#` only opens a comment as
    the first character of a line — so the whole string was one literal pattern
    matching nothing (`git check-ignore -v` confirmed no rule matched). The
    comment now sits on its own line. No other rule in that file has an inline
    comment; don't add one.
  - The files were also already **tracked** (committed in `15405a4`), and
    `.gitignore` never applies to tracked files. Cleared with
    `git rm --cached backend/query_logs.db*` (index only — the local DB is
    untouched on disk).
  - The pattern must keep the trailing `*`: SQLite holds un-checkpointed rows in
    the **`-wal`** sidecar, so `query_logs.db` can look empty (4 KB, zero tables)
    while `query_logs.db-wal` carries every question and answer — which is exactly
    what got committed. The 9 rows in that commit were synthetic test queries, not
    student traffic, so history was deliberately left alone rather than rewritten
    on an already-pushed public repo.
- Known thin spot: `mode=quick` on the non-streaming `/api/chat` logs
  `search_query` but no `route_json` — `bot.ask()` doesn't return the route. The
  streaming path (what the frontend uses) always logs it.

---

## Retrieval Pipeline — Full Detail (tune here)

### 1. Query rewrite + route — `MiniMax-M3` (thinking disabled)
`rewrite_and_route(question, history, audience)` is **one** call that returns
JSON: `search_query`, `course_codes` (normalized, no space — `"CSC 225"` →
`"CSC225"`), `program_query`, `wants_outline`, `wants_availability`, `instructor_query`
(prof name|null — reverse lookup), `term_season` (`spring|summer|fall|null`),
`term_year` (int|null),
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
  (truncated to 4000 chars, tagged HISTORICAL with its term, and numbered as
  its **own** `[n]` block, not folded into the course block — see the numbering
  contract in §4).
- ⚠️ **Two of those accessors return UNIONS, not requirement sets, and their
  block labels must keep saying so (fixed 2026-08-04).** Both were previously
  labelled as if they were authoritative, and since the block is tagged
  `source=kuali` — which `SYSTEM_PROMPT` calls the authoritative catalog — the
  answer model repeated them verbatim. Measured, not theorized:
  - `prereq_chain()` walks every prereq edge including the leaves *inside*
    `complete 1 of` groups, so it's a union of mutually exclusive alternatives.
    Labelled "Full prerequisite chain also requires (indirectly)", CSC 225 came
    out as requiring MATH 100, 101, 102, 109, 110, 120 **and** 151 — seven
    calculus courses where a student needs at most two. The label now states
    explicitly that it's a union including alternatives and must not be
    presented as required coursework or as a sequence.
  - `get_alternatives()` aggregates co-membership in *any* `1 of` group anywhere
    in the catalog, which is far weaker than equivalence: CHEM 101 → ASTR 101,
    GEOG 103, and **CHEM 102** (the next course); MATH 100 → **MATH 101**;
    CSC 115 → **CSC 110** (its own prereq) and **CSC 226**. Labelled
    "interchangeable with X wherever it's used as a prerequisite option", the
    model told students ASTR 101 substitutes for CHEM 101. The label now says
    it's co-occurrence only, explicitly warns it can contain the course's own
    prereq or a later course, and forbids claiming substitutability.
  Don't "tidy" either label back into a short phrase — the verbosity *is* the
  fix, and the short version is what produced the wrong answers.
- **A course code that isn't in the graph now emits a NOT FOUND block**
  (`graph_retrieve` returns `not_found: [{code, suggestions}]`). It used to be
  dropped silently, leaving the model with no idea a lookup had happened —
  so it answered from its own knowledge and stated invented catalog facts
  confidently ("PSYC 100" produced a fabricated prereq relationship between
  PSYC 100A and 100B). This matters far more than it sounds: **~225 course
  bases across 59 subjects exist ONLY as lettered variants** (PSYC 100, SOCI
  100, GEOG 101, LING 100, SPAN 100, BIOL 150/190, ATWP 100, most HSTR and MUS
  courses) and students type the base form. `_course_suggestions()` finds those
  variants (`PSYC100` → `PSYC100A`, `PSYC100B`) and the block tells the model to
  answer about them. With no near-miss it says so and — deliberately —
  forbids guessing at other codes the user might have meant, because the model
  otherwise invents plausible-looking ones (observed on `ENGL 135`, which is
  really a *rename* to ATWP 135; a rename/alias map would be a taxonomy concern
  and belongs in the pipeline repo).
- **Program query:** `_program_facts()` calls `search_programs()` (fuzzy,
  token-based, can return multiple candidates). `search_programs()` normalizes
  punctuation to spaces on both the query and the candidate "title code
  credential" haystack before token-matching, so parens/hyphens don't glue
  onto tokens — without this, the exact credential string the disambiguation
  flow itself displays (`"Computer Science (Bachelor of Science - Honours)"`)
  tokenizes to `(bachelor` / `honours)` and matches **nothing**, which is what
  used to trap users in a disambiguation loop. Each token then matches a whole
  **word** — by prefix for title/credential (`math`→Mathematics), exact for the
  opaque `code` — never as a substring of a flattened haystack; see patch (4)
  above for why (`"CS"` used to return 65 unrelated programs). A short
  abbreviation that isn't a real word prefix (`"CS"`, `"stats"`) therefore
  returns nothing and the caller asks for the exact name. **On multiple matches,
  `_program_facts` auto-selects the closest** via `_rank_program_matches()`
  rather than always asking: since every candidate already contains all the
  query's tokens, it ranks by *surplus* tokens (the candidate adding the least
  beyond what the user typed is the closest fit — e.g. "computer science
  honours" → standalone Honours, which adds only `{bachelor, of}`, beats
  Combined Honours, which also drags in `{and, mathematics, combined}`), tie-
  breaking on candidate size. It only auto-picks a **clear** winner; a genuine
  tie (bare `"computer science"`, where Major/Honours/Minor/General are equally
  close) still falls back to `ambiguous: True` and asks — silently guessing
  there would emit a real-but-wrong requirements list. When it does auto-pick,
  the rendered block sets `auto_selected` + `alternatives`, and instructs the
  answer model to state which program it assumed and note the user can ask
  about another. If exactly one match (or a clear auto-pick), pulls
  `get_program()` + `program_requirement_groups()` + `program_specializations()`
  + `program_courses()` (flat list of every course the program references,
  capped at 60 in the rendered block).
- ⚠️ **A zero-match program search is a NAME miss, not proof of non-existence —
  and both the retry and the block wording exist because it was treated as proof
  (fixed 2026-08-04).** Reported from production: "geog and environmental studies
  general degree" was answered with *"There isn't a 'Geography and Environmental
  Studies' combined degree at UVic"*, plus fabricated corroboration (*"UVic
  doesn't list one in the combined-major materials"* — nothing in context said
  that) and a leak of retrieval internals (*"A search of the academic calendar…
  came back with no match"*). Both General programs the student meant were
  sitting in the graph with full requirement trees: `MNR-GEGA`/`MNR-GEGS`
  (Geography BA/BSc, "General and Minor") and `MNR-ES` (Environmental Studies,
  "General and Minor"). Two independent causes, both fixed:
  - **`search_programs` is an all-tokens-AND matcher over a *single* program
    node**, so it under-matches in two ordinary cases. (a) A generic word no
    title contains kills the whole query: only 7 of 280 programs have a word
    starting with "degree" (the Law joint/double degrees, the post-degree BEds),
    so `"geography general degree"` → 0 while `"geography general"` → the 2
    correct programs. (b) A query naming **two** programs can never match one
    node — `geography` hits 9 programs, `environmental` hits 2, intersection 0 —
    and a UVic General degree *is* two areas, listed as one program per subject.
    `_program_facts` now retries via `_relaxed_program_queries()`: strip
    `_PROGRAM_FILLER` (query-side only; none of those words discriminates), then
    split on the `and`/`&`/`,`/`+`/`/` conjunction and copy any trailing
    `_PROGRAM_QUALIFIERS` word ("general", "major", "honours"…) onto whichever
    part lacks one, so "…general" applies to both subjects. Only reached **after**
    the strict search fails, which is why the split can't damage a real title
    containing "and" (`"Physical Geography and Earth and Ocean Sciences"` matches
    strictly and never gets there — verified). The relaxation lives in
    `chatbot.py`, deliberately **not** in `graph_queries.py`: that file is a copy
    of the pipeline's and already carries 4 local patches, and `search_programs`'
    strict AND semantics are still what the auto-select ranking depends on.
  - **The no-match block was a bare authoritative negative.** It rendered as
    `"No matching program found in the calendar."` tagged `source=kuali` — which
    `_ANSWER_RULES_HEAD` calls the authoritative catalog — so the model treated it
    as a fact and denied the degree existed. Same class as the `prereq_chain` /
    `get_alternatives` / `openSection` bugs: **our code asserted something the
    data did not mean.** `_program_block` now states that the lookup matches
    program *names* only, that an empty result is not evidence of non-existence,
    forbids saying a program/degree/combination isn't offered, forbids describing
    the lookup, notes that a two-subject degree is listed as two programs, and
    asks for the exact calendar name. A matching bullet in `_ANSWER_RULES_HEAD`
    (next to the multiple-candidates one) generalizes it beyond this block.
    **Don't shorten either back to a phrase** — the verbosity is the fix, exactly
    as with the graph-label labels above.
  - Note the *course* path already had this guard (the NOT FOUND block with
    near-miss suggestions); only the program path was missing it. Verified live:
    the reported query now returns both General programs and asks BA-vs-BSc; a
    genuine no-match ("underwater basketweaving") hedges correctly instead of
    denying; auto-pick, ambiguity, and the course path are unchanged.
  - ⚠️ **A multi-part split means COMPONENTS OF ONE REQUEST, not a menu — and
    `_relaxed_note` has to say so explicitly.** The first version of the note
    ended "Answer about this program and make clear which one it is", which was
    written for the single-program case and is actively wrong for a split: the
    model led with "there isn't a single program called that" and then asked the
    user to pick *one* of the two — the original wrong answer in a softer form,
    since someone naming two subjects for a General degree wants both, and each
    part is already a real program. It also bridged the gap by volunteering an
    invented "Combined Major such as Geography and Environmental Studies" that
    does not exist. The note now branches on `len(parts) > 1`: cover every part,
    don't open by denying the combined name, only ask a clarifying question for a
    part that is *itself* ambiguous (BA vs BSc) and scope it to that part, and
    never name or speculate about a combined/joint program absent from the
    blocks. Verified: answers now lay out ES General's actual course list *and*
    the Geography BA/BSc split, with no invented program. Residual, accepted:
    the model still sometimes opens with "UVic doesn't offer a single combined
    …" despite the instruction — but that statement is now *true* and is followed
    by real requirements, so it's a style wobble, not the wrong answer. Like the
    DATES and ARITHMETIC rules this is prompt-confidence, not deterministic:
    **re-measure across several runs before concluding it regressed.**
- ⚠️ **Relaxed matching can emit MORE THAN ONE program block.** A relaxed
  multi-subject result carries `parts: [<single-program result>, …]` and renders
  one numbered block per entry — so the program contribution to the block count
  is no longer a hard-coded `1`. All three walkers of the numbering contract
  (§4) derive it from the single helper **`_n_program_blocks()`**:
  `_graph_context_text` / `_n_graph_blocks` / `format_sources`. Ambiguity and
  no-match parts still take a number and contribute no source. If you add
  another program shape, change that helper — don't re-count inline in three
  places, which is precisely how the numbering drifted before.
- These accessors run as a deterministic **bulk fetch** — the router names a
  course/program, the code pulls the full fact set; the LLM does not pick
  accessors. Two `GraphStore` methods remain intentionally unused:
  `get_prereqs` (superseded by `get_eligibility`'s option-group view) and
  `course_with_programs` (a convenience wrapper that only re-bundles accessors
  already called here) — wiring either in would duplicate facts already shown.
- `_graph_context_text()` renders this into numbered `[n]` blocks tagged
  `source=kuali`, same numbering scheme as vector chunks.

### 2b. Live availability — `banner.py` (UVic Banner 9, no LLM, no vector search)
Gated live-data step, parallel to graph retrieval. Two firing conditions
(mutually exclusive): (a) **`course_codes` non-empty AND `wants_availability`**
(seats/openings/waitlist/"is X full"/section times/where-it-meets/who-teaches-it)
→ `banner_retrieve`; (b) **`instructor_query` set** (which courses does prof X
teach) → `banner_instructor_retrieve`. Case (a) is why the router populates
`course_codes` for availability questions too — not just catalog facts — or the
gate would never open. See `BANNER_API.md` for the endpoint research.
- **Source of truth vs. the static index.** Chroma/graph are a static snapshot
  from a Railway Volume; Banner is *current, per-section, real-time* enrollment
  from `banner.uvic.ca` registration self-service (no auth for class search).
  Banner owns **only** live seats/waitlist, section schedule/room, instructor,
  delivery/campus — prereqs/credits/program requirements stay with the graph.
- ⚠️ **Term default is date-aware — do NOT revert it to "earliest registerable"
  (fixed 2026-08-04).** `resolve_term()` used to return the numerically smallest
  non-"View Only" code, on the assumption that non-View-Only == {current +
  upcoming}. It isn't: UVic leaves a term registerable well past its add
  deadline, so on 2026-08-04 the default resolved to **Summer 2026 (202605)** —
  a term ending that month — when every asker meant Fall. Two costs, both
  measured: seat/schedule answers described the wrong term, and because a
  summer term carries a fraction of the catalog, most courses returned `{}`
  and the turn degraded into a punt that burned both `MAX_VERIFY_ROUNDS`
  NEED_MORE rounds (19–25s) and leaked "I don't have current registration
  data". `_default_term()` now skips any term that started more than
  `TERM_ADD_GRACE_DAYS` (30) ago when a later registerable term exists, falling
  back to the old behaviour if all are stale — so it can only change *which*
  term is picked, never whether one is. A router-named term still wins outright.
  Verified across the year: May 5→Summer, Jun 20→Fall, Aug 4→Fall, Sep 15→Fall,
  Oct 20→Spring, Jan 10→Spring, Apr 25→Summer.
- ⚠️ **A router-named term must still be a LIVE term (fixed 2026-08-04).** The
  season+year branch of `resolve_term` used to accept any code present in the
  term list — **including "(View Only)" past terms** — while the season-only
  branch already preferred registerable ones, so the *more specific* hint carried
  the *weaker* guard. Reached in production: the router intermittently invents a
  term for a question that named none ("are there seats in csc 320?" was
  rewritten to "…winter 2026"), which resolved to **202601 (Jan–Apr 2026, View
  Only)** and returned frozen seat counts for a term that had already ended —
  bypassing `_default_term` entirely, since a named term wins by design.
  `resolve_term(..., require_registerable=True)` (the default) now refuses a dead
  term and falls back to the date-aware default; pass `False` to ask what a hint
  literally names. `requested_term_code()` lets the caller see what was declined,
  and `banner_retrieve` sets **`requested_term_label`** so `_banner_context_text`
  emits an explicit note — otherwise we would silently answer about a different
  term than the user named, which is a new wrong answer rather than a fix.
  Note the router slip is intermittent (4/4 clean on re-test), so **don't
  conclude it's gone from one sample** — the guard is the durable half.
- ⚠️ **Section status comes from the SEAT COUNT, not Banner's `openSection`.**
  They are independent fields and disagree constantly: `openSection` means "the
  department has this section open for registration", not "it has free seats".
  CSC 320 A01 (Fall 2026) is `seats=0 / openSection=True`, as is most of CSC 110
  — all of which rendered as `"0 of 125 seats open — OPEN"`, the exact wrong word
  for "are there seats?". `_banner_section_line` now derives one of four states
  from all three signals: `CLOSED — not open for registration` (openSection
  false, e.g. CSC 320 T02), `SEATS AVAILABLE`, `FULL — no seats left, but
  waitlist space remains`, `FULL — no seats and no waitlist space`. Same class of
  bug as the graph-label ones: our code was asserting something the field did not
  mean.
- **`banner_retrieve(course_codes, season, year)`** (course availability) resolves
  the term (default: `_default_term`, see the ⚠️ above; router-named term
  via `term_season`/`term_year` wins — today's date is injected into the router
  prompt so "next spring" resolves), then per course: `searchResults` (all
  sections) + one `getFacultyMeetingTimes` per CRN, **fanned out over a thread pool**
  since `searchResults` returns `faculty: []`. Best-effort — returns `{}` on any
  failure so the answer never breaks.
- **`banner_instructor_retrieve(name, season, year)`** (reverse lookup: what a prof
  teaches) fires on `instructor_query`. `get_instructor` resolves the name → a
  **session-scoped ephemeral instructor code** (NOT a stable id — it changes every
  session), so the resolve + `txt_instructor` search must run back-to-back on one
  session; `banner.search_by_instructor` uses its own **dedicated session** (as every
  search here does) so that ephemeral state stays isolated, caching only the result
  (120s). A common surname returns multiple `get_instructor` matches → the block tells
  the model to ask the user to disambiguate (like the ambiguous-program flow). See the
  `get_instructor` gotcha in `BANNER_API.md`.
- **Caching is in-process + ephemeral (NOT the Volume)** — freshness is the point.
  Each search uses its **own dedicated `requests.Session`** (`_handshake_session`),
  NOT a shared one: Banner's `searchResults` replays the session's *previous* search
  unless the form is reset, so a shared session leaks one course's sections into the
  next query and races under concurrency. The `_lock` only guards the TTL cache dicts.
  120s TTL on section payloads (seat counts must stay honest), hours-long TTL on the
  term list and per-section instructor. See `BANNER_API.md` "Request flow"/"Caching".
- **Outbound concurrency: one request must not be able to starve the others
  (tuned 2026-08-04, alongside the term-default fix).** Three knobs interact:
  `MAX_OUTBOUND_CONCURRENCY` (8, global in-flight cap — this is the number that
  protects banner.uvic.ca), `MAX_COURSE_FANOUT` (half that, the per-course
  faculty fan-out), and `_SEM_ACQUIRE_TIMEOUT` (8s, how long a waiter queues).
  The fan-out used to equal the global cap and the timeout was 2s, which was
  survivable only while the term default landed on Summer — a handful of
  sections per course. Once the default became a real teaching term (CSC 110
  Fall 2026: **43** sections, MATH 100: 25), a single availability lookup took
  every global slot and concurrent Banner requests timed out into `{}`, i.e.
  into exactly the punting/leaking answer the term fix was meant to remove.
  Measured: 3 concurrent cold lookups went 2-of-3 starved → 3-of-3 served.
  Note the timeout bounds **queueing, not load** — peak concurrency is the
  semaphore either way, so giving up early buys UVic nothing and costs a wrong
  answer. Don't lower it back "to be polite"; lower `MAX_OUTBOUND_CONCURRENCY`
  if real politeness is needed. Cold-cache cost is paid once per section per
  `FACULTY_TTL` (6h).
- `_banner_context_text()` (in `chatbot.py`) renders this into numbered `[n]`
  blocks tagged `source=banner`, inserted **graph → banner → rmp → vector** so all
  share one continuous `[n]` sequence. `_assemble_context()` owns that ordering
  (used by both `ask()` and api.py's stream path).

### 2c. Professor ratings — `rmp.py` (RateMyProfessors, no LLM, no vector search)
Second gated live-data step, runs **after** Banner so it can reuse Banner's
resolved instructor names. RMP has **no official API** — `rmp.py` posts to the
internal GraphQL endpoint (`www.ratemyprofessors.com/graphql`) with a hardcoded
public Basic-auth token (base64 `test:test`, no key to provision), scoped to
UVic's base64 school id (`RMP_SCHOOL_ID`, "School-1488"). Unofficial, so
best-effort ({} on failure), like Banner. See `RMP_API.md`.
- **What it owns:** subjective *student opinion* only — avg quality/difficulty
  rating, would-take-again %, rating count, and up to `REVIEWS_N=20` recent
  written reviews. NOT official UVic data; the prompt frames it as opinion and
  requires attribution + the sample size. Who-teaches-what / seats / prereqs
  stay with Banner / graph.
- **Two firing conditions**, decided by `_rmp_retrieve_for(route, banner_facts)`:
  (a) router set `professor_query` (a named prof's quality/rating/reviews) →
  look that name up directly; (b) router set `wants_rating` on a *course*-quality
  question → chain on the instructor names Banner just resolved
  (`_instructor_names_from_banner`, both Banner shapes). Neither → no call.
  `professor_query` wins if both are present.
- **One GraphQL search call** returns the summary fields on the teacher node
  directly; a **second call** pulls the 20 reviews only for a single clean match.
  Name resolution mirrors Banner's instructor semantics: exactly one distinct
  match → summary + reviews; several (common surname) → `candidates` for the
  model to disambiguate; none → a no-match note (never fabricate a rating).
- **In-process TTL cache** keyed by queried name, `RMP_TTL=6h` (ratings barely
  move). Unlike Banner there's no per-session form-state leak, so a single shared
  `requests.Session` is used, not a dedicated-per-call one.
- `_rmp_context_text()` renders numbered `[n]` blocks tagged `source=rmp` (see
  the ordering note above). `_route_status` (api.py) has matching status lines.

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

**Metadata filtering (`department` + `topic_families`) — soft filter plus an
UNCONDITIONAL unfiltered pass**, on top of the distance cutoff. The router
predicts up to 3 `topic_families` and one `department` from the controlled
vocabulary in `vector_taxonomy.json`. `vector_retrieve()` runs a filtered pass
first (`department $in [predicted, "general / cross-departmental"]` AND/OR
`tf_<slug>` booleans) for up to `N_CONTEXT` chunks, then **always** adds up to
`N_UNFILTERED_BACKFILL` more from an unfiltered pass (deduped) — so a normal
turn sees up to 7 chunks. Both passes respect `MAX_CHUNK_DISTANCE`.

⚠️ **The unfiltered pass used to be conditional (`only if the filtered pass
returned fewer than N_CONTEXT`) and that was a wrong-answer bug — fixed
2026-08-04. The old claim here, "a wrong/narrow prediction degrades to *no
filtering benefit*, not *wrong answer*", is FALSE; don't restore it or the
conditional.** A program spanning two departments carries exactly ONE
`department` tag, so whichever department the router names, the other's
documents are filtered out — and when the filtered pass still returns a full
`N_CONTEXT`, the backfill never fires and they are never seen. Measured on the
reported failure: "geog and environmental studies general degree" routed to
`department="Environmental Studies"`, while the **Geography and Environmental
Studies Double Major** worksheets are tagged `department="Geography"`. Those
worksheets are the three globally-nearest chunks in the corpus for that query
(d=0.446/0.460/0.470) and were displaced by chunks ~0.05 worse, so the answer
told a student the degree they were asking about doesn't exist. The same class
of miss was measured on other cross-departmental programs (biology+psychology,
math+economics) — it is systematic, not a one-off.

**Why the filtered budget stays at `N_CONTEXT` instead of shrinking to make
room.** A 3-filtered + 3-unfiltered split was measured first: it fixes the
target case but evicts the filtered pass's 4th chunk, losing real content on
**5 of 10** test queries (the tuition query lost "International Undergraduate
Tuition Fees"). 4+2 keeps the same total with no losses but retrieves only 2 of
the 3 double-major chunks and misses the BA worksheet. **4+3** loses nothing
(strict superset of the old behaviour) and catches all three, at ~1.75x context;
4+4 bought nothing more. Verified end-to-end: the answer now names the Double
Major and reproduces the worksheet's requirements exactly.

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
- **Cited-sources filtering (2026-07-28).** `format_sources()` numbers every
  retrieved graph/banner/rmp/vector block, but not everything the router
  retrieves is actually relevant (see `MAX_CHUNK_DISTANCE`'s caveat above) —
  previously the full retrieved list was shown to the user regardless of
  whether the model actually used it. The `SYSTEM_PROMPT`'s **CITED
  SOURCES** section now requires the model to end every answer with a
  machine-readable `<<CITED_SOURCES: 1,3>>` (or `<<CITED_SOURCES: none>>`)
  marker reporting which `[n]` blocks it actually relied on. `answer()`/
  `answer_stream()`/`answer_verified_stream()` all strip this marker before
  any text reaches the user (`_extract_cited_sources` non-streaming,
  `_split_cited_sources` streaming — same buffer-then-decide idiom as the
  `<think>`-tag stripper, so it works even split across stream chunks) and
  return/yield the parsed numbers alongside the answer. `_filter_cited_sources`
  then trims `format_sources()`'s output down to just those numbers before
  it's sent to the frontend — `cited=None` (marker missing/malformed) fails
  open and shows everything rather than hiding real sources. In the
  streaming endpoint this means the `sources` SSE event now fires *after*
  all `token` events (previously before) — harmless, since the frontend
  already buffers sources until the answer finishes revealing regardless.
- ⚠️ **THE BLOCK-NUMBERING CONTRACT (fixed 2026-08-04) — read before touching
  any renderer.** `_filter_cited_sources` resolves the model's `[n]` against
  `format_sources()`'s `n`, so **`n` is the block's index in the assembled
  context, not its position in the source list.** Four functions assign or
  consume those numbers and must walk the *same* sequence:
  `_graph_context_text` → `_banner_context_text` → `_rmp_context_text` →
  `_build_context` (assigning), `_n_graph_blocks` (the offset banner/rmp
  continue from), and `format_sources` (what the numbers resolve to). They had
  drifted in four places, each silently attaching the **wrong** sources to an
  answer: a course outline was numbered by `format_sources` but folded into the
  course block in the context (verified: every citation after an outline was off
  by one); and an ambiguous/no-match *program*, an unmatched *RMP* name, and the
  one-block *Banner instructor* shape each took a context number that
  `format_sources` skipped. The rule now: **every numbered block advances `n`;
  only citable blocks append a source entry.** A not-found course, an ambiguity
  note, and a no-match note are instructions to the model, not pages — they
  consume a number and contribute nothing. `scratchpad/test_numbering.py`-style
  coverage (16 fact shapes, asserting context numbering is contiguous, source
  `n`s are a strictly-increasing subset of it, and each vector chunk resolves to
  its own source) is the cheapest way to re-verify after any renderer change.
  **The program contribution is variable** — a relaxed multi-subject match
  renders one block per program (see the `_n_program_blocks` note in §2) — so
  all three walkers derive that count from that one helper rather than assuming
  `1`. Re-verified after that change across 7 program shapes × 6
  course/not-found combinations.

### Tunable constants (top of `chatbot.py`)

| Const | Value | Effect |
|---|---|---|
| `VOYAGE_MODEL` | `voyage-4-large` | embedding model (query + doc must match index) |
| `MINIMAX_MODEL` | `MiniMax-M3` (official API) | query rewrite + route classifier, and final answer |
| `QUESTION_K` | 40 | question-vectors pulled from Chroma before collapsing |
| `MAX_CHUNK_DISTANCE` | 0.75 | cosine distance cutoff — chunks worse than this are dropped, not backfilled |
| `N_CONTEXT` | 4 | max distinct chunks the **filtered** pass contributes (graph blocks additive, uncapped; can be fewer if the cutoff filters hard) |
| `N_UNFILTERED_BACKFILL` | 3 | chunks the **unfiltered** pass always adds on top — so up to 7 reach the answer model. Do not make this conditional again; see the ⚠️ under "Metadata filtering" |
| `LLM_MAX_TOKENS` | 4000 | route/rewrite budget |
| `ANSWER_MAX_TOKENS` | 1500 | answer budget |
| `MAX_HISTORY_TURNS` | 6 | trailing turns kept for rewrite + answer |

### System prompt (answer model) — behavioral contract

**Two assembled prompts, one shared accuracy contract (2026-08-04).** The
answer contract is composed from three class constants in `chatbot.py` rather
than written out twice:

```
SYSTEM_PROMPT       = _ANSWER_RULES_HEAD + _DEFAULT_STYLE + _CITED_SOURCES_RULES   # default mode
QUICK_SYSTEM_PROMPT = _ANSWER_RULES_HEAD + _QUICK_STYLE   + _CITED_SOURCES_RULES   # quick mode
```

`_ANSWER_RULES_HEAD` (the SYSTEM-supplied framing + every ACCURACY rule) and
`_CITED_SOURCES_RULES` are **shared verbatim** — quick mode is narrower in
what it *volunteers*, never looser about facts, and that's guaranteed by
construction rather than by keeping two texts in sync. **Don't fork the head**;
edit it and both modes move together. `SYSTEM_PROMPT` is byte-identical to what
it was before the split (verified against git HEAD), so default mode is
unchanged. `_QUICK_STYLE` adds the SCOPE AND STYLE section: answer exactly
what was asked and nothing else — no preamble, no restating the question, no
closing/summary line, no unsolicited tips, next steps, caveats, alternatives,
or related courses/programs; length follows the question (one sentence if that
answers it); prose by default, short lists only when the answer really is a
list. Its **last bullet is load-bearing**: it explicitly exempts the
qualifications the ACCURACY rules mandate (naming a Banner/HISTORICAL term,
section codes, RMP attribution + sample size, asking on an ambiguous program
match) from "nothing extra" — without it a hard brevity rule reads as license
to drop them.

Quick mode's prompt reaches the model via `answer(…, system_prompt=…)` /
`answer_stream(…, system_prompt=…)` → `_system_prompt_with_context`. Both quick
entry points use it: `api.py`'s SSE path (via `_quick_mode_system_prompt()`,
now a thin accessor) and `bot.ask()` (non-streaming `/api/chat` + the CLI
`--ask` smoke test) — `ask()` previously used the default prompt by oversight,
which made CLI answers read longer than what the frontend actually showed.

**The old QUICK MODE nudge was removed** in the same change. It let the model
append one closing sentence suggesting the user retry with quick mode off when
it spotted a gap — the single thing quick mode was licensed to add beyond the
answer, and exactly what the new scope rules forbid. Quick mode now has no
in-band gap signal at all (it has no verification step either — that's default
mode's `answer_verified_stream`); if one is wanted back, surface it in the UI,
not in the answer text.

- **Reference material is framed as SYSTEM-supplied, and the word "chunk"
  never appears in the prompt** (explicit user direction). The prompt tells
  the model the user didn't write or see the material, that some of it may
  be irrelevant (ignore it), and that if none is relevant it should answer
  from its own knowledge — never announcing missing/insufficient context.
  Runs on `thinking: "disabled"`, so the prompt carries this behavior itself
  (see "what NOT to over-fix" #1).
- **Today's date reaches the ANSWER step, not just the router (2026-08-04).**
  `_system_prompt_with_context` injects a `TODAY'S DATE: YYYY-MM-DD` line, and
  `_ANSWER_RULES_HEAD` has a **DATES** bullet that keys off it. Two placement
  decisions are load-bearing:
  - It's computed **per call**, not baked into the prompt class constants —
    those are class-level and would freeze at import time, so a long-running
    container would answer with the date it booted on.
  - It sits **outside** the `=== BEGIN SYSTEM-SUPPLIED REFERENCE MATERIAL ===`
    delimiters. Everything inside them is framed as "gathered automatically, may
    be irrelevant, ignore what doesn't help" — precisely the wrong framing for
    the one fact the model must never ignore.

  The corpus is a static snapshot, so without a "now" the answer step read every
  date it found as still upcoming: an acting appointment whose stated term ended
  June 30 2026 was reported as who *currently* holds the role, and January 2026
  info sessions were described as "their next ones". The rule now requires
  checking any date against today before saying "current"/"upcoming"/"next",
  and covers events, deadlines, published fees/rates, and terms of office.
  Two guards in it exist because the first draft caused each failure in testing:
  it must **keep the figure with its as-of date** rather than dropping it
  (the ONECard fee went from "$25 (as of Aug 15 2025)" to a contentless "there's
  a replacement fee"), and it must **not manufacture precision** ("mid-March"
  became "March 15, 2027"). `_QUICK_STYLE`'s exemption bullet lists the date
  qualification too, so quick mode can't shed it as "extra".
  Verified: legitimately-future dates are NOT over-hedged — reading break, fall
  start, drop deadlines and residence rates all keep full detail, and some
  answers improved ("today is August 4, so your next tuition deadline is
  September 30, 2026").
  ⚠️ **This holds in DEFAULT mode only. Quick mode does not reliably apply it —
  known and accepted, don't assume otherwise.** Asked who the current president
  is, quick mode answers in the present tense about an appointment that ended
  June 30 2026: 4/4 failures, still 3/3 after adding a "this is a CORRECTNESS
  requirement, not a caveat" clause to the shared head, and still 4/4 after
  additionally restating the rule as the last bullet of `_QUICK_STYLE` for
  recency (that restatement was reverted — it bought nothing and was dead
  prompt weight). The plausible mechanism is that default mode's
  `VERIFY_ANSWER_ADDENDUM` forces an explicit sufficiency assessment before
  answering, and that deliberation is what surfaces the date check; quick mode
  is one flat generation under a "answer exactly what was asked and nothing
  else" instruction that sits *later* in the prompt than the DATES rule.
  Default is what the frontend ships (`App.tsx` defaults `thinkingMode` to
  `'default'`), so the shipped path is covered; quick mode is opt-in, plus
  `ask()` and the `--ask` CLI. If this is ever worth closing, the lead is
  giving quick mode a cheap pre-answer check rather than more prompt text —
  more prompt text has now been measured not to work.
- **ARITHMETIC rule — the model was inventing an input, not miscalculating
  (2026-08-04).** Asked "how much is tuition at UVic", the answer step correctly
  sourced `$436.21 per credit unit` and then added, unsourced, "most full-time
  students take **15 units per term** (5 courses), which works out to roughly
  **$6,540 per term**" — about **double** the real per-term cost, rendered with
  `cited=[1,2,3]` so it looked sourced. High-traffic question for an ad-driven
  audience, and silent.
  - **Root cause is a wrong-country prior, and the model's own numbers prove
    it.** "5 courses = 15 units" is only coherent if a course is 3 units — the
    US credit-hour convention. UVic courses are 1.5 units, so 5 courses is 7.5
    units/term; **15 units is the UVic *year* figure, not the term figure.**
    Corroborated independently: UVic's own viewbook sample first-year tuition
    is ~$6,414, i.e. what the model quoted as one term is really two.
  - **Diagnosed before fixing, and the diagnosis changed the fix.** The
    course-load fact IS in the corpus and is highly retrievable (d≈0.45 for
    "what is a full-time course load at UVic"; 11 chunks carry it, including the
    calendar's part-time/full-time definition) — it just isn't what a *tuition*
    query retrieves (that pulls fee tables and admissions pages, none of which
    state a load). So this is **not** a `georgebot-pipeline` corpus gap, and
    equally it is **not** fixed by retrieving the load: a single headline
    "typical term" total is misleading regardless, since it varies by program
    and excludes ancillary fees. The right behaviour is to not invent the input
    and not produce the total.
  - The rule therefore constrains **inputs, not arithmetic** — a blunt "never
    compute totals" would break the genuinely good case ("how much does one
    course cost" → 1.5 × $436.21, which now works and cites its source). It
    permits calculation when every input is stated or user-supplied, names the
    specific bad prior (unit system, not credit-hours; never assume 3-credit
    courses or a 15-credit semester), and otherwise requires the per-unit rate
    plus a pointer to UVic's tuition estimator.
  - Verified 3/3 on the failing query (fabricated total gone, one run even
    deriving the correct "5 courses × 1.5 = 7.5 units"), with the compute-when-
    given-a-load cases preserved. Where a genuinely sourced total exists the
    model still uses it *with* its stated assumptions — which is the intent, so
    don't "fix" a cited multi-term figure that names its own basis.
  - Unlike the graph-label fixes, **no line of our code was asserting anything
    false here** — the fee table, retrieval, and labels were all correct. That
    makes this a prompt-confidence fix, not a deterministic one; treat future
    regressions as "re-measure across several runs", not "the rule is gone".
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
- **CITED SOURCES**: every answer must end with a `<<CITED_SOURCES: 1,3>>`
  (or `<<CITED_SOURCES: none>>`) marker naming which `[n]` blocks were
  actually relied on — never described or mentioned in the visible answer.
  This is stripped before the user ever sees it and drives which sources
  the frontend shows (see "Cited-sources filtering" under §4 above) — it's
  the mechanism that turns "silently drop chunks that don't address the
  question" (previous bullet) into something the Sources panel actually
  reflects, instead of always listing everything retrieved.
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
- `source=rmp` blocks are third-party **student opinion** from RateMyProfessors
  (rating/difficulty/would-take-again + recent reviews) — subjective and
  self-selected, NOT official UVic data. Use only for "is this prof any good /
  what are they like" questions; attribute explicitly, always give the rating
  *with* its sample size (few ratings = weak evidence), present it as opinion,
  and never fabricate a rating (say so if there's no listing; disambiguate a
  common surname). The prompt owns this contract on `thinking: "disabled"`.
- If a program search returned multiple candidates, ask the user to
  clarify rather than guessing.
- Concise, plain text (no LaTeX) — default mode. Quick mode replaces this
  bullet with the much stricter SCOPE AND STYLE section described at the top
  of this section; everything above it is shared by both.

---

## Known issues / open items

- ⚠️ **THE PROGRAM GRAPH IS MISSING EVERY DOUBLE MAJOR — fix belongs in
  `georgebot-pipeline` (found 2026-08-04).** `program_graph.pkl` has **280
  programs and zero `Double Major` credentials.** It has `Bachelor of Science -
  Combined Major`, `Combined Honours` and `Bachelor of Fine Arts - Combined
  Major`, and the two Law *Double Degrees* — but the entire Double Major class
  is absent. Meanwhile the vector corpus carries **111 program-planning
  worksheets, ~81 of them naming two subjects**, including
  `ppw-ss-geog-es-ba.pdf` / `ppw-ss-geog-es-bsc.pdf` ("Geography and
  Environmental Studies Double Major", BA and BSc, 60 units, effective Sept
  2026). So the program exists, UVic publishes a worksheet for it, and the
  worksheet is indexed — the **graph** just can't see it.
  - This is what produced the user-reported wrong answer: the graph found no
    match, and because graph blocks are tagged `source=kuali` — which
    `_ANSWER_RULES_HEAD` calls the authoritative catalog — the model reported
    the program doesn't exist. Fixing the retrieval filter (see "Metadata
    filtering") makes the worksheet reachable and the live answer is now
    correct, but that is a **mitigation**: the graph is still wrong, so a
    question needing structured double-major facts (prereq chains, requirement
    groups, `requirements_remaining`) still can't be served from the graph.
  - **Serve-time code must therefore not present the program graph as a complete
    list of what UVic offers.** `_relaxed_note` and the program no-match block in
    `_program_block` both say explicitly that this index is incomplete for
    two-subject programs and that other reference material wins. Earlier wording
    asserted the opposite ("the catalog lists one program per subject") as fact —
    **do not reintroduce it.**
  - The real fix is regenerating `program_graph.pkl` in `georgebot-pipeline`
    (Kuali program ingest) so Double Majors land in the graph. Nothing in this
    repo can add them — this repo only serves what that one produces. When they
    land, the incompleteness warnings above should be revisited.

- **The extended-thinking plan system was removed (2026-07-27).** There used
  to be a "planner" layer where `rewrite_and_route(mode="default")` also
  classified `is_simple`/`plan`/`confidence`, and a not-simple question
  dispatched into one of three plans via `ExtendedThinking`
  (`backend/thinking.py`): `scattered_info` ("Deep Research" — an
  evaluator-driven multi-round retrieval loop), `course_planning` (multi-turn
  course scheduling with persisted slot state, `backend/scheduler.py`), and
  `situational` ("Guidance" — a fixed policy/procedure checklist).
  `course_planning` had too many bugs to be worth keeping, and rather than
  patch it in isolation, all three plans were removed together — there is no
  plan-dispatch system anymore, and `thinking.py`/`scheduler.py` are gone.
  `rewrite_and_route` no longer takes a `mode` param or returns
  `is_simple`/`plan`/`confidence` — it always returns the same routing
  fields regardless of caller. The Quick/Default toggle
  (`ExtendedThinkingToggle.tsx`, `ThinkingMode`) still exists but now means
  something simpler: Quick is the flat retrieve→answer pipeline; Default
  additionally self-verifies the answer and does one targeted re-fetch if
  the model flags a gap (`answer_verified_stream`/`MAX_VERIFY_ROUNDS`, via
  `api.py`'s `_default_verified_events`) — this was already independent of
  the plan system, just previously only reachable via the `is_simple=true`
  branch. If course planning (or research/guidance modes) get rebuilt later,
  they're starting from scratch — no scaffolding (`PlanningState`, the
  `planning_state` SSE event, `ModeTag.tsx`, `Message.mode`) was preserved.
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
  it hits steady-state speed. Self-resolving.
- ⚠️ **UNRESOLVED: the service also goes cold after a long IDLE.** A Chroma-only
  keep-warm was removed (2026-08-03) because it did not fix it, and replaced the
  same day by the full-pipeline warm probes in `backend/warmup.py` (see
  "Warm probes" below). The root cause is still unidentified — the new loop is
  as much instrumentation as mitigation. Read this before changing either.
  - **The symptom, measured against prod:** first request after ~2 days idle
    took **51.9s** server-side (`latency_ms` in the query log), then 8.2s and
    5.9s back-to-back. Railway logs showed **no container restart** — same
    process throughout — so this is not a container cold start.
  - **What was built and removed:** `WARM_INTERVAL_SECONDS` (300s) /
    `WARM_QUERIES` / `_warm_vectors` / `_warm_pass` / `warm_up` /
    `_keep_warm_loop` / `start_keep_warm` in `chatbot.py`, plus a `WARM_UP`-
    gated call block in `api.py`'s `main()`. A daemon thread re-queried Chroma
    with 6 cached query embeddings every 300s. It was confirmed **alive** in
    prod (`Keep-warm thread started (every 300s)` in the deploy log) during the
    51.9s request. It made **zero HTTP calls** by design (embeddings cached at
    boot), so it warmed only the `chroma.sqlite3` page cache — the cost
    separately measured at ~1.1s, i.e. it defended ~1s of a ~44s problem.
    Its boot pass also added 14.6s *before* uvicorn bound the port, making
    genuine container cold starts slower.
  - **Ruled out — don't re-investigate these:** container restart / Railway app
    sleep (no new `Starting Container` in the logs); the thread failing to
    start or being env-disabled (`WARM_UP`/`WARM_INTERVAL_SECONDS` are not set
    on Railway, so defaults applied).
  - **Stale outbound connection pools were the leading hypothesis and are NOT
    the answer, at least not within minutes.** `openai.OpenAI(...)` and
    `voyageai.Client(...)` (chatbot.py, in `__init__`) are constructed with no
    `timeout` and no `max_retries`, so SDK defaults (600s / 2 retries) apply —
    which predicted a ~20s hang before any retrieval. Disproved by direct
    measurement: after **16 min idle**, an SSE run timed `request → status`
    event (the router MiniMax call, which happens *before* embed/retrieve) at
    **1.97s**, first token at 3.84s, whole stream 6.95s. Fully warm.
  - **So the two data points bracket it:** ~2 days idle → 51.9s; ~16 min idle →
    6.9s. Whatever it is develops over **hours-to-days**, not minutes. That
    timescale matches the page-cache reclaim noted above (`file` cache observed
    falling 891 MB → 180 MB over 16 idle hours) — which would mean the ticker's
    idea was right but its working set (6 fixed queries, and the loop ran only
    the *unfiltered* pass while boot ran both) was far too small a slice of the
    968 MB SQLite. **This is inference, not measurement** — one 16-minute
    negative does not prove what the multi-day positive was.
  - **Next step to actually pin it:** `warmup.py`'s per-probe `route=` /
    `retrieve=` split is now logging exactly the attribution needed, once an
    hour, forever. Grep Railway for `[warm]` after a quiet stretch: if
    `route_ms` balloons it's the outbound MiniMax path; if `retrieve_ms`
    balloons it's Voyage/Chroma/the Volume page cache. That reading is what
    should finally close this out — no further experiment needs designing.

- **This repo has no venv/lockfile of its own yet** — currently dev-tested
  by borrowing `georgebot-pipeline`'s venv. Fine short-term since both
  repos share a machine during this transition, but worth fixing before
  this repo is truly standalone.
- **Frontend**: `App.tsx` uses the streaming endpoint (`askGeorgeStream` →
  `/api/chat/stream`); tokens now arrive incrementally and a client-side
  typewriter (`REVEAL_CHARS_PER_TICK`/`REVEAL_TICK_MS`) paces the reveal.
  Sources are buffered and shown only after the reveal completes. There is no
  login/auth — the app gates on a one-time `DisclaimerPage` (educational-use
  notice + a "Continue" button, state held as `acknowledged` in `App.tsx`);
  `mockData.ts` exists but is unused.

### Warm probes (`backend/warmup.py`, 2026-08-03)

Replaces the removed Chroma-only ticker. A daemon thread runs 5 synthetic
questions through the **real** pipeline every `WARM_INTERVAL_SECONDS`, so every
layer a live request touches is exercised: router LLM call, Voyage embedding,
**both** Chroma collections, course graph, Banner, RMP.

- **Probes are one-per-path**: `vector-undergrad`, `vector-faculty` (separate
  HNSW graph + SQLite pages — warming one does not warm the other), `graph`,
  `banner`, `rmp`.
- **Gates are pinned via `Probe.force`, not left to the router.** The router
  call still runs on every probe (that's what warms MiniMax, and it's half the
  timing split), but its result is overridden for the gated paths. Measured
  reason: M3's non-determinism (CLAUDE.md §1) had the *same* Banner/RMP question
  return a full route on one cycle and an empty one on the next, warming those
  layers on a coin flip. This pins warming coverage, **not** router correctness —
  the loop is not a router test.
- **RMP chains off Banner.** `_rmp_retrieve_for` only fires on `professor_query`
  OR (`wants_rating` AND non-empty `banner_facts`), so the RMP probe must open
  the Banner gate too. Chained deliberately rather than hardcoding a real
  professor's name into the repo.
- **The Banner probe's course must be OFFERED in the resolved term** or it
  returns `{}` and warms nothing (MATH 100 and ENGL 135 both came back empty in
  Summer 2026). `MISSED=banner` in the log means that first, not "Banner is
  down".
- **Never touches `querylog`** (probes call `bot.*` directly, not the HTTP
  endpoint — synthetic turns must not pollute corpus-gap analysis) and **never
  uses the chat executor**.
- **Env:** `WARM_INTERVAL_SECONDS` (default 3600), `WARM_ENABLED=0` to disable,
  `WARM_ANSWER=1` to also warm the answer generation. `WARM_ANSWER` is **off by
  default** — it would add a second MiniMax generation of up to
  `ANSWER_MAX_TOKENS` per probe against the documented token-throughput rate
  limit while warming nothing the router call hasn't (same endpoint, model,
  connection).
- ⚠️ **1h is an assumption, not a validated number.** The only interval proven
  to keep this service warm is ~16 min. If cold hits persist in the query log,
  lower `WARM_INTERVAL_SECONDS` first.
- Cycle cost ~15-25s wall, 5 router calls. Started before uvicorn binds so the
  first cycle overlaps startup — the removed boot warm-up ran *before* the port
  opened and added 14.6s to every container start, which the frontend health-gate
  had to sit through. Don't reintroduce that.
- Every cycle prints, success or failure. The removed ticker printed nothing on
  a good tick, which is why confirming it even ran took a live experiment.

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
- **Branding**: browser tab uses `frontend/public/favicon.svg` (a black
  rounded square with a white stroked "G") + `<title>GeorgeBot</title>`; the
  assistant message avatar (`MessageBubble.tsx`) and header both render a "G".
- **Palette: deliberately achromatic.** Every token in `index.css`'s `:root`
  and `.dark` is `oklch(L 0 0)` — chroma 0, pure greyscale — and `--primary`
  inverts between themes (near-black `0.205` in light, near-white `0.922` in
  dark), which is what flips the user bubbles and the "G" avatar. The only
  chromatic tokens are `--destructive` (red) and, in dark only, an unused
  `--sidebar-primary` (violet, a leftover shadcn default — there is no
  sidebar). `--chart-1..5` is a greyscale ramp and is also unused. The one
  place real color appears in the UI is `SourceBadge.tsx`: six Tailwind
  palette pairs (`100/800` light, `900/200` dark) — blue `uvic.ca`, amber
  `HEAT`, green `Catalog`, teal `Document`, rose `Live` (banner), orange
  `Rating` (rmp). No `.tsx` file contains a hex value or a gradient.

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
heat | banner | rmp` (plus occasional `calendar`). UI badge enum: `webpage →
uvic_html`, `document → uvic_docs`, `heat → heat`, `kuali → kuali`, `banner →
banner` (rose "Live" badge), `rmp → rmp` (orange "Rating" badge) — in
`SourceBadge.tsx`. Only *matched* RMP professors become a citable source (the
professor's RMP page); ambiguous/no-match RMP blocks are model prompts, not
sources.

---

## Pointers

- Everything about *how the data was built* (crawling, chunking, embedding,
  taxonomy, course graph) — `georgebot-pipeline` (private, sibling repo),
  see its root `CLAUDE.md` and `data-pipeline/v2/CLAUDE.md`.
- This repo is read-only with respect to that data at serve time.
