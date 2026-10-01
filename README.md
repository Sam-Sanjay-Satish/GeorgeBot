# GeorgeBot

An AI assistant for University of Victoria students. It answers questions about courses, prerequisites, program requirements, registration, policies, and campus services, with citations. It can also look up live class availability and professor ratings.

**Live at [georgebot.org](https://georgebot.org)**

![GeorgeBot answering a live seat availability question for CSC 225](/images/SS1.png)

I built GeorgeBot on my own, from the first scrape to production. This repo holds the serving code (FastAPI backend and React frontend). The data pipeline that builds the search index lives in a separate private repo.

## What it does

- Answers from UVic's own pages and documents, and shows only the sources the answer actually used.
- Answers prerequisite and program-requirement questions from a course graph (3,761 courses, 280 programs) instead of guessing from text.
- Shows live seats, waitlists, schedules, and instructors by querying UVic's registration system.
- Adds professor ratings from RateMyProfessors, labeled as student opinion with the sample size.
- Lets you pick an audience (undergrad, faculty, or both) and a mode. Quick mode retrieves and answers. Default mode also checks whether the retrieved material is enough and fetches more (up to two rounds) if it isn't.

## How a question is answered

```
React frontend (Vercel)
  -> POST /api/chat/stream (server-sent events)
FastAPI backend (Railway, Docker)
  1. Router (one LLM call): rewrites the query and extracts course codes,
     program, term, named entities, and topic
  2. Course and program graph lookups (deterministic, no LLM)
  3. Live registration data from Banner (only when the question needs it)
  4. Professor ratings (only when the question needs it)
  5. Hybrid vector retrieval
  6. Context assembly, then a streamed answer with citations
  7. Sources panel filtered to what the answer cited
```

### Retrieval

- **Reverse-HyDE indexing.** For every chunk of content, an LLM writes 5 realistic student questions, and I embed the questions instead of the chunk. Students ask in everyday words, not the wording of a policy page, so this closes the vocabulary gap. I picked 5 questions per chunk from a recall sweep: recall@1 went from 0.90 with one question to 1.0 at three, and I added a cushion.
- **Hybrid search.** Three arms are fused with Reciprocal Rank Fusion: dense search over the question vectors, dense search over the chunk text, and BM25 scoped to a named entity (like a professor, building, or program). I tested seven arms offline first. BM25 over the whole query added noise on generic questions, but BM25 on just the entity phrase was the cleanest signal. Entity detection is folded into the existing router call, so it adds no extra LLM call.
- **Distance cutoff.** Off-topic queries retrieve nothing rather than being padded with loosely related chunks.

### Course and program graph

UVic's academic calendar API is parsed into typed requirement trees (17 node types, 96.6% of prerequisites fully structured) and stored as NetworkX graphs. Prerequisite chains, cross-listings, and program requirements are answered by graph lookups. Fuzzy program matching uses whole-word matching, after an early version let "CS" match 65 unrelated programs.

### Live data

![GeorgeBot summarizing RateMyProfessors ratings, labeled as student opinion](/images/SS2.png)

UVic's registration system (Banner 9) is queried for seats, waitlists, schedules, and instructors. Each lookup gets its own session, because Banner replays the previous search on a reused session, which leaked one course's sections into another's results. RateMyProfessors is queried through its unofficial GraphQL endpoint. Both integrations fail gracefully: if either is unavailable, the answer says it has no live data and never invents any.

## Data pipeline (private repo)

The pipeline was rebuilt three times (v1, v2, v2.2). The final version is a set of modular stages with JSONL in and out:

1. **Crawl** about 15K pages and documents across 30 UVic hosts, recall-first with trap detection and resumable checkpoints.
2. **Parse** HTML and PDFs (including a layout model for tables and headings, with a fallback to plain text when the layout output looks lossy).
3. **Scope and audience filter**, sorting each document into undergrad, faculty, or grad. About half the corpus turned out to be junk, so each drop is logged with a reason.
4. **Two-pass lossless topic split**: one pass decides topic boundaries, a second finds exact start markers, and the text is sliced with zero loss.
5. **Label and relevance filter** against a closed taxonomy, with exact-match validation after paraphrased labels slipped through an earlier validator.
6. **Reverse-HyDE questions and embedding** with Voyage into two Chroma collections.

The result is about 12.6K chunks indexed as about 63K question vectors. The final pipeline has 202 pytest tests.

## Measured results

| What I tested | Setup | Result |
|---|---|---|
| Hybrid vs. dense-only retrieval | 37 real student questions, full pipeline | Cited sources rose from 79 to 99, and chunks retrieved from 234 to 252, at about +0.4 s latency. Five regressions were found and documented |
| LLM provider migration | 40-question golden set through the real pipeline | DeepSeek Flash: 6.6 s median vs. 13.0 s for the larger model, and about $4.40 vs. $18.70 per 1,000 turns |
| Reasoning on vs. off | Router and answer step | Disabling "thinking" on the router was about 90x faster. On the answer step it cut average latency from 7.8 s to 4.8 s with no leaked retrieval internals in 8/8 test questions |
| Questions per chunk | 40 documents, 40 golden queries | Recall plateaued at 3 questions, so I use 5 |

## Production hardening

- **Rate limiting built for campus wifi.** UVic's network puts every student behind one IP, so a plain per-IP limit would throttle the whole campus as one user. Limits are per-device token buckets nested inside a looser per-IP bucket, plus a daily cap.
- **Bounded concurrency.** A dedicated worker pool, an in-flight request cap that rejects immediately instead of queueing, and a wall-clock timeout per turn.
- **Capped outbound traffic.** Router output is validated and capped before it triggers any Banner or RateMyProfessors calls, and all in-memory caches are bounded with TTLs.
- **Observability.** A SQLite query log (it stores no retrieved text), a token-gated admin dashboard that fails closed, and hourly synthetic probes through the real pipeline with per-phase timing.
- **Deployment.** Multi-stage Docker image on Railway with a persistent volume (about 1.4 GB of indexes and graphs), seeded over `railway ssh` with checksum verification and an atomic directory swap. The frontend is on Vercel.

## Tech stack

- **Backend:** Python 3.14, FastAPI, Uvicorn, server-sent events, SQLite
- **Models:** DeepSeek V4.1 Flash (routing and answers), Voyage `voyage-4-large` (embeddings)
- **Retrieval:** ChromaDB, BM25 (`rank-bm25`), NumPy, Reciprocal Rank Fusion, NetworkX
- **Frontend:** React 19, TypeScript, Vite, Tailwind CSS v4, shadcn/ui
- **Infrastructure:** Docker, Railway, Vercel

## Known limitations

- No graduate-student coverage yet. The assistant is instructed not to apply undergrad rules to grad students.

## Repo layout

```
backend/
  api.py              FastAPI app, rate limiting, concurrency, SSE
  chatbot.py          router, context assembly, answer and verify loop
  graph_queries.py    course and program graph accessors
  hybrid_retrieve.py  dense + dense + entity-scoped BM25, fused with RRF
  banner.py           live registration data
  rmp.py              professor ratings
  querylog.py         query logging
  warmup.py           hourly synthetic probes
  ttlcache.py         bounded TTL cache
frontend/             React + TypeScript app
```

## Running locally

The search index and graphs aren't included in this repo, so a full local run needs the artifacts built by the private pipeline. Set these environment variables: `DEEPSEEK_API_KEY`, `VOYAGE_API_KEY`, `DATA_DIR`, and optionally `ADMIN_TOKEN`, `HYBRID_RETRIEVAL_ENABLED`, and `CORS_ALLOW_ORIGINS`.

```
python3 backend/chatbot.py --ask "What are the prerequisites for CSC 225?"
python3 backend/api.py
cd frontend && npm run dev
```