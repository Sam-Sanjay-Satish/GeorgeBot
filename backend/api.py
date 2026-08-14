#!/usr/bin/env python3
"""
GeorgeBot — FastAPI RAG backend (for wiring up the React frontend during testing).

Reuses the retrieval + Claude pipeline from chatbot.py (the GeorgeBot engine) and
serves it over FastAPI/uvicorn instead of Flask.

Usage:
  python3 backend/api.py                  # run the server (default :5001)
  python3 backend/api.py --port 8000      # custom port

HTTP API:
  GET  /health
  POST /api/chat          {question, history?}  -> {answer, sources, ...}  (JSON)
  POST /api/chat/stream   {question, history?}  -> text/event-stream (SSE)
                          SSE events: status, sources, token, done, error
                          ("status" fires before tokens with a short, templated
                          phase message, e.g. "Looking up CSC 225…")
  GET  /admin             query-log viewer (HTML; prompts for the admin token)
  GET  /api/admin/*       query-log JSON/CSV, gated on ADMIN_TOKEN

Every chat turn is recorded to the query log (querylog.py) — best-effort, so a
logging failure never affects the answer.

Env (.env): MINIMAX_SUB_KEY, VOYAGE_API_KEY, ADMIN_TOKEN (admin routes)

Rate limiting (chat endpoints only; all optional, sane defaults):
  RATE_LIMIT_QUICK_PER_MIN / RATE_LIMIT_DEFAULT_PER_MIN  per-device refill rates
  RATE_LIMIT_BURST                                       per-device bucket size
  RATE_LIMIT_IP_QUICK_PER_MIN / RATE_LIMIT_IP_DEFAULT_PER_MIN  per-IP refill rates
  RATE_LIMIT_IP_BURST                                    per-IP bucket size
  DAILY_CHAT_CAP                                         global daily ceiling
  RATE_LIMIT_ENABLED=0                                   disable (load tests)

Concurrency (chat endpoints; all optional):
  CHAT_EXECUTOR_WORKERS   bounded thread pool for blocking chat work (default 16)
  MAX_INFLIGHT_CHAT       in-flight chat requests before 503 (default 12)
  CHAT_TIMEOUT_SECONDS    hard wall-clock ceiling per chat turn (default 45)
"""

import argparse
import asyncio
import csv
import hmac
import io
import itertools
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

import querylog
from admin_page import ADMIN_HTML
from banner import banner_instructor_retrieve, banner_retrieve
from chatbot import (
    DEFAULT_AUDIENCE,
    MAX_VERIFY_ROUNDS,
    VALID_AUDIENCES,
    GeorgeBot,
    _filter_cited_sources,
)

VALID_MODES = ("quick", "default")
# "quick" is the fallback on purpose: "default" is the expensive verify path
# (up to 3 MiniMax generations), so an unrecognized/omitted mode must land on
# the cheap pipeline, not the costly one. The frontend always sends its mode
# explicitly, so this only changes what bare API callers get.
DEFAULT_MODE = "quick"

# ----------------------------------------------------------------
# Rate limiting for the chat endpoints (the ones that cost real money:
# every request is at least one Voyage embedding + two MiniMax calls).
# Token buckets in TWO tiers, both of which must have a token for a request
# to pass, with a stricter bucket for mode="default" (the verify path can run
# three full generations), plus a global daily request ceiling as a spend
# backstop. CORS is browser-only policy and is NOT a defence here — these
# endpoints are effectively public API.
#
# WHY TWO TIERS (added 2026-08-04). Keying only on IP was wrong for this
# app's actual users: UVic campus wifi NATs the whole student body behind one
# public address, so a 10/min per-IP bucket meant ~10 questions per minute for
# ALL of campus and 429s for people who had asked nothing. The tiers now split
# the two jobs the single bucket was failing to do at once:
#
#   device tier — X-Client-Id, a UUID the frontend generates once and keeps in
#     localStorage. Strict (RATE_QUICK_PER_MIN), and it's what stops one
#     person spamming. Separates users behind a shared IP, which is the point.
#   IP tier — the real client address. Loose (RATE_IP_QUICK_PER_MIN, sized for
#     a whole NATed campus), and it's what stops one machine rotating fake
#     device IDs from having no limit at all.
#
# The device ID is client-supplied and therefore trivially spoofable — that is
# accepted, not overlooked. An attacker who rotates it per request falls
# through to the IP tier, and past that to MAX_INFLIGHT_CHAT and
# DAILY_CHAT_CAP, which are the real spend/load backstops. The device tier is
# a fairness mechanism between honest clients, not an authentication one.
#
# A request with NO usable X-Client-Id keys its device bucket on the IP
# instead, so bare API callers (curl, scripts) still get the strict limit —
# only browsers that opt in by sending an ID get per-device treatment.
#
# In-process state: correct for a single Railway replica. If this app ever
# scales past one replica, the buckets stop being global — move them to a
# shared store (e.g. Redis) at that point.
# ----------------------------------------------------------------

RATE_QUICK_PER_MIN = float(os.getenv("RATE_LIMIT_QUICK_PER_MIN", "10"))
RATE_DEFAULT_PER_MIN = float(os.getenv("RATE_LIMIT_DEFAULT_PER_MIN", "4"))
RATE_BURST = float(os.getenv("RATE_LIMIT_BURST", "5"))  # bucket capacity

# Per-IP tier: sized to hold a whole NATed campus, not one person. It exists
# to bound device-ID rotation, so it should stay well above what any single
# user can consume through the device tier and well below what would drain
# DAILY_CHAT_CAP in minutes.
RATE_IP_QUICK_PER_MIN = float(os.getenv("RATE_LIMIT_IP_QUICK_PER_MIN", "120"))
RATE_IP_DEFAULT_PER_MIN = float(os.getenv("RATE_LIMIT_IP_DEFAULT_PER_MIN", "40"))
RATE_IP_BURST = float(os.getenv("RATE_LIMIT_IP_BURST", "40"))

DAILY_CHAT_CAP = int(os.getenv("DAILY_CHAT_CAP", "2000"))  # all IPs combined
_MAX_BUCKETS = 10_000  # bound the attacker-keyed dict (unique-IP/ID floods)

# Device IDs are attacker-controlled strings, so they're never used raw as
# dict keys: anything not matching this (or longer than _MAX_CLIENT_ID) is
# discarded and the request falls back to IP-keyed strict limiting.
_CLIENT_ID_RE = re.compile(r"\A[A-Za-z0-9_-]{8,64}\Z")

_rate_lock = threading.Lock()
# (tier, key, mode) -> (tokens, last_ts); tier is "dev" or "ip"
_buckets: dict[tuple[str, str, str], tuple[float, float]] = {}
_daily = {"day": "", "count": 0}


def _client_ip(request: Request) -> str:
    """Best-effort client IP for rate-limit bucketing.

    On Railway every request arrives through the platform proxy, so
    request.client.host is the proxy, not the user. The proxy appends the
    real client to X-Forwarded-For; the RIGHTMOST entry is the one added by
    the trusted hop — anything left of it is client-supplied and spoofable,
    so it must not be used (or one attacker could rotate fake IPs and dodge
    the limiter, and worse, bucket real users together)."""
    xff = request.headers.get("X-Forwarded-For", "")
    if xff:
        return xff.rsplit(",", 1)[-1].strip()
    return request.client.host if request.client else "unknown"


def _client_device(request: Request) -> str | None:
    """The client's self-reported device ID, or None if absent/malformed.

    Spoofable by design (see the tier comment above) — validated only so an
    attacker can't use it to grow the bucket dict with huge keys."""
    cid = request.headers.get("X-Client-Id", "").strip()
    return cid if _CLIENT_ID_RE.match(cid) else None


def _take_token(tier: str, key: str, mode: str, per_min: float,
                burst: float, now: float) -> float | None:
    """Refill this bucket and spend one token.

    Returns None on success, or the seconds to wait if the bucket is empty.
    Caller must hold _rate_lock. The bucket is written back either way, so a
    rejected request still records its refill clock."""
    bkey = (tier, key, mode)
    tokens, last = _buckets.get(bkey, (burst, now))
    tokens = min(burst, tokens + (now - last) * per_min / 60.0)
    if tokens < 1.0:
        _buckets[bkey] = (tokens, now)
        return (1.0 - tokens) * 60.0 / per_min if per_min > 0 else 60.0
    _buckets[bkey] = (tokens - 1.0, now)
    return None


def _peek_token(tier: str, key: str, mode: str, per_min: float,
                burst: float, now: float) -> float | None:
    """Like _take_token but spends nothing — used to check the second tier
    before either is charged, so a request rejected by one tier doesn't
    silently drain the other's budget."""
    tokens, last = _buckets.get((tier, key, mode), (burst, now))
    tokens = min(burst, tokens + (now - last) * per_min / 60.0)
    if tokens < 1.0:
        return (1.0 - tokens) * 60.0 / per_min if per_min > 0 else 60.0
    return None


def _check_rate_limit(request: Request, mode: str) -> None:
    """Raise 429 (a bucket is empty) or 503 (daily ceiling hit).

    Token bucket per tier: starts full, refills continuously at the per-mode
    rate, each request spends one from BOTH tiers. RATE_LIMIT_ENABLED=0
    disables the whole thing (load tests, local batch runs)."""
    if os.getenv("RATE_LIMIT_ENABLED", "1").lower() in ("0", "false", "no"):
        return
    is_default = mode == "default"
    dev_per_min = RATE_DEFAULT_PER_MIN if is_default else RATE_QUICK_PER_MIN
    ip_per_min = RATE_IP_DEFAULT_PER_MIN if is_default else RATE_IP_QUICK_PER_MIN

    ip = _client_ip(request)
    # No usable device ID -> the strict tier keys on the IP, i.e. exactly the
    # old single-tier behaviour for non-browser callers.
    device = _client_device(request) or ip
    now = time.monotonic()

    with _rate_lock:
        # Global daily backstop: even a distributed abuser can't spend more
        # than DAILY_CHAT_CAP requests' worth of API credits in a UTC day.
        today = time.strftime("%Y-%m-%d", time.gmtime())
        if _daily["day"] != today:
            _daily["day"], _daily["count"] = today, 0
        if _daily["count"] >= DAILY_CHAT_CAP:
            raise HTTPException(
                status_code=503,
                detail="GeorgeBot has reached its daily capacity — please try again tomorrow.",
                headers={"Retry-After": "3600"})

        # Both tiers must have a token. Check the IP tier without spending
        # first, so a request the device tier is about to reject doesn't also
        # burn IP budget that its NAT neighbours are sharing.
        retry = _peek_token("ip", ip, mode, ip_per_min, RATE_IP_BURST, now)
        if retry is None:
            retry = _take_token("dev", device, mode, dev_per_min, RATE_BURST, now)
            if retry is None:
                _take_token("ip", ip, mode, ip_per_min, RATE_IP_BURST, now)
                _daily["count"] += 1
        if retry is not None:
            raise HTTPException(
                status_code=429,
                detail="Too many requests — please slow down.",
                headers={"Retry-After": str(int(retry) + 1)})

        # Keep the bucket dict bounded: it's keyed on client IPs and device
        # IDs, both of which an attacker controls. Evict idle entries first (a
        # full bucket refills from zero in under a minute at any sane rate, so
        # dropping stale state is harmless); a clear() only happens under an
        # active unique-key flood, where resetting bursts is the lesser evil.
        if len(_buckets) > _MAX_BUCKETS:
            cutoff = now - 600
            for k in [k for k, v in _buckets.items() if v[1] < cutoff]:
                del _buckets[k]
            if len(_buckets) > _MAX_BUCKETS:
                _buckets.clear()


# ----------------------------------------------------------------
# Concurrency control. Every route used to be a sync `def`, so Starlette ran
# them all on AnyIO's default 40-thread pool — ~40 concurrent slow chat turns
# (15s Banner/RMP timeouts, up to 3 MiniMax generations) starved EVERY route
# including /health, which lets Railway's healthcheck cycle the container
# right when it's under load. Now the chat endpoints are async and push all
# blocking work (retrieval, LLM calls, SQLite logging) onto this bounded,
# dedicated executor, with:
#   - an in-flight request cap that rejects excess load with 503+Retry-After
#     (never an unbounded queue), and
#   - a hard wall-clock ceiling per chat turn (CHAT_TIMEOUT_SECONDS).
# /health is async and touches nothing blocking, so it can't be starved.
# ----------------------------------------------------------------

CHAT_EXECUTOR_WORKERS = int(os.getenv("CHAT_EXECUTOR_WORKERS", "16"))
MAX_INFLIGHT_CHAT = int(os.getenv("MAX_INFLIGHT_CHAT", "12"))
CHAT_TIMEOUT_SECONDS = float(os.getenv("CHAT_TIMEOUT_SECONDS", "45"))

# A timed-out turn can't be interrupted mid-LLM-call — its worker thread runs
# to completion in the background. This pool being bounded is what caps how
# much of that orphaned work can pile up.
_chat_executor = ThreadPoolExecutor(max_workers=CHAT_EXECUTOR_WORKERS,
                                    thread_name_prefix="chat")

# In-flight chat slots, id -> expiry (time.monotonic). Touched only from the
# event-loop thread (async endpoints + async-generator teardown), so plain
# dict ops are race-free and no lock is needed. Each slot expires a bit past
# the request's own wall-clock ceiling, so any leak path (e.g. a response
# torn down before its generator ever ran) self-heals instead of permanently
# eating capacity.
_chat_slots: dict[int, float] = {}
_slot_ids = itertools.count()


def _acquire_chat_slot() -> int:
    """Claim an in-flight chat slot, or raise 503 immediately (never queue)."""
    now = time.monotonic()
    for sid, expires in list(_chat_slots.items()):
        if expires < now:
            del _chat_slots[sid]
    if len(_chat_slots) >= MAX_INFLIGHT_CHAT:
        raise HTTPException(
            status_code=503,
            detail="GeorgeBot is answering a lot of questions right now — "
                   "please try again in a moment.",
            headers={"Retry-After": "10"})
    sid = next(_slot_ids)
    _chat_slots[sid] = now + CHAT_TIMEOUT_SECONDS + 30.0
    return sid


def _release_chat_slot(slot_id: int) -> None:
    _chat_slots.pop(slot_id, None)


class ChatRequest(BaseModel):
    question: str = ""
    history: list[dict] = []
    # Which v2.2 corpus to search: "undergrad" | "faculty" | "both". Chosen by
    # the user via the frontend toggle; anything unrecognized falls back to the
    # default rather than erroring.
    audience: str = DEFAULT_AUDIENCE
    # "quick": today's flat retrieve->answer pipeline.
    # "default": retrieve->answer->self-verify, with one targeted re-fetch if
    # the model flags a gap (see _default_verified_events).
    mode: str = DEFAULT_MODE


def _clean_audience(value: str | None) -> str:
    return value if value in VALID_AUDIENCES else DEFAULT_AUDIENCE


def _clean_mode(value: str | None) -> str:
    return value if value in VALID_MODES else DEFAULT_MODE


def _course_label(code: str) -> str:
    """Display form of a normalized course code: "CSC225" -> "CSC 225"."""
    for i, ch in enumerate(code):
        if ch.isdigit():
            return f"{code[:i]} {code[i:]}".strip()
    return code


def _route_status(route: dict) -> str:
    """Short, Claude.ai-style status line derived purely from the route.

    Templated from known state (no LLM call, no latency). Falls back to a
    plain "Searching…" when nothing more specific is known."""
    if route.get("professor_query"):
        return f"Checking student ratings for {route['professor_query']}…"
    instr = route.get("instructor_query")
    if instr:
        return f"Looking up what {instr} is teaching…"
    codes = route.get("course_codes") or []
    if codes and route.get("wants_availability"):
        if len(codes) == 1:
            return f"Looking up current {_course_label(codes[0])} availability…"
        return "Looking up current class availability…"
    if codes:
        if len(codes) == 1:
            return f"Looking up {_course_label(codes[0])}…"
        return "Looking up courses…"
    if route.get("wants_rating"):
        return "Checking student ratings…"
    if route.get("program_query"):
        return "Checking program requirements…"
    return "Finding relevant UVic documents…"


def _require_admin(request) -> None:
    """Gate the /api/admin/* query-log endpoints on ADMIN_TOKEN.

    Fails CLOSED: with no ADMIN_TOKEN set in the environment the log is simply
    unreachable over HTTP (503), never open. The token may arrive as the
    X-Admin-Token header (what the admin page sends) or a ?token= query param
    (so the CSV link works as a plain browser download)."""
    expected = os.getenv("ADMIN_TOKEN", "")
    if not expected:
        raise HTTPException(status_code=503, detail="ADMIN_TOKEN is not configured")
    supplied = request.headers.get("X-Admin-Token") or \
        request.query_params.get("token") or ""
    if not hmac.compare_digest(supplied, expected):
        raise HTTPException(status_code=401, detail="bad admin token")


def _drain_events(events) -> dict:
    """Collect a `{type, data}` event generator (`_default_verified_events` --
    the same shape `/api/chat/stream` maps to SSE frames) into one
    JSON-friendly dict, for the non-streaming endpoint and the CLI."""
    answer_parts: list[str] = []
    sources: list[dict] = []
    error = None
    for ev in events:
        etype, data = ev.get("type"), ev.get("data")
        if etype == "token":
            answer_parts.append(data)
        elif etype == "sources":
            sources = data
        elif etype in ("error", "log_error"):
            error = data
    if error is not None:
        return {"error": error}
    return {"answer": "".join(answer_parts), "sources": sources,
            "n_chunks": len(sources)}


def _default_verified_events(bot: GeorgeBot, question: str, history: list[dict],
                             audience: str, route: dict):
    """Default mode's answering path: retrieve, then a combined verify-answer
    loop (`bot.answer_verified_stream`) capped at `MAX_VERIFY_ROUNDS` gated
    rounds before a forced plain answer. Yields `{type, data}` events that
    both the SSE handler and `_drain_events` can treat uniformly.

    Nothing is forwarded to the caller for a NEED_MORE round (per
    `answer_verified_stream`, it never yields a "token" before a verdict is
    known) -- `sources` and `token` events for a round only appear once that
    round's answer is confirmed SUFFICIENT, so a discarded round is never
    shown to the user. `sources` is emitted after all tokens (not before),
    once the model has reported which numbered blocks it actually relied on
    (see `_filter_cited_sources`) -- the frontend buffers sources until the
    answer finishes revealing regardless, so this adds no visible delay.

    A "failed" tuple-kind from `answer_verified_stream`/`answer_stream` (every
    retry produced an empty response -- see their docstrings) is surfaced here
    as a `log_error` event: an internal-only signal, not one of the SSE event
    types the frontend knows about. `generate()` (api.py's streaming handler)
    intercepts it to log the turn as a real query-log error instead of "ok";
    `_drain_events` does the same for the non-streaming endpoint/CLI. The
    user-visible apology "token" is still forwarded normally either way --
    this only fixes what gets logged, not what the client sees (see the
    2026-08-15 empty-stream-logged-as-ok bug this closes).
    """
    chunks, graph_facts, banner_facts, rmp_facts = bot.retrieve_with_route(route, audience)
    context, n_prefix_blocks = bot._assemble_context(chunks, graph_facts, banner_facts, rmp_facts)
    has_context = bool(chunks) or n_prefix_blocks > 0
    yield {"type": "status", "data": "Reading through sources…" if has_context else "Thinking…"}

    for _round in range(MAX_VERIFY_ROUNDS):
        need_more, cited = None, None
        for kind, payload in bot.answer_verified_stream(question, context, history):
            if kind == "token":
                yield {"type": "token", "data": payload}
            elif kind == "cited":
                cited = payload
            elif kind == "failed":
                yield {"type": "log_error", "data": payload}
            else:  # "need_more"
                need_more = payload
        if need_more is None:
            sources = _filter_cited_sources(
                bot.format_sources(chunks, graph_facts, banner_facts, rmp_facts), cited)
            yield {"type": "sources", "data": sources}
            yield {"type": "done"}
            return

        # Targeted re-fetch only for the piece the model flagged -- narrow by
        # design (see VERIFY_ANSWER_ADDENDUM). Nothing shown to the user yet.
        yield {"type": "status", "data": "Double-checking that…"}
        if need_more.get("search_query"):
            chunks = bot.vector_retrieve(
                need_more["search_query"], audience=audience,
                topic_families=route["topic_families"], department=route["department"])
        if (need_more.get("term_season") or need_more.get("term_year")) and route["course_codes"]:
            season = need_more.get("term_season") or route["term_season"]
            year = need_more.get("term_year") or route["term_year"]
            if route["wants_availability"]:
                banner_facts = banner_retrieve(route["course_codes"], season, year)
            elif route["instructor_query"]:
                banner_facts = banner_instructor_retrieve(route["instructor_query"], season, year)
        context, _ = bot._assemble_context(chunks, graph_facts, banner_facts, rmp_facts)

    # MAX_VERIFY_ROUNDS gated rounds exhausted, still NEED_MORE -> force a
    # plain answer (no verify wrapper) instead of gating a third time.
    cited = None
    for kind, payload in bot.answer_stream(question, context, history):
        if kind == "token":
            yield {"type": "token", "data": payload}
        elif kind == "failed":
            yield {"type": "log_error", "data": payload}
        else:  # "cited"
            cited = payload
    sources = _filter_cited_sources(
        bot.format_sources(chunks, graph_facts, banner_facts, rmp_facts), cited)
    yield {"type": "sources", "data": sources}
    yield {"type": "done"}


def _chat_turn_sync(bot: GeorgeBot, question: str, history: list[dict],
                    audience: str, mode: str) -> dict:
    """One full non-streaming chat turn — the blocking body of /api/chat.

    Runs inside the bounded chat executor, never on the event loop. Owns the
    turn's query-log lifecycle end to end: if the endpoint times out and stops
    waiting on the future, this keeps running to completion in its worker
    thread and still finishes its own log row (the bounded executor caps how
    much such orphaned work can pile up)."""
    session_id, turn_index = querylog.resolve_session(history)
    row_id = querylog.start_turn(
        session_id=session_id, turn_index=turn_index, question=question,
        history=history, audience=audience, mode=mode)
    t0 = time.monotonic()
    try:
        if mode == "quick":
            # bot.ask() doesn't hand back the route, only search_query.
            result = bot.ask(question, history, audience=audience)
        else:
            # retrieve -> answer -> self-verify, drained into one response.
            route = bot.rewrite_and_route(question, history, audience)
            querylog.update_route(row_id, route)
            events = _default_verified_events(bot, question, history,
                                              audience, route)
            result = _drain_events(events)
            result.setdefault("search_query", route["search_query"])
    except Exception as e:
        querylog.finish_turn(
            row_id, latency_ms=int((time.monotonic() - t0) * 1000),
            status="error", error=str(e))
        raise

    querylog.finish_turn(
        row_id, search_query=result.get("search_query"),
        answer=result.get("answer"), sources=result.get("sources") or [],
        latency_ms=int((time.monotonic() - t0) * 1000),
        status="error" if result.get("error") else "ok",
        error=result.get("error"))
    return result


def _start_stream_log(question: str, history: list[dict], audience: str,
                      mode: str) -> int | None:
    """resolve_session + start_turn for the streaming path. Both touch SQLite
    (on network-attached storage in prod), so this runs in the chat executor
    rather than on the event loop."""
    session_id, turn_index = querylog.resolve_session(history)
    return querylog.start_turn(
        session_id=session_id, turn_index=turn_index, question=question,
        history=history, audience=audience, mode=mode)


def create_app(bot: GeorgeBot) -> FastAPI:
    app = FastAPI(title="GeorgeBot")

    # CORS allow-list. Prod origins come from CORS_ALLOW_ORIGINS (comma-
    # separated, e.g. "https://georgebot.ca,https://www.georgebot.ca"); local
    # dev origins are always included so `npm run dev` keeps working. Set the
    # env var on Railway to your Vercel URL + custom domain(s). If it's unset
    # AND CORS_ALLOW_ALL isn't truthy, we fall back to dev-only origins (safe
    # default — a stray browser origin gets blocked, not silently allowed).
    dev_origins = [
        "http://localhost:5173", "http://127.0.0.1:5173",  # Vite dev server
        "http://localhost:3000", "http://127.0.0.1:3000",
    ]
    env_origins = [o.strip() for o in os.getenv("CORS_ALLOW_ORIGINS", "").split(",") if o.strip()]
    if os.getenv("CORS_ALLOW_ALL", "").lower() in ("1", "true", "yes"):
        allow_origins = ["*"]
    else:
        allow_origins = env_origins + dev_origins

    app.add_middleware(
        CORSMiddleware,
        allow_origins=allow_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Chunk count is snapshotted ONCE at startup: /health must stay cheap and
    # non-blocking (async, no per-hit Chroma/SQLite touch) so slow chat
    # traffic can never starve it — Railway cycles the container when its
    # healthcheck stalls, which used to be trivially inducible (issue 3). The
    # count only changes on a Volume re-seed + restart anyway, so a boot-time
    # snapshot stays accurate for the life of the process.
    try:
        chunk_count = sum(c.count() for c in bot.collections.values())
    except Exception:
        chunk_count = -1

    @app.get("/health")
    async def health():
        return {"status": "ok", "chunks": chunk_count}

    @app.post("/api/chat")
    async def chat(req: ChatRequest, request: Request):
        mode = _clean_mode(req.mode)
        # Rate-limit before doing ANY work (no log row, no session lookup) so
        # a rejected request is as close to free as possible.
        _check_rate_limit(request, mode)

        question = (req.question or "").strip()
        if not question:
            return {"error": "missing 'question'"}

        audience = _clean_audience(req.audience)
        slot = _acquire_chat_slot()  # 503 when saturated — never queue
        try:
            loop = asyncio.get_running_loop()
            return await asyncio.wait_for(
                loop.run_in_executor(
                    _chat_executor, _chat_turn_sync,
                    bot, question, req.history, audience, mode),
                timeout=CHAT_TIMEOUT_SECONDS)
        except TimeoutError:
            # The worker can't be interrupted mid-LLM-call; it finishes (and
            # logs its own row) in the bounded executor. The client just
            # stops waiting.
            raise HTTPException(
                status_code=504,
                detail="GeorgeBot took too long answering — please try again.")
        finally:
            _release_chat_slot(slot)

    @app.post("/api/chat/stream")
    async def chat_stream(req: ChatRequest, request: Request):
        mode = _clean_mode(req.mode)
        # Raised here (not inside the stream) so a limited request gets a
        # real 429/503 status line instead of a 200 SSE stream.
        _check_rate_limit(request, mode)

        question = (req.question or "").strip()
        if not question:
            return {"error": "missing 'question'"}

        audience = _clean_audience(req.audience)
        slot = _acquire_chat_slot()  # 503 (real status line) when saturated
        loop = asyncio.get_running_loop()

        # The turn is logged in two steps (see querylog.start_turn): the row is
        # written HERE, before a single token exists, because a client that
        # disconnects mid-stream tears the response down before any end-of-turn
        # code in `generate()` runs. Such a row stays 'started' and is later
        # swept to 'abandoned'.
        try:
            row_id = await loop.run_in_executor(
                _chat_executor, _start_stream_log,
                question, req.history, audience, mode)
        except BaseException:
            _release_chat_slot(slot)
            raise

        # Set when the client has gone away, the wall clock expired, or the
        # stream finished. The pump thread checks it between frames, and
        # _finish() checks it so a torn-down turn can never record itself as
        # a clean 'ok'.
        cancelled = threading.Event()

        def generate():
            t0 = time.monotonic()
            answer_parts: list[str] = []
            logged_sources: list[dict] = []

            def _finish(status: str, error: str | None = None):
                # Once the async side has torn the stream down, this row is
                # no longer ours to complete: it either stays 'started'
                # (client disconnect -> swept to 'abandoned') or was already
                # finished as a timeout error by sse_events(). Never let a
                # cancelled turn record itself as 'ok'.
                if cancelled.is_set():
                    return
                querylog.finish_turn(
                    row_id, answer="".join(answer_parts) or None,
                    sources=logged_sources,
                    latency_ms=int((time.monotonic() - t0) * 1000),
                    status=status, error=error)

            try:
                # Route is computed once up front (regardless of mode) so we
                # can emit an early "status" event during the gap before
                # retrieval/answer tokens start streaming. Status text is
                # templated from route state — no extra LLM call, no added
                # latency beyond the router call itself.
                route = bot.rewrite_and_route(question, req.history, audience)
                querylog.update_route(row_id, route)
                yield f"event: status\ndata: {json.dumps(_route_status(route))}\n\n"

                if mode == "default":
                    events = _default_verified_events(bot, question, req.history, audience, route)
                else:  # mode == "quick"
                    events = None

                if events is not None:
                    failed_reason = None
                    for ev in events:
                        etype = ev.get("type")
                        data = ev.get("data", {})
                        if etype == "token":
                            answer_parts.append(data)
                        elif etype == "sources":
                            logged_sources = data
                        elif etype == "log_error":
                            # Internal-only signal (see _default_verified_events)
                            # -- never forwarded as an SSE frame, the frontend
                            # doesn't know this event type. The user-visible
                            # apology "token" for this turn was already/will
                            # still be forwarded normally below.
                            failed_reason = data
                            continue
                        yield f"event: {etype}\ndata: {json.dumps(data)}\n\n"
                    _finish("error" if failed_reason else "ok", failed_reason)
                    return

                # Quick mode: today's flat retrieve -> answer pipeline, with a
                # sparing nudge in the answer prompt if there's a real gap.
                chunks, graph_facts, banner_facts, rmp_facts = bot.retrieve_with_route(
                    route, audience)
                context, n_prefix_blocks = bot._assemble_context(
                    chunks, graph_facts, banner_facts, rmp_facts)

                # Second status covers the answer-generation wait (the biggest
                # gap). If nothing was retrieved, "Reading sources…" would be
                # misleading, so fall back to a neutral "Thinking…".
                has_context = bool(chunks) or n_prefix_blocks > 0
                yield f"event: status\ndata: {json.dumps('Reading through sources…' if has_context else 'Thinking…')}\n\n"

                # Sources are emitted after all tokens (not before), once the
                # model has reported which numbered blocks it actually relied
                # on (see `_filter_cited_sources`) -- the frontend buffers
                # sources until the answer finishes revealing regardless, so
                # this adds no visible delay.
                cited = None
                failed_reason = None
                for kind, payload in bot.answer_stream(
                        question, context, req.history,
                        system_prompt=bot._quick_mode_system_prompt()):
                    if kind == "token":
                        answer_parts.append(payload)
                        yield f"event: token\ndata: {json.dumps(payload)}\n\n"
                    elif kind == "failed":
                        # Every retry (see answer_stream's docstring) came back
                        # empty -- the apology "token" above is still shown to
                        # the user normally; this only affects query-log status.
                        failed_reason = payload
                    else:  # "cited"
                        cited = payload
                sources = _filter_cited_sources(
                    bot.format_sources(chunks, graph_facts, banner_facts, rmp_facts), cited)
                logged_sources = sources
                yield f"event: sources\ndata: {json.dumps(sources)}\n\n"
                yield "event: done\ndata: {}\n\n"
                _finish("error" if failed_reason else "ok", failed_reason)
            except Exception as e:
                _finish("error", str(e))
                yield f"event: error\ndata: {json.dumps(str(e))}\n\n"

        async def sse_events():
            """Async SSE relay: runs the blocking `generate()` on the bounded
            chat executor and forwards its frames through an asyncio.Queue,
            so the event loop (and /health) never blocks on pipeline work.

            Client disconnect cancels THIS generator (CancelledError at the
            queue.get) — unlike the old sync path, we DO observe the teardown
            now. The two-step query-log semantics are preserved regardless:
            the finally below only flags `cancelled` and frees the slot, it
            never finishes the log row, so an abandoned turn's row stays
            'started' and is swept to 'abandoned' exactly as before.
            """
            queue: asyncio.Queue = asyncio.Queue()
            done_marker = object()

            def _emit(item) -> None:
                try:
                    loop.call_soon_threadsafe(queue.put_nowait, item)
                except RuntimeError:  # event loop already closed (shutdown)
                    cancelled.set()

            def _pump() -> None:
                it = generate()
                try:
                    while not cancelled.is_set():
                        try:
                            frame = next(it)
                        except StopIteration:
                            break
                        _emit(frame)
                finally:
                    # close() raises GeneratorExit inside generate() at its
                    # current yield; generate() has no finally and its
                    # `except Exception` doesn't catch GeneratorExit, so a
                    # cancelled turn's row is left 'started' (-> 'abandoned').
                    it.close()
                    _emit(done_marker)

            _chat_executor.submit(_pump)
            deadline = loop.time() + CHAT_TIMEOUT_SECONDS
            try:
                while True:
                    remaining = deadline - loop.time()
                    if remaining <= 0:
                        raise TimeoutError
                    item = await asyncio.wait_for(queue.get(), timeout=remaining)
                    if item is done_marker:
                        break
                    yield item
            except TimeoutError:
                # Hard wall-clock ceiling. The pump can't interrupt an
                # in-flight upstream call, but every one of those has its own
                # timeout and the executor is bounded, so it winds down on its
                # own. Record the timeout off-loop; `cancelled` (set first)
                # keeps the worker's _finish from overwriting it with 'ok'.
                cancelled.set()
                _chat_executor.submit(
                    querylog.finish_turn, row_id,
                    latency_ms=int(CHAT_TIMEOUT_SECONDS * 1000),
                    status="error",
                    error=f"wall-clock timeout ({int(CHAT_TIMEOUT_SECONDS)}s)")
                msg = "GeorgeBot took too long answering — please try again."
                yield f"event: error\ndata: {json.dumps(msg)}\n\n"
            finally:
                cancelled.set()
                _release_chat_slot(slot)

        return StreamingResponse(
            sse_events(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # ----------------------------------------------------------------
    # Admin: query-log viewer (see querylog.py + admin_page.py).
    # ----------------------------------------------------------------

    @app.get("/admin", response_class=HTMLResponse)
    def admin_page():
        # The page itself is unauthenticated -- it holds no data, it just
        # prompts for the token and calls the (gated) endpoints below.
        return HTMLResponse(ADMIN_HTML)

    @app.get("/api/admin/stats")
    def admin_stats(request: Request):
        _require_admin(request)
        return querylog.stats()

    @app.get("/api/admin/logs")
    def admin_logs(request: Request, limit: int = 100, offset: int = 0,
                   q: str | None = None, status: str | None = None,
                   since: str | None = None, zero_sources: bool = False):
        _require_admin(request)
        return querylog.recent_turns(limit=min(limit, 1000), offset=offset, q=q,
                                     status=status, since=since,
                                     zero_sources=zero_sources)

    @app.get("/api/admin/sessions")
    def admin_sessions(request: Request, limit: int = 50, offset: int = 0):
        _require_admin(request)
        return querylog.sessions(limit=min(limit, 1000), offset=offset)

    @app.get("/api/admin/sessions/{session_id}")
    def admin_session(request: Request, session_id: str):
        _require_admin(request)
        return querylog.session_turns(session_id)

    @app.get("/api/admin/logs.csv")
    def admin_logs_csv(request: Request, q: str | None = None,
                       status: str | None = None, since: str | None = None,
                       zero_sources: bool = False, limit: int = 100000):
        _require_admin(request)
        rows = querylog.recent_turns(limit=limit, q=q, status=status,
                                     since=since, zero_sources=zero_sources)
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=list(querylog.TURN_COLUMNS),
                                extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
        return StreamingResponse(
            iter([buf.getvalue()]), media_type="text/csv",
            headers={"Content-Disposition": 'attachment; filename="georgebot_logs.csv"'})

    return app


def main() -> None:
    # Railway injects PORT and expects a 0.0.0.0 bind; both are overridable
    # via env var or CLI flag for local dev (e.g. python3 api.py --port 8000).
    parser = argparse.ArgumentParser(description="GeorgeBot FastAPI backend")
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", 5001)),
                         help="server port (default 5001, or $PORT if set)")
    parser.add_argument("--host", default=os.getenv("HOST", "0.0.0.0"),
                         help="server host (default 0.0.0.0)")
    args = parser.parse_args()

    import uvicorn

    import warmup

    bot = GeorgeBot()
    app = create_app(bot)
    # Started before the port binds so the first cycle overlaps startup instead
    # of delaying it (see warmup.start). Daemon thread, never the chat executor.
    warmup.start(bot)
    print(f"Serving GeorgeBot (FastAPI) on http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
