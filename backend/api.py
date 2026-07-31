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
"""

import argparse
import csv
import hmac
import io
import json
import os
import time

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
DEFAULT_MODE = "default"


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
        elif etype == "error":
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
        else:  # "cited"
            cited = payload
    sources = _filter_cited_sources(
        bot.format_sources(chunks, graph_facts, banner_facts, rmp_facts), cited)
    yield {"type": "sources", "data": sources}
    yield {"type": "done"}


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

    @app.get("/health")
    def health():
        return {"status": "ok",
                "chunks": sum(c.count() for c in bot.collections.values())}

    @app.post("/api/chat")
    def chat(req: ChatRequest):
        question = (req.question or "").strip()
        if not question:
            return {"error": "missing 'question'"}

        audience = _clean_audience(req.audience)
        mode = _clean_mode(req.mode)
        session_id, turn_index = querylog.resolve_session(req.history)
        row_id = querylog.start_turn(
            session_id=session_id, turn_index=turn_index, question=question,
            history=req.history, audience=audience, mode=mode)
        t0 = time.monotonic()
        try:
            if mode == "quick":
                # bot.ask() doesn't hand back the route, only search_query.
                result = bot.ask(question, req.history, audience=audience)
            else:
                # retrieve -> answer -> self-verify, drained into one response.
                route = bot.rewrite_and_route(question, req.history, audience)
                querylog.update_route(row_id, route)
                events = _default_verified_events(bot, question, req.history,
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

    @app.post("/api/chat/stream")
    def chat_stream(req: ChatRequest):
        question = (req.question or "").strip()
        if not question:
            return {"error": "missing 'question'"}

        audience = _clean_audience(req.audience)
        mode = _clean_mode(req.mode)
        session_id, turn_index = querylog.resolve_session(req.history)
        # The turn is logged in two steps (see querylog.start_turn): the row is
        # written HERE, before a single token exists, because a client that
        # disconnects mid-stream never lets `generate()` reach any end-of-turn
        # code -- Starlette cancels the response without closing the generator.
        # Such a row stays 'started' and is later swept to 'abandoned'.
        row_id = querylog.start_turn(
            session_id=session_id, turn_index=turn_index, question=question,
            history=req.history, audience=audience, mode=mode)

        def generate():
            t0 = time.monotonic()
            answer_parts: list[str] = []
            logged_sources: list[dict] = []

            def _finish(status: str, error: str | None = None):
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
                    for ev in events:
                        etype = ev.get("type")
                        data = ev.get("data", {})
                        if etype == "token":
                            answer_parts.append(data)
                        elif etype == "sources":
                            logged_sources = data
                        yield f"event: {etype}\ndata: {json.dumps(data)}\n\n"
                    _finish("ok")
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
                for kind, payload in bot.answer_stream(
                        question, context, req.history,
                        system_prompt=bot._quick_mode_system_prompt()):
                    if kind == "token":
                        answer_parts.append(payload)
                        yield f"event: token\ndata: {json.dumps(payload)}\n\n"
                    else:  # "cited"
                        cited = payload
                sources = _filter_cited_sources(
                    bot.format_sources(chunks, graph_facts, banner_facts, rmp_facts), cited)
                logged_sources = sources
                yield f"event: sources\ndata: {json.dumps(sources)}\n\n"
                yield "event: done\ndata: {}\n\n"
                _finish("ok")
            except Exception as e:
                _finish("error", str(e))
                yield f"event: error\ndata: {json.dumps(str(e))}\n\n"

        return StreamingResponse(
            generate(),
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

    bot = GeorgeBot()
    app = create_app(bot)
    print(f"Serving GeorgeBot (FastAPI) on http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
