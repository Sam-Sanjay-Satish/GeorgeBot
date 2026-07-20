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
                          SSE events: mode, status, sources, token, done, error
                          ("status" fires before tokens with a short, templated
                          phase message, e.g. "Looking up CSC 225…"; "mode"
                          fires first, only in mode="default" when an
                          extended-thinking plan is dispatched — see
                          thinking.PLAN_LABELS)

Env (.env): MINIMAX_SUB_KEY, VOYAGE_API_KEY
"""

import argparse
import json
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from banner import banner_instructor_retrieve, banner_retrieve
from chatbot import DEFAULT_AUDIENCE, MAX_VERIFY_ROUNDS, VALID_AUDIENCES, GeorgeBot
from thinking import ExtendedThinking

VALID_MODES = ("quick", "default")
DEFAULT_MODE = "default"


class ChatRequest(BaseModel):
    question: str = ""
    history: list[dict] = []
    # Which v2.2 corpus to search: "undergrad" | "faculty" | "both". Chosen by
    # the user via the frontend toggle; anything unrecognized falls back to the
    # default rather than erroring.
    audience: str = DEFAULT_AUDIENCE
    # "quick" (today's flat retrieve->answer pipeline) | "default" (the
    # planner also decides simple vs. one of the three extended-thinking
    # plans -- see chatbot.rewrite_and_route(mode=...) and thinking.py).
    # Replaces the old boolean `extended_thinking` toggle.
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


def _drain_events(events) -> dict:
    """Collect a `{type, data}` event generator (`ExtendedThinking.run` or
    `_default_simple_events` -- the same shapes `/api/chat/stream` maps to
    SSE frames) into one JSON-friendly dict, for the non-streaming endpoint
    and the CLI. A `clarify` event terminates the turn without a real answer
    -- surfaced as the answer text with `needs_clarification` set so callers
    can render it like any other answer."""
    answer_parts: list[str] = []
    sources: list[dict] = []
    needs_clarification = False
    mode_label = None
    error = None
    for ev in events:
        etype, data = ev.get("type"), ev.get("data")
        if etype == "token":
            answer_parts.append(data)
        elif etype == "sources":
            sources = data
        elif etype == "clarify":
            answer_parts = [data]
            needs_clarification = True
        elif etype == "mode":
            mode_label = data
        elif etype == "error":
            error = data
    if error is not None:
        return {"error": error}
    result = {"answer": "".join(answer_parts), "sources": sources,
              "n_chunks": len(sources)}
    if needs_clarification:
        result["needs_clarification"] = True
    if mode_label:
        result["mode_label"] = mode_label
    return result


def _default_simple_events(bot: GeorgeBot, question: str, history: list[dict],
                           audience: str, route: dict):
    """Default mode's "simple" path: retrieve, then a combined verify-answer
    loop (`bot.answer_verified_stream`) capped at `MAX_VERIFY_ROUNDS` gated
    rounds before a forced plain answer. Yields the same `{type, data}` event
    shape as `ExtendedThinking.run()`, so both the SSE handler and
    `_drain_events` can treat them uniformly.

    Nothing is forwarded to the caller for a NEED_MORE round (per
    `answer_verified_stream`, it never yields a "token" before a verdict is
    known) -- `sources` and `token` events for a round only appear once that
    round's answer is confirmed SUFFICIENT, so a discarded round is never
    shown to the user.
    """
    chunks, graph_facts, banner_facts, rmp_facts = bot.retrieve_with_route(route, audience)
    context, n_prefix_blocks = bot._assemble_context(chunks, graph_facts, banner_facts, rmp_facts)
    has_context = bool(chunks) or n_prefix_blocks > 0
    yield {"type": "status", "data": "Reading through sources…" if has_context else "Thinking…"}

    for _round in range(MAX_VERIFY_ROUNDS):
        need_more, sources_emitted = None, False
        for kind, payload in bot.answer_verified_stream(question, context, history):
            if kind == "token":
                if not sources_emitted:
                    yield {"type": "sources",
                           "data": bot.format_sources(chunks, graph_facts, banner_facts, rmp_facts)}
                    sources_emitted = True
                yield {"type": "token", "data": payload}
            else:  # "need_more"
                need_more = payload
        if need_more is None:
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
    yield {"type": "sources",
           "data": bot.format_sources(chunks, graph_facts, banner_facts, rmp_facts)}
    for text in bot.answer_stream(question, context, history):
        yield {"type": "token", "data": text}
    yield {"type": "done"}


def create_app(bot: GeorgeBot) -> FastAPI:
    app = FastAPI(title="GeorgeBot")
    thinker = ExtendedThinking(bot)

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
        if mode == "quick":
            return bot.ask(question, req.history, audience=audience)

        # mode == "default": one planner call decides simple vs. one of the
        # three extended-thinking plans; both paths yield the same {type,
        # data} event shape, drained here into one JSON response.
        route = bot.rewrite_and_route(question, req.history, audience, mode=mode)
        if route["is_simple"]:
            events = _default_simple_events(bot, question, req.history, audience, route)
        else:
            events = thinker.run(question, req.history, audience,
                                 route, route["plan"], route["confidence"])
        result = _drain_events(events)
        result.setdefault("search_query", route["search_query"])
        return result

    @app.post("/api/chat/stream")
    def chat_stream(req: ChatRequest):
        question = (req.question or "").strip()
        if not question:
            return {"error": "missing 'question'"}

        audience = _clean_audience(req.audience)
        mode = _clean_mode(req.mode)

        def generate():
            try:
                # Route is computed once up front (regardless of mode) so we
                # can emit an early "status" event during the gap before
                # retrieval/answer tokens start streaming. Status text is
                # templated from route state — no extra LLM call, no added
                # latency beyond the planner call itself.
                route = bot.rewrite_and_route(question, req.history, audience, mode=mode)
                yield f"event: status\ndata: {json.dumps(_route_status(route))}\n\n"

                if mode == "default" and not route["is_simple"]:
                    # Not-simple: dispatch into one of the three preset plans.
                    # The planner already classified + downgraded low-
                    # confidence picks, so this no longer re-classifies.
                    events = thinker.run(question, req.history, audience,
                                         route, route["plan"], route["confidence"])
                elif mode == "default":  # is_simple
                    events = _default_simple_events(bot, question, req.history, audience, route)
                else:  # mode == "quick"
                    events = None

                if events is not None:
                    for ev in events:
                        etype = ev.get("type")
                        data = ev.get("data", {})
                        yield f"event: {etype}\ndata: {json.dumps(data)}\n\n"
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

                # Emit sources first so the UI can render them immediately.
                sources = bot.format_sources(chunks, graph_facts, banner_facts, rmp_facts)
                yield f"event: sources\ndata: {json.dumps(sources)}\n\n"
                for text in bot.answer_stream(question, context, req.history,
                                              system_prompt=bot._quick_mode_system_prompt()):
                    yield f"event: token\ndata: {json.dumps(text)}\n\n"
                yield "event: done\ndata: {}\n\n"
            except Exception as e:
                yield f"event: error\ndata: {json.dumps(str(e))}\n\n"

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

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
