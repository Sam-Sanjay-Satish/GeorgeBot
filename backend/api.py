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
from chatbot import DEFAULT_AUDIENCE, VALID_AUDIENCES, GeorgeBot
from thinking import ExtendedThinking


class ChatRequest(BaseModel):
    question: str = ""
    history: list[dict] = []
    # Which v2.2 corpus to search: "undergrad" | "faculty" | "both". Chosen by
    # the user via the frontend toggle; anything unrecognized falls back to the
    # default rather than erroring.
    audience: str = DEFAULT_AUDIENCE
    # Extended-thinking toggle (frontend). When true, the streaming endpoint
    # routes into the multi-mode orchestrator (thinking.ExtendedThinking) instead
    # of the single-shot retrieve->answer path. Only the SSE endpoint honors it.
    extended_thinking: bool = False


def _clean_audience(value: str | None) -> str:
    return value if value in VALID_AUDIENCES else DEFAULT_AUDIENCE


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
        return bot.ask(question, req.history, audience=_clean_audience(req.audience))

    @app.post("/api/chat/stream")
    def chat_stream(req: ChatRequest):
        question = (req.question or "").strip()
        if not question:
            return {"error": "missing 'question'"}

        audience = _clean_audience(req.audience)

        # Extended-thinking path: the orchestrator classifies the query into one
        # of three modes and yields event dicts we map straight to SSE frames
        # (including a `clarify` event that terminates the turn without an answer
        # when it needs more info from the user). See thinking.ExtendedThinking.
        if req.extended_thinking:
            def generate_thinking():
                try:
                    for ev in thinker.run(question, req.history, audience):
                        etype = ev.get("type")
                        data = ev.get("data", {})
                        yield f"event: {etype}\ndata: {json.dumps(data)}\n\n"
                    # thinker.run always ends with a done/clarify/error event; a
                    # trailing done is harmless if the client already saw clarify.
                except Exception as e:
                    yield f"event: error\ndata: {json.dumps(str(e))}\n\n"

            return StreamingResponse(
                generate_thinking(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )

        def generate():
            try:
                # Retrieval is inlined (rather than one bot.retrieve() call) so we
                # can emit "status" events at each phase transition during the gap
                # before answer tokens start streaming. Status text is templated
                # from route state — no extra LLM call, no added latency.
                route = bot.rewrite_and_route(question, req.history, audience)
                yield f"event: status\ndata: {json.dumps(_route_status(route))}\n\n"

                graph_facts = {}
                if route["course_codes"] or route["program_query"]:
                    graph_facts = bot.graph_retrieve(
                        route["course_codes"], route["program_query"],
                        route["wants_outline"], route["completed_courses"],
                    )
                # Gated live-data step — same firing condition as chatbot.retrieve().
                banner_facts = {}
                if route["course_codes"] and route["wants_availability"]:
                    banner_facts = banner_retrieve(
                        route["course_codes"], route["term_season"], route["term_year"],
                    )
                elif route["instructor_query"]:
                    banner_facts = banner_instructor_retrieve(
                        route["instructor_query"], route["term_season"], route["term_year"],
                    )
                # RMP ratings — runs after Banner so it can reuse its instructor
                # names (same logic as chatbot.retrieve()). Best-effort.
                rmp_facts = bot._rmp_retrieve_for(route, banner_facts)
                chunks = bot.vector_retrieve(
                    route["search_query"], audience=audience,
                    topic_families=route["topic_families"],
                    department=route["department"],
                )

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
                for text in bot.answer_stream(question, context, req.history):
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
