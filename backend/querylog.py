#!/usr/bin/env python3
"""
Query log — a per-turn record of what users actually ask GeorgeBot.

Purpose is corpus-gap analysis: which questions retrieve nothing, which land the
wrong department, which get a generic non-answer. Every chat turn writes one row
holding the user's question, the router's rewritten `search_query` + full route
metadata, and the final answer text. Retrieved chunk *text* is deliberately NOT
stored (only the cited sources' titles/urls) — this is a usage log, not a cache.

Storage is a SQLite file on the same Railway Volume that holds chroma_db/ and
graph_data/, so it survives redeploys. Note it is the only NON-reproducible file
on that volume: a re-seed that wipes /data must not delete it (see CLAUDE.md).

Everything here is BEST-EFFORT: no function raises, so a logging failure can't
break a user's answer (same posture as banner.py / rmp.py).

A turn is written in two steps — `start_turn` on arrival, `finish_turn` when the
answer completes — see the comment above `start_turn` for why that matters.

Env:
  DATA_DIR            — volume mount; the DB lands at $DATA_DIR/query_logs.db
  QUERY_LOG_DB        — absolute override for that path
  QUERY_LOG_ENABLED   — "0"/"false"/"no" turns logging off entirely (no-ops)
"""

import hashlib
import json
import os
import sqlite3
import sys
import threading
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

BASE_DIR = Path(__file__).parent          # backend/

# Same resolution scheme as chatbot.py's CHROMA_DIR/TAXONOMY_FILE (that module's
# _DATA_DIR is private, so recompute rather than import it):
#   QUERY_LOG_DB > $DATA_DIR/query_logs.db > backend/query_logs.db
_DATA_DIR = Path(os.environ["DATA_DIR"]) if os.getenv("DATA_DIR") else BASE_DIR
DB_PATH = Path(os.getenv("QUERY_LOG_DB", _DATA_DIR / "query_logs.db"))

ENABLED = os.getenv("QUERY_LOG_ENABLED", "1").lower() not in ("0", "false", "no")

_lock = threading.Lock()
_conn: sqlite3.Connection | None = None

SCHEMA = """
CREATE TABLE IF NOT EXISTS turns (
  id            INTEGER PRIMARY KEY AUTOINCREMENT,
  session_id    TEXT NOT NULL,
  turn_index    INTEGER NOT NULL,
  ts            TEXT NOT NULL,      -- ISO8601 UTC
  question      TEXT NOT NULL,
  search_query  TEXT,               -- rewritten query from rewrite_and_route
  answer        TEXT,               -- full answer (partial if abandoned)
  audience      TEXT,               -- undergrad | faculty | both
  mode          TEXT,               -- quick | default
  route_json    TEXT,               -- whole route dict, JSON
  n_sources     INTEGER,
  source_titles TEXT,               -- JSON [{title,url,source}] of CITED sources
  latency_ms    INTEGER,
  status        TEXT,               -- started | ok | error | abandoned
  error         TEXT
);
CREATE INDEX IF NOT EXISTS idx_turns_session ON turns(session_id, turn_index);
CREATE INDEX IF NOT EXISTS idx_turns_ts ON turns(ts);

-- Maps a conversation's history-prefix signature to its session, so the next
-- turn in the same chat can be grouped with it (see resolve_session).
CREATE TABLE IF NOT EXISTS session_chain (
  sig         TEXT PRIMARY KEY,
  session_id  TEXT NOT NULL,
  turn_index  INTEGER NOT NULL,
  updated     TEXT NOT NULL
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _connect() -> sqlite3.Connection:
    """Lazily open the single shared connection. Callers must hold `_lock`."""
    global _conn
    if _conn is None:
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False: the chat endpoints are sync `def`, so FastAPI
        # runs them across a threadpool. `_lock` is what actually serializes us.
        conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.executescript(SCHEMA)
        conn.commit()
        _conn = conn
    return _conn


def _sig(user_texts: list[str]) -> str:
    """Signature of a conversation prefix: sha256 over its ordered user turns."""
    return hashlib.sha256(
        json.dumps(user_texts, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def _user_texts(history: list[dict] | None) -> list[str]:
    """The user-role contents of a history list, in order.

    User turns only — the assistant text the frontend sends back is the
    *revealed* answer and can drift from what we logged; the user's own words
    can't."""
    return [(m.get("content") or "").strip()
            for m in (history or []) if m.get("role") == "user"]


def resolve_session(history: list[dict] | None) -> tuple[str, int]:
    """Group this turn into a conversation, returning (session_id, turn_index).

    There is no client-supplied session id (deliberately — no frontend change).
    Instead the frontend sends the full untruncated history on every request
    (frontend/src/lib/api.ts `toHistory`), so the chain of user questions
    identifies the conversation:

      - empty history      -> a brand-new session (fresh uuid, turn 0)
      - otherwise          -> look up this prefix in `session_chain`; a hit
                              continues that session, a miss starts a new one.

    `log_turn` then records the prefix INCLUDING this question, so the next
    request in the same chat finds it. Caveat: two users whose questions are
    identical word-for-word from turn 1 collapse into one session. Acceptable
    for a corpus-gap tool.
    """
    if not ENABLED:
        return ("", 0)
    prior = _user_texts(history)
    if not prior:
        return (uuid.uuid4().hex, 0)
    try:
        with _lock:
            row = _connect().execute(
                "SELECT session_id, turn_index FROM session_chain WHERE sig = ?",
                (_sig(prior),),
            ).fetchone()
        if row:
            return (row["session_id"], row["turn_index"] + 1)
    except Exception as e:                                    # best-effort
        print(f"[querylog] session lookup failed: {e}", file=sys.stderr)
    return (uuid.uuid4().hex, 0)


# A turn is written in two steps -- INSERT on arrival, UPDATE on completion --
# rather than once at the end. This is what makes abandoned turns observable:
# when the user closes the tab mid-answer, Starlette cancels the streaming
# response WITHOUT closing our generator (no GeneratorExit, no `finally`), so
# end-of-turn code simply never runs. Verified against this stack -- don't
# "simplify" this back into a single write in a finally block; it silently
# drops exactly the turns worth seeing. Rows left in 'started' are swept to
# 'abandoned' by `sweep_stale`.

def start_turn(*, session_id: str, turn_index: int, question: str,
               history: list[dict] | None = None, audience: str | None = None,
               mode: str | None = None) -> int | None:
    """Record an arriving turn; returns its row id (None if logging is off/failed).

    Also advances `session_chain` here, not at completion, so the *next* turn
    of this chat groups correctly even if this one is abandoned."""
    if not ENABLED:
        return None
    try:
        with _lock:
            conn = _connect()
            cur = conn.execute(
                """INSERT INTO turns (session_id, turn_index, ts, question,
                       audience, mode, status)
                   VALUES (?,?,?,?,?,?, 'started')""",
                (session_id, turn_index, _now(), question, audience, mode),
            )
            sig = _sig(_user_texts(history) + [(question or "").strip()])
            conn.execute(
                """INSERT INTO session_chain (sig, session_id, turn_index, updated)
                   VALUES (?,?,?,?)
                   ON CONFLICT(sig) DO UPDATE SET
                       session_id=excluded.session_id,
                       turn_index=excluded.turn_index,
                       updated=excluded.updated""",
                (sig, session_id, turn_index, _now()),
            )
            conn.commit()
            return cur.lastrowid
    except Exception as e:                                    # best-effort
        print(f"[querylog] start_turn failed: {e}", file=sys.stderr)
        return None


def update_route(row_id: int | None, route: dict | None) -> None:
    """Attach the router's output as soon as it's known.

    Deliberately separate from `finish_turn`: an abandoned turn still keeps its
    rewritten query and route, which is most of the gap-analysis value."""
    if not ENABLED or row_id is None or not route:
        return
    try:
        with _lock:
            conn = _connect()
            conn.execute(
                "UPDATE turns SET search_query = ?, route_json = ? WHERE id = ?",
                (route.get("search_query"),
                 json.dumps(route, ensure_ascii=False), row_id),
            )
            conn.commit()
    except Exception as e:                                    # best-effort
        print(f"[querylog] update_route failed: {e}", file=sys.stderr)


def finish_turn(row_id: int | None, *, answer: str | None = None,
                search_query: str | None = None,
                sources: list[dict] | None = None,
                latency_ms: int | None = None, status: str = "ok",
                error: str | None = None) -> None:
    """Complete a turn started by `start_turn`. Never raises."""
    if not ENABLED or row_id is None:
        return
    try:
        titles = [{"title": s.get("title"), "url": s.get("url"),
                   "source": s.get("source")} for s in (sources or [])]
        with _lock:
            conn = _connect()
            conn.execute(
                """UPDATE turns SET answer = ?, n_sources = ?, source_titles = ?,
                       latency_ms = ?, status = ?, error = ?,
                       search_query = COALESCE(?, search_query)
                     WHERE id = ?""",
                (answer, len(sources or []),
                 json.dumps(titles, ensure_ascii=False) if titles else None,
                 latency_ms, status, error, search_query, row_id),
            )
            conn.commit()
    except Exception as e:                                    # best-effort
        print(f"[querylog] finish_turn failed: {e}", file=sys.stderr)


# No real turn runs this long; anything still 'started' past it was abandoned
# by the client (or died with the process).
STALE_SECONDS = 600


def sweep_stale() -> None:
    """Reclassify long-'started' turns as 'abandoned'. Called before admin reads."""
    if not ENABLED:
        return
    try:
        with _lock:
            conn = _connect()
            # Cutoff computed in Python so it matches _now()'s exact format --
            # SQLite's datetime('now') would render differently ("… 11:20:00"
            # vs "…T11:20:00+00:00") and the string comparison would never fire.
            cutoff = (datetime.now(timezone.utc)
                      - timedelta(seconds=STALE_SECONDS)).isoformat(timespec="seconds")
            conn.execute(
                "UPDATE turns SET status = 'abandoned' "
                " WHERE status = 'started' AND ts < ?",
                (cutoff,),
            )
            conn.commit()
    except Exception as e:                                    # best-effort
        print(f"[querylog] sweep failed: {e}", file=sys.stderr)


# --------------------------------------------------------------------------
# Read side — used by the admin endpoints in api.py.
# --------------------------------------------------------------------------

TURN_COLUMNS = ("id", "session_id", "turn_index", "ts", "question",
                "search_query", "answer", "audience", "mode", "route_json",
                "n_sources", "source_titles", "latency_ms", "status", "error")


def _rows(sql: str, params: tuple = ()) -> list[dict]:
    with _lock:
        cur = _connect().execute(sql, params)
        return [dict(r) for r in cur.fetchall()]


def recent_turns(limit: int = 100, offset: int = 0, q: str | None = None,
                 status: str | None = None, since: str | None = None,
                 zero_sources: bool = False) -> list[dict]:
    """Flat, newest-first turn list with optional filters.

    `q` substring-matches the question, rewritten query and answer;
    `zero_sources` isolates turns that cited nothing — the primary gap signal."""
    sweep_stale()
    where, params = [], []
    if q:
        where.append("(question LIKE ? OR search_query LIKE ? OR answer LIKE ?)")
        params += [f"%{q}%"] * 3
    if status:
        where.append("status = ?")
        params.append(status)
    if since:
        where.append("ts >= ?")
        params.append(since)
    if zero_sources:
        where.append("status = 'ok' AND COALESCE(n_sources, 0) = 0")
    clause = f"WHERE {' AND '.join(where)}" if where else ""
    params += [limit, offset]
    return _rows(f"SELECT * FROM turns {clause} ORDER BY id DESC LIMIT ? OFFSET ?",
                 tuple(params))


def sessions(limit: int = 50, offset: int = 0) -> list[dict]:
    """One row per conversation, most recently active first."""
    sweep_stale()
    return _rows(
        """SELECT session_id,
                  COUNT(*)                                   AS n_turns,
                  MIN(ts)                                    AS started,
                  MAX(ts)                                    AS ended,
                  SUM(CASE WHEN status != 'ok' THEN 1 ELSE 0 END) AS n_problem,
                  SUM(CASE WHEN status = 'ok' AND COALESCE(n_sources,0) = 0
                           THEN 1 ELSE 0 END)
                                                             AS n_no_sources,
                  (SELECT question FROM turns t2
                    WHERE t2.session_id = t.session_id
                    ORDER BY t2.id LIMIT 1)                   AS first_question
             FROM turns t
            GROUP BY session_id
            ORDER BY MAX(id) DESC
            LIMIT ? OFFSET ?""",
        (limit, offset),
    )


def session_turns(session_id: str) -> list[dict]:
    """Full transcript of one conversation, in order."""
    return _rows("SELECT * FROM turns WHERE session_id = ? ORDER BY id",
                 (session_id,))


def stats() -> dict:
    """Headline counters for the admin page."""
    sweep_stale()
    base = _rows(
        """SELECT COUNT(*) AS n_turns,
                  COUNT(DISTINCT session_id) AS n_sessions,
                  SUM(CASE WHEN status = 'ok' AND COALESCE(n_sources,0) = 0
                           THEN 1 ELSE 0 END)
                      AS n_no_sources,
                  SUM(CASE WHEN status = 'error' THEN 1 ELSE 0 END) AS n_error,
                  SUM(CASE WHEN status IN ('abandoned','started') THEN 1 ELSE 0 END)
                      AS n_abandoned,
                  MAX(ts) AS last_ts
             FROM turns"""
    )[0]
    base["top_departments"] = _rows(
        """SELECT json_extract(route_json, '$.department') AS department,
                  COUNT(*) AS n
             FROM turns
            WHERE json_extract(route_json, '$.department') IS NOT NULL
            GROUP BY 1 ORDER BY n DESC LIMIT 10"""
    )
    base["db_path"] = str(DB_PATH)
    return base


def _main() -> None:
    """CLI peek: `python3 backend/querylog.py [N]` prints the last N turns."""
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    print(f"db: {DB_PATH}  (exists={DB_PATH.exists()})")
    print(json.dumps(stats(), indent=2, default=str))
    for t in reversed(recent_turns(limit=n)):
        print(f"\n[{t['ts']}] {t['session_id'][:8]}#{t['turn_index']} "
              f"({t['status']}, {t['n_sources']} src)")
        print(f"  Q: {t['question']}")
        print(f"  →  {t['search_query']}")
        print(f"  A: {(t['answer'] or '')[:200]}")


if __name__ == "__main__":
    _main()
