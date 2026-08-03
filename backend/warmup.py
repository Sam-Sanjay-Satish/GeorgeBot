"""Periodic full-pipeline warm probes + per-phase timing heartbeat.

WHY THIS EXISTS (read before changing the design)
-------------------------------------------------
An earlier keep-warm (removed 2026-08-03, see CLAUDE.md "Known issues") replayed
6 cached query embeddings against Chroma every 300s and made **zero HTTP calls**
by design. It was confirmed alive in prod during a request that still took
**51.9s** after ~2 days idle, so it defended ~1.1s of a ~44s problem. The lesson:
warming a subset of the path proves nothing about the path.

This module instead runs *real questions through the real pipeline* — router LLM
call, Voyage embedding, both Chroma collections, the course graph, Banner and
RMP — so every layer a live request touches is exercised by the warm cycle.

The root cause of the multi-day cold is still **unidentified**. That is why each
probe logs per-phase timings: the heartbeat doubles as the instrumentation that
should finally show *what* decays. The split below mirrors the SSE attribution
used to rule out stale connection pools (route time vs. everything after it):
if `route_ms` balloons on a cold cycle it's the outbound LLM path; if
`retrieve_ms` balloons it's Voyage/Chroma/the Volume page cache.

DESIGN CONSTRAINTS
------------------
* **Never touches querylog.** Probes call `bot.*` directly rather than going
  through api.py, so synthetic turns cannot pollute the corpus-gap log. Do not
  "simplify" this by calling the HTTP endpoint.
* **Never uses the chat executor.** Warming must not consume a worker slot a
  real request could have had.
* **Best-effort everywhere.** A probe failure logs and moves on; the loop
  catches everything so a background thread can never die silently or take the
  app down. Warming is an optimization, never a serving precondition.
* **Observable.** Every cycle prints, success or failure. The removed ticker
  printed nothing on a good tick, which is why confirming it even ran required
  a live experiment.
"""
from __future__ import annotations

import os
import threading
import time
import traceback
from datetime import datetime, timezone

# Default 1h per the operator's request. NOTE: the only interval empirically
# proven to keep this service warm is ~16 min (a probe after 16 min idle came
# back fully warm at 6.95s end-to-end); 2 days idle produced 51.9s. Nothing
# between those two points has been measured, so 3600s is an assumption, not a
# validated number. If cold hits persist in the query log, drop this first.
WARM_INTERVAL_SECONDS = float(os.getenv("WARM_INTERVAL_SECONDS", "3600"))

# Warming the *answer* step is off by default. It would add a second MiniMax
# generation of up to ANSWER_MAX_TOKENS per probe against a documented
# account-level token-throughput rate limit (CLAUDE.md "Known issues"), while
# warming nothing the router call hasn't already warmed — same endpoint, same
# model, same connection. Set WARM_ANSWER=1 to include it anyway.
WARM_ANSWER = os.getenv("WARM_ANSWER", "0").lower() not in ("0", "false", "no")

WARM_ENABLED = os.getenv("WARM_ENABLED", "1").lower() not in ("0", "false", "no")


class Probe:
    """One synthetic question aimed at one retrieval path.

    `force` pins the routing fields that open this probe's gate, applied on top
    of the real router output. The router call still happens for every probe —
    that is what warms the MiniMax path, and it is half the timing split — but
    its *result* is not trusted to open the gate, because MiniMax-M3 is not
    deterministic even at temperature 0 (CLAUDE.md §1). Observed directly here:
    "who teaches CSC 225 and are they any good" returned
    course_codes=['CSC225'], wants_availability=True on one run and an empty
    route on the next, so Banner and RMP were warmed only intermittently. A
    warmer that skips a layer on a coin flip is the exact false-confidence
    failure of the ticker this replaces.

    This pins *warming* coverage, not router correctness — the warm loop is not
    a router test. Downstream breakage (Banner down, course not offered, RMP
    unreachable) still shows up as MISSED in the log.

    `expect` names the gates that should fire; it is logged against what
    actually fired rather than enforced.
    """

    __slots__ = ("label", "question", "audience", "expect", "force")

    def __init__(self, label: str, question: str, audience: str,
                 expect: tuple[str, ...], force: dict | None = None):
        self.label = label
        self.question = question
        self.audience = audience
        self.expect = expect
        self.force = force or {}


# One probe per retrieval path. Between them these touch: the router LLM, the
# Voyage query embedding, BOTH Chroma collections (undergrad and faculty have
# disjoint tf_* vocabularies and separate HNSW graphs — warming one does not
# warm the other), the course/program graph, Banner, and RMP.
PROBES = (
    Probe("vector-undergrad",
          "what student support services does UVic offer",
          "undergrad", ("vector",)),
    # The faculty collection is a separate HNSW graph + its own SQLite pages.
    Probe("vector-faculty",
          "what is the policy on research grant administration and ethics approval",
          "faculty", ("vector",)),
    # course_codes -> graph_retrieve (prereqs/credits/cross-listings). The graph
    # itself is in-heap after boot, so this mostly re-touches its pages if the
    # host has swapped them out; the Chroma leg is the real work.
    Probe("graph",
          "what are the prerequisites for CSC 225",
          "undergrad", ("graph", "vector"),
          force={"course_codes": ["CSC225"]}),
    # course_codes + wants_availability -> banner_retrieve. Banner's 120s
    # SECTIONS_TTL means seat data is ALWAYS re-fetched (freshness is the point),
    # so what this actually keeps populated is the 6h TERMS_TTL term list and the
    # 6h FACULTY_TTL per-CRN instructor cache — both comfortably longer than a 1h
    # cycle — plus the TCP/TLS path to banner.uvic.ca.
    #
    # The course must actually be OFFERED in the term Banner resolves to, or the
    # probe returns {} and warms nothing: MATH 100 and ENGL 135 both came back
    # empty in Summer 2026 for exactly this reason. If `MISSED=banner` starts
    # showing up in the logs, that is the likely cause — swap in a course running
    # that term rather than assuming Banner is down.
    Probe("banner",
          "how many seats are left in CSC 225",
          "undergrad", ("banner", "vector"),
          force={"course_codes": ["CSC225"], "wants_availability": True}),
    # RMP chains off Banner: `_rmp_retrieve_for` only looks up names when
    # `professor_query` is set OR `wants_rating` AND banner_facts is non-empty.
    # So the question has to open the Banner gate too — "who teaches X" sets
    # wants_availability, "are they any good" sets wants_rating. A plain "is X a
    # good course" sets only wants_rating, leaves Banner shut, and silently
    # retrieves no ratings at all (verified against the live router).
    # Chained deliberately rather than hardcoding a real professor's name.
    Probe("rmp",
          "who teaches CSC 225 and are they any good",
          "undergrad", ("banner", "rmp", "vector"),
          force={"course_codes": ["CSC225"], "wants_availability": True,
                 "wants_rating": True}),
)


def _gates_hit(chunks, graph_facts: dict, banner_facts: dict, rmp_facts: dict) -> list[str]:
    """Which retrieval paths actually produced something this probe."""
    hit = []
    if graph_facts and (graph_facts.get("courses") or graph_facts.get("program")):
        hit.append("graph")
    if banner_facts:
        hit.append("banner")
    if rmp_facts:
        hit.append("rmp")
    if chunks:
        hit.append("vector")
    return hit


def _run_probe(bot, probe: Probe) -> None:
    """Run one probe end-to-end and print its phase timings.

    Route and retrieval are timed separately on purpose — that split is the
    whole diagnostic value of this loop (see module docstring).
    """
    t0 = time.monotonic()
    try:
        route = bot.rewrite_and_route(probe.question, [], probe.audience)
        t_route = time.monotonic()
        route.update(probe.force)   # pin this probe's gate — see Probe docstring

        chunks, graph_facts, banner_facts, rmp_facts = bot.retrieve_with_route(
            route, probe.audience)
        t_retrieve = time.monotonic()

        answer_ms = ""
        if WARM_ANSWER:
            context, _ = bot._assemble_context(chunks, graph_facts, banner_facts, rmp_facts)
            bot.answer(probe.question, context, [])
            answer_ms = f" answer={int((time.monotonic() - t_retrieve) * 1000)}ms"

        hit = _gates_hit(chunks, graph_facts, banner_facts, rmp_facts)
        missed = [g for g in probe.expect if g not in hit]
        note = f"  MISSED={','.join(missed)}" if missed else ""
        print(f"[warm] {probe.label:<17} total={int((time.monotonic() - t0) * 1000):>6}ms "
              f"route={int((t_route - t0) * 1000):>6}ms "
              f"retrieve={int((t_retrieve - t_route) * 1000):>6}ms{answer_ms} "
              f"chunks={len(chunks)} gates={','.join(hit) or '-'}{note}", flush=True)
    except Exception as e:  # noqa: BLE001 - one bad probe must not stop the cycle
        print(f"[warm] {probe.label:<17} FAILED after "
              f"{int((time.monotonic() - t0) * 1000)}ms: {type(e).__name__}: {e}",
              flush=True)


def run_cycle(bot) -> None:
    """One full pass over every probe. Safe to call directly (CLI/testing)."""
    started = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    t0 = time.monotonic()
    print(f"[warm] cycle start {started}", flush=True)
    for probe in PROBES:
        _run_probe(bot, probe)
    print(f"[warm] cycle done in {time.monotonic() - t0:.1f}s "
          f"(next in {WARM_INTERVAL_SECONDS:.0f}s)", flush=True)


def _loop(bot, interval: float) -> None:
    while True:
        try:
            run_cycle(bot)
        except Exception:  # noqa: BLE001 - a background thread must never die
            traceback.print_exc()
        time.sleep(interval)


def start(bot, interval: float = WARM_INTERVAL_SECONDS) -> None:
    """Start the warm loop on a daemon thread.

    Call this BEFORE uvicorn.run() binds the port: the first cycle then runs
    concurrently with startup rather than delaying it. The removed boot warm-up
    ran *before* the port opened and added 14.6s to every container start, which
    the frontend's health-gate poll had to sit through — don't reintroduce that.

    Daemon, so it never holds the process open at shutdown.
    """
    if not WARM_ENABLED or interval <= 0:
        print("[warm] disabled", flush=True)
        return
    threading.Thread(target=_loop, args=(bot, interval),
                     name="warm-probe", daemon=True).start()
    print(f"[warm] probe thread started — {len(PROBES)} probes every {interval:.0f}s "
          f"(answer step {'ON' if WARM_ANSWER else 'off'})", flush=True)
