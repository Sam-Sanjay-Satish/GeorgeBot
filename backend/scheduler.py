#!/usr/bin/env python3
"""
Deterministic schedule solver for GeorgeBot's `course_planning` extended-thinking
mode.

This is **not** an LLM. It takes a flat list of candidate course sections (the
exact shape `banner._normalize_section` produces — see `banner.py`) plus a
structured `Constraints` object, and returns ranked, non-conflicting schedule
combinations. The planning pipeline (`thinking.py`) enumerates eligible courses
from the graph, fetches their live sections from Banner, then hands them here;
the answer model narrates whatever this returns.

Why deterministic code and not the LLM: conflict detection is exact interval
math, and letting a model "eyeball" overlapping meeting times is exactly the
kind of thing it gets subtly wrong. The LLM's only job upstream is to *parse the
user's prose into the `Constraints` object below* — never to apply it.

Design notes / known limitations (deliberate for v1, documented so they aren't
mistaken for bugs):
  - **One section per course.** Each distinct `subject_course` contributes at
    most one section to a schedule (typically the lecture). Linked lab/tutorial
    sections are NOT auto-bundled with their lecture — Banner exposes them as
    separate sections and we don't model the co-req linkage. The narration
    layer should tell the student to add the matching lab.
  - **Banner meeting caveats flow through.** `meeting` may be `None` (async/TBA)
    — those sections never conflict and are always schedulable. Only
    `meetingsFaculty[0]` is read by Banner, so a section with multiple distinct
    weekly patterns is represented by its first pattern only. No room data.
  - **Bounded search.** The eligible list can be large; enumerating every subset
    is combinatorial. We cap the number of courses considered and the search
    nodes, and return only the top `max_results` by preference score. This is a
    heuristic, not a guaranteed-optimal timetable optimizer.

`Constraints` (the structured object `thinking.py` builds from the user's slots;
matches the plan-file schema in the design doc):

    {
      "include_courses": ["CSC225"],       # each MUST appear in every schedule
      "exclude_courses": ["MATH100"],      # dropped from candidates entirely
      "soft_preferences": [                # scored, not hard-filtered here
        {"type": "no_time_before", "value": "1000"},   # HHMM
        {"type": "no_time_after",  "value": "1600"},   # HHMM
        {"type": "no_days",        "value": ["Fr"]},   # day codes to avoid
        {"type": "delivery",       "value": "online"}, # online | in_person
        {"type": "max_courses",    "value": 5},        # cap schedule size
      ],
    }

Hard filters (exclude/include existence, delivery=hard) are normally applied by
the caller before the candidate list reaches here, but this module re-enforces
include/exclude and `max_courses` structurally so it is safe to call directly.
"""

from __future__ import annotations

# Bounds on the search so a big eligible list can't blow up (see "Bounded
# search" above). Tuned for a realistic term: a couple dozen candidate courses,
# a handful of sections each. These are heuristics, not correctness limits.
MAX_COURSES_CONSIDERED = 14   # distinct courses fed into the combination search
MAX_SEARCH_NODES = 200_000    # DFS node budget before we stop and rank what we have
DEFAULT_TARGET_COURSES = 5    # schedule size when the user didn't say max_courses
MAX_RESULTS = 5               # top schedules returned


# ---------------------------------------------------------------------------
# Time / meeting helpers
# ---------------------------------------------------------------------------

def _to_minutes(hhmm: str | None) -> int | None:
    """"1430" -> 870 (minutes since midnight). None / malformed -> None."""
    if not hhmm or len(hhmm) != 4 or not hhmm.isdigit():
        return None
    h, m = int(hhmm[:2]), int(hhmm[2:])
    if h > 23 or m > 59:
        return None
    return h * 60 + m


def _meeting_intervals(section: dict) -> list[tuple[str, int, int]]:
    """(day, begin_min, end_min) tuples for a section's meeting, one per day.

    Returns [] for async/TBA sections (no meeting, or no parseable day+time) —
    an empty interval set means "never conflicts, always schedulable".
    """
    m = section.get("meeting")
    if not m:
        return []
    days = m.get("days") or []
    begin = _to_minutes(m.get("begin"))
    end = _to_minutes(m.get("end"))
    if not days or begin is None or end is None or end <= begin:
        return []
    return [(d, begin, end) for d in days]


def _conflicts(a_intervals: list[tuple[str, int, int]],
               b_intervals: list[tuple[str, int, int]]) -> bool:
    """True if two sections share a day with overlapping [begin, end) times."""
    for da, ba, ea in a_intervals:
        for db, bb, eb in b_intervals:
            if da == db and ba < eb and bb < ea:
                return True
    return False


# ---------------------------------------------------------------------------
# Soft-preference scoring
# ---------------------------------------------------------------------------

def _pref_violations(section: dict, intervals: list[tuple[str, int, int]],
                     soft_prefs: list[dict]) -> list[str]:
    """Which soft preferences THIS section violates, as human-readable strings.

    Async sections (no intervals) can't violate a time/day preference — they
    only interact with `delivery`. Unknown preference types are ignored (the
    parser upstream should already have dropped them, but be defensive).
    """
    out: list[str] = []
    for pref in soft_prefs:
        ptype = pref.get("type")
        val = pref.get("value")
        if ptype == "no_time_before":
            lo = _to_minutes(val)
            if lo is not None and any(b < lo for _d, b, _e in intervals):
                out.append(f"starts before {val[:2]}:{val[2:]}")
        elif ptype == "no_time_after":
            hi = _to_minutes(val)
            if hi is not None and any(e > hi for _d, _b, e in intervals):
                out.append(f"ends after {val[:2]}:{val[2:]}")
        elif ptype == "no_days":
            avoid = {d for d in (val or [])}
            hit = sorted({d for d, _b, _e in intervals if d in avoid})
            if hit:
                out.append(f"meets on {', '.join(hit)}")
        elif ptype == "delivery":
            want = (val or "").lower()
            delivery = (section.get("delivery") or "").lower()
            if want == "online" and "online" not in delivery:
                out.append("not online")
            elif want in ("in_person", "in-person", "face-to-face") \
                    and "face" not in delivery and "person" not in delivery:
                out.append("not in-person")
    return out


# ---------------------------------------------------------------------------
# Candidate preparation
# ---------------------------------------------------------------------------

class _Candidate:
    """A schedulable section plus its precomputed intervals + violation strings."""
    __slots__ = ("section", "code", "intervals", "violations")

    def __init__(self, section: dict, soft_prefs: list[dict]) -> None:
        self.section = section
        self.code = section.get("subject_course") or ""
        self.intervals = _meeting_intervals(section)
        self.violations = _pref_violations(section, self.intervals, soft_prefs)


def _group_candidates(sections: list[dict], soft_prefs: list[dict],
                      exclude: set[str]) -> dict[str, list[_Candidate]]:
    """Group sections by course code, dropping excluded courses. Within each
    course, sections with fewer preference violations come first (so the search
    prefers nicer sections early and better schedules surface sooner)."""
    by_course: dict[str, list[_Candidate]] = {}
    for s in sections:
        code = s.get("subject_course") or ""
        if not code or code in exclude:
            continue
        by_course.setdefault(code, []).append(_Candidate(s, soft_prefs))
    for cands in by_course.values():
        cands.sort(key=lambda c: len(c.violations))
    return by_course


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

def solve_schedule(sections: list[dict], constraints: dict | None = None,
                   max_results: int = MAX_RESULTS) -> list[dict]:
    """Return up to `max_results` ranked, conflict-free schedules.

    `sections` — Banner-shaped section dicts (see module docstring).
    `constraints` — the structured object (see module docstring); may be None.

    Each returned schedule:
        {"sections":   [<section dict>, ...],   # one per included course
         "courses":    ["CSC225", ...],
         "score":      float,                    # higher = fewer pref violations
         "prefs_met":  bool,                     # True if zero violations
         "notes":      ["CSC226 A01 meets on Fr", ...]}  # per-section violations

    Returns [] when no conflict-free schedule of the target size exists (e.g. an
    over-constrained include set whose sections all collide) — the caller should
    surface that plainly rather than inventing a timetable.
    """
    constraints = constraints or {}
    include = {c for c in (constraints.get("include_courses") or [])}
    exclude = {c for c in (constraints.get("exclude_courses") or [])}
    soft_prefs = constraints.get("soft_preferences") or []

    # max_courses can arrive either as its own key or inside soft_preferences.
    target = constraints.get("max_courses")
    for pref in soft_prefs:
        if pref.get("type") == "max_courses":
            try:
                target = int(pref["value"])
            except (TypeError, ValueError, KeyError):
                pass

    by_course = _group_candidates(sections, soft_prefs, exclude)

    # An include_courses entry with no offered/eligible section can never be
    # satisfied — bail with an empty result so the caller says so honestly.
    missing_includes = [c for c in include if c not in by_course]
    if missing_includes:
        return []

    # Order courses: required includes first (they must appear), then the rest
    # ordered by how "clean" their best section is, capped so the search stays
    # bounded. Includes are never dropped by the cap.
    other_codes = [c for c in by_course if c not in include]
    other_codes.sort(key=lambda c: len(by_course[c][0].violations))
    ordered = list(include) + other_codes
    ordered = ordered[:max(len(include), MAX_COURSES_CONSIDERED)]

    if target is None:
        target = min(DEFAULT_TARGET_COURSES, len(ordered))
    target = max(len(include), min(target, len(ordered)))

    results: list[dict] = []
    nodes = 0

    def _score(chosen: list[_Candidate]) -> float:
        # Fewer total violations = higher score. Normalize lightly so schedules
        # of equal size compare on violation count.
        violations = sum(len(c.violations) for c in chosen)
        return -float(violations)

    def _record(chosen: list[_Candidate]) -> None:
        notes = [f"{c.code} {c.section.get('section', '?')}: {', '.join(c.violations)}"
                 for c in chosen if c.violations]
        results.append({
            "sections": [c.section for c in chosen],
            "courses": [c.code for c in chosen],
            "score": _score(chosen),
            "prefs_met": all(not c.violations for c in chosen),
            "notes": notes,
        })

    def _dfs(idx: int, chosen: list[_Candidate], chosen_intervals: list) -> None:
        nonlocal nodes
        if nodes >= MAX_SEARCH_NODES:
            return
        # Enough courses picked — record a schedule and stop deepening this branch.
        if len(chosen) == target:
            _record(chosen)
            return
        # Not enough courses remain to reach the target — prune.
        remaining = len(ordered) - idx
        if len(chosen) + remaining < target:
            return
        for j in range(idx, len(ordered)):
            code = ordered[j]
            must_take = code in include
            for cand in by_course[code]:
                nodes += 1
                if nodes >= MAX_SEARCH_NODES:
                    return
                if any(_conflicts(cand.intervals, iv) for iv in chosen_intervals):
                    continue
                chosen.append(cand)
                chosen_intervals.append(cand.intervals)
                _dfs(j + 1, chosen, chosen_intervals)
                chosen.pop()
                chosen_intervals.pop()
                if len(results) >= max_results * 40:  # plenty gathered; stop early
                    return
            # A required include can't be skipped: if we advance past it without
            # taking it, this branch is invalid.
            if must_take:
                return

    _dfs(0, [], [])

    # Rank by score (fewest violations), then prefer larger schedules, and
    # dedup identical section sets that different branch orders can produce.
    results.sort(key=lambda r: (-r["score"], -len(r["courses"])))
    seen: set[tuple] = set()
    unique: list[dict] = []
    for r in results:
        key = tuple(sorted(s.get("crn", "") for s in r["sections"]))
        if key in seen:
            continue
        seen.add(key)
        unique.append(r)
        if len(unique) >= max_results:
            break
    return unique


# ---------------------------------------------------------------------------
# Self-test (no API keys / network needed): python3 backend/scheduler.py
# ---------------------------------------------------------------------------

def _demo_section(code: str, sec: str, days: list[str], begin: str | None,
                  end: str | None, delivery: str = "Face-to-face",
                  crn: str | None = None) -> dict:
    meeting = ({"days": days, "begin": begin, "end": end} if days or begin else None)
    return {
        "crn": crn or f"{code}{sec}", "subject_course": code, "section": sec,
        "title": code, "schedule_type": "Lecture", "seats_available": 5,
        "max_enrollment": 40, "enrollment": 35, "open": True, "delivery": delivery,
        "meeting": meeting, "instructors": [],
    }


def _run_self_test() -> None:
    sections = [
        _demo_section("CSC225", "A01", ["Mo", "We", "Th"], "0930", "1020"),
        _demo_section("CSC226", "A01", ["Mo", "We", "Th"], "0930", "1020"),  # conflicts w/ CSC225 A01
        _demo_section("CSC226", "A02", ["Mo", "We", "Th"], "1030", "1120"),
        _demo_section("MATH122", "A01", ["Tu", "Fr"], "1400", "1520"),
        _demo_section("SENG265", "A01", ["Mo", "We", "Th"], "1130", "1220"),
        _demo_section("ECON180", "A01", [], None, None, delivery="Online"),  # async
    ]

    # 1. Basic conflict-free schedule of 4.
    res = solve_schedule(sections, {"soft_preferences": [{"type": "max_courses", "value": 4}]})
    assert res, "expected at least one schedule"
    for sched in res:
        ivs = [_meeting_intervals(s) for s in sched["sections"]]
        for a in range(len(ivs)):
            for b in range(a + 1, len(ivs)):
                assert not _conflicts(ivs[a], ivs[b]), "returned a conflicting schedule!"
    print(f"[ok] basic solve returned {len(res)} conflict-free schedule(s) of size 4")

    # 2. Include a course that only has a conflicting section vs another include -> empty.
    res2 = solve_schedule(
        sections,
        {"include_courses": ["CSC225", "CSC226"],
         "soft_preferences": [{"type": "max_courses", "value": 2}]},
    )
    # CSC225 A01 conflicts with CSC226 A01 but CSC226 A02 is free -> solvable.
    assert res2, "CSC225+CSC226 should be solvable via CSC226 A02"
    codes = res2[0]["courses"]
    assert "CSC225" in codes and "CSC226" in codes
    picked_226 = next(s for s in res2[0]["sections"] if s["subject_course"] == "CSC226")
    assert picked_226["section"] == "A02", "should have avoided the conflicting A01"
    print("[ok] include set resolves to the non-conflicting section")

    # 3. Missing include -> empty.
    res3 = solve_schedule(sections, {"include_courses": ["PHYS102"]})
    assert res3 == [], "unknown include course should yield no schedule"
    print("[ok] unsatisfiable include returns empty")

    # 4. Soft preference scoring: avoiding Friday ranks MATH122 (Fri) lower.
    res4 = solve_schedule(
        sections,
        {"soft_preferences": [{"type": "no_days", "value": ["Fr"]},
                              {"type": "max_courses", "value": 3}]},
    )
    assert res4, "expected schedules"
    assert res4[0]["score"] >= res4[-1]["score"]
    top_has_friday = any("Fr" in (s.get("meeting") or {}).get("days", [])
                         for s in res4[0]["sections"])
    assert not top_has_friday, "top-ranked schedule should avoid Friday when possible"
    print("[ok] soft-preference scoring ranks Friday-free schedules first")

    # 5. Async section never conflicts.
    res5 = solve_schedule(
        [_demo_section("CSC225", "A01", ["Mo"], "0930", "1020"),
         _demo_section("ECON180", "A01", [], None, None, delivery="Online")],
        {"soft_preferences": [{"type": "max_courses", "value": 2}]},
    )
    assert res5 and len(res5[0]["courses"]) == 2
    print("[ok] async course schedules alongside a timed one")

    print("\nAll scheduler self-tests passed.")


if __name__ == "__main__":
    _run_self_test()
