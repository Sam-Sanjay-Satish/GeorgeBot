#!/usr/bin/env python3
"""
Banner — live UVic class availability (seats / waitlist / schedule / instructor).

A self-contained live-data source for GeorgeBot, structured like `graph_queries.py`:
pure data-fetching + caching, NO rendering (the caller — `chatbot.py` — renders the
fact blocks, keeping the numbered-`[n]` scheme consistent across graph/banner/vector).

Unlike the Chroma / graph artifacts (a static snapshot served from a Railway Volume),
Banner is *current, per-section, real-time* enrollment straight from UVic's Ellucian
Banner 9 registration self-service at `banner.uvic.ca`. Read-only class search needs
no auth — just a `JSESSIONID` cookie with a term bound into the session (3-step
handshake below). See `BANNER_API.md` for the full endpoint/field research.

This module owns ONLY what the static index can't give:
  - live seats / waitlist counts        (the headline)
  - per-section schedule (days/time/room)
  - instructor (name + email) for the term
  - delivery mode & campus per section
Prereqs / credits / program requirements stay owned by the graph path.

Caching is deliberately **in-process and ephemeral** (module-level dicts, TTL-keyed) —
freshness is the whole point, so a survives-restart cache (Railway Volume) would be
exactly wrong. Seat counts get a short 120s TTL; the term list and per-section
instructor get an hours-long TTL (they barely move).

CLI smoke test (no LLM):
  python3 backend/banner.py --course CSC225
  python3 backend/banner.py --course CSC225 --term 202609
"""

from __future__ import annotations

import argparse
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import requests

from ttlcache import BoundedTTLCache

BASE = "https://banner.uvic.ca/StudentRegistrationSsb/ssb"
# Undocumented internal endpoints — present as a real browser, don't look like a bare
# script (per BANNER_API.md "Cautions").
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)
HTTP_TIMEOUT = 15  # seconds per request

SECTIONS_TTL = 120       # seat counts move minute-to-minute during registration
FACULTY_TTL = 6 * 3600   # instructor for a section barely changes
TERMS_TTL = 6 * 3600     # valid-term list barely changes

# Outbound-amplification guard (2026-08-01, audit issue 4). `course_codes` reach
# this module from an LLM router — chatbot.py caps + pattern-validates them at
# parse time, but this bounds the module itself no matter who calls it.
MAX_COURSES_PER_CALL = int(os.environ.get("BANNER_MAX_COURSES", "5"))
# Global ceiling on simultaneous in-flight HTTP requests to banner.uvic.ca
# across ALL threads/requests (handshakes, searches, and the per-CRN faculty
# fan-out all count). When saturated, a waiter gives up after
# _SEM_ACQUIRE_TIMEOUT and the lookup degrades to "no live data" ({} from the
# best-effort top-level entries) instead of queueing more load onto UVic.
MAX_OUTBOUND_CONCURRENCY = int(os.environ.get("BANNER_MAX_CONCURRENCY", "8"))
_SEM_ACQUIRE_TIMEOUT = 2  # seconds to wait for a slot before giving up

# Season -> the MM half of a YYYYMM Banner term code.
#   01 = Second Term  (Jan-Apr, "Spring")
#   05 = Summer Session (May-Aug)
#   09 = First Term   (Sep-Dec, "Fall")
_SEASON_MM = {"spring": "01", "summer": "05", "fall": "09"}
_MM_SEASON = {"01": "Spring", "05": "Summer", "09": "Fall"}

# ---------------------------------------------------------------------------
# Module-level state: TTL caches guarded by a lock. Sessions are NOT shared —
# each search uses its own dedicated session (see `_handshake_session`), so the
# lock only guards the cache dicts. The FastAPI SSE generator is a sync def run
# in a threadpool, so concurrent requests can hit these caches at once.
# ---------------------------------------------------------------------------
#
# The caches are BOUNDED (audit issue 5): their keys are user-driven — arbitrary
# course codes and instructor names — and as plain dicts they were never pruned,
# so a loop of unique keys grew them until the container OOM-died. BoundedTTLCache
# keeps the same (t, value) entry shape and the TTL checks below, and adds real
# eviction (expired on read, least-recently-used past maxsize). Sizes are far
# above real traffic; they only bite under abuse.
_lock = threading.Lock()
_sections_cache = BoundedTTLCache(512, SECTIONS_TTL)    # (term,subject,number) -> (t, sections)
_faculty_cache = BoundedTTLCache(4096, FACULTY_TTL)     # (term, crn) -> (t, faculty)
_instructor_cache = BoundedTTLCache(512, SECTIONS_TTL)  # (term, name) -> (t, result); 120s, as its reader checks
_terms_cache: tuple[float, list] | None = None         # (t, [{code, description}]) — single entry, unbounded growth impossible
_outbound_sem = threading.BoundedSemaphore(MAX_OUTBOUND_CONCURRENCY)


def _outbound(call, *args, **kwargs):
    """Run one outbound HTTP call (a bound `session.get`/`session.post`) under the
    global concurrency cap. The semaphore is held only for the network call itself
    and is never held while waiting on `_lock` (cache writes happen after release),
    so the lock->semaphore nesting in `get_terms` can't deadlock. Raises
    RuntimeError when saturated ("give up fast"); the best-effort top-level
    entries catch it and return {}."""
    if not _outbound_sem.acquire(timeout=_SEM_ACQUIRE_TIMEOUT):
        raise RuntimeError("banner.uvic.ca outbound concurrency limit reached")
    try:
        return call(*args, **kwargs)
    finally:
        _outbound_sem.release()


def split_course_code(code: str) -> tuple[str, str]:
    """"CSC225" -> ("CSC", "225"). Splits at the first digit. Also the basis for
    the display form used in api.py's `_course_label`."""
    code = (code or "").replace(" ", "").upper()
    for i, ch in enumerate(code):
        if ch.isdigit():
            return code[:i], code[i:]
    return code, ""


def term_label(code: str) -> str:
    """"202609" -> "Fall 2026". Falls back to the raw code if unparseable."""
    if code and len(code) == 6 and code.isdigit():
        season = _MM_SEASON.get(code[4:])
        if season:
            return f"{season} {code[:4]}"
    return code


# ---------------------------------------------------------------------------
# Session handshake + low-level fetch
# ---------------------------------------------------------------------------

def _new_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT})
    return s


def _handshake_session(term: str) -> requests.Session:
    """A fresh session with `term` bound into it (JSESSIONID + term selected — both
    required before `searchResults` returns anything; see BANNER_API.md).

    Deliberately a NEW session per call rather than a shared module-level one. Banner's
    `searchResults` returns the *session's previous* search unless the search form is
    reset, so a shared session leaks one course's results into the next query (and
    concurrent requests would race on that per-session form state). A dedicated session
    per search gives each its own isolated form state — no stale bleed, no lock held
    across the network. Cheap: two quick calls, and results are cached (see callers)."""
    s = _new_session()
    _outbound(s.get, f"{BASE}/registration", timeout=HTTP_TIMEOUT)            # sets JSESSIONID
    _outbound(s.post, f"{BASE}/term/search?mode=search", data={"term": term},  # binds the term
              timeout=HTTP_TIMEOUT)
    return s


# ---------------------------------------------------------------------------
# Term list + resolution
# ---------------------------------------------------------------------------

def get_terms() -> list[dict]:
    """`[{code, description}]` newest-first. Past terms are marked "(View Only)" in
    the description. No session needed. Cached for hours."""
    global _terms_cache
    with _lock:
        if _terms_cache and (time.monotonic() - _terms_cache[0] < TERMS_TTL):
            return _terms_cache[1]
        s = _new_session()
        resp = _outbound(s.get, f"{BASE}/classSearch/getTerms",
                         params={"searchTerm": "", "offset": 1, "max": 20},
                         timeout=HTTP_TIMEOUT)
        terms = resp.json() or []
        _terms_cache = (time.monotonic(), terms)
        return terms


def _is_registerable(t: dict) -> bool:
    return "(View Only)" not in (t.get("description") or "")


def resolve_term(season: str | None = None, year: int | None = None) -> str | None:
    """Resolve a Banner term code from an optional season/year hint.

    - No hint          -> the current/upcoming registerable term (the smallest,
                          i.e. earliest still-open, non-"View Only" code). UVic marks
                          past terms "View Only", so the non-View-Only set is exactly
                          {current + upcoming}.
    - season (+ year)  -> build YYYYMM from the hint and confirm it exists in the term
                          list; season-only picks the nearest registerable term of that
                          season.
    Returns None only if Banner returned no terms at all.
    """
    terms = get_terms()
    if not terms:
        return None
    registerable = sorted((t["code"] for t in terms if _is_registerable(t)))
    all_codes = {t["code"] for t in terms}

    season = (season or "").strip().lower() or None
    mm = _SEASON_MM.get(season) if season else None

    if mm and year:
        code = f"{year}{mm}"
        if code in all_codes:
            return code
        # Named a term Banner doesn't list — fall through to the default.
    elif mm:
        # Season only: nearest registerable term of that season (else any listed one).
        matches = [c for c in registerable if c.endswith(mm)]
        if not matches:
            matches = sorted(c for c in all_codes if c.endswith(mm))
        if matches:
            return matches[0]

    if registerable:
        return registerable[0]
    return sorted(all_codes)[0] if all_codes else None


# ---------------------------------------------------------------------------
# Section search + instructor
# ---------------------------------------------------------------------------

def _fetch_faculty(session: requests.Session, term: str, crn: str) -> list[dict]:
    """Instructor(s) for a CRN — `searchResults` returns `faculty: []`, so this
    separate call is the only way to get the prof. Cached for hours."""
    key = (term, crn)
    with _lock:
        hit = _faculty_cache.get(key)
        if hit and (time.monotonic() - hit[0] < FACULTY_TTL):
            return hit[1]
    resp = _outbound(session.get, f"{BASE}/searchResults/getFacultyMeetingTimes",
                     params={"term": term, "courseReferenceNumber": crn},
                     timeout=HTTP_TIMEOUT)
    fmt = (resp.json() or {}).get("fmt") or []
    faculty: list[dict] = []
    for entry in fmt:
        for f in entry.get("faculty") or []:
            faculty.append({
                "name": f.get("displayName"),
                "email": f.get("emailAddress"),
                "primary": bool(f.get("primaryIndicator")),
            })
    # Dedup by name (a prof can appear once per meeting-time entry).
    seen, uniq = set(), []
    for f in faculty:
        if f["name"] and f["name"] not in seen:
            seen.add(f["name"])
            uniq.append(f)
    with _lock:
        _faculty_cache[key] = (time.monotonic(), uniq)
    return uniq


def _meeting_from_section(raw: dict) -> dict | None:
    """Pull the inline schedule (no extra call) from `meetingsFaculty[].meetingTime`.

    Only days/times are kept: UVic's feed never populates `room` (always "None
    specified") and its `building` is an unlabeled numeric code, so neither is surfaced.
    """
    mf = raw.get("meetingsFaculty") or []
    if not mf:
        return None
    mt = mf[0].get("meetingTime") or {}
    days = [d[:2].capitalize() for d in
            ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")
            if mt.get(d)]
    return {
        "days": days,                       # e.g. ["Mo", "We", "Th"]
        "begin": mt.get("beginTime"),       # "1430" (24h HHMM)
        "end": mt.get("endTime"),           # "1520"
        "start_date": mt.get("startDate"),
        "end_date": mt.get("endDate"),
    }


def search_sections(subject: str, number: str, term: str) -> list[dict]:
    """All sections (A01, A02, B01, …) for a course in a term, normalized to the
    fields we serve. Cached 120s per (term, subject, number). Instructor is fetched
    per section (cached longer)."""
    key = (term, subject, number)
    with _lock:
        hit = _sections_cache.get(key)
        if hit and (time.monotonic() - hit[0] < SECTIONS_TTL):
            return hit[1]

    session = _handshake_session(term)   # dedicated session — no cross-course bleed
    resp = _outbound(
        session.get,
        f"{BASE}/searchResults/searchResults",
        params={"txt_subject": subject, "txt_courseNumber": number, "txt_term": term,
                "pageOffset": 0, "pageMaxSize": 50,
                "sortColumn": "subjectDescription", "sortDirection": "asc"},
        timeout=HTTP_TIMEOUT,
    )
    data = (resp.json() or {}).get("data") or []

    # Instructor needs a separate call per section (searchResults returns
    # faculty: []). These are independent round-trips, so fan them out over a small
    # thread pool — a course with N sections goes from N sequential GETs to ~1 wall-
    # clock GET on a cold cache. The cache is _lock-guarded and a requests.Session's
    # GETs share urllib3's thread-safe connection pool, so concurrent use is fine.
    crns = [str(raw.get("courseReferenceNumber") or "") for raw in data]
    unique_crns = [c for c in dict.fromkeys(crns) if c]
    faculty_by_crn: dict[str, list] = {}
    if unique_crns:
        with ThreadPoolExecutor(max_workers=min(8, len(unique_crns))) as pool:
            faculty_by_crn = dict(zip(
                unique_crns,
                pool.map(lambda c: _fetch_faculty(session, term, c), unique_crns),
            ))

    sections = [_normalize_section(raw, faculty_by_crn.get(crn, []))
                for raw, crn in zip(data, crns)]

    with _lock:
        _sections_cache[key] = (time.monotonic(), sections)
    return sections


def _normalize_section(raw: dict, instructors: list[dict]) -> dict:
    """Shape one raw Banner section object into the fields we serve. `instructors`
    is passed in — the course path fetches it per CRN, the instructor-search path
    already knows who it searched for."""
    return {
        "crn": str(raw.get("courseReferenceNumber") or ""),
        "subject_course": raw.get("subjectCourse"),        # "CSC225"
        "section": raw.get("sequenceNumber"),              # "A01"
        "title": raw.get("courseTitle"),
        "schedule_type": raw.get("scheduleTypeDescription"),   # Lecture / Lab / Tutorial
        "seats_available": raw.get("seatsAvailable"),
        "max_enrollment": raw.get("maximumEnrollment"),
        "enrollment": raw.get("enrollment"),
        "open": bool(raw.get("openSection")),
        "wait_available": raw.get("waitAvailable"),
        "wait_count": raw.get("waitCount"),
        "wait_capacity": raw.get("waitCapacity"),
        "delivery": raw.get("instructionalMethodDescription"),  # Face-to-face / Online
        "campus": raw.get("campusDescription"),
        "credits": raw.get("creditHours"),
        "meeting": _meeting_from_section(raw),
        "instructors": instructors,
    }


# ---------------------------------------------------------------------------
# Instructor -> courses reverse lookup
# ---------------------------------------------------------------------------

def _get_instructor_matches(session: requests.Session, term: str, name: str) -> list[dict]:
    """`get_instructor` autocomplete -> [{code, description}]. NOTE the `code` is a
    SESSION-SCOPED ephemeral id (it changes every session), so it's only valid for a
    `txt_instructor` search issued on this same session — never cache or reuse it."""
    resp = _outbound(session.get, f"{BASE}/classSearch/get_instructor",
                     params={"term": term, "searchTerm": name, "offset": 1, "max": 15},
                     timeout=HTTP_TIMEOUT)
    return resp.json() or []


def search_by_instructor(name: str, term: str) -> dict:
    """Reverse lookup: which sections a professor teaches in `term`.

    Returns `{"instructor": <resolved name|None>, "candidates": [names],
    "sections": [section, ...]}`:
      - exactly one instructor matched -> `instructor` set, `sections` populated.
      - multiple matched (common surname) -> `candidates` lists them, `sections` empty
        (caller asks the user to disambiguate).
      - none matched -> all empty.

    Uses a DEDICATED session (like every search here) for the whole get_instructor ->
    txt_instructor sequence. The instructor `code` is session-ephemeral, so the two
    calls must share one session; a throwaway session keeps that fragile state isolated.
    Cached 120s by (term, name)."""
    key = (term, name.strip().lower())
    with _lock:
        hit = _instructor_cache.get(key)
        if hit and (time.monotonic() - hit[0] < SECTIONS_TTL):
            return hit[1]

    session = _handshake_session(term)

    matches = _get_instructor_matches(session, term, name)
    # Fall back to the surname alone if a full "First Last" query found nothing.
    if not matches and " " in name.strip():
        matches = _get_instructor_matches(session, term, name.strip().split()[-1])

    # Dedup by display name (one person can appear once per term-assignment).
    descs = list(dict.fromkeys(m.get("description") for m in matches if m.get("description")))

    if not descs:
        result = {"instructor": None, "candidates": [], "sections": []}
    elif len(descs) > 1:
        result = {"instructor": None, "candidates": descs, "sections": []}
    else:
        code = matches[0]["code"]
        resp = _outbound(
            session.get,
            f"{BASE}/searchResults/searchResults",
            params={"txt_instructor": code, "txt_term": term, "pageOffset": 0,
                    "pageMaxSize": 100, "sortColumn": "subjectDescription",
                    "sortDirection": "asc"},
            timeout=HTTP_TIMEOUT,
        )
        data = (resp.json() or {}).get("data") or []
        # We already know the instructor (we searched by them); no per-CRN faculty call.
        known = [{"name": descs[0], "email": None, "primary": True}]
        sections = [_normalize_section(raw, known) for raw in data]
        result = {"instructor": descs[0], "candidates": [descs[0]], "sections": sections}

    with _lock:
        _instructor_cache[key] = (time.monotonic(), result)
    return result


def _group_by_course(sections: list[dict]) -> list[dict]:
    """Group flat sections into `[{code, sections}]` by subjectCourse, preserving
    first-seen order (Banner already sorts by subject)."""
    courses: dict[str, dict] = {}
    for s in sections:
        code = s.get("subject_course") or "?"
        courses.setdefault(code, {"code": code, "sections": []})["sections"].append(s)
    return list(courses.values())


# ---------------------------------------------------------------------------
# Top-level entries — gated calls from chatbot.py
# ---------------------------------------------------------------------------

def banner_retrieve(course_codes: list[str], season: str | None = None,
                    year: int | None = None) -> dict:
    """Live section data for the given normalized course codes.

    Best-effort: any failure (network, session, bad JSON, term resolution) returns an
    empty dict so the answer pipeline degrades to no-Banner rather than erroring —
    same spirit as the router's vector-only fallback. Returns:

        {"term": "202609", "term_label": "Fall 2026",
         "courses": [{"code": "CSC225", "sections": [ ... ]}]}

    Courses with no sections found in the resolved term are dropped. At most
    MAX_COURSES_PER_CALL codes are honored (each code costs ~10 outbound calls
    on a cold cache — see the outbound-amplification guard at the top).
    """
    try:
        term = resolve_term(season, year)
        if not term:
            return {}
        courses = []
        for code in course_codes[:MAX_COURSES_PER_CALL]:
            subject, num = split_course_code(code)
            if not subject or not num:
                continue
            sections = search_sections(subject, num, term)
            if sections:
                courses.append({"code": code, "sections": sections})
        if not courses:
            return {}
        return {"kind": "availability", "term": term, "term_label": term_label(term),
                "courses": courses}
    except Exception as e:  # noqa: BLE001 — best-effort, never break the answer
        import sys
        print(f"  [banner_retrieve] failed, skipping live availability: {e}",
              file=sys.stderr)
        return {}


def banner_instructor_retrieve(name: str, season: str | None = None,
                               year: int | None = None) -> dict:
    """Reverse lookup: which courses/sections a professor teaches in the resolved term.

    Best-effort (returns {} on failure, like `banner_retrieve`). Returns one of:

        # resolved to one instructor
        {"kind":"instructor", "term", "term_label", "instructor": "Quinton Yong",
         "instructor_query": "Yong", "courses": [{"code","sections":[...]}]}
        # ambiguous name — caller asks the user to pick
        {"kind":"instructor", ..., "instructor": None, "candidates": ["A","B"]}
        # no classes found for that name this term
        {"kind":"instructor", ..., "instructor": None, "candidates": [], "courses": []}
    """
    try:
        term = resolve_term(season, year)
        if not term:
            return {}
        res = search_by_instructor(name, term)
        out = {"kind": "instructor", "term": term, "term_label": term_label(term),
               "instructor_query": name, "instructor": res["instructor"],
               "candidates": res.get("candidates", []),
               "courses": _group_by_course(res["sections"])}
        return out
    except Exception as e:  # noqa: BLE001 — best-effort, never break the answer
        import sys
        print(f"  [banner_instructor_retrieve] failed, skipping: {e}", file=sys.stderr)
        return {}


# ---------------------------------------------------------------------------
# CLI smoke test (no LLM) — see module docstring
# ---------------------------------------------------------------------------

def _fmt_time(hhmm: str | None) -> str:
    if not hhmm or len(hhmm) != 4:
        return "?"
    return f"{hhmm[:2]}:{hhmm[2:]}"


def _print_section(s: dict, show_course: bool = False) -> None:
    sched = s.get("meeting")
    when = ""
    if sched:
        days = "".join(sched["days"])
        when = f"  {days} {_fmt_time(sched['begin'])}-{_fmt_time(sched['end'])}"
    prof = ", ".join(i["name"] for i in s["instructors"]) or "TBA"
    wl = f"  WL {s['wait_count']}/{s['wait_capacity']}" if s.get("wait_capacity") else ""
    full = "" if s["open"] else "  [FULL]"
    label = f"{s.get('subject_course', '')} " if show_course else ""
    print(f"{label}{s['section']:<4} {s['schedule_type']:<8} "
          f"seats {s['seats_available']}/{s['max_enrollment']} "
          f"({s['enrollment']} enrolled){wl}{full}{when}  — {prof}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Banner live class availability — smoke test")
    parser.add_argument("--course", default=None, help="normalized course code, e.g. CSC225")
    parser.add_argument("--instructor", default=None, help="professor name, e.g. 'Yong' (reverse lookup)")
    parser.add_argument("--term", default=None, help="explicit YYYYMM term (default: current/upcoming)")
    parser.add_argument("--season", default=None, choices=list(_SEASON_MM),
                        help="season hint if no explicit --term")
    parser.add_argument("--year", type=int, default=None, help="year hint if no explicit --term")
    args = parser.parse_args()
    if not args.course and not args.instructor:
        parser.error("pass --course or --instructor")

    term = args.term or resolve_term(args.season, args.year)
    print(f"Resolved term: {term} ({term_label(term or '')})\n")

    if args.instructor:
        res = search_by_instructor(args.instructor, term)
        if res["candidates"] and not res["instructor"]:
            print(f"Multiple instructors match '{args.instructor}': "
                  + "; ".join(res["candidates"]))
            return
        if not res["sections"]:
            print(f"No classes found for an instructor matching '{args.instructor}'.")
            return
        print(f"{res['instructor']} teaches:")
        for c in _group_by_course(res["sections"]):
            print(f"  {c['code']}:")
            for s in c["sections"]:
                print("  ", end="")
                _print_section(s)
        return

    subject, num = split_course_code(args.course)
    sections = search_sections(subject, num, term)
    if not sections:
        print("No sections found.")
        return
    for s in sections:
        _print_section(s)


if __name__ == "__main__":
    main()
