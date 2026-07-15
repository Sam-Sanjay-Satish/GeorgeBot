#!/usr/bin/env python3
"""
RMP — RateMyProfessors ratings/reviews for UVic professors.

A self-contained live-data source for GeorgeBot, structured like `banner.py`:
pure data-fetching + caching, NO rendering (the caller — `chatbot.py` — renders
the fact blocks so the numbered-`[n]` scheme stays consistent across
graph/banner/rmp/vector).

RateMyProfessors has **no official API**. Every wrapper hits the same internal
GraphQL endpoint (`www.ratemyprofessors.com/graphql`) with a hardcoded public
Basic-auth token (base64 "test:test") — there is no key to provision. It's
unofficial and can break if RMP changes its schema, so this module is
**best-effort**: any failure returns `{}` and the answer degrades to no-RMP,
exactly like `banner.py`.

What it owns (and nothing else): third-party *student opinion* — average
quality/difficulty rating, would-take-again %, rating count, and recent written
reviews. This is subjective, self-selected data; it is NOT an authoritative UVic
source (the prompt frames it accordingly). Official facts (who teaches what,
seats, prereqs) stay with Banner / graph.

Scope is fixed to UVic: `RMP_SCHOOL_ID` is UVic's base64 relay id
("School-1488"), so we skip school-search entirely and search teachers directly.

Caching is in-process + ephemeral (module-level dict, TTL-keyed), like Banner —
ratings barely move, so an hours-long TTL is fine. Unlike Banner there's no
per-session form-state leak, so a single shared `requests.Session` is safe.

CLI smoke test (no LLM):
  python3 backend/rmp.py --name "Yong"
  python3 backend/rmp.py --name "Quinton Yong" --name "Nishant Mehta"
"""

from __future__ import annotations

import argparse
import sys
import threading
import time

import requests

RMP_GRAPHQL = "https://www.ratemyprofessors.com/graphql"
# Public constant hardcoded in every RMP wrapper — base64("test:test"). Not a
# secret, not per-user; RMP's read-only GraphQL accepts it unauthenticated.
RMP_AUTH = "Basic dGVzdDp0ZXN0"
# UVic's RMP school, as the base64 relay id the GraphQL API expects
# (base64("School-1488")). The teacher search REQUIRES this encoded form — the
# raw "1488" is rejected. See https://www.ratemyprofessors.com/school/1488
RMP_SCHOOL_ID = "U2Nob29sLTE0ODg="
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)
HTTP_TIMEOUT = 15    # seconds per request
REVIEWS_N = 20       # recent reviews pulled for a matched professor
RMP_TTL = 6 * 3600   # ratings barely move — hours (mirrors Banner's FACULTY_TTL)

# ---------------------------------------------------------------------------
# Module-level state: one TTL cache guarded by a lock, keyed by the queried name.
# A single shared Session is fine here (unlike Banner — RMP has no per-session
# search-form state to leak). The FastAPI SSE generator runs sync in a
# threadpool, so concurrent requests can hit this at once.
# ---------------------------------------------------------------------------
_lock = threading.Lock()
_rmp_cache: dict[str, tuple[float, dict]] = {}   # name.lower() -> (t, professor entry)
_session: requests.Session | None = None

_SEARCH_QUERY = (
    "query($text:String!,$schoolID:ID!){newSearch{teachers(query:{text:$text,"
    "schoolID:$schoolID}){edges{node{id firstName lastName avgRating avgDifficulty "
    "numRatings wouldTakeAgainPercent department legacyId}}}}}"
)
_REVIEWS_QUERY = (
    "query($id:ID!,$n:Int!){node(id:$id){... on Teacher{ratings(first:$n){edges{node{"
    "comment class date qualityRating difficultyRatingRounded wouldTakeAgain "
    "ratingTags thumbsUpTotal}}}}}}"
)


def _get_session() -> requests.Session:
    global _session
    with _lock:
        if _session is None:
            s = requests.Session()
            s.headers.update({
                "User-Agent": USER_AGENT,
                "Authorization": RMP_AUTH,
                "Content-Type": "application/json",
            })
            _session = s
        return _session


def _graphql(query: str, variables: dict) -> dict:
    """POST a GraphQL query, return the `data` object ({} on any error/absence)."""
    resp = _get_session().post(RMP_GRAPHQL, json={"query": query, "variables": variables},
                               timeout=HTTP_TIMEOUT)
    resp.raise_for_status()
    return (resp.json() or {}).get("data") or {}


def _full_name(node: dict) -> str:
    return f"{node.get('firstName', '')} {node.get('lastName', '')}".strip()


def _search_teachers(name: str) -> list[dict]:
    """Teacher nodes at UVic matching `name` (RMP's search is already fuzzy).
    Each node carries the summary fields directly — no second lookup needed."""
    data = _graphql(_SEARCH_QUERY, {"text": name.strip(), "schoolID": RMP_SCHOOL_ID})
    edges = (((data.get("newSearch") or {}).get("teachers") or {}).get("edges")) or []
    return [e["node"] for e in edges if e.get("node")]


def _split_tags(raw: str | None) -> list[str]:
    """RMP joins a review's tags with "--"; split and clean them."""
    if not raw:
        return []
    return [t.strip() for t in raw.split("--") if t.strip()]


def _fetch_reviews(teacher_id: str) -> list[dict]:
    """Up to REVIEWS_N recent written reviews for a teacher (by relay id)."""
    data = _graphql(_REVIEWS_QUERY, {"id": teacher_id, "n": REVIEWS_N})
    edges = (((data.get("node") or {}).get("ratings") or {}).get("edges")) or []
    reviews = []
    for e in edges:
        r = e.get("node") or {}
        reviews.append({
            "comment": (r.get("comment") or "").strip(),
            "course": r.get("class"),                        # "CSC110"
            "date": (r.get("date") or "")[:10],              # "2025-12-08"
            "quality": r.get("qualityRating"),               # 1-5
            "difficulty": r.get("difficultyRatingRounded"),  # 1-5
            "would_take_again": r.get("wouldTakeAgain"),     # 1 / 0 / -1(unknown)
            "tags": _split_tags(r.get("ratingTags")),
        })
    return reviews


def _wta_percent(node: dict) -> int | None:
    """wouldTakeAgainPercent is a float, or -1 when RMP has no data."""
    pct = node.get("wouldTakeAgainPercent")
    if pct is None or pct < 0:
        return None
    return round(pct)


def _professor_entry(name: str) -> dict:
    """Resolve one queried name to a professor entry (cached by name).

    Mirrors Banner's instructor semantics:
      - exactly one distinct teacher matched -> full summary + reviews (matched=True)
      - several matched (common surname)      -> candidates listed, no reviews
      - none matched                          -> matched=False, empty candidates
    """
    key = name.strip().lower()
    with _lock:
        hit = _rmp_cache.get(key)
        if hit and (time.monotonic() - hit[0] < RMP_TTL):
            return hit[1]

    nodes = _search_teachers(name)
    # Dedup by full display name (RMP can surface the same person more than once).
    by_name: dict[str, dict] = {}
    for node in nodes:
        by_name.setdefault(_full_name(node), node)

    if not by_name:
        entry = {"query": name, "name": None, "matched": False, "candidates": []}
    elif len(by_name) > 1:
        candidates = [f"{nm} ({nd.get('department') or 'Unknown dept'})"
                      for nm, nd in by_name.items()]
        entry = {"query": name, "name": None, "matched": False, "candidates": candidates}
    else:
        node = next(iter(by_name.values()))
        num = node.get("numRatings") or 0
        entry = {
            "query": name,
            "name": _full_name(node),
            "matched": True,
            "candidates": [],
            "rating": node.get("avgRating") if num else None,
            "difficulty": node.get("avgDifficulty") if num else None,
            "would_take_again": _wta_percent(node) if num else None,
            "num_ratings": num,
            "department": node.get("department"),
            "legacy_id": node.get("legacyId"),
            "reviews": _fetch_reviews(node["id"]) if num else [],
        }

    with _lock:
        _rmp_cache[key] = (time.monotonic(), entry)
    return entry


# ---------------------------------------------------------------------------
# Top-level entry — gated call from chatbot.py
# ---------------------------------------------------------------------------

def rmp_retrieve(names: list[str]) -> dict:
    """RateMyProfessors ratings/reviews for one or more professor names.

    Best-effort: any failure (network, bad JSON, schema drift) returns {} so the
    answer pipeline degrades to no-RMP rather than erroring — same spirit as
    Banner and the router's vector-only fallback. Returns:

        {"kind": "rmp", "professors": [ <entry>, ... ]}

    where each entry is a matched professor (summary + reviews), an ambiguous
    name (candidates to disambiguate), or a no-match note. Duplicate/empty names
    are ignored; {} if nothing usable was requested.
    """
    try:
        seen: set[str] = set()
        professors = []
        for name in names:
            key = (name or "").strip().lower()
            if not key or key in seen:
                continue
            seen.add(key)
            professors.append(_professor_entry(name))
        if not professors:
            return {}
        return {"kind": "rmp", "professors": professors}
    except Exception as e:  # noqa: BLE001 — best-effort, never break the answer
        print(f"  [rmp_retrieve] failed, skipping ratings: {e}", file=sys.stderr)
        return {}


# ---------------------------------------------------------------------------
# CLI smoke test (no LLM) — see module docstring
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="RateMyProfessors lookup — smoke test")
    parser.add_argument("--name", action="append", default=[],
                        help="professor name (repeatable)")
    args = parser.parse_args()
    if not args.name:
        parser.error("pass at least one --name")

    facts = rmp_retrieve(args.name)
    if not facts:
        print("No RMP data (lookup failed or nothing matched).")
        return
    for p in facts["professors"]:
        print("=" * 60)
        if not p["matched"]:
            if p["candidates"]:
                print(f'"{p["query"]}" is ambiguous: ' + "; ".join(p["candidates"]))
            else:
                print(f'No RMP match for "{p["query"]}".')
            continue
        if not p["num_ratings"]:
            print(f'{p["name"]} ({p.get("department")}) — on RMP but no ratings yet.')
            continue
        wta = f"{p['would_take_again']}%" if p["would_take_again"] is not None else "n/a"
        print(f"{p['name']} — {p.get('department')}")
        print(f"  rating {p['rating']}/5   difficulty {p['difficulty']}/5   "
              f"would take again {wta}   ({p['num_ratings']} ratings)")
        for r in p["reviews"][:5]:
            tags = f"  [{', '.join(r['tags'])}]" if r["tags"] else ""
            print(f"  - {r['course'] or '?'} ({r['date']}) q{r['quality']}/d{r['difficulty']}{tags}")
            print(f"    {r['comment'][:200]}")


if __name__ == "__main__":
    main()
