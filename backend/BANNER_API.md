# Banner API — live class availability (research notes)

Research for a planned feature: real-time class **availability, seat counts,
section schedule, and instructor** from UVic's Banner registration system.
Everything below was verified live against `banner.uvic.ca` on 2026-07-15
(Fall 2026 / term `202609`). No auth required for read-only class search.

**Status:** IMPLEMENTED (2026-07-15) in `backend/banner.py`, wired into the gated
retrieval path in `chatbot.py`/`api.py`. This doc is the endpoint/field reference it
was built from. Two live features ship: **course availability** (`banner_retrieve`,
router `wants_availability`) and **instructor → courses reverse lookup**
(`banner_instructor_retrieve`, router `instructor_query`).

---

## What it is (and why it's not in the pipeline)

UVic runs **Ellucian Banner 9 Student Registration Self-Service** at
`https://banner.uvic.ca/StudentRegistrationSsb/ssb`. The registration web UI is
backed by plain JSON endpoints that the page calls under the hood — the same
ones we call here. No login for class search.

This is fundamentally different from everything GeorgeBot serves today. The
Chroma / graph artifacts are a **static snapshot** built in `georgebot-pipeline`.
Banner is **current, per-section, real-time** enrollment. So this is a new
**live-data source**, not a retrieval tune — it does NOT belong in the vector
index or the graph. It should be a **gated tool/retrieval step** the answer
pipeline calls, parallel to how graph retrieval fires only when a course code
is present (see "Integration plan" below).

**Scope boundary (important):** Banner overlaps with data we already serve
better from Kuali/graph (prereqs, credits, restrictions, description). Keep
Banner tight — it should only own what the static index *can't* give:

1. **Live seat / waitlist counts** (the headline feature)
2. **Section-level schedule** (days, times, building/room per A01/B01)
3. **Instructor for the term** (name + email)
4. **Delivery mode & campus** per section

Let the existing graph path keep owning prereqs / credits / program requirements.

---

## Request flow (session-based — verified working)

Banner requires a `JSESSIONID` cookie AND you must **select a term** into the
session before searching. Three steps:

1. `GET  /ssb/registration`
   → establishes the session cookie (`JSESSIONID`).
2. `POST /ssb/term/search?mode=search`  body: `term=202609`
   → binds that term to the session. **Required** — `searchResults` returns
   empty without it.
3. `GET  /ssb/searchResults/searchResults?txt_subject=CSC&txt_courseNumber=225&txt_term=202609&pageOffset=0&pageMaxSize=10`
   → the actual data (JSON).

Hold ONE `requests.Session` (cookie jar) and reuse it across questions.
Re-handshake only when it expires (401 / empty results after previously
working). Optionally keep one session per term.

### Base + example (copy-paste smoke test)

```bash
JAR=cookies.txt
BASE=https://banner.uvic.ca/StudentRegistrationSsb/ssb
TERM=202609

curl -s -c $JAR -b $JAR "$BASE/registration" -o /dev/null
curl -s -c $JAR -b $JAR -X POST "$BASE/term/search?mode=search" --data "term=$TERM" -o /dev/null
curl -s -c $JAR -b $JAR "$BASE/searchResults/searchResults?txt_subject=CSC&txt_courseNumber=225&txt_term=$TERM&pageMaxSize=10"
```

Set a real `User-Agent` header in the client (these are undocumented internal
endpoints — don't look like a bare script).

---

## Endpoints

### `getTerms` — list valid terms (no session needed)
```
GET /ssb/classSearch/getTerms?searchTerm=&offset=1&max=10
```
Returns `[{code, description}]`. Term code format is `YYYYMM`:
- `202609` = First Term Sep–Dec 2026 (Fall)
- `202601` = Second Term Jan–Apr 2026 (Spring)
- `202605` = Summer Session May–Aug 2026

Terms marked `(View Only)` in the description are past/closed. For "current
term" resolution: pick the nearest **non–"View Only"** code. Cache for hours
(barely changes).

### `get_subject` — subject codes (CSC, MATH, …)
```
GET /ssb/classSearch/get_subject?term=202609&searchTerm=&offset=1&max=...
```

### `searchResults` — the main data (clean JSON) ⭐
```
GET /ssb/searchResults/searchResults
    ?txt_subject=CSC
    &txt_courseNumber=225
    &txt_term=202609
    &pageOffset=0
    &pageMaxSize=10
    &sortColumn=subjectDescription
    &sortDirection=asc
```
Response: `{ success, totalCount, data: [ <section>, ... ], ... }`.
One `<section>` object per section (A01, A02, B01, …). Full field set below.

### `getFacultyMeetingTimes` — instructor + schedule (clean JSON)
```
GET /ssb/searchResults/getFacultyMeetingTimes?term=202609&courseReferenceNumber=10780
```
Needed only because `searchResults` returns `faculty: []` (empty). This call
populates the instructor. Keyed by CRN (`courseReferenceNumber`). One call per
section if you want the prof.

### `get_instructor` + instructor search — reverse lookup (what a prof teaches) ⭐
Two-step, and there's a **sharp gotcha**. First resolve the name:
```
GET /ssb/classSearch/get_instructor?term=202609&searchTerm=Yong&offset=1&max=15
→ [{ "code": "92738", "description": "Quinton Yong" }, ...]
```
Then search that instructor's sections (across all subjects):
```
GET /ssb/searchResults/searchResults?txt_instructor=92738&txt_term=202609&pageMaxSize=100
→ same {success, totalCount, data:[<section>]} shape as a normal search
```
**GOTCHA: the `get_instructor` `code` is a SESSION-SCOPED EPHEMERAL id, not a stable
PIDM.** It increments every session (observed `92736 → 92737 → 92738` across three
handshakes), and using a code from a *different* session (or a stale one) fails —
either HTTP 500 `Cannot get property 'pidm' on null object`, or a wrong-instructor
result. So `get_instructor` and the `txt_instructor` search **must run on the same
session, back-to-back.** In this codebase (`banner.search_by_instructor`) that's done
with a **dedicated throwaway session per lookup** (own handshake → get_instructor →
search), keeping the fragile ephemeral state off the shared module session; only the
*result* is cached (120s by term+name). `get_instructor` may return multiple people
(common surname) → disambiguate with the user. Sections returned this way also have
`faculty: []`, but you already know the instructor (you searched by them), so no
`getFacultyMeetingTimes` call is needed.

### HTML-only endpoints — SKIP (overlap with graph, must scrape HTML)
- `POST /ssb/searchResults/getClassDetails`
- `POST /ssb/searchResults/getSectionPrerequisites`
- `POST /ssb/searchResults/getRestrictions`

Return rendered HTML fragments, not JSON. Content (prereqs, level/program
restrictions, description) duplicates what Kuali/graph already serve cleanly.
Not worth scraping.

---

## `searchResults` — full section field set (verified: CSC 225 A01, Fall 2026)

### Availability (the point of the feature)
| Field | Meaning | Example |
|---|---|---|
| `seatsAvailable` | open seats | `11` |
| `maximumEnrollment` | section cap | `100` |
| `enrollment` | currently enrolled | `89` |
| `openSection` | is it open | `true` |
| `waitAvailable` | open waitlist spots | `25` |
| `waitCapacity` | waitlist cap | `25` |
| `waitCount` | current waitlist size | `0` |

### Identity / classification
| Field | Example |
|---|---|
| `courseReferenceNumber` (CRN) | `"10780"` |
| `sequenceNumber` (section) | `"A01"` |
| `subject` / `subjectCourse` | `"CSC"` / `"CSC225"` |
| `subjectDescription` | `"Computer Science"` |
| `courseNumber` / `courseTitle` | `"225"` / `"Algorithms and Data Structures I"` |
| `scheduleTypeDescription` | `"Lecture"` (also Lab / Tutorial) |
| `linkIdentifier` | `"A1"` (which lecture a lab/tutorial links to) |
| `isSectionLinked` | `true` |
| `instructionalMethodDescription` | `"Face-to-face"` (also Online) |
| `campusDescription` | `"Main (Gordon Head)"` |
| `partOfTerm` | `"1"` |
| `creditHours` (+ `creditHourLow/High`) | `1.5` |

### Cross-listing
`crossList`, `crossListAvailable`, `crossListCapacity`, `crossListCount`
(all `None` when not cross-listed).

### Embedded meeting times — `meetingsFaculty[].meetingTime`
No extra call needed for schedule; it's inline. Shape:
```json
{
  "beginTime": "1430", "endTime": "1520",      // 24h HHMM
  "monday": true, "tuesday": false, "wednesday": true,
  "thursday": true, "friday": false, "saturday": false, "sunday": false,
  "building": "148", "buildingDescription": "None specified",
  "room": "None specified",
  "campusDescription": "Main (Gordon Head)",
  "startDate": "Sep 09, 2026", "endDate": "Dec 06, 2026",
  "hoursWeek": 3.0, "creditHourSession": 1.5,
  "meetingScheduleType": "LEC", "meetingTypeDescription": "Every Week"
}
```
`faculty` inside `meetingsFaculty` is `[]` in `searchResults` — use
`getFacultyMeetingTimes` for the instructor.

### `getFacultyMeetingTimes` — instructor shape
```json
{ "fmt": [ { "faculty": [ {
  "displayName": "Quinton Yong",
  "emailAddress": "quintonyong@uvic.ca",
  "primaryIndicator": true,
  "bannerId": "87480"
} ], "meetingTime": { ...same as above... } } ] }
```

### Live sample (Fall 2026, pulled 2026-07-15)
```
CSC225 A01  seats 11/100 (89 enrolled)  WL  0/25   Lecture  MTuWThF? MWTh 14:30-15:20  Quinton Yong
CSC225 A02  seats 14/20  ( 6 enrolled)  WL  0/25
CSC225 A03  seats 37/50  (13 enrolled)  WL  0/25
CSC225 B01  seats  7/30  (23 enrolled)  no WL
CSC225 B02  seats  1/30  (29 enrolled)  no WL
CSC225 B03  seats  0/30  (30 enrolled)  full, no WL
```

---

## Caching — in-process, ephemeral (NOT a Railway Volume)

The value of Banner is *freshness* (seat counts move minute-to-minute during
registration). Persistent, survives-restart cache is exactly wrong here — it'd
serve stale numbers after a redeploy. The Railway Volume is for the ~1.2 GB
static artifacts; cache does NOT belong there.

Use **in-process memory, TTL-keyed:**

1. **Session** (`JSESSIONID` + bound term): module-level `requests.Session`,
   reused across questions. Re-handshake only on expiry. This is the expensive
   part (the 3-request handshake) — cache it hardest.
2. **`searchResults` payloads**, keyed by `(term, subject, courseNumber)`,
   **short TTL 60–300s**. Long enough to absorb a follow-up burst and shield
   Banner from repeat hits; short enough that seat counts stay honest.
3. **Term list** (`getTerms`): cache hours.

Minimal shape (no lib needed; `cachetools.TTLCache` if you want eviction free):
```python
_cache: dict[tuple, tuple[float, list]] = {}   # key -> (fetched_at, sections)
TTL = 120
```

**Railway caveat:** if the backend runs multiple replicas/workers, each has its
own in-memory cache + session. That's fine — independent, just a few more
handshakes, nothing to coordinate. Cold start = empty cache; first availability
question eats the full handshake (~1–2s), same self-resolving cold-start story
as Chroma. Cross-replica cache would be a Redis add-on (not a Volume) — not
worth it for a 2-min TTL. Skip it.

---

## Integration plan (tool-use, gated retrieval)

Mirror the existing graph-retrieval pattern (fires only when the router names a
course). Do NOT use true LLM function-calling — the router already extracts
`course_codes`, so extend that path deterministically.

1. **Router intent flag.** In `rewrite_and_route`, add a boolean
   `wants_availability` (sibling to `wants_outline`) — set when the question is
   about seats / openings / waitlist / "is X full" / section times / when/where
   it meets / who teaches it. No extra LLM call.

2. **New module `backend/banner.py`** — self-contained, like `graph_queries.py`:
   session handling + TTL cache + term resolution +
   `search_sections(subject, number, term) -> list[dict]`. Returns
   already-rendered fact blocks (like `_graph_context_text`).

3. **Gated step in `chatbot.py`**, after graph retrieval:
   ```python
   if course_codes and wants_availability:
       banner_blocks = banner_retrieve(course_codes, term)
   ```

4. **Term resolution helper.** Pick current term from `getTerms` (nearest
   non–"View Only"). Cache hours. (Later: let router parse a term from the
   question, e.g. "next spring".)

5. **Context injection — same numbering scheme.** Render sections into numbered
   `[n]` blocks tagged `source=banner`, appended alongside graph + vector blocks
   in `_build_context`, through the same SYSTEM-message framing.

6. **Prompt rule** (one line in `SYSTEM_PROMPT`): Banner blocks are **live /
   current** enrollment (contrast `kuali` = catalog, `HISTORICAL` = past
   outlines). Cite section codes (A01) and note when a section is full or
   waitlist-only.

7. **Sources / UI.** Add `banner` to `format_sources()` and a badge enum entry
   (`banner → banner`), like existing `kuali` / `heat` handling.

### Feature-boundary decision (TODO — pick before building)
- **Availability-only** = 1 call (`searchResults`), simplest.
- **Availability + schedule + instructor** = 2 calls
  (`+ getFacultyMeetingTimes`), the complete "section info" answer.
  (Schedule alone is free — it's inline in `searchResults`. The 2nd call is
  only for the instructor name/email.)

---

## Cautions

- **No official API contract.** Undocumented internal endpoints; can change
  without notice. Cache aggressively, set a real User-Agent, don't hammer.
- **Rate limiting / blocks** possible under heavy repeat hits — the TTL cache
  is the main defense.
- **Term selection matters** — default to current/upcoming registerable term;
  wrong term = empty or stale-looking results.
- `searchResults` requires the `term/search` handshake first, or returns empty.
