# RateMyProfessors (RMP) — endpoint research

Notes backing `backend/rmp.py`. RMP has **no official/public API**; this is the
same internal GraphQL endpoint every open-source wrapper uses. It's undocumented
and can change without notice — hence `rmp.py` is strictly **best-effort**
(returns `{}` on any failure, so the answer degrades to no-RMP).

## Endpoint & auth

- **URL:** `POST https://www.ratemyprofessors.com/graphql`
- **Auth header:** `Authorization: Basic dGVzdDp0ZXN0` — base64 of `test:test`.
  A public constant hardcoded in every wrapper; not a secret, not per-user. RMP's
  read-only GraphQL accepts it unauthenticated.
- Send a browser-ish `User-Agent` and `Content-Type: application/json`.

## School id (UVic)

The teacher search requires the school as a **base64 relay id**, NOT the raw
number. UVic is school `1488` (`ratemyprofessors.com/school/1488`);
base64(`"School-1488"`) = **`U2Nob29sLTE0ODg=`** → `RMP_SCHOOL_ID`. The raw
`"1488"` is rejected. (Teacher relay ids are likewise base64, e.g.
`base64("Teacher-2793234")` — the search returns them ready-to-use.)

## Queries

**1. Search teachers by name at a school** — the node is a full `Teacher`, so the
summary fields come back in this one call (no separate getTeacher needed):

```graphql
query($text:String!,$schoolID:ID!){
  newSearch{ teachers(query:{text:$text, schoolID:$schoolID}){
    edges{ node{
      id firstName lastName avgRating avgDifficulty numRatings
      wouldTakeAgainPercent department legacyId
    } }
  } }
}
```
- `avgRating` / `avgDifficulty`: 0–5 (0 when `numRatings == 0`).
- `wouldTakeAgainPercent`: float %, or `-1` when unknown.
- `legacyId`: builds the public URL
  `https://www.ratemyprofessors.com/professor/{legacyId}` (used as the source).
- Search is fuzzy; a common surname returns several teachers → disambiguate.

**2. Recent reviews for a matched teacher** (by relay `id` from query 1):

```graphql
query($id:ID!,$n:Int!){
  node(id:$id){ ... on Teacher{
    ratings(first:$n){ edges{ node{
      comment class date qualityRating difficultyRatingRounded
      wouldTakeAgain ratingTags thumbsUpTotal
    } } }
  } }
}
```
- `class`: the course the review is for (e.g. `CSC110`).
- `ratingTags`: a single string, tags joined by `--` (split + strip).
- `wouldTakeAgain`: `1` / `0` / `-1` (unknown) per review.

## Caching

In-process, ephemeral, `_lock`-guarded dict keyed by the queried name,
`RMP_TTL = 6h` (ratings barely move). Unlike Banner there is **no per-session
search-form state to leak**, so a single shared `requests.Session` is fine — no
dedicated-session-per-call dance.

## Cautions

- Unofficial: schema/token can break — keep it best-effort.
- Present results as **subjective student opinion**, never official data — the
  answer-model system prompt enforces this (`source=rmp` bullet in `chatbot.py`).
- Be a polite client (real UA, low volume, cache) on an undocumented endpoint.

## Smoke test

```bash
python3 backend/rmp.py --name "Yong"
python3 backend/rmp.py --name "Quinton Yong" --name "Zhang"   # match + ambiguous
```
