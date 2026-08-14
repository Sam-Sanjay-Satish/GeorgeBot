"""Entity-routed hybrid vector retrieval — an additive alternative to
`GeorgeBot.vector_retrieve()`'s dense-on-reverse-HyDE-questions search.

Fuses (via Reciprocal Rank Fusion) up to three ranked chunk lists:
  A. dense-on-questions   — the EXISTING `vector_retrieve()` call, unchanged
  B. sparse-on-entity     — BM25 over chunk text, scoped to just a named
                             entity phrase (a facility/person/institution
                             name) when the router found one -- full-query
                             BM25 on generic/common-word queries reliably
                             degraded results in offline testing and is
                             deliberately excluded
  C. dense-on-chunks      — brute-force cosine search embedding the CHUNK
                             TEXT directly (not reverse-HyDE questions),
                             against a small (~12.5K vector) in-memory
                             numpy matrix -- no ANN/Chroma index needed at
                             this corpus size

Entirely gated behind `HYBRID_RETRIEVAL_ENABLED` (env var, default off).
When the flag is off, `maybe_load()` returns None with zero file I/O and
this module's numpy/rank_bm25 imports are the only cost paid (a few ms at
process start, not per-request) -- see chatbot.py's __init__ for how that's
wired: `self.hybrid = hybrid_retrieve.maybe_load()`.

Artifacts (see the Georgebot Testing/sparse-retrieval-test repo for how
these were built -- full corpus, not samples, already validated offline
against 37 real logged questions through the real answer pipeline):
    $HYBRID_DIR/chunk_embeddings_{undergrad,faculty}.npy   (float32, 1024-d)
    $HYBRID_DIR/chunk_ids_{undergrad,faculty}.json         (parallel array of chunk_id)
    $HYBRID_DIR/bm25_{undergrad,faculty}.pkl               (prebuilt BM25Okapi + chunk_ids)
    $HYBRID_DIR/chunk_metadata_{undergrad,faculty}.jsonl   (chunk_id -> text/title/origin/...)
"""
from __future__ import annotations

import json
import os
import pickle
import re
import string
import sys
from dataclasses import dataclass
from pathlib import Path

BASE_DIR = Path(__file__).parent
_DATA_DIR = Path(os.environ["DATA_DIR"]) if os.getenv("DATA_DIR") else BASE_DIR
HYBRID_DIR = Path(os.getenv("HYBRID_DIR", _DATA_DIR / "hybrid_retrieve"))

HYBRID_RETRIEVAL_ENABLED = os.getenv("HYBRID_RETRIEVAL_ENABLED", "0").lower() not in ("0", "false", "no")

AUDIENCES = ("undergrad", "faculty")

# Kept in sync with chatbot.py's constants of the same name (can't import
# them directly -- chatbot.py imports this module, not the other way
# around). TOP_K_FUSED matches chatbot.py's N_CONTEXT(4) + N_UNFILTERED_
# BACKFILL(3) exactly, so context length / answer-LLM token cost is
# unchanged by adding this path. MAX_CHUNK_DISTANCE matches chatbot.py's
# vector_retrieve() cutoff, applied here to the dense-on-chunks arm for
# the same reason it's applied there: don't force in a genuinely
# off-topic chunk just to hit a count.
TOP_K_FUSED = 7
MAX_CHUNK_DISTANCE = 0.75
RRF_K = 60

# BM25 tokenizer -- MUST match the one used to build the bm25_*.pkl files
# byte-for-byte, or query-time scores are meaningless against the indexed
# vocabulary. Course-code-aware: merges "CSC" + "225" into an extra
# "csc225" token alongside the two separate tokens.
_ALPHA_2_4 = re.compile(r"^[a-zA-Z]{2,4}$")
_DIGIT_3 = re.compile(r"^\d{3}[a-zA-Z]?$")
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def bm25_tokenize(text: str) -> list[str]:
    text = text.lower().translate(_PUNCT_TABLE)
    tokens = text.split()
    result: list[str] = []
    for i, tok in enumerate(tokens):
        result.append(tok)
        if i + 1 < len(tokens):
            nxt = tokens[i + 1]
            if _ALPHA_2_4.match(tok) and _DIGIT_3.match(nxt):
                result.append(tok + nxt)
    return result


def rrf(ranked_lists: list[list[str]], k: int = RRF_K) -> list[str]:
    """Reciprocal rank fusion. Returns chunk_ids sorted best-first."""
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, cid in enumerate(ranked):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=scores.__getitem__, reverse=True)


@dataclass(frozen=True)
class HybridIndex:
    chunk_vectors: dict[str, "object"]       # audience -> normalized np.ndarray (n, 1024)
    chunk_ids: dict[str, list[str]]          # audience -> list[str], parallel to chunk_vectors rows
    bm25: dict[str, tuple["object", list[str]]]  # audience -> (BM25Okapi, chunk_ids)
    metadata: dict[str, dict]                # chunk_id -> {text, title, origin, source, document_type, department, topic_families}


def _artifact_paths() -> dict[str, Path]:
    paths = {}
    for aud in AUDIENCES:
        paths[f"chunk_embeddings_{aud}"] = HYBRID_DIR / f"chunk_embeddings_{aud}.npy"
        paths[f"chunk_ids_{aud}"] = HYBRID_DIR / f"chunk_ids_{aud}.json"
        paths[f"bm25_{aud}"] = HYBRID_DIR / f"bm25_{aud}.pkl"
        paths[f"chunk_metadata_{aud}"] = HYBRID_DIR / f"chunk_metadata_{aud}.jsonl"
    return paths


def maybe_load() -> HybridIndex | None:
    """Returns None (zero file I/O) if HYBRID_RETRIEVAL_ENABLED is false.
    Otherwise loads and validates every artifact, failing loudly with a
    specific missing-file message if any are absent -- a production boot
    that silently degrades to legacy retrieval on a missing artifact is a
    worse surprise than one that refuses to start."""
    if not HYBRID_RETRIEVAL_ENABLED:
        return None

    import numpy as np

    paths = _artifact_paths()
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        print("ERROR: HYBRID_RETRIEVAL_ENABLED=1 but artifact(s) missing:", file=sys.stderr)
        for m in missing:
            print(f"  - {m}", file=sys.stderr)
        sys.exit(1)

    chunk_vectors, chunk_ids, bm25, metadata = {}, {}, {}, {}
    for aud in AUDIENCES:
        arr = np.load(paths[f"chunk_embeddings_{aud}"])
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        chunk_vectors[aud] = arr / norms
        with open(paths[f"chunk_ids_{aud}"]) as f:
            chunk_ids[aud] = json.load(f)
        with open(paths[f"bm25_{aud}"], "rb") as f:
            d = pickle.load(f)
            bm25[aud] = (d["bm25"], d["chunk_ids"])
        with open(paths[f"chunk_metadata_{aud}"]) as f:
            for line in f:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    metadata[rec["chunk_id"]] = rec

    total_vecs = sum(len(v) for v in chunk_ids.values())
    print(f"  Hybrid retrieval index loaded: {total_vecs:,} chunk vectors, "
          f"{len(metadata):,} metadata records")
    return HybridIndex(chunk_vectors=chunk_vectors, chunk_ids=chunk_ids, bm25=bm25, metadata=metadata)


def _audiences(audience: str) -> tuple[str, ...]:
    return AUDIENCES if audience == "both" else (audience,)


def dense_chunk_search(index: HybridIndex, qvec, audience: str, k: int) -> list[str]:
    """Arm C: brute-force cosine search over chunk-text embeddings."""
    import numpy as np

    q = qvec / np.linalg.norm(qvec)
    cands: list[tuple[str, float]] = []
    for aud in _audiences(audience):
        matrix = index.chunk_vectors[aud]
        ids = index.chunk_ids[aud]
        sims = matrix @ q
        top = np.argsort(-sims)[:k]
        for i in top:
            dist = 1.0 - float(sims[i])
            if dist <= MAX_CHUNK_DISTANCE:
                cands.append((ids[i], dist))
    cands.sort(key=lambda x: x[1])
    return [cid for cid, _ in cands[:k]]


def sparse_entity_search(index: HybridIndex, entity_query: str, audience: str, k: int) -> list[str]:
    """Arm B: BM25 scoped to just the entity phrase -- never the full
    query, which is what let common words drown out real matches in
    offline testing."""
    cands: list[tuple[str, float]] = []
    for aud in _audiences(audience):
        bm25, ids = index.bm25[aud]
        scores = bm25.get_scores(bm25_tokenize(entity_query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        for i in ranked[:k]:
            if scores[i] <= 0:
                break
            cands.append((ids[i], float(scores[i])))
    cands.sort(key=lambda x: x[1], reverse=True)
    return [cid for cid, _ in cands[:k]]


def _resolve_chunk(cid: str, a_chunks_by_id: dict[str, dict], metadata: dict[str, dict]) -> dict | None:
    """Reconstruct the {chunk_id, text, metadata} shape `_build_context`/
    `format_sources` already expect, reusing an arm-A chunk dict directly
    when available (byte-identical to what vector_retrieve() would have
    produced) and falling back to the hybrid metadata lookup otherwise."""
    if cid in a_chunks_by_id:
        return a_chunks_by_id[cid]
    rec = metadata.get(cid)
    if not rec:
        return None
    return {
        "chunk_id": cid,
        "text": rec.get("text", ""),
        "metadata": {
            "chunk_id": cid,
            "doc_id": rec.get("doc_id", ""),
            "origin": rec.get("origin", ""),
            "title": rec.get("title", ""),
            "source": rec.get("source", ""),
            "document_type": rec.get("document_type", ""),
            "department": rec.get("department", ""),
            "topic_families": " | ".join(rec.get("topic_families") or []),
        },
    }


def retrieve(bot, route: dict, audience: str, index: HybridIndex) -> list[dict]:
    """Entity-routed hybrid retrieval. `bot` is the GeorgeBot instance (used
    for `.vector_retrieve()` and `._embed_query()`, both unchanged).

    Sequential by design, not threaded: arm A (Chroma, a network call)
    dominates wall-clock; arms B and C are in-process CPU work over a
    ~12.5K-row corpus (sub-10ms each). The whole request already runs
    inside api.py's bounded chat ThreadPoolExecutor -- nesting a second
    thread pool here to "parallelize" work that's this cheap risks pool
    starvation for no measurable benefit.
    """
    search_query = route["search_query"]
    entities = route.get("named_entities") or []  # .get(), never route[...] -- see rewrite_and_route's
                                                    # except-branch fallback dict, which omits this key

    # Arm A: the exact existing production call, unchanged. Guarantees this
    # arm is byte-identical to today's vector_retrieve() output -- see the
    # design note in the implementation plan for why this isn't
    # reimplemented against _query_candidates/_merge_collapse directly.
    a_chunks = bot.vector_retrieve(
        search_query, audience=audience,
        topic_families=route["topic_families"], department=route["department"],
    )
    a_chunks_by_id = {c["chunk_id"]: c for c in a_chunks}
    a_ranked = list(a_chunks_by_id.keys())

    qvec = bot._embed_query(search_query)
    c_ranked = dense_chunk_search(index, qvec, audience, TOP_K_FUSED)

    lists = [a_ranked, c_ranked]
    if entities:
        b_ranked = sparse_entity_search(index, " ".join(entities), audience, TOP_K_FUSED)
        lists.append(b_ranked)

    fused_ids = rrf(lists)[:TOP_K_FUSED]
    out = []
    for cid in fused_ids:
        ch = _resolve_chunk(cid, a_chunks_by_id, index.metadata)
        if ch:
            out.append(ch)
    return out
