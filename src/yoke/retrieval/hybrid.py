"""Hybrid retrieval: dense + sparse search merged with RRF."""

import json
import logging
import sqlite3
import time
from pathlib import Path

from pydantic import BaseModel

from yoke.ingestion.embedding import embed_texts
from yoke.retrieval.dense import dense_search
from yoke.retrieval.fusion import rrf_merge
from yoke.retrieval.models import RetrievalResult
from yoke.retrieval.sparse import sparse_search

logger = logging.getLogger(__name__)


class RetrievalTimings(BaseModel):
    """Latency breakdown for a retrieval call."""

    embedding_ms: float
    dense_ms: float
    sparse_ms: float
    rrf_ms: float
    total_ms: float


def retrieve(
    query: str,
    db_path: Path,
    bm25_path: Path,
    k: int = 10,
    top_n: int = 50,
) -> list[RetrievalResult]:
    """Run hybrid retrieval: embed, dense search, sparse search, RRF merge.

    Args:
        query: Natural language query.
        db_path: Path to the SQLite database.
        bm25_path: Path to the .bm25.json index file.
        k: Number of final results to return.
        top_n: Number of candidates from each retriever before fusion.

    Returns:
        Top-k RetrievalResult objects sorted by RRF score descending.
    """
    results, timings = retrieve_with_timings(query, db_path, bm25_path, k=k, top_n=top_n)

    logger.info(
        "Retrieval: %.0fms total (embed=%.0fms dense=%.0fms sparse=%.0fms rrf=%.0fms) "
        "query=%r k=%d",
        timings.total_ms,
        timings.embedding_ms,
        timings.dense_ms,
        timings.sparse_ms,
        timings.rrf_ms,
        query[:60],
        k,
    )

    return results


def retrieve_with_timings(
    query: str,
    db_path: Path,
    bm25_path: Path,
    k: int = 10,
    top_n: int = 50,
) -> tuple[list[RetrievalResult], RetrievalTimings]:
    """Like retrieve(), but also returns the latency breakdown."""
    t_start = time.perf_counter()

    query_embedding = embed_texts([query])[0]
    t_embed = time.perf_counter()

    dense_results = dense_search(query_embedding, db_path, top_n=top_n)
    t_dense = time.perf_counter()

    sparse_results = sparse_search(query, bm25_path, db_path, top_n=top_n)
    t_sparse = time.perf_counter()

    merged = rrf_merge(dense_results, sparse_results, k_rrf=60)
    t_rrf = time.perf_counter()

    top_k_ids = [cid for cid, _, _, _ in merged[:k]]
    results = _fetch_chunk_metadata(db_path, top_k_ids, merged[:k])

    t_end = time.perf_counter()

    timings = RetrievalTimings(
        embedding_ms=(t_embed - t_start) * 1000,
        dense_ms=(t_dense - t_embed) * 1000,
        sparse_ms=(t_sparse - t_dense) * 1000,
        rrf_ms=(t_rrf - t_sparse) * 1000,
        total_ms=(t_end - t_start) * 1000,
    )

    return results, timings


def _fetch_chunk_metadata(
    db_path: Path,
    chunk_ids: list[int],
    merged: list[tuple[int, float, int | None, int | None]],
) -> list[RetrievalResult]:
    """Fetch chunk text and metadata from SQLite for the given IDs."""
    if not chunk_ids:
        return []

    # Build lookup from merged results
    merge_map = {cid: (score, d_rank, s_rank) for cid, score, d_rank, s_rank in merged}

    conn = sqlite3.connect(str(db_path))
    try:
        placeholders = ",".join("?" for _ in chunk_ids)
        rows = conn.execute(
            f"SELECT c.id, c.chunk_text, c.context_summary, c.page_numbers, d.filename "
            f"FROM chunks c JOIN documents d ON c.doc_id = d.id "
            f"WHERE c.id IN ({placeholders})",
            chunk_ids,
        ).fetchall()
    finally:
        conn.close()

    # Build lookup by chunk_id
    row_map = {row[0]: row for row in rows}

    # Assemble results in merged order
    results: list[RetrievalResult] = []
    for cid in chunk_ids:
        row = row_map.get(cid)
        if row is None:
            continue
        score, d_rank, s_rank = merge_map[cid]
        results.append(
            RetrievalResult(
                chunk_id=cid,
                chunk_text=row[1],
                context_summary=row[2],
                source_file=row[4],
                page_numbers=json.loads(row[3]),
                rrf_score=score,
                dense_rank=d_rank,
                sparse_rank=s_rank,
            )
        )

    return results
