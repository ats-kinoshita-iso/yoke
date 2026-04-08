"""Sparse (BM25) retrieval."""

import sqlite3
from pathlib import Path

import numpy as np

from yoke.ingestion.store import load_bm25_index


def sparse_search(
    query: str,
    bm25_path: Path,
    db_path: Path,
    top_n: int = 50,
) -> list[tuple[int, float]]:
    """Find the top-N chunks by BM25 score.

    Args:
        query: Raw query string.
        bm25_path: Path to the .bm25.json index file.
        db_path: Path to the SQLite database (for chunk ID mapping).
        top_n: Number of results to return.

    Returns:
        List of (chunk_id, bm25_score) sorted by score descending.
    """
    index = load_bm25_index(bm25_path)

    # Tokenize with the same method used during index construction
    # (see store.py:build_bm25_index — text.lower().split())
    tokenized_query = query.lower().split()
    if not tokenized_query:
        return []

    scores = index.get_scores(tokenized_query)

    # Map BM25 positions back to chunk IDs using the same ordering
    # invariant as build_bm25_index: ORDER BY doc_id, chunk_index
    conn = sqlite3.connect(str(db_path))
    try:
        chunk_ids = [
            row[0]
            for row in conn.execute(
                "SELECT id FROM chunks ORDER BY doc_id, chunk_index"
            ).fetchall()
        ]
    finally:
        conn.close()

    if len(chunk_ids) != len(scores):
        raise ValueError(
            f"BM25 index has {len(scores)} entries but DB has {len(chunk_ids)} chunks. "
            "Index may be out of sync — re-run ingestion."
        )

    top_indices = np.argsort(scores)[::-1][:top_n]
    # Filter out zero-score entries (no query term overlap)
    return [
        (chunk_ids[i], float(scores[i]))
        for i in top_indices
        if scores[i] > 0.0
    ]
