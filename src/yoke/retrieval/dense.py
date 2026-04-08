"""Dense (embedding) retrieval via cosine similarity."""

import sqlite3
from pathlib import Path

import numpy as np


def dense_search(
    query_embedding: list[float],
    db_path: Path,
    top_n: int = 50,
) -> list[tuple[int, float]]:
    """Find the top-N chunks by cosine similarity to the query embedding.

    Args:
        query_embedding: The query vector (same dimensionality as stored embeddings).
        db_path: Path to the SQLite database.
        top_n: Number of results to return.

    Returns:
        List of (chunk_id, cosine_similarity) sorted by similarity descending.
    """
    conn = sqlite3.connect(str(db_path))
    try:
        rows = conn.execute(
            "SELECT id, embedding FROM chunks ORDER BY doc_id, chunk_index"
        ).fetchall()
    finally:
        conn.close()

    if not rows:
        return []

    chunk_ids = [row[0] for row in rows]
    # Each embedding is stored as float32 bytes
    embeddings = np.array(
        [np.frombuffer(row[1], dtype=np.float32) for row in rows]
    )

    q_vec = np.array(query_embedding, dtype=np.float32)

    # L2-normalize for cosine similarity via dot product
    emb_norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    emb_norms[emb_norms == 0] = 1.0
    embeddings_normed = embeddings / emb_norms

    q_norm = np.linalg.norm(q_vec)
    if q_norm > 0:
        q_vec = q_vec / q_norm

    similarities = embeddings_normed @ q_vec
    top_indices = np.argsort(similarities)[::-1][:top_n]

    return [(chunk_ids[i], float(similarities[i])) for i in top_indices]
