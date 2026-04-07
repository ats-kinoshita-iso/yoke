"""SQLite storage and BM25 index building."""

import json
import sqlite3
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from yoke.ingestion.models import EnrichedChunk

_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filename TEXT NOT NULL UNIQUE,
    full_text TEXT NOT NULL,
    ingested_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id INTEGER NOT NULL REFERENCES documents(id),
    chunk_index INTEGER NOT NULL,
    chunk_text TEXT NOT NULL,
    context_summary TEXT NOT NULL,
    enriched_text TEXT NOT NULL,
    embedding BLOB NOT NULL,
    page_numbers TEXT NOT NULL,
    UNIQUE(doc_id, chunk_index)
);
"""


def init_db(db_path: Path) -> sqlite3.Connection:
    """Create the database and tables if they don't exist."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.executescript(_SCHEMA)
    return conn


def store_document(
    conn: sqlite3.Connection,
    filename: str,
    full_text: str,
    chunks: list[EnrichedChunk],
    embeddings: list[list[float]],
) -> int:
    """Store a document and its chunks. Upserts by filename.

    Returns the document ID.
    """
    cursor = conn.cursor()
    cursor.execute("BEGIN")
    try:
        # Upsert document
        cursor.execute(
            "SELECT id FROM documents WHERE filename = ?", (filename,)
        )
        row = cursor.fetchone()
        if row:
            doc_id = row[0]
            cursor.execute(
                "UPDATE documents SET full_text = ?, ingested_at = datetime('now') "
                "WHERE id = ?",
                (full_text, doc_id),
            )
            cursor.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
        else:
            cursor.execute(
                "INSERT INTO documents (filename, full_text) VALUES (?, ?)",
                (filename, full_text),
            )
            doc_id = cursor.lastrowid

        # Insert chunks
        for chunk, embedding in zip(chunks, embeddings):
            emb_blob = np.array(embedding, dtype=np.float32).tobytes()
            cursor.execute(
                "INSERT INTO chunks "
                "(doc_id, chunk_index, chunk_text, context_summary, "
                "enriched_text, embedding, page_numbers) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    doc_id,
                    chunk.chunk_index,
                    chunk.chunk_text,
                    chunk.context_summary,
                    chunk.enriched_text,
                    emb_blob,
                    json.dumps(chunk.page_numbers),
                ),
            )

        conn.commit()
    except Exception:
        conn.rollback()
        raise
    return doc_id


def build_bm25_index(conn: sqlite3.Connection, bm25_path: Path) -> BM25Okapi:
    """Build a BM25 index over all raw chunk texts in the database.

    Stores the tokenized corpus as JSON (not pickle) to avoid arbitrary
    code execution on deserialization. The BM25Okapi index is reconstructed
    from the token lists at load time.
    """
    cursor = conn.execute(
        "SELECT chunk_text FROM chunks ORDER BY doc_id, chunk_index"
    )
    texts = [row[0] for row in cursor.fetchall()]

    if not texts:
        raise ValueError("No chunks in database to build BM25 index")

    tokenized = [text.lower().split() for text in texts]
    index = BM25Okapi(tokenized)

    # Save tokenized corpus as JSON instead of pickling the index
    bm25_path = bm25_path.with_suffix(".json")
    bm25_path.parent.mkdir(parents=True, exist_ok=True)
    bm25_path.write_text(json.dumps(tokenized), encoding="utf-8")

    return index


def load_bm25_index(bm25_path: Path) -> BM25Okapi:
    """Load a BM25 index from a JSON tokenized corpus file."""
    tokenized = json.loads(bm25_path.read_text(encoding="utf-8"))
    return BM25Okapi(tokenized)
