"""Unit tests for the retrieval module.

Tests fusion logic (pure math, no API calls) and integration test with
a real ingested corpus (requires OPENAI_API_KEY).
"""

import json
import os
import sqlite3
from pathlib import Path

import numpy as np
import pytest

from yoke.retrieval.fusion import rrf_merge
from yoke.retrieval.models import RetrievalResult


# =========================================================================
# Unit tests for RRF fusion — no external dependencies
# =========================================================================


class TestRRFMerge:
    """Test the Reciprocal Rank Fusion merge logic."""

    def test_single_list_dense_only(self) -> None:
        """When only dense results exist, RRF scores from dense alone."""
        dense = [(1, 0.9), (2, 0.8), (3, 0.7)]
        sparse: list[tuple[int, float]] = []
        merged = rrf_merge(dense, sparse, k_rrf=60)

        assert len(merged) == 3
        # Order should follow dense ranking
        assert [cid for cid, _, _, _ in merged] == [1, 2, 3]
        # Dense ranks should be 1, 2, 3
        assert merged[0] == (1, 1 / (60 + 1), 1, None)
        assert merged[1] == (2, 1 / (60 + 2), 2, None)
        assert merged[2] == (3, 1 / (60 + 3), 3, None)

    def test_single_list_sparse_only(self) -> None:
        """When only sparse results exist, RRF scores from sparse alone."""
        dense: list[tuple[int, float]] = []
        sparse = [(10, 5.0), (20, 3.0)]
        merged = rrf_merge(dense, sparse, k_rrf=60)

        assert len(merged) == 2
        assert merged[0] == (10, 1 / (60 + 1), None, 1)
        assert merged[1] == (20, 1 / (60 + 2), None, 2)

    def test_overlapping_results(self) -> None:
        """Chunks in both lists get contributions from both."""
        dense = [(1, 0.95), (2, 0.90)]
        sparse = [(2, 8.0), (3, 5.0)]
        merged = rrf_merge(dense, sparse, k_rrf=60)

        # Chunk 2 appears in both — should have the highest RRF
        chunk_2 = next(m for m in merged if m[0] == 2)
        expected_score_2 = 1 / (60 + 2) + 1 / (60 + 1)  # dense rank 2, sparse rank 1
        assert abs(chunk_2[1] - expected_score_2) < 1e-10
        assert chunk_2[2] == 2  # dense_rank
        assert chunk_2[3] == 1  # sparse_rank

        # Chunk 1 only in dense
        chunk_1 = next(m for m in merged if m[0] == 1)
        expected_score_1 = 1 / (60 + 1)
        assert abs(chunk_1[1] - expected_score_1) < 1e-10
        assert chunk_1[3] is None  # no sparse rank

        # Chunk 2 should be ranked first (highest RRF)
        assert merged[0][0] == 2

    def test_empty_inputs(self) -> None:
        """Empty inputs produce empty output."""
        assert rrf_merge([], []) == []

    def test_no_overlap(self) -> None:
        """Disjoint lists produce all items, each with one contribution."""
        dense = [(1, 0.9)]
        sparse = [(2, 5.0)]
        merged = rrf_merge(dense, sparse, k_rrf=60)

        assert len(merged) == 2
        # Same rank (1) in their respective lists — same RRF score
        assert abs(merged[0][1] - merged[1][1]) < 1e-10

    def test_k_rrf_parameter(self) -> None:
        """Different k_rrf values produce different scores."""
        dense = [(1, 0.9)]
        sparse: list[tuple[int, float]] = []

        merged_60 = rrf_merge(dense, sparse, k_rrf=60)
        merged_10 = rrf_merge(dense, sparse, k_rrf=10)

        # Smaller k_rrf gives higher score (1/(10+1) > 1/(60+1))
        assert merged_10[0][1] > merged_60[0][1]

    def test_order_is_score_descending(self) -> None:
        """Output is sorted by RRF score descending."""
        dense = [(1, 0.9), (2, 0.8), (3, 0.7), (4, 0.6), (5, 0.5)]
        sparse = [(5, 10.0), (4, 8.0), (3, 6.0), (2, 4.0), (1, 2.0)]
        merged = rrf_merge(dense, sparse, k_rrf=60)

        scores = [score for _, score, _, _ in merged]
        assert scores == sorted(scores, reverse=True)


# =========================================================================
# Integration test — requires OPENAI_API_KEY
# =========================================================================


def _openai_key_available() -> bool:
    return bool(os.environ.get("OPENAI_API_KEY"))


@pytest.fixture
def ingested_db(tmp_path: Path) -> tuple[Path, Path]:
    """Create a small test DB with 5 chunks and a BM25 index."""
    from yoke.ingestion.models import EnrichedChunk
    from yoke.ingestion.embedding import embed_texts
    from yoke.ingestion.store import build_bm25_index, init_db, store_document

    chunks_data = [
        "A topological manifold is a Hausdorff, second-countable, locally Euclidean space.",
        "A smooth atlas is a collection of smoothly compatible charts covering M.",
        "Stereographic projection from north and south poles covers the sphere S^n.",
        "The fundamental group of a topological manifold is countable.",
        "Every topological manifold is paracompact and locally compact.",
    ]

    enriched = [
        EnrichedChunk(
            chunk_index=i,
            chunk_text=text,
            context_summary=f"Test chunk {i}.",
            enriched_text=f"[Context: Test chunk {i}.]\n\n{text}",
            page_numbers=[i + 1],
            source_file="test.txt",
        )
        for i, text in enumerate(chunks_data)
    ]

    embeddings = embed_texts([ec.enriched_text for ec in enriched])

    db_path = tmp_path / "test.db"
    conn = init_db(db_path)
    store_document(conn, "test.txt", "\n".join(chunks_data), enriched, embeddings)
    bm25_path = db_path.with_suffix(".bm25.json")
    build_bm25_index(conn, bm25_path)
    conn.close()

    return db_path, bm25_path


@pytest.mark.skipif(not _openai_key_available(), reason="OPENAI_API_KEY not set")
class TestHybridRetrieval:
    """End-to-end integration test with real embeddings."""

    def test_retrieve_returns_results(
        self, ingested_db: tuple[Path, Path]
    ) -> None:
        """retrieve() returns a list of RetrievalResult objects."""
        from yoke.retrieval import retrieve

        db_path, bm25_path = ingested_db
        results = retrieve("topological manifold", db_path, bm25_path, k=3)

        assert len(results) == 3
        assert all(isinstance(r, RetrievalResult) for r in results)
        # Scores should be descending
        scores = [r.rrf_score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_retrieve_finds_relevant_chunk(
        self, ingested_db: tuple[Path, Path]
    ) -> None:
        """A query about Hausdorff should surface the manifold definition chunk."""
        from yoke.retrieval import retrieve

        db_path, bm25_path = ingested_db
        results = retrieve("Hausdorff space definition", db_path, bm25_path, k=3)

        top_texts = [r.chunk_text.lower() for r in results]
        assert any("hausdorff" in t for t in top_texts)

    def test_retrieve_with_timings(
        self, ingested_db: tuple[Path, Path]
    ) -> None:
        """retrieve_with_timings returns results + timing breakdown."""
        from yoke.retrieval import retrieve_with_timings, RetrievalTimings

        db_path, bm25_path = ingested_db
        results, timings = retrieve_with_timings(
            "smooth atlas", db_path, bm25_path, k=2
        )

        assert len(results) == 2
        assert isinstance(timings, RetrievalTimings)
        assert timings.total_ms > 0
        assert timings.embedding_ms >= 0
        assert timings.dense_ms >= 0
        assert timings.sparse_ms >= 0

    def test_each_result_has_metadata(
        self, ingested_db: tuple[Path, Path]
    ) -> None:
        """Every result should have all metadata fields populated."""
        from yoke.retrieval import retrieve

        db_path, bm25_path = ingested_db
        results = retrieve("paracompact", db_path, bm25_path, k=5)

        for r in results:
            assert r.chunk_id > 0
            assert len(r.chunk_text) > 0
            assert len(r.context_summary) > 0
            assert r.source_file == "test.txt"
            assert len(r.page_numbers) > 0
            assert r.rrf_score > 0
            # At least one of dense/sparse rank should be set
            assert r.dense_rank is not None or r.sparse_rank is not None
