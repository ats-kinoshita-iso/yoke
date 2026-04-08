"""Phase 1 ingestion pipeline evals — chunking, enrichment, and embedding quality.

These evals define success criteria for the document ingestion pipeline.
They test against the Lee Ch.1 fixture (tests/fixtures/docs-math/lee_ch1.txt).

Run: uv run pytest evals/phase1_ingestion_eval.py -v
"""

import io
import json
import re
import sys
from pathlib import Path

import numpy as np
import pytest
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from evals._judge import judge as judge_dispatch

# Fix Windows console encoding for Unicode math symbols
if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer, encoding="utf-8", errors="replace"
    )

from yoke.ingestion.chunking import chunk_text
from yoke.ingestion.enrichment import enrich_chunk
from yoke.ingestion.embedding import embed_texts

load_dotenv(override=True)

FIXTURE_PATH = (
    Path(__file__).resolve().parent.parent
    / "tests" / "fixtures" / "docs-math" / "lee_ch1.txt"
)
RESULTS_DIR = Path(__file__).resolve().parent / "results"


def _load_fixture() -> str:
    return FIXTURE_PATH.read_text(encoding="utf-8")


def _sentence_split(text: str) -> list[str]:
    """Split text into sentences for coverage checking.

    Uses a simple regex: split on period/question-mark/exclamation followed
    by whitespace or end-of-string. Filters out very short fragments (< 20
    chars) that are likely headings or artifacts.
    """
    raw = re.split(r"(?<=[.?!])\s+", text)
    return [s.strip() for s in raw if len(s.strip()) >= 20]


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 characters per token for English prose."""
    return len(text) // 4


# ===========================================================================
# Eval 1: Chunk coverage — every sentence appears in at least one chunk
# ===========================================================================


class TestChunkCoverage:
    """Verify that chunking preserves all content from the original document."""

    def test_every_sentence_in_at_least_one_chunk(self) -> None:
        """Every sentence (>=20 chars) in the original must appear in some chunk."""
        full_text = _load_fixture()
        chunks = chunk_text(full_text, source_file="lee_ch1.txt")

        sentences = _sentence_split(full_text)
        assert len(sentences) > 50, (
            f"Sentence split produced only {len(sentences)} sentences — "
            "check the fixture or splitting logic"
        )

        # Build a single string from all chunk texts for substring search
        all_chunk_text = "\n".join(c.text for c in chunks)

        missing: list[str] = []
        for sentence in sentences:
            # Normalize whitespace for matching (chunks may re-wrap lines)
            normalized = " ".join(sentence.split())
            # Check if the normalized sentence appears in any chunk
            # (also normalize chunk text for comparison)
            normalized_chunks = " ".join(all_chunk_text.split())
            if normalized not in normalized_chunks:
                missing.append(sentence[:80])

        coverage = 1.0 - len(missing) / len(sentences)
        print(f"\n  Chunk coverage: {coverage:.1%} ({len(missing)} missing of {len(sentences)})")
        if missing:
            print(f"  First 5 missing sentences:")
            for s in missing[:5]:
                print(f"    - \"{s}...\"")

        assert coverage >= 0.99, (
            f"Chunk coverage {coverage:.1%} below 99% threshold. "
            f"{len(missing)} sentences missing."
        )


# ===========================================================================
# Eval 2: Chunk size — all chunks within tolerance of 512-token target
# ===========================================================================


class TestChunkSize:
    """Verify chunks are within the target token range."""

    def test_chunk_sizes_within_tolerance(self) -> None:
        """At least 90% of chunks should be 400-600 tokens (~1600-2400 chars)."""
        full_text = _load_fixture()
        chunks = chunk_text(full_text, source_file="lee_ch1.txt")

        assert len(chunks) > 0, "No chunks produced"

        sizes = [_estimate_tokens(c.text) for c in chunks]
        in_range = sum(1 for s in sizes if 400 <= s <= 600)
        pct_in_range = in_range / len(sizes)

        print(f"\n  Total chunks: {len(chunks)}")
        print(f"  Token sizes — min: {min(sizes)}, max: {max(sizes)}, "
              f"mean: {sum(sizes)/len(sizes):.0f}")
        print(f"  In range (400-600 tokens): {in_range}/{len(sizes)} ({pct_in_range:.0%})")

        # The last chunk is allowed to be short
        sizes_except_last = sizes[:-1] if len(sizes) > 1 else sizes
        in_range_except_last = sum(1 for s in sizes_except_last if 400 <= s <= 600)
        pct_except_last = in_range_except_last / len(sizes_except_last) if sizes_except_last else 0

        assert pct_except_last >= 0.90, (
            f"Only {pct_except_last:.0%} of chunks (excluding last) are in the "
            f"400-600 token range. Need >= 90%."
        )

    def test_no_empty_chunks(self) -> None:
        """No chunk should be empty or near-empty."""
        full_text = _load_fixture()
        chunks = chunk_text(full_text, source_file="lee_ch1.txt")

        for i, c in enumerate(chunks):
            assert len(c.text.strip()) >= 50, (
                f"Chunk {i} is too short ({len(c.text.strip())} chars): "
                f"\"{c.text[:50]}...\""
            )


# ===========================================================================
# Eval 3: Overlap — consecutive chunks share ~10% content
# ===========================================================================


class TestChunkOverlap:
    """Verify consecutive chunks share approximately 10% overlap."""

    def test_consecutive_overlap(self) -> None:
        """Adjacent chunks should share overlapping text at their boundaries."""
        full_text = _load_fixture()
        chunks = chunk_text(full_text, source_file="lee_ch1.txt")

        assert len(chunks) >= 3, f"Need >= 3 chunks to test overlap, got {len(chunks)}"

        overlaps: list[float] = []
        for i in range(len(chunks) - 1):
            current = chunks[i].text
            next_chunk = chunks[i + 1].text

            # Find the longest suffix of current that is a prefix of next
            overlap_len = 0
            # Check: does the end of current appear at the start of next?
            # Try progressively shorter suffixes of current
            max_check = min(len(current), len(next_chunk), len(current) // 2)
            for length in range(max_check, 0, -1):
                suffix = current[-length:]
                if next_chunk.startswith(suffix):
                    overlap_len = length
                    break

            overlap_pct = overlap_len / len(current) if len(current) > 0 else 0
            overlaps.append(overlap_pct)

        avg_overlap = sum(overlaps) / len(overlaps)
        pairs_with_overlap = sum(1 for o in overlaps if o > 0.02)

        print(f"\n  Chunk pairs: {len(overlaps)}")
        print(f"  Average overlap: {avg_overlap:.1%}")
        print(f"  Pairs with >2% overlap: {pairs_with_overlap}/{len(overlaps)}")
        print(f"  Overlap range: {min(overlaps):.1%} - {max(overlaps):.1%}")

        # At least 80% of pairs should have meaningful overlap
        assert pairs_with_overlap / len(overlaps) >= 0.80, (
            f"Only {pairs_with_overlap}/{len(overlaps)} chunk pairs have >2% overlap. "
            f"Expected >= 80%."
        )

        # Average overlap should be roughly 5-15% (target is 10%)
        assert 0.05 <= avg_overlap <= 0.20, (
            f"Average overlap {avg_overlap:.1%} outside expected range 5%-20%."
        )


# ===========================================================================
# Eval 4: Context summary quality — LLM-as-judge
# ===========================================================================


class ContextSummaryScore(BaseModel):
    accuracy: int = Field(ge=1, le=5)
    situating_value: int = Field(ge=1, le=5)
    conciseness: int = Field(ge=1, le=5)
    reasoning: str


SUMMARY_JUDGE_TOOL = {
    "name": "score_summary",
    "description": "Score the contextual summary for a document chunk.",
    "input_schema": {
        "type": "object",
        "properties": {
            "accuracy": {
                "type": "integer",
                "description": (
                    "1-5. Does the summary correctly describe what the chunk "
                    "covers? 5 = perfectly accurate, 1 = wrong or misleading."
                ),
            },
            "situating_value": {
                "type": "integer",
                "description": (
                    "1-5. Does the summary add context that is NOT obvious from "
                    "the chunk alone? Does it reference how the chunk fits into "
                    "the broader document? 5 = excellent situating context, "
                    "1 = just restates the chunk."
                ),
            },
            "conciseness": {
                "type": "integer",
                "description": (
                    "1-5. Is the summary 2-3 sentences with no filler? "
                    "5 = perfectly concise, 3 = slightly verbose, "
                    "1 = way too long or rambling."
                ),
            },
            "reasoning": {
                "type": "string",
                "description": "Brief explanation of the scores.",
            },
        },
        "required": ["accuracy", "situating_value", "conciseness", "reasoning"],
    },
}

SUMMARY_JUDGE_SYSTEM = (
    "You are an evaluation judge. You will be given:\n"
    "1. A chunk of text from a document\n"
    "2. A contextual summary that was generated to situate this chunk within "
    "the broader document\n"
    "3. The full document for reference\n\n"
    "Score the summary using the score_summary tool.\n\n"
    "Accuracy: Does the summary correctly describe the chunk's content?\n"
    "Situating value: Does the summary add context BEYOND what's in the chunk? "
    "A good summary references how this chunk relates to the document's overall "
    "structure, what comes before/after, or why this section matters.\n"
    "Conciseness: Is the summary 2-3 sentences? No filler, no repetition."
)

DEFAULT_JUDGE_MODEL = "claude-haiku-4-5-20251001"


def _judge_summary(
    chunk_text: str,
    summary: str,
    full_text: str,
    judge_model: str,
) -> "ContextSummaryScore":
    """Score a summary using the shared judge dispatch."""
    return judge_dispatch(
        model=judge_model,
        system=SUMMARY_JUDGE_SYSTEM,
        user_prompt=(
            f"Chunk text:\n{chunk_text}\n\n"
            f"Contextual summary:\n{summary}\n\n"
            f"Full document (for reference):\n{full_text[:8000]}..."
        ),
        tool=SUMMARY_JUDGE_TOOL,
        score_cls=ContextSummaryScore,
    )


def _ollama_available() -> bool:
    """Check if Ollama is running locally."""
    try:
        import httpx
        resp = httpx.get("http://localhost:11434/api/tags", timeout=2.0)
        return resp.status_code == 200
    except Exception:
        return False


def _openai_key_available() -> bool:
    """Check if OPENAI_API_KEY is set."""
    import os
    return bool(os.environ.get("OPENAI_API_KEY"))


class TestContextSummary:
    """Score contextual summaries using an LLM judge."""

    @pytest.mark.skipif(
        not _ollama_available(),
        reason="Ollama not running — skip enrichment eval",
    )
    def test_context_summary_quality(self, judge_model: str) -> None:
        """5 sample chunks should get average accuracy >= 3.0 and situating_value >= 3.0."""
        full_text = _load_fixture()
        chunks = chunk_text(full_text, source_file="lee_ch1.txt")

        # Pick 5 evenly-spaced chunks (not first/last which may be atypical)
        n = len(chunks)
        indices = [n // 6, n // 3, n // 2, 2 * n // 3, 5 * n // 6]
        sample_chunks = [chunks[i] for i in indices if i < n]

        scores: list[ContextSummaryScore] = []

        for i, chunk in enumerate(sample_chunks):
            # Generate the contextual summary
            summary = enrich_chunk(full_text, chunk)

            # Judge the summary
            score = _judge_summary(
                chunk.text, summary, full_text, judge_model or DEFAULT_JUDGE_MODEL,
            )
            scores.append(score)
            print(
                f"\n  Chunk {indices[i]}: accuracy={score.accuracy} "
                f"situating={score.situating_value} "
                f"conciseness={score.conciseness}"
            )
            print(f"    {score.reasoning}")

        assert len(scores) == len(sample_chunks), "Judge failed on some chunks"

        avg_accuracy = sum(s.accuracy for s in scores) / len(scores)
        avg_situating = sum(s.situating_value for s in scores) / len(scores)
        avg_conciseness = sum(s.conciseness for s in scores) / len(scores)

        print(f"\n  Average accuracy:       {avg_accuracy:.1f}")
        print(f"  Average situating value: {avg_situating:.1f}")
        print(f"  Average conciseness:     {avg_conciseness:.1f}")

        assert avg_accuracy >= 3.0, (
            f"Average accuracy {avg_accuracy:.1f} below threshold 3.0"
        )
        assert avg_situating >= 3.0, (
            f"Average situating value {avg_situating:.1f} below threshold 3.0"
        )


# ===========================================================================
# Eval 5: Embedding retrieval — queries retrieve relevant chunks
# ===========================================================================


class TestEmbeddingRetrieval:
    """Verify that embeddings support accurate retrieval for known queries.

    Instead of cosine similarity (which CLAUDE.md prohibits as an eval metric),
    this eval tests a task-specific capability: given a natural-language query,
    does embedding-based retrieval return a chunk that actually contains the
    answer?
    """

    @pytest.mark.skipif(
        not _openai_key_available(),
        reason="OPENAI_API_KEY not set — skip embedding eval",
    )
    def test_retrieval_finds_relevant_chunks(self) -> None:
        """Queries about specific topics should retrieve chunks containing those topics."""
        full_text = _load_fixture()
        chunks = chunk_text(full_text, source_file="lee_ch1.txt")

        n = len(chunks)
        assert n >= 5, f"Need >= 5 chunks for this eval, got {n}"

        # Define queries and keywords that MUST appear in the top-3 retrieved chunks
        queries_and_keywords = [
            ("What is a topological manifold?", ["topological manifold"]),
            ("What is a coordinate chart?", ["coordinate", "chart"]),
            ("What is the Hausdorff property?", ["hausdorff"]),
        ]

        # Embed all chunks
        chunk_texts = [c.text for c in chunks]
        query_texts = [q for q, _ in queries_and_keywords]
        all_texts = chunk_texts + query_texts

        all_embeddings = embed_texts(all_texts)
        chunk_embeddings = np.array(all_embeddings[:len(chunks)])
        query_embeddings = np.array(all_embeddings[len(chunks):])

        # Normalize for dot-product ranking
        chunk_norms = np.linalg.norm(chunk_embeddings, axis=1, keepdims=True)
        chunk_norms[chunk_norms == 0] = 1.0
        chunk_unit = chunk_embeddings / chunk_norms

        hits = 0
        total = len(queries_and_keywords)
        top_k = 3

        for i, (query, keywords) in enumerate(queries_and_keywords):
            q_vec = query_embeddings[i]
            q_norm = np.linalg.norm(q_vec)
            if q_norm > 0:
                q_vec = q_vec / q_norm
            scores = chunk_unit @ q_vec
            top_indices = np.argsort(scores)[::-1][:top_k]

            # Check if any top-K chunk contains all required keywords
            found = False
            for idx in top_indices:
                text_lower = chunks[idx].text.lower()
                if all(kw.lower() in text_lower for kw in keywords):
                    found = True
                    break

            if found:
                hits += 1
                print(f"\n  ✓ Query: \"{query}\" — found in top-{top_k}")
            else:
                print(f"\n  ✗ Query: \"{query}\" — NOT found in top-{top_k}")
                for idx in top_indices:
                    print(f"      chunk {idx}: \"{chunks[idx].text[:80]}...\"")

        recall = hits / total
        print(f"\n  Retrieval recall@{top_k}: {hits}/{total} ({recall:.0%})")

        assert recall >= 0.66, (
            f"Retrieval recall@{top_k} is {recall:.0%} ({hits}/{total}). "
            f"Expected >= 66% of queries to find relevant chunks."
        )


# ===========================================================================
# Summary & results
# ===========================================================================


class TestIngestionSummary:
    """Collect and report all ingestion eval metrics."""

    def test_write_results(self) -> None:
        """Run all metrics and write results JSON."""
        full_text = _load_fixture()
        chunks = chunk_text(full_text, source_file="lee_ch1.txt")

        # Chunk size stats
        sizes = [_estimate_tokens(c.text) for c in chunks]
        sizes_except_last = sizes[:-1] if len(sizes) > 1 else sizes
        in_range = sum(1 for s in sizes_except_last if 400 <= s <= 600)

        # Coverage stats
        sentences = _sentence_split(full_text)
        all_chunk_text = " ".join(" ".join(c.text.split()) for c in chunks)
        missing = sum(
            1 for s in sentences
            if " ".join(s.split()) not in all_chunk_text
        )

        summary = {
            "phase": "phase1_ingestion",
            "corpus": "Lee, Introduction to Smooth Manifolds, Ch.1 (pp.19-49)",
            "total_chunks": len(chunks),
            "chunk_size_min_tokens": min(sizes),
            "chunk_size_max_tokens": max(sizes),
            "chunk_size_mean_tokens": round(sum(sizes) / len(sizes)),
            "pct_chunks_in_range": round(in_range / len(sizes_except_last), 2),
            "sentence_coverage": round(1.0 - missing / len(sentences), 4),
            "sentences_missing": missing,
            "sentences_total": len(sentences),
        }

        print(f"\n  Phase 1 Ingestion Eval Summary")
        print(f"  {'=' * 40}")
        for k, v in summary.items():
            print(f"    {k}: {v}")

        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        results_path = RESULTS_DIR / "phase1_ingestion.json"
        results_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"\n  Results written to {results_path}")


# ===========================================================================
# Standalone runner
# ===========================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
