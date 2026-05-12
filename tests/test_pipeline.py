"""Unit tests for yoke.pipeline — offline tests (no API keys needed)."""

from yoke.pipeline import PipelineResult, format_context
from yoke.retrieval.models import RetrievalResult
from yoke.retrieval.hybrid import RetrievalTimings


def _make_chunk(
    chunk_id: int,
    text: str,
    source_file: str = "test.txt",
    page_numbers: list[int] | None = None,
) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        chunk_text=text,
        context_summary="",
        source_file=source_file,
        page_numbers=page_numbers or [1],
        rrf_score=0.5,
        dense_rank=1,
        sparse_rank=1,
    )


def _make_timings() -> RetrievalTimings:
    return RetrievalTimings(
        embedding_ms=10, dense_ms=20, sparse_ms=15, rrf_ms=1, total_ms=46
    )


class TestFormatContext:
    def test_numbered_blocks(self) -> None:
        chunks = [
            _make_chunk(1, "First chunk.", page_numbers=[1, 2]),
            _make_chunk(2, "Second chunk.", page_numbers=[3]),
        ]
        ctx = format_context(chunks)
        assert "[1] Source: test.txt, pages 1, 2" in ctx
        assert "[2] Source: test.txt, pages 3" in ctx
        assert "First chunk." in ctx
        assert "Second chunk." in ctx

    def test_empty_chunks(self) -> None:
        assert format_context([]) == ""

    def test_single_chunk(self) -> None:
        chunks = [_make_chunk(1, "Only chunk.", page_numbers=[5])]
        ctx = format_context(chunks)
        assert "[1] Source: test.txt, pages 5" in ctx
        assert "---" not in ctx  # No separator for single chunk

    def test_multiple_sources(self) -> None:
        chunks = [
            _make_chunk(1, "From A.", source_file="a.pdf", page_numbers=[1]),
            _make_chunk(2, "From B.", source_file="b.pdf", page_numbers=[2, 3]),
        ]
        ctx = format_context(chunks)
        assert "a.pdf" in ctx
        assert "b.pdf" in ctx


class TestPipelineResultCitations:
    def test_parses_citations(self) -> None:
        result = PipelineResult(
            answer="See [1] and [3]. Also [1] again.",
            sources=[_make_chunk(1, "x"), _make_chunk(2, "y"), _make_chunk(3, "z")],
            retrieval_timings=_make_timings(),
            generation_ms=100,
        )
        assert result.cited_chunk_numbers == [1, 3, 1]

    def test_no_citations(self) -> None:
        result = PipelineResult(
            answer="I don't have enough information to answer this.",
            sources=[_make_chunk(1, "x")],
            retrieval_timings=_make_timings(),
            generation_ms=50,
        )
        assert result.cited_chunk_numbers == []

    def test_total_ms(self) -> None:
        result = PipelineResult(
            answer="test",
            sources=[],
            retrieval_timings=_make_timings(),
            generation_ms=100,
        )
        assert result.total_ms == 146.0  # 46 + 100

    def test_does_not_parse_non_numeric_brackets(self) -> None:
        result = PipelineResult(
            answer="The set [a, b] is open. See [2].",
            sources=[_make_chunk(1, "x"), _make_chunk(2, "y")],
            retrieval_timings=_make_timings(),
            generation_ms=50,
        )
        assert result.cited_chunk_numbers == [2]
