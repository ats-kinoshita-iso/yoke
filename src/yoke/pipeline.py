"""Phase 1 query pipeline: hybrid retrieval + grounded generation with citations.

Usage:
    from yoke.pipeline import query, PipelineResult

    result = query("What is a topological manifold?", db_path, bm25_path)
    print(result.answer)
    for s in result.sources:
        print(s.source_file, s.page_numbers)
"""

import re
import time
from pathlib import Path

from pydantic import BaseModel

from yoke.config import YokeSettings, generate as llm_generate
from yoke.retrieval import RetrievalResult, RetrievalTimings, retrieve_with_timings

GENERATION_MODEL = YokeSettings().generation_model

GENERATION_SYSTEM = """\
You are a precise question-answering assistant. Answer ONLY from the \
provided context chunks. Follow these rules:

1. Base every claim on a specific chunk. Cite chunks by number, e.g. [1], [3].
2. If multiple chunks support a claim, cite all of them.
3. If the context does not contain enough information, respond with exactly: \
"I don't have enough information to answer this."
4. Do not use knowledge outside the provided context."""


class PipelineResult(BaseModel):
    """Result from the query pipeline."""

    answer: str
    sources: list[RetrievalResult]
    retrieval_timings: RetrievalTimings
    generation_ms: float

    @property
    def total_ms(self) -> float:
        return self.retrieval_timings.total_ms + self.generation_ms

    @property
    def cited_chunk_numbers(self) -> list[int]:
        """Parse [N] citation references from the answer."""
        return [int(m) for m in re.findall(r"\[(\d+)\]", self.answer)]


def format_context(chunks: list[RetrievalResult]) -> str:
    """Format retrieved chunks as numbered context blocks with source info."""
    blocks: list[str] = []
    for i, c in enumerate(chunks, 1):
        pages = ", ".join(str(p) for p in c.page_numbers)
        blocks.append(
            f"[{i}] Source: {c.source_file}, pages {pages}\n{c.chunk_text}"
        )
    return "\n\n---\n\n".join(blocks)


def query(
    question: str,
    db_path: Path,
    bm25_path: Path,
    *,
    k: int = 10,
    model: str = GENERATION_MODEL,
) -> PipelineResult:
    """End-to-end query: retrieve top-k chunks, generate answer with citations.

    Args:
        question: Natural language question.
        db_path: Path to the SQLite database.
        bm25_path: Path to the BM25 index file.
        k: Number of chunks to retrieve.
        model: Model spec for generation, e.g. "claude-sonnet-4-20250514"
            or "ollama/gemma4:e4b".

    Returns:
        PipelineResult with answer, sources, and timing info.
    """
    # 1. Hybrid retrieval
    results, timings = retrieve_with_timings(question, db_path, bm25_path, k=k)

    # 2. Format context
    context = format_context(results)

    # 3. Generate answer
    prompt = f"Context:\n{context}\n\nQuestion: {question}"
    t0 = time.perf_counter()
    answer = llm_generate(model, prompt, system=GENERATION_SYSTEM)
    gen_ms = (time.perf_counter() - t0) * 1000

    return PipelineResult(
        answer=answer,
        sources=results,
        retrieval_timings=timings,
        generation_ms=gen_ms,
    )
