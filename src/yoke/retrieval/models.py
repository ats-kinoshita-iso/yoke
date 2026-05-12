"""Pydantic data models for the retrieval module."""

from pydantic import BaseModel


class RetrievalResult(BaseModel):
    """A single retrieved chunk with ranking metadata."""

    chunk_id: int
    chunk_text: str
    context_summary: str
    source_file: str
    page_numbers: list[int]
    rrf_score: float
    dense_rank: int | None  # None if chunk wasn't in dense top-N
    sparse_rank: int | None  # None if chunk wasn't in sparse top-N
