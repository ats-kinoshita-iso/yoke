"""Pydantic data models for the ingestion pipeline."""

from pathlib import Path

from pydantic import BaseModel


class PageText(BaseModel):
    """A single page of extracted text."""
    page_number: int
    text: str


class Chunk(BaseModel):
    """A chunk of text with metadata, before enrichment."""
    chunk_index: int
    text: str
    page_numbers: list[int]
    source_file: str


class EnrichedChunk(BaseModel):
    """A chunk after contextual enrichment."""
    chunk_index: int
    chunk_text: str
    context_summary: str
    enriched_text: str
    page_numbers: list[int]
    source_file: str


class IngestResult(BaseModel):
    """Result of an ingestion run."""
    documents_processed: int
    total_chunks: int
    db_path: str
    bm25_path: str
    errors: list[str]
