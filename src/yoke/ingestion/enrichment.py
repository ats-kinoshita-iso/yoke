"""Contextual Retrieval: LLM-generated chunk summaries."""

from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import Callable

from langfuse.decorators import observe

from yoke.config import YokeSettings, parse_model_spec
from yoke.ingestion.models import Chunk, EnrichedChunk
from yoke.models import ModelClient, get_model_client

logger = logging.getLogger(__name__)

ENRICHMENT_SYSTEM = (
    "You write exactly 3 sentences. No markdown. No bullet points. No lists."
)

ENRICHMENT_PROMPT = """\
Document structure:
{doc_outline}

Chunk from this document:
{chunk_text}

Write exactly 3 sentences:
1. What specific concept, definition, example, or theorem this chunk discusses.
2. What topic the document covers just before this chunk.
3. What topic the document covers just after this chunk."""


def _get_doc_outline(full_text: str) -> str:
    """Extract section headings and key structural markers from the document."""
    lines = full_text.split("\n")
    outline_parts: list[str] = []

    # Get title from first non-empty lines
    for line in lines[:5]:
        if line.strip():
            outline_parts.append(line.strip())

    # Find section-like headings (short lines, often title case or all caps)
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        # Heuristic: lines that are short, start with a capital, and are
        # surrounded by blank lines are likely headings
        if (len(stripped) < 80
            and stripped[0].isupper()
            and (i == 0 or not lines[i - 1].strip())
            and not stripped.endswith(",")
            and not stripped.endswith(";")):
            if stripped not in outline_parts:
                outline_parts.append(stripped)

    # Also grab first ~200 chars for context
    preamble = " ".join(full_text[:300].split())
    return f"Preamble: {preamble}\n\nSections found:\n" + "\n".join(f"- {p}" for p in outline_parts[:20])


def _estimate_tokens(text: str) -> int:
    return len(text) // 4


def _clean_summary(raw: str) -> str:
    """Post-process LLM output: strip markdown, take first 2-3 sentences."""
    # Remove markdown headers, bullet points, bold markers
    text = re.sub(r"^#+\s+", "", raw, flags=re.MULTILINE)
    text = re.sub(r"^\s*[\*\-\d]+[\.\)]\s*", "", text, flags=re.MULTILINE)
    text = text.replace("**", "").replace("*", "")
    # Collapse to single paragraph
    text = " ".join(text.split())
    # Take first 3 sentences
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return " ".join(sentences[:3]).strip()


async def _enrich_one(
    full_text: str,
    chunk: Chunk,
    client: ModelClient,
    semaphore: asyncio.Semaphore,
    doc_outline: str | None = None,
    on_complete: Callable[[], None] | None = None,
) -> EnrichedChunk:
    """Generate a contextual summary for a single chunk."""
    summary_text = doc_outline or _get_doc_outline(full_text)
    prompt = ENRICHMENT_PROMPT.format(
        doc_outline=summary_text, chunk_text=chunk.text
    )
    input_tokens = _estimate_tokens(prompt)

    async with semaphore:
        raw = await client.complete(prompt, system=ENRICHMENT_SYSTEM)

    summary = _clean_summary(raw)
    output_tokens = _estimate_tokens(summary)
    logger.info(
        "Enriched chunk %d: ~%d input tokens, ~%d output tokens",
        chunk.chunk_index, input_tokens, output_tokens,
    )

    enriched_text = f"[Context: {summary}]\n\n{chunk.text}"

    if on_complete is not None:
        on_complete()

    return EnrichedChunk(
        chunk_index=chunk.chunk_index,
        chunk_text=chunk.text,
        context_summary=summary,
        enriched_text=enriched_text,
        page_numbers=chunk.page_numbers,
        source_file=chunk.source_file,
    )


@observe(name="enrich_chunks")
async def enrich_chunks(
    full_text: str,
    chunks: list[Chunk],
    client: ModelClient,
    *,
    max_concurrent: int = 3,
    on_chunk_complete: Callable[[], None] | None = None,
) -> list[EnrichedChunk]:
    """Add contextual summaries to all chunks in parallel.

    Args:
        full_text: The full document text for context.
        chunks: Chunks to enrich.
        client: ModelClient to use for summary generation.
        max_concurrent: Max parallel requests (semaphore size).
        on_chunk_complete: Optional callback invoked after each chunk finishes.

    Returns:
        List of EnrichedChunk in the same order as input.
    """
    semaphore = asyncio.Semaphore(max_concurrent)
    doc_outline = _get_doc_outline(full_text)
    total_input = _estimate_tokens(full_text) * len(chunks)
    logger.info(
        "Enriching %d chunks (~%d total input tokens, $0.00 if local model)",
        len(chunks), total_input,
    )

    tasks = [
        _enrich_one(
            full_text, chunk, client, semaphore, doc_outline,
            on_complete=on_chunk_complete,
        )
        for chunk in chunks
    ]
    results = await asyncio.gather(*tasks)
    return list(results)


def _complete_sync(
    provider: str, model_name: str, prompt: str, system: str
) -> str:
    """Synchronous LLM completion that works inside or outside an event loop.

    Uses sync HTTP clients directly to avoid asyncio.run() conflicts with
    pytest-asyncio. Delegates to the same backends as ModelClient.
    """
    if provider == "ollama":
        import httpx
        from yoke.models import OllamaClient
        # Use the same base_url as OllamaClient would
        base_url = "http://localhost:11434"
        payload = {
            "model": model_name,
            "prompt": prompt,
            "system": system,
            "stream": False,
        }
        resp = httpx.post(
            f"{base_url}/api/generate", json=payload, timeout=300.0,
        )
        resp.raise_for_status()
        return resp.json()["response"]
    elif provider == "claude":
        import anthropic
        ac = anthropic.Anthropic()
        msg = ac.messages.create(
            model=model_name,
            max_tokens=1024,
            temperature=0,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text
    else:
        raise ValueError(f"No sync fallback for provider: {provider}")


def enrich_chunk(full_text: str, chunk: Chunk) -> str:
    """Synchronous single-chunk enrichment for eval use.

    Returns the context summary string (not the full enriched text).
    """
    settings = YokeSettings()
    provider, model_name = parse_model_spec(settings.summary_model)

    doc_outline = _get_doc_outline(full_text)
    prompt = ENRICHMENT_PROMPT.format(
        doc_outline=doc_outline, chunk_text=chunk.text
    )

    try:
        asyncio.get_running_loop()
        # Inside a running event loop — use sync client
        raw = _complete_sync(provider, model_name, prompt, ENRICHMENT_SYSTEM)
    except RuntimeError:
        # No event loop — safe to use asyncio.run
        client = get_model_client(provider, model_name)
        raw = asyncio.run(client.complete(prompt, system=ENRICHMENT_SYSTEM))

    return _clean_summary(raw)
