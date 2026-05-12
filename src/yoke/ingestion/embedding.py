"""OpenAI embedding generation with batching and cost logging."""

from __future__ import annotations

import asyncio
import logging

import openai
from langfuse.decorators import observe

logger = logging.getLogger(__name__)

# text-embedding-3-small pricing: $0.02 per 1M tokens
COST_PER_TOKEN = 0.02 / 1_000_000


def _estimate_tokens(text: str) -> int:
    return len(text) // 4


@observe(name="embed_texts_async")
async def embed_texts_async(
    texts: list[str],
    *,
    model: str = "text-embedding-3-small",
    batch_size: int = 100,
    max_concurrent: int = 10,
) -> list[list[float]]:
    """Generate embeddings for a list of texts using OpenAI API.

    Args:
        texts: Texts to embed.
        model: OpenAI embedding model name.
        batch_size: Number of texts per API call.
        max_concurrent: Max parallel API calls.

    Returns:
        List of embedding vectors, one per input text.
    """
    if not texts:
        return []

    client = openai.AsyncOpenAI()
    semaphore = asyncio.Semaphore(max_concurrent)

    async def _embed_batch(
        start: int, batch: list[str]
    ) -> tuple[int, int, list[list[float]]]:
        async with semaphore:
            response = await client.embeddings.create(
                model=model,
                input=batch,
            )
        batch_tokens = response.usage.total_tokens
        logger.info(
            "Embedded batch [%d:%d] — %d tokens ($%.6f)",
            start, start + len(batch), batch_tokens,
            batch_tokens * COST_PER_TOKEN,
        )
        batch_embeddings = [obj.embedding for obj in response.data]
        return start, batch_tokens, batch_embeddings

    batches = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        batches.append((i, batch))

    results = await asyncio.gather(*[_embed_batch(s, b) for s, b in batches])

    # Assemble results after all tasks complete — no concurrent mutation
    all_embeddings: list[list[float]] = [[] for _ in texts]
    total_tokens = 0
    for start, batch_tokens, batch_embeddings in results:
        total_tokens += batch_tokens
        for i, emb in enumerate(batch_embeddings):
            all_embeddings[start + i] = emb

    cost = total_tokens * COST_PER_TOKEN
    logger.info(
        "Embedding complete: %d texts, %d tokens, $%.6f",
        len(texts), total_tokens, cost,
    )

    return all_embeddings


def embed_texts(
    texts: list[str],
    *,
    model: str = "text-embedding-3-small",
) -> list[list[float]]:
    """Synchronous wrapper for embed_texts_async (for eval use).

    Handles the case where an event loop is already running (e.g. pytest-asyncio)
    by using the synchronous OpenAI client instead.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is not None:
        # We're inside a running event loop — use sync client
        return _embed_texts_sync(texts, model=model)

    return asyncio.run(embed_texts_async(texts, model=model))


def _embed_texts_sync(
    texts: list[str],
    *,
    model: str = "text-embedding-3-small",
    batch_size: int = 100,
) -> list[list[float]]:
    """Synchronous embedding using the sync OpenAI client."""
    if not texts:
        return []

    client = openai.OpenAI()
    all_embeddings: list[list[float]] = [[] for _ in texts]
    total_tokens = 0

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        response = client.embeddings.create(model=model, input=batch)
        batch_tokens = response.usage.total_tokens
        total_tokens += batch_tokens
        logger.info(
            "Embedded batch [%d:%d] — %d tokens ($%.6f)",
            i, i + len(batch), batch_tokens, batch_tokens * COST_PER_TOKEN,
        )
        for j, embedding_obj in enumerate(response.data):
            all_embeddings[i + j] = embedding_obj.embedding

    cost = total_tokens * COST_PER_TOKEN
    logger.info("Embedding complete: %d texts, %d tokens, $%.6f", len(texts), total_tokens, cost)
    return all_embeddings
