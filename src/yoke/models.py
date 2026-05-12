"""
Model provider abstraction.

Supports three backends:
  - "claude" -> Anthropic API (claude-sonnet-4-20250514, claude-haiku-4-5-20251001, etc.)
  - "ollama" -> Local models via Ollama (gemma4:e2b, gemma4:e4b, etc.)
  - "openai" -> OpenAI-compatible APIs (for embeddings, or future use)

Usage:
  client = get_model_client("claude", "claude-haiku-4-5-20251001")
  response = await client.complete(prompt, system=system_prompt)
"""

from typing import Protocol, runtime_checkable

import anthropic
import httpx


@runtime_checkable
class ModelClient(Protocol):
    async def complete(self, prompt: str, *, system: str | None = None) -> str: ...


class ClaudeClient:
    def __init__(self, model: str) -> None:
        self._client = anthropic.AsyncAnthropic()
        self._model = model

    async def complete(self, prompt: str, *, system: str | None = None) -> str:
        message = await self._client.messages.create(
            model=self._model,
            max_tokens=1024,
            temperature=0,
            system=system or "",
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text


class OllamaClient:
    def __init__(self, model: str, *, base_url: str = "http://localhost:11434") -> None:
        self._model = model
        self._base_url = base_url

    async def complete(self, prompt: str, *, system: str | None = None) -> str:
        async with httpx.AsyncClient(timeout=300.0) as client:
            payload: dict = {"model": self._model, "prompt": prompt, "stream": False}
            if system:
                payload["system"] = system
            resp = await client.post(f"{self._base_url}/api/generate", json=payload)
            resp.raise_for_status()
            return resp.json()["response"]


def get_model_client(
    provider: str,
    model_name: str,
    *,
    langfuse: object | None = None,
) -> ModelClient:
    """Create a model client, optionally wrapped with Langfuse tracing.

    Args:
        provider: Backend provider ("claude", "ollama").
        model_name: Model identifier for the provider.
        langfuse: A Langfuse instance. When provided, the returned client
            is wrapped with ``TracedModelClient`` for automatic span recording.
    """
    match provider:
        case "claude":
            inner: ModelClient = ClaudeClient(model_name)
        case "ollama":
            inner = OllamaClient(model_name)
        case _:
            raise ValueError(f"Unknown provider: {provider}")

    if langfuse is not None:
        from yoke.tracing import TracedModelClient

        return TracedModelClient(inner, model_name=model_name, langfuse=langfuse)

    return inner
