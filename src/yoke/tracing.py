"""Langfuse tracing — client init, TracedModelClient wrapper, score attachment."""

from __future__ import annotations

import os
import time
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from langfuse import Langfuse

from yoke.scoring import QualityScore

# Context variable for propagating the active trace ID through async call chains.
_current_trace_id: ContextVar[str | None] = ContextVar(
    "_current_trace_id", default=None
)


def init_tracing() -> Langfuse | None:
    """Initialise a Langfuse client from environment variables.

    Returns None (no-op) when LANGFUSE_PUBLIC_KEY or LANGFUSE_SECRET_KEY
    are empty or missing.
    """
    public_key = os.environ.get("LANGFUSE_PUBLIC_KEY", "")
    secret_key = os.environ.get("LANGFUSE_SECRET_KEY", "")

    if not public_key or not secret_key:
        return None

    from langfuse import Langfuse

    host = os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com")
    return Langfuse(
        public_key=public_key,
        secret_key=secret_key,
        host=host,
    )


def flush_tracing(langfuse: Langfuse | None) -> None:
    """Flush pending Langfuse events. Safe to call with None."""
    if langfuse is not None:
        langfuse.flush()


def get_current_trace_id() -> str | None:
    """Return the active Langfuse trace ID, or None if no trace is active."""
    return _current_trace_id.get()


def set_current_trace_id(trace_id: str | None) -> None:
    """Set the active Langfuse trace ID in the current async context."""
    _current_trace_id.set(trace_id)


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 characters per token for English prose."""
    return max(1, len(text) // 4)


class TracedModelClient:
    """Wrapper that records every ``complete()`` call as a Langfuse generation.

    When *langfuse* is ``None`` this is a zero-overhead pass-through.
    """

    def __init__(
        self,
        inner: Any,  # ModelClient protocol
        model_name: str,
        langfuse: Langfuse | None,
    ) -> None:
        self._inner = inner
        self._model_name = model_name
        self._langfuse = langfuse
        self._trace: Any | None = None

    def _ensure_trace(self) -> Any:
        """Lazily create (or return the existing) Langfuse trace."""
        if self._trace is None and self._langfuse is not None:
            self._trace = self._langfuse.trace(name="model-client")
            set_current_trace_id(self._trace.id)
        return self._trace

    async def complete(self, prompt: str, *, system: str | None = None) -> str:
        """Delegate to the inner client, optionally recording a generation span."""
        if self._langfuse is None:
            return await self._inner.complete(prompt, system=system)

        trace = self._ensure_trace()

        # Build the input payload logged to Langfuse
        input_payload = {"prompt": prompt}
        if system is not None:
            input_payload["system"] = system

        generation = trace.generation(
            name="llm-complete",
            model=self._model_name,
            input=input_payload,
        )

        t0 = time.perf_counter()
        result = await self._inner.complete(prompt, system=system)
        latency_ms = (time.perf_counter() - t0) * 1000

        prompt_tokens = _estimate_tokens(prompt + (system or ""))
        completion_tokens = _estimate_tokens(result)

        generation.end(
            output=result,
            usage={
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            },
            metadata={"latency_ms": round(latency_ms, 2)},
        )

        return result


async def attach_scores_to_trace(
    *,
    langfuse: Langfuse | None,
    trace_id: str,
    score: QualityScore,
) -> None:
    """Post faithfulness and relevance scores to a Langfuse trace.

    No-op when *langfuse* is None.
    """
    if langfuse is None:
        return

    langfuse.score(
        name="faithfulness",
        value=score.faithfulness,
        trace_id=trace_id,
        comment=score.reasoning,
    )
    langfuse.score(
        name="relevance",
        value=score.relevance,
        trace_id=trace_id,
        comment=score.reasoning,
    )
