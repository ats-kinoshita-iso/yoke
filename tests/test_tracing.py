"""Unit tests for the tracing module — TracedModelClient, init, no-op, context propagation."""

import asyncio
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from yoke.tracing import (
    TracedModelClient,
    init_tracing,
    flush_tracing,
    get_current_trace_id,
    attach_scores_to_trace,
)
from yoke.scoring import QualityScore, score_response
from yoke.models import ModelClient


# ---------------------------------------------------------------------------
# TracedModelClient — pass-through behavior
# ---------------------------------------------------------------------------


class TestTracedModelClientPassthrough:
    """TracedModelClient must delegate to the inner client and return its result."""

    async def test_returns_inner_result(self) -> None:
        inner = AsyncMock(spec=ModelClient)
        inner.complete.return_value = "hello"

        traced = TracedModelClient(inner, model_name="m", langfuse=None)
        assert await traced.complete("prompt") == "hello"

    async def test_forwards_system_param(self) -> None:
        inner = AsyncMock(spec=ModelClient)
        inner.complete.return_value = "ok"

        traced = TracedModelClient(inner, model_name="m", langfuse=None)
        await traced.complete("p", system="sys")
        inner.complete.assert_called_once_with("p", system="sys")

    async def test_satisfies_model_client_protocol(self) -> None:
        """TracedModelClient must be a valid ModelClient."""
        inner = AsyncMock(spec=ModelClient)
        inner.complete.return_value = ""

        traced = TracedModelClient(inner, model_name="m", langfuse=None)
        assert isinstance(traced, ModelClient)


# ---------------------------------------------------------------------------
# TracedModelClient — span recording with mock Langfuse
# ---------------------------------------------------------------------------


class TestTracedModelClientSpans:
    """When Langfuse is provided, TracedModelClient must record generation spans."""

    def _make_mocks(self) -> tuple[MagicMock, MagicMock, MagicMock]:
        mock_langfuse = MagicMock()
        mock_trace = MagicMock()
        mock_trace.id = "test-trace-id"
        mock_generation = MagicMock()
        mock_langfuse.trace.return_value = mock_trace
        mock_trace.generation.return_value = mock_generation
        return mock_langfuse, mock_trace, mock_generation

    async def test_creates_generation_span(self) -> None:
        inner = AsyncMock(spec=ModelClient)
        inner.complete.return_value = "answer"
        mock_lf, mock_trace, mock_gen = self._make_mocks()

        traced = TracedModelClient(inner, model_name="test-model", langfuse=mock_lf)
        await traced.complete("question")

        mock_trace.generation.assert_called_once()

    async def test_generation_span_includes_model_name(self) -> None:
        inner = AsyncMock(spec=ModelClient)
        inner.complete.return_value = "answer"
        mock_lf, mock_trace, mock_gen = self._make_mocks()

        traced = TracedModelClient(inner, model_name="claude-sonnet-4-20250514", langfuse=mock_lf)
        await traced.complete("question")

        call_kwargs = mock_trace.generation.call_args.kwargs
        assert call_kwargs.get("model") == "claude-sonnet-4-20250514"

    async def test_generation_span_includes_input_output(self) -> None:
        inner = AsyncMock(spec=ModelClient)
        inner.complete.return_value = "42"
        mock_lf, mock_trace, mock_gen = self._make_mocks()

        traced = TracedModelClient(inner, model_name="m", langfuse=mock_lf)
        await traced.complete("meaning of life")

        call_kwargs = mock_trace.generation.call_args.kwargs
        assert "meaning of life" in str(call_kwargs.get("input", ""))

        end_kwargs = mock_gen.end.call_args.kwargs
        assert "42" in str(end_kwargs.get("output", ""))

    async def test_generation_span_includes_usage(self) -> None:
        inner = AsyncMock(spec=ModelClient)
        inner.complete.return_value = "short"
        mock_lf, mock_trace, mock_gen = self._make_mocks()

        traced = TracedModelClient(inner, model_name="m", langfuse=mock_lf)
        await traced.complete("a prompt")

        end_kwargs = mock_gen.end.call_args.kwargs
        usage = end_kwargs.get("usage", {})
        assert usage.get("prompt_tokens", 0) > 0
        assert usage.get("completion_tokens", 0) > 0

    async def test_multiple_calls_share_trace(self) -> None:
        inner = AsyncMock(spec=ModelClient)
        inner.complete.return_value = "r"
        mock_lf, mock_trace, _ = self._make_mocks()

        traced = TracedModelClient(inner, model_name="m", langfuse=mock_lf)
        await traced.complete("q1")
        await traced.complete("q2")

        assert mock_trace.generation.call_count == 2


# ---------------------------------------------------------------------------
# init_tracing — configuration-based initialization
# ---------------------------------------------------------------------------


class TestInitTracing:
    """init_tracing() must return a Langfuse client or None based on config."""

    def test_returns_none_without_keys(self) -> None:
        with patch.dict(os.environ, {
            "LANGFUSE_PUBLIC_KEY": "",
            "LANGFUSE_SECRET_KEY": "",
        }, clear=False):
            result = init_tracing()
            assert result is None

    def test_returns_none_with_missing_keys(self) -> None:
        env = os.environ.copy()
        env.pop("LANGFUSE_PUBLIC_KEY", None)
        env.pop("LANGFUSE_SECRET_KEY", None)
        with patch.dict(os.environ, env, clear=True):
            result = init_tracing()
            assert result is None


# ---------------------------------------------------------------------------
# flush_tracing — safe no-op
# ---------------------------------------------------------------------------


class TestFlushTracing:

    def test_noop_with_none(self) -> None:
        """flush_tracing(None) must not raise."""
        flush_tracing(None)

    def test_calls_flush_on_langfuse(self) -> None:
        mock_lf = MagicMock()
        flush_tracing(mock_lf)
        mock_lf.flush.assert_called_once()


# ---------------------------------------------------------------------------
# get_current_trace_id — context variable
# ---------------------------------------------------------------------------


class TestGetCurrentTraceId:

    def test_returns_none_outside_trace(self) -> None:
        assert get_current_trace_id() is None


# ---------------------------------------------------------------------------
# attach_scores_to_trace
# ---------------------------------------------------------------------------


class TestAttachScores:

    async def test_posts_faithfulness_and_relevance(self) -> None:
        mock_lf = MagicMock()
        score = QualityScore(faithfulness=4, relevance=5, reasoning="solid")

        await attach_scores_to_trace(
            langfuse=mock_lf,
            trace_id="t-123",
            score=score,
        )

        calls = mock_lf.score.call_args_list
        names = {c.kwargs["name"] for c in calls}
        assert "faithfulness" in names
        assert "relevance" in names

    async def test_noop_with_none_langfuse(self) -> None:
        """attach_scores_to_trace with langfuse=None must not raise."""
        score = QualityScore(faithfulness=3, relevance=3, reasoning="ok")
        await attach_scores_to_trace(langfuse=None, trace_id="t", score=score)


# ---------------------------------------------------------------------------
# QualityScore model
# ---------------------------------------------------------------------------


class TestQualityScoreModel:

    def test_valid_score(self) -> None:
        s = QualityScore(faithfulness=5, relevance=4, reasoning="good")
        assert s.faithfulness == 5
        assert s.relevance == 4

    def test_rejects_out_of_range(self) -> None:
        with pytest.raises(Exception):
            QualityScore(faithfulness=6, relevance=4, reasoning="bad")

        with pytest.raises(Exception):
            QualityScore(faithfulness=3, relevance=0, reasoning="bad")
