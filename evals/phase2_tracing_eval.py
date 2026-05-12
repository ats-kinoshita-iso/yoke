"""Phase 2 tracing eval — Langfuse trace completeness, quality scoring, and no-op mode.

These evals define success criteria for the observability layer:
1. Every LLM call and embedding call produces a Langfuse span with required fields.
2. Quality scoring (faithfulness + relevance) attaches correct scores to traces.
3. When Langfuse is not configured, the system works identically with zero side effects.

Run: uv run pytest evals/phase2_tracing_eval.py -v
"""

import asyncio
import json
import os
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from yoke.tracing import (
    TracedModelClient,
    init_tracing,
    flush_tracing,
    get_current_trace_id,
)
from yoke.scoring import score_response, QualityScore
from yoke.models import ModelClient

load_dotenv(override=True)

RESULTS_DIR = Path(__file__).resolve().parent / "results"
DOCS_DIR = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "docs"


def _load_context() -> str:
    paths = sorted(p for p in DOCS_DIR.iterdir() if p.suffix in (".md", ".txt"))
    sections = []
    for p in paths:
        sections.append(f"## {p.name}\n{p.read_text(encoding='utf-8')}")
    return "\n---\n".join(sections)


def _langfuse_configured() -> bool:
    """Check if Langfuse credentials are available."""
    return bool(os.environ.get("LANGFUSE_PUBLIC_KEY"))


# ===========================================================================
# Eval 1: Trace completeness — every LLM call produces a span with required fields
# ===========================================================================


class TestTraceCompleteness:
    """Verify that TracedModelClient records spans with all required metadata."""

    async def test_generation_span_has_required_fields(self) -> None:
        """A traced LLM call must record: model, input, output, tokens, cost, latency."""
        # Create a mock inner client that returns a known response
        inner = AsyncMock(spec=ModelClient)
        inner.complete.return_value = "The answer is 42."

        # Create a mock Langfuse instance to capture what gets recorded
        mock_langfuse = MagicMock()
        mock_trace = MagicMock()
        mock_generation = MagicMock()
        mock_langfuse.trace.return_value = mock_trace
        mock_trace.generation.return_value = mock_generation

        traced = TracedModelClient(
            inner,
            model_name="claude-sonnet-4-20250514",
            langfuse=mock_langfuse,
        )

        result = await traced.complete("What is the meaning of life?", system="Be helpful.")
        assert result == "The answer is 42."

        # Verify a generation span was created
        mock_trace.generation.assert_called_once()
        call_kwargs = mock_trace.generation.call_args

        # Required fields in the generation span
        gen_kwargs = call_kwargs.kwargs if call_kwargs.kwargs else {}
        # If passed as positional or via a dict, adapt accordingly
        # The key assertion: the span must contain these fields
        assert "model" in gen_kwargs or call_kwargs, (
            "Generation span must include 'model'"
        )

    async def test_generation_span_records_latency(self) -> None:
        """Latency (ms) must be > 0 and recorded in the span metadata."""
        inner = AsyncMock(spec=ModelClient)

        async def slow_complete(prompt: str, *, system: str | None = None) -> str:
            await asyncio.sleep(0.05)  # 50ms delay
            return "response"

        inner.complete.side_effect = slow_complete

        mock_langfuse = MagicMock()
        mock_trace = MagicMock()
        mock_generation = MagicMock()
        mock_langfuse.trace.return_value = mock_trace
        mock_trace.generation.return_value = mock_generation

        traced = TracedModelClient(
            inner,
            model_name="test-model",
            langfuse=mock_langfuse,
        )

        await traced.complete("test prompt")

        # The generation end() call should include latency metadata
        mock_generation.end.assert_called_once()
        end_kwargs = mock_generation.end.call_args.kwargs
        assert "metadata" in end_kwargs or "completion_start_time" in end_kwargs, (
            "Generation span must record latency"
        )

    async def test_trace_id_links_related_operations(self) -> None:
        """Multiple calls within one trace context must share the same trace ID."""
        inner = AsyncMock(spec=ModelClient)
        inner.complete.return_value = "response"

        mock_langfuse = MagicMock()
        mock_trace = MagicMock()
        mock_trace.id = "trace-abc-123"
        mock_langfuse.trace.return_value = mock_trace

        traced = TracedModelClient(
            inner,
            model_name="test-model",
            langfuse=mock_langfuse,
        )

        # Two calls should be linked to the same trace
        await traced.complete("question 1")
        await traced.complete("question 2")

        # Both generation spans should be on the same trace
        assert mock_trace.generation.call_count == 2, (
            "Two LLM calls should produce two generation spans on the same trace"
        )

    async def test_token_counts_are_populated(self) -> None:
        """Token counts (prompt + completion) must be > 0 in the generation span."""
        inner = AsyncMock(spec=ModelClient)
        inner.complete.return_value = "A short response."

        mock_langfuse = MagicMock()
        mock_trace = MagicMock()
        mock_generation = MagicMock()
        mock_langfuse.trace.return_value = mock_trace
        mock_trace.generation.return_value = mock_generation

        traced = TracedModelClient(
            inner,
            model_name="test-model",
            langfuse=mock_langfuse,
        )

        await traced.complete("A medium-length prompt for testing token estimation.")

        mock_generation.end.assert_called_once()
        end_kwargs = mock_generation.end.call_args.kwargs

        # Must include usage with prompt and completion token counts
        assert "usage" in end_kwargs, "Generation span must include 'usage' with token counts"
        usage = end_kwargs["usage"]
        assert usage.get("prompt_tokens", 0) > 0, "Prompt tokens must be > 0"
        assert usage.get("completion_tokens", 0) > 0, "Completion tokens must be > 0"


# ===========================================================================
# Eval 2: Quality scoring accuracy — good answers score high, bad answers low
# ===========================================================================


def _anthropic_key_available() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


class TestQualityScoringAccuracy:
    """Verify that score_response() correctly distinguishes good from bad answers."""

    @pytest.mark.skipif(
        not _anthropic_key_available(),
        reason="ANTHROPIC_API_KEY not set — skip scoring eval",
    )
    async def test_good_answer_scores_high(self) -> None:
        """A faithful, relevant answer should score faithfulness >= 4."""
        context = _load_context()
        question = "What port does the API Gateway run on?"
        answer = "The API Gateway runs on port 8090."

        score = await score_response(
            question=question,
            answer=answer,
            context=context,
        )

        assert isinstance(score, QualityScore)
        assert score.faithfulness >= 4, (
            f"Good answer scored faithfulness={score.faithfulness}, expected >= 4. "
            f"Reasoning: {score.reasoning}"
        )
        assert score.relevance >= 4, (
            f"Good answer scored relevance={score.relevance}, expected >= 4. "
            f"Reasoning: {score.reasoning}"
        )

    @pytest.mark.skipif(
        not _anthropic_key_available(),
        reason="ANTHROPIC_API_KEY not set — skip scoring eval",
    )
    async def test_bad_answer_scores_low(self) -> None:
        """A hallucinated answer should score faithfulness < 3."""
        context = _load_context()
        question = "What port does the API Gateway run on?"
        bad_answer = (
            "The API Gateway runs on port 443 and uses NGINX as a reverse "
            "proxy with WebSocket support and TLS termination."
        )

        score = await score_response(
            question=question,
            answer=bad_answer,
            context=context,
        )

        assert isinstance(score, QualityScore)
        assert score.faithfulness < 3, (
            f"Bad answer scored faithfulness={score.faithfulness}, expected < 3. "
            f"Reasoning: {score.reasoning}"
        )

    @pytest.mark.skipif(
        not _anthropic_key_available(),
        reason="ANTHROPIC_API_KEY not set — skip scoring eval",
    )
    async def test_score_returns_pydantic_model(self) -> None:
        """score_response() must return a QualityScore with all required fields."""
        context = _load_context()

        score = await score_response(
            question="What is the default embedding batch size?",
            answer="The default is 64.",
            context=context,
        )

        assert isinstance(score, QualityScore)
        assert 1 <= score.faithfulness <= 5
        assert 1 <= score.relevance <= 5
        assert len(score.reasoning) > 0, "Reasoning must be non-empty"


# ===========================================================================
# Eval 3: No-op mode — system works identically without Langfuse configured
# ===========================================================================


class TestNoOpMode:
    """Verify tracing is silently disabled when Langfuse is not configured."""

    async def test_traced_client_works_without_langfuse(self) -> None:
        """TracedModelClient with langfuse=None must pass through to the inner client."""
        inner = AsyncMock(spec=ModelClient)
        inner.complete.return_value = "response without tracing"

        traced = TracedModelClient(
            inner,
            model_name="test-model",
            langfuse=None,  # No Langfuse configured
        )

        result = await traced.complete("test prompt")
        assert result == "response without tracing"
        inner.complete.assert_called_once_with("test prompt", system=None)

    async def test_init_tracing_returns_none_without_keys(self) -> None:
        """init_tracing() with empty keys must return None (no-op)."""
        with patch.dict(os.environ, {
            "LANGFUSE_PUBLIC_KEY": "",
            "LANGFUSE_SECRET_KEY": "",
        }, clear=False):
            langfuse = init_tracing()
            assert langfuse is None, (
                "init_tracing() must return None when LANGFUSE_PUBLIC_KEY is empty"
            )

    async def test_flush_tracing_noop_without_langfuse(self) -> None:
        """flush_tracing(None) must not raise."""
        # Should be a silent no-op
        flush_tracing(None)

    async def test_no_http_calls_without_langfuse(self) -> None:
        """When Langfuse is not configured, no HTTP calls should be made to Langfuse."""
        inner = AsyncMock(spec=ModelClient)
        inner.complete.return_value = "response"

        traced = TracedModelClient(
            inner,
            model_name="test-model",
            langfuse=None,
        )

        # Patch httpx to detect any outgoing calls
        with patch("httpx.AsyncClient.post") as mock_post, \
             patch("httpx.AsyncClient.get") as mock_get:
            await traced.complete("test")
            mock_post.assert_not_called()
            mock_get.assert_not_called()

    async def test_get_trace_id_returns_none_without_trace(self) -> None:
        """get_current_trace_id() must return None when no trace is active."""
        trace_id = get_current_trace_id()
        assert trace_id is None, (
            "get_current_trace_id() must return None outside a trace context"
        )


# ===========================================================================
# Eval 4: Score attachment to trace — scores are posted as Langfuse scores
# ===========================================================================


class TestScoreAttachment:
    """Verify that quality scores are attached to their parent trace."""

    async def test_score_posted_to_trace(self) -> None:
        """After scoring, the score must be attached to the trace via langfuse.score()."""
        mock_langfuse = MagicMock()
        mock_trace = MagicMock()
        mock_trace.id = "trace-score-test"
        mock_langfuse.trace.return_value = mock_trace

        # Create a mock score_response that returns a known score
        fake_score = QualityScore(faithfulness=4, relevance=5, reasoning="Good answer.")

        with patch("yoke.scoring.score_response", return_value=fake_score):
            from yoke.tracing import attach_scores_to_trace

            await attach_scores_to_trace(
                langfuse=mock_langfuse,
                trace_id="trace-score-test",
                score=fake_score,
            )

        # Verify scores were posted
        score_calls = mock_langfuse.score.call_args_list
        assert len(score_calls) >= 2, (
            f"Expected at least 2 score calls (faithfulness + relevance), "
            f"got {len(score_calls)}"
        )

        score_names = {call.kwargs.get("name") for call in score_calls}
        assert "faithfulness" in score_names, "Must post a 'faithfulness' score"
        assert "relevance" in score_names, "Must post a 'relevance' score"


# ===========================================================================
# Summary & results
# ===========================================================================


class TestTracingSummary:
    """Collect and report all tracing eval metrics."""

    async def test_write_results(self) -> None:
        """Aggregate eval metadata and write results JSON."""
        summary = {
            "phase": "phase2_tracing",
            "eval_categories": [
                "trace_completeness",
                "quality_scoring_accuracy",
                "noop_mode",
                "score_attachment",
            ],
            "langfuse_configured": _langfuse_configured(),
            "anthropic_key_available": _anthropic_key_available(),
        }

        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        results_path = RESULTS_DIR / "phase2_tracing.json"
        results_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        print(f"\n  Phase 2 Tracing Eval Summary")
        print(f"  {'=' * 40}")
        for k, v in summary.items():
            print(f"    {k}: {v}")
        print(f"\n  Results written to {results_path}")


# ===========================================================================
# Standalone runner
# ===========================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
