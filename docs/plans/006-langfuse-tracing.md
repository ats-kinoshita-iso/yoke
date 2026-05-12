# Plan 006: Langfuse Tracing & Quality Scoring

**Status:** Proposed
**Date:** 2026-04-07

## Goal

Instrument all LLM calls, tool use, and retrieval operations with Langfuse
tracing. After each agent response, run a quality scoring step (faithfulness +
relevance) and log those scores as trace metadata. This gives us cost tracking,
latency monitoring, and per-response quality signals — all linked by a single
trace ID.

## Context

The system already logs token counts and cost estimates to Python's `logging`
module (`embedding.py`, `enrichment.py`), but those logs are ephemeral and
unstructured. The README promises Langfuse integration but no code exists yet.

Current LLM call sites that need instrumentation:
- `src/yoke/models.py` — `ClaudeClient.complete()`, `OllamaClient.complete()`
- `src/yoke/ingestion/enrichment.py` — `enrich_chunks()` (contextual summaries)
- `src/yoke/ingestion/embedding.py` — `embed_texts_async()` (OpenAI embeddings)
- `evals/phase0_eval.py` — `_async_ask()`, `_async_judge()` (eval LLM calls)

Future call sites (not yet implemented, but the tracing layer must accommodate):
- `src/yoke/agent/` — LangGraph agent orchestration (tool calls, reasoning steps)
- `src/yoke/retrieval/` — hybrid search (dense + sparse retrieval)

The evaluator subagent (`.claude/agents/evaluator.md`) defines the scoring
rubric: faithfulness, relevance, completeness on 1-5. We will implement a
runtime version of this as an async scoring function.

## Design

### Architecture overview

```
┌──────────────────────────────────────────────────┐
│  Application Code                                │
│  (models.py, enrichment.py, embedding.py, agent) │
│                                                  │
│  Uses: @observe decorator or manual span API     │
└────────────┬─────────────────────────────────────┘
             │  spans / generations / scores
             ▼
┌──────────────────────────────────────────────────┐
│  src/yoke/tracing.py                             │
│  - init_tracing() → configure Langfuse client    │
│  - trace context manager for request-level trace │
│  - score_response() → async quality scoring      │
└────────────┬─────────────────────────────────────┘
             │  HTTP (async, batched)
             ▼
┌──────────────────────────────────────────────────┐
│  Langfuse Server (cloud or self-hosted)          │
│  - Traces, spans, generations, scores            │
│  - Dashboard: latency, cost, quality over time   │
└──────────────────────────────────────────────────┘
```

### Key decisions

**1. Use the Langfuse Python SDK's `@observe` decorator (not the LangChain
   integration)**

The project explicitly avoids LangChain chains. Langfuse's `@observe`
decorator and manual `langfuse.generation()` / `langfuse.span()` calls work
at the function level without any framework dependency. When the LangGraph
agent is built later, we can use the LangGraph callback handler OR continue
with manual instrumentation — both work.

*Trade-off:* Manual instrumentation is more code but gives full control over
what is traced and how spans are named. The LangChain auto-instrumentation
would conflict with the "no LangChain chains" principle and would miss
non-LangChain calls (Ollama, direct Anthropic, OpenAI embeddings).

**2. Instrument at the `ModelClient` protocol level**

Wrap `ClaudeClient` and `OllamaClient` with a `TracedModelClient` decorator
that records each `complete()` call as a Langfuse "generation" span. This
captures input/output, latency, token count, and cost without modifying each
call site.

For embeddings (`embed_texts_async`), use `@observe` on the function since
it doesn't go through `ModelClient`.

*Trade-off:* A decorator/wrapper approach means all calls through
`ModelClient` are automatically traced. The alternative — instrumenting each
call site individually — is more explicit but requires changes in every file
that makes an LLM call.

**3. Quality scoring as an async post-processing step**

After each agent response, fire an async task that:
1. Calls the judge model (claude-haiku) to score faithfulness + relevance
2. Posts scores to Langfuse via `trace.score()`
3. Does NOT block the response to the user

This mirrors the existing `_async_judge()` pattern in `phase0_eval.py` but
runs in production, not just during evals.

*Trade-off:* Async scoring adds ~1 extra LLM call per response (haiku is
cheap: ~$0.001/call). The latency does not affect the user since it runs
after the response is returned. The alternative — scoring only during evals —
gives less visibility into production quality.

**4. Trace ID propagation via context variables**

Use Python `contextvars` to propagate the Langfuse trace ID through async
call chains. Each incoming request (API endpoint or CLI invocation) creates a
new trace, and all downstream LLM calls, retrieval operations, and tool uses
inherit that trace ID automatically.

### Data captured per span type

| Span type    | Fields captured                                           |
|-------------|-----------------------------------------------------------|
| Generation  | model, input, output, tokens (prompt + completion), cost, latency |
| Retrieval   | query, top-K results, retrieval method (dense/sparse/hybrid), latency |
| Tool use    | tool name, input args, output, latency                     |
| Score       | trace_id, name (faithfulness/relevance), value (1-5), comment |

### Configuration

Add to `YokeSettings` in `config.py`:

```python
langfuse_public_key: str = ""       # LANGFUSE_PUBLIC_KEY
langfuse_secret_key: str = ""       # LANGFUSE_SECRET_KEY
langfuse_host: str = "https://cloud.langfuse.com"  # LANGFUSE_HOST
tracing_enabled: bool = True        # YOKE_TRACING_ENABLED
quality_scoring_enabled: bool = True  # YOKE_QUALITY_SCORING_ENABLED
```

When `langfuse_public_key` is empty, tracing is silently disabled (no-op).
This ensures the system works without Langfuse configured (local dev, CI).

## Files to create or modify

### New files

| File | Purpose |
|------|---------|
| `src/yoke/tracing.py` | Langfuse client init, `TracedModelClient` wrapper, `@traced` decorator, `score_response()`, trace context manager |
| `src/yoke/scoring.py` | Runtime quality scoring: faithfulness + relevance judge using claude-haiku, Pydantic models for scores |
| `evals/phase2_tracing_eval.py` | Eval: verify traces are emitted with correct structure, scores are attached, no-op when disabled |
| `tests/test_tracing.py` | Unit tests: `TracedModelClient` wrapping, context propagation, no-op mode |

### Modified files

| File | Change |
|------|--------|
| `pyproject.toml` | Add `langfuse>=2.0` dependency |
| `src/yoke/config.py` | Add Langfuse settings fields |
| `src/yoke/models.py` | Wrap clients with `TracedModelClient` in `get_model_client()` |
| `src/yoke/ingestion/embedding.py` | Add `@observe` decorator to `embed_texts_async()` |
| `src/yoke/ingestion/enrichment.py` | Add `@observe` decorator to `enrich_chunks()` |
| `src/yoke/ingestion/pipeline.py` | Create root trace per `ingest_directory()` call, flush on completion |
| `src/yoke/baseline.py` | Create root trace per `ask()` call (existing baseline path) |

## Acceptance criteria

1. **Every LLM call is traced.** A call to `ClaudeClient.complete()` or
   `OllamaClient.complete()` produces a Langfuse "generation" span with:
   input text, output text, model name, token counts, estimated cost, and
   latency in ms.

2. **Embedding calls are traced.** Each `embed_texts_async()` batch produces
   a Langfuse span with: model, batch size, token count, cost, and latency.

3. **Trace linking works.** All spans from a single `ingest_directory()` or
   `ask()` invocation share the same trace ID. Navigating to a trace in the
   Langfuse UI shows the full call tree.

4. **Quality scores are attached.** After each `ask()` response, a
   faithfulness score (1-5) and relevance score (1-5) are posted to the
   trace as Langfuse scores.

5. **Graceful degradation.** When `LANGFUSE_PUBLIC_KEY` is not set, tracing
   is silently disabled. No errors, no performance impact. All existing tests
   pass without Langfuse configured.

6. **Cost tracking is accurate.** Traced cost matches the existing manual
   cost calculations in `embedding.py` (within 1% tolerance).

7. **Latency overhead is minimal.** Tracing adds < 5ms per LLM call (Langfuse
   SDK batches HTTP sends asynchronously).

## Evals needed before implementation

### Eval 1: Trace completeness (evals/phase2_tracing_eval.py)

Run a small end-to-end ingestion (3 docs) and a baseline `ask()` call with
Langfuse configured. Then query the Langfuse API to verify:

- [ ] Each LLM call produced a generation span
- [ ] Each embedding batch produced a span
- [ ] All spans within one operation share a trace ID
- [ ] Token counts are populated and > 0
- [ ] Cost fields are populated and match expected pricing
- [ ] Latency fields are populated and > 0

### Eval 2: Quality scoring accuracy

Using the existing Phase 0 QA pairs, run `score_response()` on known-good
and known-bad answers. Verify:

- [ ] Good answers (from Phase 0 baseline) score faithfulness >= 4
- [ ] Known-bad calibration answers score faithfulness < 3
- [ ] Scores are posted to the Langfuse trace as metadata
- [ ] Scoring does not block the response (async)

### Eval 3: No-op mode

Run the full test suite with `LANGFUSE_PUBLIC_KEY` unset:

- [ ] All existing tests pass unchanged
- [ ] No Langfuse-related errors in logs
- [ ] No HTTP calls to Langfuse endpoints (verified via `pytest-httpx`)

## Implementation order

1. **Add `langfuse` dependency** and config fields (smallest change, unblocks
   the rest)
2. **Write `src/yoke/tracing.py`** with `init_tracing()`, `TracedModelClient`,
   and the no-op fallback
3. **Write `tests/test_tracing.py`** — unit tests for the tracing module
4. **Instrument `models.py`** — wrap `get_model_client()` return values
5. **Instrument ingestion** — `@observe` on `embed_texts_async()` and
   `enrich_chunks()`, root trace in `ingest_directory()`
6. **Write `src/yoke/scoring.py`** — runtime quality scorer
7. **Instrument `baseline.py`** — root trace + quality scoring on `ask()`
8. **Write `evals/phase2_tracing_eval.py`** — trace completeness + scoring
   accuracy eval
9. **Run full test/eval suite** to verify no regressions

## Risks and mitigations

| Risk | Mitigation |
|------|-----------|
| Langfuse SDK adds latency to hot path | SDK batches sends async; verified by acceptance criterion #7 |
| Langfuse server unavailable | SDK has built-in retry + flush timeout; no-op mode for CI |
| Quality scoring costs add up | Haiku is ~$0.001/call; scoring can be disabled via config |
| Token count not available from Ollama | Estimate via `len(text) // 4`; flag as estimate in span metadata |
| Context var propagation breaks in thread pools | Use `asyncio.TaskGroup` (not thread executors) for concurrency |
| Circular import: tracing.py imports models.py for type hints | Use `TYPE_CHECKING` guard or accept `ModelClient` as protocol param |
