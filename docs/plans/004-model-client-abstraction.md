# Plan 004: Model Client Abstraction Layer

**Status:** Proposed
**Date:** 2026-04-07

## Goal

Replace hardcoded `anthropic.Anthropic()` / `anthropic.AsyncAnthropic()` calls
scattered across `baseline.py` and `phase0_eval.py` with a provider-agnostic
`ModelClient` protocol. This enables swapping between Claude, Ollama (local),
and future providers without changing call sites.

## Context

Currently the codebase has:
- `src/yoke/baseline.py` — hardcodes `anthropic.Anthropic()` with
  `claude-sonnet-4-20250514`
- `evals/phase0_eval.py` — hardcodes `anthropic.AsyncAnthropic()` with
  `claude-sonnet-4-20250514` and `claude-haiku-4-5-20251001`
- `src/models.py` — exists with a docstring describing the intended design
  but no implementation

The user's docstring in `src/models.py` specifies three backends: `claude`,
`ollama`, and `openai`. This plan covers `claude` and `ollama`; `openai` is
mentioned for future use (embeddings) and is out of scope for now.

## Files to Create

| File | Purpose |
|---|---|
| `src/yoke/models.py` | `ModelClient` protocol, `ClaudeClient`, `OllamaClient`, `get_model_client()` factory |
| `src/yoke/config.py` | Pydantic Settings class reading `YOKE_*` env vars with defaults |
| `tests/test_models.py` | Tests for factory function and `OllamaClient` (httpx mocked) |

## Files to Modify

| File | Change |
|---|---|
| `pyproject.toml` | Add `httpx` to main dependencies (currently dev-only; needed at runtime for Ollama) |

## Files to Delete

| File | Reason |
|---|---|
| `src/models.py` | Placeholder docstring only — real implementation goes to `src/yoke/models.py` (inside the package) |

## Design

### `ModelClient` Protocol

```python
from typing import Protocol

class ModelClient(Protocol):
    async def complete(self, prompt: str, *, system: str | None = None) -> str:
        ...
```

A minimal protocol — just `complete()`. No streaming, no tool use, no
embeddings. Those can be added as separate protocols later when needed.
This keeps the interface tight and testable.

### `ClaudeClient`

```python
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
```

Wraps `anthropic.AsyncAnthropic`. The API key is read from the environment
by the Anthropic SDK itself — no need to pass it explicitly.

**Decision: async-only.** The current `baseline.py` uses the sync client,
but the eval already uses async. Making `ModelClient` async-only is the right
call — `baseline.py` can use `asyncio.run()` to call it. This avoids
maintaining both sync and async interfaces.

### `OllamaClient`

```python
class OllamaClient:
    def __init__(self, model: str, *, base_url: str = "http://localhost:11434") -> None:
        self._model = model
        self._base_url = base_url

    async def complete(self, prompt: str, *, system: str | None = None) -> str:
        async with httpx.AsyncClient() as client:
            payload: dict = {"model": self._model, "prompt": prompt, "stream": False}
            if system:
                payload["system"] = system
            resp = await client.post(f"{self._base_url}/api/generate", json=payload)
            resp.raise_for_status()
            return resp.json()["response"]
```

Direct `httpx` calls to Ollama's REST API. No SDK dependency.
`stream: False` returns the full response in one shot.

**Trade-off: new `httpx.AsyncClient` per call vs. persistent client.**
Creating per-call is simpler and avoids lifecycle management. For a KM system
that makes a handful of LLM calls per query, the overhead is negligible. If
we later need high-throughput Ollama calls, we can add a persistent client
with connection pooling.

### `get_model_client()` Factory

```python
def get_model_client(provider: str, model_name: str) -> ModelClient:
    match provider:
        case "claude":
            return ClaudeClient(model_name)
        case "ollama":
            return OllamaClient(model_name)
        case _:
            raise ValueError(f"Unknown provider: {provider}")
```

Simple dispatch. No registry, no plugin system. Two providers, two branches.

### `src/yoke/config.py`

```python
from pydantic_settings import BaseSettings

class YokeSettings(BaseSettings):
    generation_model: str = "claude-sonnet-4-20250514"
    judge_model: str = "claude-haiku-4-5-20251001"
    summary_model: str = "ollama/gemma4:e2b"
    embedding_model: str = "text-embedding-3-small"

    model_config = SettingsConfigDict(env_prefix="YOKE_")
```

Models that include a `/` (e.g., `ollama/gemma4:e2b`) encode both provider
and model name. A helper function parses this:

```python
def parse_model_spec(spec: str) -> tuple[str, str]:
    """Parse 'provider/model' -> (provider, model). Default provider is 'claude'."""
    if "/" in spec:
        provider, model = spec.split("/", 1)
        return provider, model
    return "claude", spec
```

This means `claude-sonnet-4-20250514` → `("claude", "claude-sonnet-4-20250514")`
and `ollama/gemma4:e2b` → `("ollama", "gemma4:e2b")`.

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `YOKE_GENERATION_MODEL` | `claude-sonnet-4-20250514` | Model for answer generation |
| `YOKE_JUDGE_MODEL` | `claude-haiku-4-5-20251001` | Model for LLM-as-judge eval scoring |
| `YOKE_SUMMARY_MODEL` | `ollama/gemma4:e2b` | Model for document summarization |
| `YOKE_EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model (not used by `ModelClient`) |

## Tests

### `tests/test_models.py`

**1. Factory tests (no mocks, no network):**

```python
def test_get_model_client_claude():
    client = get_model_client("claude", "claude-sonnet-4-20250514")
    assert isinstance(client, ClaudeClient)

def test_get_model_client_ollama():
    client = get_model_client("ollama", "gemma4:e2b")
    assert isinstance(client, OllamaClient)

def test_get_model_client_unknown_raises():
    with pytest.raises(ValueError, match="Unknown provider"):
        get_model_client("fake", "fake-model")
```

**2. `parse_model_spec` tests:**

```python
def test_parse_model_spec_with_provider():
    assert parse_model_spec("ollama/gemma4:e2b") == ("ollama", "gemma4:e2b")

def test_parse_model_spec_default_provider():
    assert parse_model_spec("claude-sonnet-4-20250514") == ("claude", "claude-sonnet-4-20250514")
```

**3. OllamaClient tests (mock httpx):**

Per CLAUDE.md: "Tests use pytest with real assertions (no meaningless mocks)."
The httpx mock here is justified — it replaces an external network call to
localhost:11434, which may not be running. The assertions verify real behavior:
correct URL, payload structure, and response parsing.

```python
async def test_ollama_complete(httpx_mock):
    httpx_mock.add_response(json={"response": "The answer is 42."})
    client = OllamaClient("gemma4:e2b")
    result = await client.complete("What is the answer?", system="Be concise.")
    assert result == "The answer is 42."
    # Verify the request payload
    request = httpx_mock.get_requests()[0]
    body = json.loads(request.content)
    assert body["model"] == "gemma4:e2b"
    assert body["prompt"] == "What is the answer?"
    assert body["system"] == "Be concise."
    assert body["stream"] is False

async def test_ollama_complete_no_system(httpx_mock):
    httpx_mock.add_response(json={"response": "Hello."})
    client = OllamaClient("gemma4:e2b")
    result = await client.complete("Hi")
    assert result == "Hello."
    request = httpx_mock.get_requests()[0]
    body = json.loads(request.content)
    assert "system" not in body
```

**Test dependency:** `pytest-httpx` for httpx mocking. Add to `[project.optional-dependencies] dev`.

## Acceptance Criteria

1. `from yoke.models import ModelClient, ClaudeClient, OllamaClient, get_model_client` works
2. `from yoke.config import YokeSettings, parse_model_spec` works
3. `get_model_client("claude", "claude-sonnet-4-20250514")` returns a `ClaudeClient`
4. `get_model_client("ollama", "gemma4:e2b")` returns an `OllamaClient`
5. `get_model_client("fake", "x")` raises `ValueError`
6. `parse_model_spec("ollama/gemma4:e2b")` returns `("ollama", "gemma4:e2b")`
7. `parse_model_spec("claude-sonnet-4-20250514")` returns `("claude", "claude-sonnet-4-20250514")`
8. `OllamaClient.complete()` sends correct payload to `/api/generate` (verified via httpx mock)
9. All tests pass: `uv run pytest tests/test_models.py -v`
10. Existing eval still passes (no regressions)

## Evals Before Implementation

This feature is infrastructure — it does not change any LLM behavior or
output quality. The existing Phase 0 eval (`evals/phase0_eval.py`) serves as
the regression test: it must still pass after refactoring to use `ModelClient`.

No new eval is needed. The acceptance criteria are covered by unit tests.

**Future eval opportunity:** Once `OllamaClient` is wired up, run the Phase 0
eval with `YOKE_GENERATION_MODEL=ollama/gemma4:e2b` to establish a local-model
baseline. This is a follow-up task, not a prerequisite.

## Architectural Decisions

### 1. Protocol vs. ABC

**Decision:** `typing.Protocol` (structural subtyping).

**Trade-off:** A Protocol lets any class with a matching `complete()` method
satisfy `ModelClient` without inheriting from it. This is more Pythonic and
makes testing easier (any mock with `complete()` works). An ABC would force
inheritance but provide no benefit here — there's no shared implementation
to inherit.

### 2. Async-only vs. sync+async

**Decision:** Async-only. `complete()` is `async def`.

**Trade-off:** The sync `baseline.py` will need `asyncio.run()` to call
`complete()`. This is a minor inconvenience but avoids maintaining two
parallel interfaces (`complete()` and `acomplete()`). The codebase is already
async-first (eval, future FastAPI endpoints).

### 3. `src/yoke/models.py` vs. `src/models.py`

**Decision:** `src/yoke/models.py` — inside the package.

**Trade-off:** `src/models.py` currently exists as a placeholder outside the
package. Moving it inside means `from yoke.models import ...` works naturally.
The placeholder file should be deleted to avoid confusion.

### 4. Config via pydantic-settings vs. raw `os.getenv()`

**Decision:** `pydantic-settings` with `BaseSettings`.

**Trade-off:** pydantic-settings is already a dependency. It provides
validation, type coercion, `.env` file support, and a single source of truth
for all config. Raw `os.getenv()` would be simpler but scatters defaults
across the codebase.

### 5. `provider/model` string format vs. separate env vars

**Decision:** Single string with `provider/model` format, parsed by
`parse_model_spec()`. Bare model names default to `claude`.

**Trade-off:** One env var per model role is simpler than separate
`YOKE_GENERATION_PROVIDER` + `YOKE_GENERATION_MODEL` pairs. The `/` convention
is borrowed from Ollama's own model naming and is intuitive. The downside is
that Claude model names can't contain `/`, but they don't in practice.

## Implementation Order

1. Add `pytest-httpx` to dev dependencies in `pyproject.toml`
2. Move `httpx` from dev to main dependencies in `pyproject.toml`
3. Create `src/yoke/config.py` with `YokeSettings` and `parse_model_spec()`
4. Create `src/yoke/models.py` with `ModelClient`, `ClaudeClient`, `OllamaClient`, `get_model_client()`
5. Delete `src/models.py` (placeholder)
6. Write `tests/test_models.py`
7. Run tests, verify all pass
8. (Follow-up, not this PR) Refactor `baseline.py` and `phase0_eval.py` to use `ModelClient`
