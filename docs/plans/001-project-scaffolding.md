# Plan 001: Project Scaffolding

**Status:** Proposed
**Date:** 2026-04-06

## Goal

Set up the Python project with uv, create `pyproject.toml` with all core
dependencies, establish the directory structure defined in README.md, and update
the README to reflect the actual runnable state.

## Context

The repo currently contains only `CLAUDE.md`, `README.md`, and `.claude/`
configs. The README already describes the intended project structure and tech
stack but none of it exists on disk yet. This plan bootstraps the project so
that `uv sync` succeeds and the directory layout matches the documented
architecture.

## Files to Create

| File | Purpose |
|---|---|
| `pyproject.toml` | Project metadata, dependencies, Python version constraint |
| `src/__init__.py` | Make `src` a package |
| `src/ingestion/__init__.py` | Ingestion sub-package |
| `src/retrieval/__init__.py` | Retrieval sub-package |
| `src/agent/__init__.py` | Agent orchestration sub-package |
| `src/memory/__init__.py` | Memory layer sub-package |
| `src/api/__init__.py` | FastAPI app sub-package |
| `evals/__init__.py` | Eval suite package |
| `tests/__init__.py` | Test suite package |
| `tests/conftest.py` | Shared pytest fixtures |
| `.env.example` | Template for required environment variables |
| `.gitignore` | Python/uv ignores |
| `.python-version` | Pin Python 3.12 for uv |

## Files to Modify

| File | Change |
|---|---|
| `README.md` | Add "Project Status" section noting scaffolding is complete; ensure Getting Started instructions match reality |

## Dependencies (`pyproject.toml`)

### Core
- `fastapi` — API layer
- `uvicorn[standard]` — ASGI server
- `langchain-core` — base abstractions (no high-level chains)
- `langgraph` — agent orchestration
- `anthropic` — Claude API client
- `pgvector` — PostgreSQL vector extension bindings
- `psycopg[binary]` — PostgreSQL driver (async-capable)
- `sqlalchemy` — ORM / query builder for pgvector integration
- `pydantic` — data models (pulled in by FastAPI, pinned explicitly)
- `pydantic-settings` — settings management from env vars
- `python-dotenv` — `.env` file loading

### Dev / Eval
- `pytest` — test runner
- `pytest-asyncio` — async test support
- `ragas` — RAG evaluation framework
- `httpx` — test client for FastAPI

All dependencies should use `>=` lower-bound pins (e.g. `fastapi>=0.115`),
not exact pins, to let uv resolve the latest compatible versions. The lockfile
(`uv.lock`) provides reproducibility.

## Acceptance Criteria

1. `uv sync` completes without errors
2. `uv run python -c "import src; print('ok')"` succeeds
3. `uv run pytest tests/ --collect-only` discovers the test package (0 tests collected is fine)
4. `.env.example` documents all required env vars: `ANTHROPIC_API_KEY`, `DATABASE_URL`
5. `.gitignore` covers `__pycache__`, `.venv`, `.env`, `uv.lock` (lock is committed but editor artifacts are not), `*.egg-info`
6. Directory structure on disk matches the structure documented in README.md

## Evals Before Implementation

This is a scaffolding task, not a feature — no behavioral evals are needed.
The acceptance criteria above serve as the verification checklist. However,
a smoke test should be added:

- `tests/test_smoke.py` — a single test that imports the top-level package
  and asserts the import succeeds. This establishes the eval/test pattern
  from day one.

## Architectural Decisions

### 1. Package layout: `src/` flat package vs `src/yoke/` namespace

**Decision:** Use `src/yoke/` as the installable package (standard src-layout).

**Trade-off:** A `src/yoke/` layout means imports look like `from yoke.agent import ...`
which is cleaner than `from src.agent import ...`. It also follows the
[src layout convention](https://packaging.python.org/en/latest/discussions/src-layout-vs-flat-layout/)
that uv and modern Python tooling expect. This means the directory tree from
the README should be adjusted: `src/yoke/` replaces `src/`.

**Updated structure:**
```
yoke/
├── src/
│   └── yoke/
│       ├── __init__.py
│       ├── ingestion/
│       ├── retrieval/
│       ├── agent/
│       ├── memory/
│       └── api/
├── evals/
├── tests/
├── docs/
├── pyproject.toml
└── ...
```

### 2. Python version: 3.12 vs 3.13

**Decision:** Require `>=3.12` in pyproject.toml, pin `3.12` in `.python-version`.

**Trade-off:** 3.12 has the widest library compatibility today. 3.13 is fine
but some scientific/ML deps lag. The `.python-version` file lets uv auto-install
the right interpreter.

### 3. Database driver: `psycopg[binary]` vs `asyncpg`

**Decision:** `psycopg[binary]` (psycopg3).

**Trade-off:** psycopg3 supports both sync and async, integrates natively with
SQLAlchemy 2.0, and works with pgvector's SQLAlchemy extension. asyncpg is
faster for pure-async workloads but lacks SQLAlchemy integration without
adapters.

## Implementation Order

1. Create `.python-version` and `.gitignore`
2. Create `pyproject.toml`
3. Run `uv sync` to generate lockfile and venv
4. Create directory structure with `__init__.py` files
5. Create `.env.example`
6. Create `tests/conftest.py` and `tests/test_smoke.py`
7. Run acceptance criteria checks
8. Update `README.md` to reflect src-layout adjustment
9. Commit: `feat: scaffold project structure with uv and core dependencies`
