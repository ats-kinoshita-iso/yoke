# Agentic Knowledge Management System

## Project Overview
An agentic knowledge management harness built in Python. The system ingests
documents, indexes them for hybrid retrieval, and uses an LLM agent to
autonomously search, evaluate, and synthesize answers with memory persistence.

## Architecture
- Python 3.12+ with uv for dependency management
- FastAPI for the API layer
- PostgreSQL with pgvector for hybrid (dense + sparse) retrieval
- LangGraph for agent orchestration
- Anthropic Claude API for LLM inference
- RAGAS + custom evals for quality measurement

## Conventions
- Type hints on all function signatures
- Pydantic models for all data structures
- Tests use pytest with real assertions (no meaningless mocks)
- Every new feature requires an eval before implementation
- Commits follow conventional commit format

## What NOT to do
- Do not use LangChain's high-level chains — use LangGraph for orchestration
- Do not use cosine similarity as an eval metric — use task-specific evals
- Do not put retrieval logic inside the agent prompt — expose it as tools
