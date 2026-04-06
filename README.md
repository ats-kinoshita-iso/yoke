# Yoke

An agentic knowledge management harness built with Claude Code and the Anthropic API.

Yoke wraps an LLM in a production-grade orchestration layer — managing context,
tools, memory, and control flow — to turn a passive Q&A system into a research
assistant that plans, retrieves, evaluates, and iterates autonomously.

---

## Architecture

| Layer | Technology |
|---|---|
| Ingestion & Indexing | Custom pipeline + contextual chunking |
| Retrieval | PostgreSQL + pgvector (dense + BM25 + RRF) |
| Agent Orchestration | LangGraph |
| LLM Inference | Anthropic Claude API |
| Memory | Persistent store across sessions |
| Evaluation | RAGAS + custom eval suite |
| Observability | Langfuse (tracing, cost, quality scoring) |
| API Layer | FastAPI |
| Dependency Management | uv + pyproject.toml |

---

## Getting Started

**Prerequisites:** Python 3.12+, PostgreSQL with pgvector, a Claude API key (or Claude subscription for Claude Code)
```bash
git clone https://github.com/ats-kinoshita-iso/yoke.git
cd yoke
uv sync
cp .env.example .env   # add your ANTHROPIC_API_KEY and DB config
```

### Run with Claude Code
```bash
claude
```

Claude Code will read `CLAUDE.md` and load the project context automatically.
Use `/plan`, `/eval-first`, and `/review` slash commands for structured development.

---

## Development Workflow

Yoke follows an **eval-driven development** loop:

1. **Write evals first** (`/eval-first`) — define what "working" means before building
2. **Implement** — let Claude Code build iteratively against your evals
3. **Measure** — run the eval suite and compare against the baseline
4. **Commit** only when scores improve

Every new capability must demonstrate measurable improvement over the baseline.
Complexity is earned, not assumed.

---

## Project Structure
yoke/
├── CLAUDE.md              # Agent constitution — read this first
├── .claude/commands/      # Slash commands: /plan, /eval-first, /review
├── src/
│   ├── ingestion/         # Document pipeline and chunking
│   ├── retrieval/         # Hybrid search (dense + BM25 + reranking)
│   ├── agent/             # LangGraph orchestration and tools
│   ├── memory/            # Persistent memory layer
│   └── api/               # FastAPI endpoints
├── evals/                 # Eval cases, baselines, and regression suite
├── docs/
│   └── architecture.md    # Architecture decision records
└── tests/

---

## Evaluation
```bash
uv run pytest evals/          # run full eval suite
uv run pytest evals/retrieval # retrieval-specific evals only
```

Scores are logged to Langfuse with per-trace quality metadata (faithfulness, relevance, latency, cost).

---

## Key Design Principles

- **Simplest solution first.** Each layer of complexity must earn its place through measurable eval improvement.
- **Retrieval as a tool.** Retrieval logic is never embedded in the prompt — it is exposed as an agent tool.
- **Everything is swappable.** Model, embedding, retrieval, and orchestration layers each sit behind a clean interface.
- **MCP for integrations.** All external tool integrations use the Model Context Protocol for portability.
- **Evals are regression tests.** Every shipped capability has a corresponding eval that must continue to pass.

---

## Resources

- [Anthropic Engineering Blog](https://www.anthropic.com/engineering)
- [Building Effective Agents](https://www.anthropic.com/research/building-effective-agents)
- [Claude Code Quickstart](https://code.claude.com/docs/en/quickstart)
- [LangGraph Docs](https://langchain-ai.github.io/langgraph/)
- [RAGAS](https://github.com/explodinggradients/ragas)
- [Model Context Protocol](https://modelcontextprotocol.io)

---

## License

Apache 2.0
