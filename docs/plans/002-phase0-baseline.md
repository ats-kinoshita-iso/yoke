# Plan 002: Phase 0 — Naive Baseline + LLM-as-Judge Eval

**Status:** Proposed
**Date:** 2026-04-06

## Goal

Establish a measurable baseline for the KM system by building the simplest
possible "stuff everything into the prompt" approach, then scoring it with an
LLM-as-judge eval. Every future phase (chunking, retrieval, agent) must
demonstrate improvement over these numbers.

## Context

The project scaffolding is complete (Plan 001). All `src/yoke/` sub-packages
exist but are empty. No test documents exist yet. This plan introduces:

- A set of test documents to serve as the knowledge base
- A naive baseline that concatenates all documents into the prompt
- An eval harness that scores answers on faithfulness and relevance

Per CLAUDE.md: "Every new feature requires an eval before implementation."
Phase 0 *is* the eval — it defines the scoring methodology and baseline
numbers that all future work is measured against.

## Files to Create

| File | Purpose |
|---|---|
| `tests/fixtures/docs/` | Directory of 3-5 markdown/text files used as the test knowledge base |
| `tests/fixtures/docs/architecture.md` | Document describing the system architecture |
| `tests/fixtures/docs/getting-started.md` | Document with setup/onboarding instructions |
| `tests/fixtures/docs/api-reference.md` | Document with API endpoint descriptions |
| `tests/fixtures/docs/troubleshooting.md` | Document with common issues and solutions |
| `src/yoke/baseline.py` | `phase0_baseline` module — reads docs, builds prompt, calls Claude |
| `evals/phase0_eval.py` | Eval script — 10 QA pairs, LLM-as-judge scoring, summary output |
| `evals/conftest.py` | Shared eval fixtures (e.g., test doc path, API client) |

## Files to Modify

| File | Change |
|---|---|
| `pyproject.toml` | Add `[project.scripts]` entry for `yoke-baseline = "yoke.baseline:main"` so the baseline can be invoked as a CLI command |

## Design

### `src/yoke/baseline.py`

```
ask(question: str, docs_dir: Path) -> str
```

1. Glob `docs_dir` for `*.md` and `*.txt` files
2. Read and concatenate contents, separated by `\n---\n` with filename headers
3. Build a single `messages` call to Claude:
   - System prompt: "Answer the question using only the provided context.
     If the context doesn't contain the answer, say so."
   - User message: `"Context:\n{context}\n\nQuestion: {question}"`
4. Return the assistant response text

**Model:** `claude-sonnet-4-20250514` (cost-effective for baseline; eval judge
uses the same model for consistency).

**No streaming** — the eval needs the full response as a string.

A `main()` function wraps this for CLI use:
```
uv run yoke-baseline --docs-dir tests/fixtures/docs "How do I set up the API?"
```

### `evals/phase0_eval.py`

**Test documents:** 3-5 synthetic markdown files that cover architecture,
setup, API reference, and troubleshooting. These are purpose-built for the
eval — they contain known facts that the QA pairs reference. Keeping them
small (~200-400 lines total) ensures the full context fits in a single
prompt with room to spare.

**QA pairs** (10 total, stored as a list of dicts in the eval file):

The pairs should span these categories:
- **Direct lookup** (3): answer is stated verbatim in one document
- **Cross-document synthesis** (3): answer requires combining info from 2+ docs
- **Reasoning over context** (2): answer requires inference from stated facts
- **Unanswerable** (2): answer is NOT in the documents — model should say so

Each pair is a dict:
```python
{"question": str, "expected_answer": str, "category": str}
```

**LLM-as-judge scoring:**

For each QA pair, send a judge prompt to Claude with:
- The original question
- The expected answer
- The model's actual answer
- The source context (so the judge can verify faithfulness)

The judge returns two integer scores:
- **Faithfulness (1-5):** Is the answer supported by the provided context?
  5 = fully grounded, 1 = hallucinated
- **Relevance (1-5):** Does the answer address the question asked?
  5 = directly answers, 1 = off-topic

Use structured output (`response_model` or tool-use JSON) to extract scores
reliably. Parse with a Pydantic model:

```python
class JudgeScore(BaseModel):
    faithfulness: int = Field(ge=1, le=5)
    relevance: int = Field(ge=1, le=5)
    reasoning: str
```

**Output format:**

```
Phase 0 Baseline Eval Results
=============================
Q1 [direct]   faithfulness=5 relevance=5  "What port does the API run on?"
Q2 [cross]    faithfulness=4 relevance=4  "How do I configure retrieval..."
...
------------------------------
Average faithfulness: 4.2
Average relevance:    4.1
Total questions:      10
Unanswerable correct: 2/2
```

Results are also written to `evals/results/phase0_baseline.json` for
programmatic comparison in future phases.

### Test Documents Strategy

The test documents are **synthetic** — written specifically for this eval,
not pulled from a real codebase. This is intentional:

- We control exactly what facts are present, so expected answers are unambiguous
- We can design cross-document dependencies deliberately
- Documents stay small enough to fit in a single prompt (~4K tokens total)
- No risk of eval contamination from model training data

Each document should be 50-100 lines of realistic-looking technical
documentation with specific, verifiable facts (port numbers, config keys,
error codes, etc.).

## Acceptance Criteria

1. `uv run yoke-baseline --docs-dir tests/fixtures/docs "What is the default port?"` returns a coherent answer
2. `uv run pytest evals/phase0_eval.py -v` runs all 10 QA pairs and prints the summary
3. Average faithfulness score >= 3.0 (sanity check — naive approach should do OK on small context)
4. Average relevance score >= 3.0
5. Results JSON is written to `evals/results/phase0_baseline.json`
6. The eval is deterministic enough to re-run: uses `temperature=0` for both baseline and judge calls

## Evals Before Implementation

This task **is** the eval. The eval file (`phase0_eval.py`) is the primary
deliverable. However, the eval itself needs validation:

- **Judge calibration check:** Include 2 "control" pairs where the model
  answer is intentionally wrong (hardcoded bad answers). The judge must score
  these below 3 on faithfulness. If it doesn't, the judge prompt needs tuning
  before the eval is trustworthy.
- **Unanswerable detection:** The 2 unanswerable questions must score >= 4
  on faithfulness when the model correctly declines to answer.

## Architectural Decisions

### 1. Standalone script vs. pytest test

**Decision:** The eval is a pytest-compatible module in `evals/` (already in
`testpaths` per `pyproject.toml`), but also runnable standalone via
`python -m evals.phase0_eval`.

**Trade-off:** pytest integration means `uv run pytest evals/` runs all evals
alongside tests. Standalone mode is useful for quick iteration. Both modes
share the same code.

### 2. Judge model choice: same model vs. stronger model

**Decision:** Use the same model (`claude-sonnet-4-20250514`) for both baseline
and judge in Phase 0.

**Trade-off:** Using a stronger model (Opus) as judge would be more reliable
but costs more per run. Since Phase 0 is about establishing the methodology,
not getting perfect scores, Sonnet is sufficient. We can upgrade the judge
in later phases if score variance is too high.

### 3. Structured output: tool-use vs. JSON mode

**Decision:** Use Anthropic's tool-use to extract `JudgeScore` as structured
JSON from the judge response.

**Trade-off:** Tool-use gives guaranteed schema compliance. JSON mode with
`json.loads()` is simpler but fragile if the model wraps the JSON in markdown
fences or adds commentary. Tool-use is the more robust path.

### 4. Test docs location: `tests/fixtures/` vs. `evals/fixtures/`

**Decision:** `tests/fixtures/docs/` — shared between unit tests and evals.

**Trade-off:** Evals and tests may eventually need different fixtures, but
for Phase 0 there's no reason to duplicate. A single source of truth for
test documents avoids drift.

### 5. Baseline in `src/yoke/` vs. top-level script

**Decision:** `src/yoke/baseline.py` — part of the package, not a loose script.

**Trade-off:** Putting it in the package means it's importable by evals,
tests, and future phases. A top-level `phase0_baseline.py` would be simpler
but becomes an orphan as the project grows. The baseline is a real piece of
the system — it deserves to live in `src/`.

## Implementation Order

1. Write the test documents in `tests/fixtures/docs/` (4 files)
2. Implement `src/yoke/baseline.py` with `ask()` and `main()`
3. Add the `[project.scripts]` entry to `pyproject.toml`
4. Write `evals/phase0_eval.py` with 10 QA pairs + 2 judge calibration controls
5. Create `evals/results/` directory
6. Run the eval, verify acceptance criteria
7. Commit baseline scores to `evals/results/phase0_baseline.json`
8. Commit: `feat: add phase 0 naive baseline and LLM-as-judge eval`

## Cost Estimate

Per eval run (12 questions including 2 calibration controls):
- 12 baseline calls: ~4K input tokens + ~500 output tokens each ≈ 54K tokens
- 12 judge calls: ~5K input tokens + ~200 output tokens each ≈ 62K tokens
- Total: ~116K tokens per run ≈ $0.50 with Sonnet pricing

Cheap enough to run frequently during development.
