# Plan 007: Phase 1 Query Pipeline (RAG Generation + Comparison Eval)

**Status:** Draft
**Depends on:** 006 (Hybrid Retrieval)

## Goal

Create `src/yoke/pipeline.py` — a reusable query pipeline that ties hybrid
retrieval to LLM generation with source citations. Update the eval suite so
Phase 0 (baseline) and Phase 1 (retrieval pipeline) scores are produced side
by side for direct comparison.

---

## 1. Files to Create or Modify

| File | Action | Purpose |
|------|--------|---------|
| `src/yoke/pipeline.py` | **Create** | Query pipeline: retrieve → format context → generate answer with citations |
| `evals/phase1_pipeline_eval.py` | **Create** | Eval: run both pipelines on the same QA pairs, produce comparison table |
| `src/yoke/query.py` | **Modify** | Refactor to use `pipeline.py` instead of inline generation logic |
| `pyproject.toml` | **Modify** | Add `yoke-query` script entry point |

### Not changing
- Retrieval modules (`src/yoke/retrieval/`) — already complete and tested
- Ingestion modules — no changes needed
- Phase 0/Phase 2 eval files — existing evals remain untouched; the new comparison eval reads their saved results

---

## 2. Design: `src/yoke/pipeline.py`

### Public API

```python
@dataclass
class PipelineResult:
    answer: str
    sources: list[RetrievalResult]   # The chunks used
    retrieval_timings: RetrievalTimings
    generation_ms: float

def query(
    question: str,
    db_path: Path,
    bm25_path: Path,
    *,
    k: int = 10,
    model: str = "claude-sonnet-4-20250514",
) -> PipelineResult:
    """End-to-end: retrieve → format → generate → return with sources."""
```

### Context formatting

Each chunk becomes a numbered block:

```
[1] Source: lee_ch1.txt, pages 23-24
<chunk text>

[2] Source: lee_ch1.txt, pages 25
<chunk text>
...
```

Numbering lets the model cite `[1]`, `[2]`, etc. in its answer.

### Generation prompt

**System:**
```
You are a precise question-answering assistant. Answer ONLY from the
provided context chunks. Follow these rules:

1. Base every claim on a specific chunk. Cite chunks by number, e.g. [1], [3].
2. If multiple chunks support a claim, cite all of them.
3. If the context does not contain enough information, respond with:
   "I don't have enough information to answer this."
4. Do not use knowledge outside the provided context.
```

**User message:**
```
Context:
{formatted_chunks}

Question: {question}
```

### Design decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Return type | Pydantic `PipelineResult` (or dataclass) | Follows project convention; enables structured eval inspection |
| Chunk formatting | Numbered `[1]...[N]` with source + pages | Enables citation tracking; cleaner than file paths inline |
| Model client | Direct `anthropic.Anthropic()` | `query.py` already uses this; consistent; avoids unnecessary abstraction |
| Temperature | 0 | Deterministic for evals |
| Max tokens | 1024 | Matches existing `query.py` |
| k default | 10 | Task says "top 10 chunks" |

### Why not use `ModelClient` abstraction?

The `ModelClient` protocol in `src/yoke/models.py` is async and designed for
the agent orchestrator. The pipeline is sync (matching retrieval). Using
`anthropic.Anthropic()` directly keeps it simple. When the agent layer
arrives, it will call `pipeline.query()` as a tool — the abstraction
boundary is at the agent level, not inside the pipeline.

---

## 3. Design: Eval Comparison

### `evals/phase1_pipeline_eval.py`

**Purpose:** Run the same QA pairs through both Phase 0 (baseline) and
Phase 1 (retrieval pipeline), score with the same LLM judge, and produce a
side-by-side comparison.

**QA pairs:** Reuse the 11 QA pairs from `phase0_math_eval.py` (4 direct,
3 cross-reference, 2 reasoning, 2 unanswerable). These are the canonical
eval set for the math corpus.

**Eval flow:**
1. Ingest `lee_ch1.txt` into a temp DB (reuse `_ingest_fixture` from phase2 eval)
2. For each QA pair:
   - Run Phase 0: `baseline.ask(question, docs_dir)`
   - Run Phase 1: `pipeline.query(question, db_path, bm25_path)`
   - Judge both answers with the same judge (faithfulness, relevance, precision)
3. Produce comparison table and write results JSON

**Assertions (Phase 1 must beat or match Phase 0):**
- Average faithfulness: Phase 1 ≥ Phase 0
- Average relevance: Phase 1 ≥ Phase 0 − 0.5 (margin for eval noise)
- Citation accuracy: ≥ 80% of cited chunk numbers actually support the claim
  (new metric — ensures citations aren't hallucinated)

**New metric — citation grounding:**
For each answer, parse `[N]` citation references. For each cited chunk N,
check whether the chunk text actually supports the adjacent claim. This is
judged by the LLM judge alongside faithfulness/relevance. This prevents the
model from "citation washing" — adding references that don't actually
support the text.

**Comparison table output:**

```
Phase 0 vs Phase 1 Comparison (Lee Ch.1, 11 QA pairs)
================================================================
                          Phase 0    Phase 1    Delta
Avg faithfulness:           4.20       4.50     +0.30
Avg relevance:              4.00       4.30     +0.30
Avg precision:              3.80       4.10     +0.30
Unanswerable correct:        2/2        2/2       —
Citation grounding:           —        92%        —
================================================================
```

**Results file:** `evals/results/phase1_pipeline.json` containing:
- Per-question scores for both pipelines
- Aggregate comparison metrics
- Citation grounding breakdown

---

## 4. Modifications to `src/yoke/query.py`

Refactor the CLI to delegate to `pipeline.query()`:

```python
# Before: inline _generate_answer() with own prompt
# After:  result = pipeline.query(question, db_path, bm25_path, k=k, model=model)
```

Keep the CLI flags (`--no-generate`, `--db`, `--bm25`, `-k`, `--model`).
Keep the diagnostics output. The CLI becomes a thin wrapper around the
pipeline module.

---

## 5. Acceptance Criteria

1. `from yoke.pipeline import query, PipelineResult` works
2. `pipeline.query(question, db_path, bm25_path)` returns a `PipelineResult`
   with a non-empty answer and a list of sources
3. The answer cites chunk numbers `[N]` that correspond to actual retrieved chunks
4. For unanswerable questions, the answer contains "I don't have enough
   information"
5. `uv run pytest evals/phase1_pipeline_eval.py -v` passes:
   - Phase 1 faithfulness ≥ Phase 0 faithfulness
   - Phase 1 relevance ≥ Phase 0 relevance − 0.5
   - Citation grounding ≥ 80%
6. `uv run python -m yoke.query "What is a topological manifold?" --db data/yoke.db`
   still works (CLI unchanged from user perspective)
7. Comparison table is printed to stdout during eval run

---

## 6. Eval-First Checklist

Per project convention ("every new feature requires an eval before
implementation"), the implementation order is:

1. **Write the eval first** (`evals/phase1_pipeline_eval.py`)
   - Define QA pairs (reuse from phase0_math_eval)
   - Define judge with citation grounding metric
   - Define comparison table format
   - Import `pipeline.query` — this will fail until implementation exists
2. **Write `src/yoke/pipeline.py`** — make the eval pass
3. **Refactor `src/yoke/query.py`** — delegate to pipeline
4. **Run full eval suite** — verify no regressions

---

## 7. Risks and Mitigations

| Risk | Mitigation |
|------|-----------|
| LLM ignores citation instructions | System prompt is explicit; eval measures citation grounding directly |
| Phase 1 scores lower than Phase 0 on small corpus | 0.5 margin on relevance; Phase 0 sees entire corpus (~80KB) which fits in context — retrieval shines on larger corpora |
| Rate limits during eval (both pipelines + judge = 3x calls) | Reuse existing `_throttle()` pattern; run Phase 0 answers first, then Phase 1, to batch by pipeline |
| Citation parsing brittle | Simple regex `\[(\d+)\]` is sufficient; edge cases (e.g., `[1,2]`) handled by splitting on comma |

---

## 8. Future Work (Out of Scope)

- Streaming generation (for API/CLI UX)
- Reranking retrieved chunks before generation
- Multi-turn conversation with memory
- Agent orchestration via LangGraph (Phase 3+)
