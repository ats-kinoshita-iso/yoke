# Plan 006 — Hybrid Retrieval Module

**Status:** Draft
**Depends on:** 005-document-ingestion-pipeline (complete)

## Goal

Implement a hybrid retrieval module that combines dense (embedding) and sparse
(BM25) search using Reciprocal Rank Fusion, returning ranked chunks from the
ingested corpus. This is the first retrieval layer the agent will use as a tool.

## Inputs

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query` | `str` | — | Natural language query |
| `db_path` | `Path` | — | Path to the SQLite database |
| `bm25_path` | `Path` | — | Path to the `.bm25.json` index |
| `k` | `int` | `10` | Number of final results to return |

## Output

`list[RetrievalResult]` — sorted by RRF score descending.

```python
class RetrievalResult(BaseModel):
    chunk_id: int            # chunks.id in SQLite
    chunk_text: str
    context_summary: str
    source_file: str
    page_numbers: list[int]
    rrf_score: float
    dense_rank: int | None   # None if chunk wasn't in dense top-50
    sparse_rank: int | None  # None if chunk wasn't in sparse top-50
```

## Processing Steps

1. **Embed the query** — call the same OpenAI `text-embedding-3-small` model via
   `embed_texts()` from `yoke.ingestion.embedding`.
2. **Dense search** — load all chunk embeddings from SQLite, compute cosine
   similarity against the query embedding (numpy dot product on L2-normalized
   vectors), return top 50 `(chunk_id, score)` pairs.
3. **Sparse search** — load the BM25 index via `load_bm25_index()`, tokenize
   the query with the same `text.lower().split()` tokenizer, call
   `bm25.get_scores(tokenized_query)`, return top 50 `(chunk_id, score)` pairs.
   Chunk position `i` in BM25 maps to the `i`-th row from
   `SELECT id FROM chunks ORDER BY doc_id, chunk_index`.
4. **RRF merge** — for each chunk appearing in either list, compute:
   `rrf_score = Σ 1 / (k_rrf + rank)` where `k_rrf = 60` and rank is 1-based
   position in the respective list. Chunks absent from a list contribute 0 for
   that list.
5. **Return top k** — sort by `rrf_score` desc, fetch chunk metadata for the
   top k, return `list[RetrievalResult]`.

## Files to Create / Modify

### New files

| File | Purpose |
|------|---------|
| `src/yoke/retrieval/__init__.py` | Re-export `retrieve` and `RetrievalResult` |
| `src/yoke/retrieval/models.py` | `RetrievalResult` Pydantic model |
| `src/yoke/retrieval/dense.py` | `dense_search(query_embedding, db_path, top_n=50) → list[(chunk_id, score)]` |
| `src/yoke/retrieval/sparse.py` | `sparse_search(query, bm25_path, db_path, top_n=50) → list[(chunk_id, score)]` |
| `src/yoke/retrieval/fusion.py` | `rrf_merge(dense_results, sparse_results, k_rrf=60) → list[(chunk_id, rrf_score, dense_rank, sparse_rank)]` |
| `src/yoke/retrieval/hybrid.py` | `retrieve(query, db_path, bm25_path, k=10) → list[RetrievalResult]` — orchestrator |
| `evals/phase2_retrieval_eval.py` | Retrieval quality eval (must exist before implementation) |
| `tests/test_retrieval.py` | Unit tests for fusion logic + integration test |

### Modified files

| File | Change |
|------|--------|
| `src/yoke/retrieval/__init__.py` | Replace empty file with public API exports |

## Architectural Decisions

### 1. Load all embeddings into memory for dense search

**Decision:** Fetch all embeddings from SQLite into a numpy matrix and compute
cosine similarity in one vectorized operation.

**Trade-off:** At 1000 chunks × 1536 dims × 4 bytes ≈ 6 MB, this fits
comfortably in memory and is faster than row-by-row comparison. Won't scale
past ~100k chunks, but that's far beyond current scope. When we migrate to
PostgreSQL + pgvector, this becomes a single SQL query with an index.

**Alternative rejected:** Building a FAISS index — unnecessary complexity for
the current corpus size.

### 2. BM25 chunk-ID mapping via ordered query

**Decision:** The BM25 index is built over chunks in `(doc_id, chunk_index)`
order. To map position `i` back to `chunks.id`, fetch IDs in the same order:
`SELECT id FROM chunks ORDER BY doc_id, chunk_index`.

**Trade-off:** Requires this ordering invariant to hold. It does today because
`build_bm25_index` uses this exact ORDER BY. Adding a comment to both
locations to couple them explicitly.

### 3. Synchronous API (not async)

**Decision:** The `retrieve()` function is synchronous. The only async
dependency is embedding, which already has a sync wrapper (`embed_texts`).

**Rationale:** The agent orchestrator (LangGraph) will call retrieval as a tool.
Tool functions can be sync — LangGraph handles the event loop. Keeps the module
simple and testable without async fixtures.

### 4. No caching of loaded embeddings / BM25 index

**Decision:** Each call to `retrieve()` loads the embedding matrix and BM25
index from disk.

**Rationale:** Simplicity first. The agent will typically do 1–3 retrieval calls
per question. Loading 6 MB from SQLite + a JSON file is ~50ms — negligible
vs. LLM latency. Caching can be added later if profiling shows it matters.

### 5. No reranking

**Decision:** Per the requirements, no cross-encoder reranking. RRF output is
the final ranking.

**Rationale:** Add complexity only when evals show it's needed.

## Eval Design (phase2_retrieval_eval.py)

The eval must exist and pass before the module is considered complete.

### Test corpus

Use the existing math corpus (`tests/fixtures/docs-math/lee_ch1.txt`) ingested
into a temp SQLite DB + BM25 index via the pipeline.

### Eval metrics

| Metric | Description | Target |
|--------|-------------|--------|
| **Recall@10** | For each eval query, do ≥1 of the expected chunks appear in top 10? | ≥ 80% of queries |
| **MRR (Mean Reciprocal Rank)** | Average of 1/rank of first relevant result | ≥ 0.5 |
| **Fusion lift** | Recall@10 of hybrid ≥ max(dense-only, sparse-only) | Hybrid ≥ best single |

### Eval queries (≥5)

Hand-written queries covering:

1. **Exact term match** — favours BM25 (e.g. "definition of a topological space")
2. **Semantic paraphrase** — favours dense (e.g. "what makes a set open?" when
   the text says "a subset U is open if…")
3. **Multi-hop** — needs both (e.g. "relationship between compactness and
   closed subsets")
4. **Notation-heavy** — stress test (e.g. "Hausdorff condition T2")
5. **Negative / out-of-scope** — should return low-confidence results (e.g.
   "explain quantum entanglement")

Each query has a list of expected chunk substrings that should appear in the
retrieved results.

## Acceptance Criteria

1. `uv run pytest tests/test_retrieval.py` passes — unit tests for RRF math
   and integration test for end-to-end retrieval.
2. `uv run pytest evals/phase2_retrieval_eval.py` passes — all metric targets
   met.
3. `from yoke.retrieval import retrieve, RetrievalResult` works as the public
   API.
4. No new dependencies required (numpy, rank-bm25, openai already in
   pyproject.toml).
5. Type annotations on all function signatures; Pydantic model for output.

## Implementation Order

1. Write `evals/phase2_retrieval_eval.py` (skeleton with queries + metrics,
   will fail until retrieval exists)
2. `src/yoke/retrieval/models.py` — `RetrievalResult`
3. `src/yoke/retrieval/fusion.py` — pure function, easy to unit test
4. `src/yoke/retrieval/dense.py`
5. `src/yoke/retrieval/sparse.py`
6. `src/yoke/retrieval/hybrid.py` — wire together
7. `src/yoke/retrieval/__init__.py` — public exports
8. `tests/test_retrieval.py` — unit + integration tests
9. Run evals, adjust if needed
