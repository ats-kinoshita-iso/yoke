# Plan 005: Document Ingestion Pipeline (Phase 1)

**Status:** Proposed
**Date:** 2026-04-07

## Goal

Build the document ingestion pipeline that transforms a directory of PDF files
into a SQLite database with enriched chunks and embeddings, plus a BM25 index
file — ready for hybrid retrieval. This is the transition from Phase 0's
"stuff everything in context" to Phase 1's "retrieve the right chunks."

## Context

Phase 0 proved the system works when the full document fits in context (~19K
tokens for Lee Ch.1). But a full textbook (726 pages, ~300K+ tokens) or a
directory of multiple PDFs will not fit. We need:

1. **Chunking** — split documents into retrieval-sized pieces
2. **Contextual enrichment** — prepend a situating summary to each chunk
   (Anthropic's Contextual Retrieval pattern) so embeddings capture meaning
   in context, not in isolation
3. **Dense embeddings** — for semantic similarity search
4. **Sparse index (BM25)** — for keyword/term matching
5. **Persistent storage** — SQLite for structured data + embeddings, pickle
   for the BM25 index

The existing `src/yoke/extract.py` handles PDF → text. This plan builds the
pipeline from text → indexed storage.

### Key dependencies already available

- `pymupdf` — PDF extraction (already in `extract.py`)
- `httpx` — for Ollama API calls (already in `models.py`)
- `pydantic` — data models
- `src/yoke/models.py` — `ModelClient` protocol with `OllamaClient`
- `src/yoke/config.py` — `YokeSettings` with `YOKE_SUMMARY_MODEL`

### New dependencies needed

- `openai` — for `text-embedding-3-small` embeddings
- `numpy` — for embedding arrays
- `rank-bm25` — BM25 implementation

## Design

### Pipeline overview

```
PDF directory
    │
    ▼
┌──────────────┐
│  1. Extract  │  extract.py (existing)
│  PDF → text  │  Returns: list[PageText(page_num, text)]
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  2. Chunk    │  chunking.py (new)
│  text→chunks │  Returns: list[Chunk]
└──────┬───────┘
       │
       ▼
┌──────────────────┐
│  3. Enrich       │  enrichment.py (new)
│  chunk→enriched  │  Uses YOKE_SUMMARY_MODEL (ollama/gemma4:e2b)
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│  4. Embed        │  embedding.py (new)
│  enriched→vector │  Uses text-embedding-3-small
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│  5. Index & Store│  store.py (new)
│  SQLite + BM25   │  Persists everything
└──────────────────┘
```

### Data models (`src/yoke/ingestion/models.py`)

```python
from pydantic import BaseModel

class PageText(BaseModel):
    """A single page of extracted text."""
    page_number: int
    text: str

class Chunk(BaseModel):
    """A chunk of text with metadata, before enrichment."""
    chunk_index: int
    text: str
    page_numbers: list[int]  # pages this chunk spans
    source_file: str

class EnrichedChunk(BaseModel):
    """A chunk after contextual enrichment."""
    chunk_index: int
    chunk_text: str           # original chunk text
    context_summary: str      # 2-3 sentence situating summary
    enriched_text: str        # summary prepended to chunk text
    page_numbers: list[int]
    source_file: str

class DocumentRecord(BaseModel):
    """A fully processed document ready for storage."""
    filename: str
    full_text: str
    chunks: list[EnrichedChunk]
    embeddings: list[list[float]]  # one embedding per chunk
```

### Step 1: Extract (`src/yoke/extract.py` — modify)

Add a function that returns structured page-level text:

```python
def extract_pdf_by_pages(pdf_path: Path) -> list[PageText]:
    """Extract all pages from a PDF, returning per-page text with page numbers."""
```

The existing `extract_pdf_pages()` returns a single concatenated string. The
new function preserves page boundaries so chunks can track which pages they
span.

### Step 2: Chunk (`src/yoke/ingestion/chunking.py`)

Recursive character text splitting with overlap:

```python
def chunk_document(
    pages: list[PageText],
    source_file: str,
    *,
    target_size: int = 2000,    # characters (~512 tokens)
    overlap: int = 200,          # characters (~50 tokens)
) -> list[Chunk]:
    """Split page text into overlapping chunks with metadata."""
```

**Algorithm:**
1. Concatenate all page text, tracking character-offset → page-number mapping
2. Try splitting at paragraph breaks (`\n\n`) first
3. If a paragraph exceeds `target_size`, split at sentence boundaries (`. `)
4. If a sentence exceeds `target_size`, split at word boundaries (` `)
5. Apply overlap by prepending the last `overlap` characters from the
   previous chunk
6. Assign `page_numbers` by looking up which pages the chunk's character
   range spans

**Why not use LangChain's `RecursiveCharacterTextSplitter`?**
CLAUDE.md says "Do not use LangChain's high-level chains." The text splitter
is technically not a chain, but building our own is simple (~50 lines), gives
us full control over the page-number tracking, and avoids pulling in
`langchain-text-splitters` as a dependency. The algorithm is well-understood.

### Step 3: Enrich (`src/yoke/ingestion/enrichment.py`)

Contextual Retrieval pattern — give the LLM the full document + one chunk,
ask for a situating summary:

```python
async def enrich_chunks(
    full_text: str,
    chunks: list[Chunk],
    client: ModelClient,
    *,
    max_concurrent: int = 4,
) -> list[EnrichedChunk]:
    """Add contextual summaries to each chunk using an LLM."""
```

**Prompt template:**
```
<document>
{full_text}
</document>

Here is the chunk we want to situate within the whole document:
<chunk>
{chunk_text}
</chunk>

Write a short (2-3 sentence) summary that explains what this chunk covers
and how it relates to the rest of the document. Focus on context that would
help someone searching for this information.
```

**Model:** `YOKE_SUMMARY_MODEL` (default `ollama/gemma4:e2b`) via
`OllamaClient`. This is a local model — cheap and fast for bulk operations.

**Concurrency:** Use `asyncio.Semaphore(max_concurrent)` to limit parallel
requests to the local Ollama instance. Default 4 is conservative; can be
tuned based on hardware.

**Full document in every call:** Yes, this sends the full document text with
every chunk. For a 30-page chapter (~19K tokens), this is ~19K × N_chunks
total input tokens. With a local model this is free in dollar terms but slow.
For a 30-page chapter with ~40 chunks, expect ~5-10 minutes on modest GPU
hardware. This is acceptable for batch ingestion.

**Enriched text format:**
```
[Context: {summary}]

{original_chunk_text}
```

### Step 4: Embed (`src/yoke/ingestion/embedding.py`)

```python
async def embed_chunks(
    chunks: list[EnrichedChunk],
    *,
    model: str = "text-embedding-3-small",
    batch_size: int = 100,
) -> list[list[float]]:
    """Generate embeddings for enriched chunk text using OpenAI API."""
```

**Model:** `text-embedding-3-small` (1536 dimensions, $0.02/1M tokens).
For 40 chunks of ~2500 chars each → ~25K tokens → ~$0.0005 per document.

**Batching:** OpenAI's embedding API accepts up to 2048 inputs per request.
We batch at 100 to stay well under limits and provide progress granularity.

**Client:** Use `openai.AsyncOpenAI()` directly. The API key is read from
`OPENAI_API_KEY` env var by the SDK. No need for a `ModelClient` wrapper —
embeddings have a different interface than text completion.

### Step 5: Index & Store (`src/yoke/ingestion/store.py`)

#### SQLite schema

```sql
CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filename TEXT NOT NULL UNIQUE,
    full_text TEXT NOT NULL,
    ingested_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id INTEGER NOT NULL REFERENCES documents(id),
    chunk_index INTEGER NOT NULL,
    chunk_text TEXT NOT NULL,
    context_summary TEXT NOT NULL,
    enriched_text TEXT NOT NULL,
    embedding BLOB NOT NULL,
    page_numbers TEXT NOT NULL,  -- JSON array, e.g. "[1, 2]"
    UNIQUE(doc_id, chunk_index)
);
```

**Why SQLite, not PostgreSQL/pgvector?**
The architecture doc says PostgreSQL + pgvector for production. But for
Phase 1 we want:
- Zero infrastructure setup (no Docker, no PostgreSQL server)
- Single-file database, easy to inspect and version
- Focus on the ingestion pipeline, not the retrieval infrastructure

SQLite is the right choice for building and testing the ingestion pipeline.
Migration to PostgreSQL + pgvector happens when we build the retrieval layer
with proper vector search. The schema is intentionally compatible — the same
tables can be created in PostgreSQL with `pgvector` adding a `vector(1536)`
column type instead of `BLOB`.

**Embedding storage:** Stored as `BLOB` using `numpy.ndarray.tobytes()`.
Loaded with `numpy.frombuffer(blob, dtype=np.float32)`. This is compact and
fast for SQLite. When we migrate to pgvector, these become native vector
columns.

#### BM25 index

```python
def build_bm25_index(chunks: list[EnrichedChunk]) -> BM25Okapi:
    """Build a BM25 index over raw chunk text (not enriched)."""
```

- Uses `rank-bm25.BM25Okapi`
- Tokenization: simple whitespace + lowercase (sufficient for English prose)
- Indexed on `chunk_text` (raw), NOT `enriched_text` — the contextual summary
  would dilute keyword signal
- Serialized with `pickle.dump()` alongside the SQLite database

#### Pipeline orchestrator (`src/yoke/ingestion/pipeline.py`)

```python
async def ingest_directory(
    pdf_dir: Path,
    db_path: Path,
    *,
    summary_model: str = "ollama/gemma4:e2b",
    embedding_model: str = "text-embedding-3-small",
) -> IngestResult:
    """Ingest all PDFs in a directory into a SQLite database + BM25 index."""
```

**`IngestResult`:**
```python
class IngestResult(BaseModel):
    documents_processed: int
    total_chunks: int
    db_path: Path
    bm25_path: Path
    errors: list[str]
```

**Flow:**
1. Glob `pdf_dir` for `*.pdf`
2. For each PDF:
   a. Extract pages → `list[PageText]`
   b. Chunk → `list[Chunk]`
   c. Enrich → `list[EnrichedChunk]` (async, rate-limited)
   d. Embed → `list[list[float]]` (async, batched)
   e. Store document + chunks in SQLite
3. Build BM25 index over all chunks across all documents
4. Save BM25 index as `{db_path}.bm25.pkl`
5. Return `IngestResult`

**Error handling:** If a single PDF fails (corrupt, empty, extraction error),
log the error, skip it, continue with the rest. Report failures in
`IngestResult.errors`.

### CLI entry point

Add a script to `pyproject.toml`:

```toml
[project.scripts]
yoke-baseline = "yoke.baseline:main"
yoke-ingest = "yoke.ingestion.pipeline:main"
```

Usage:
```bash
uv run yoke-ingest --pdf-dir /path/to/pdfs --db-path data/yoke.db
```

## Files to Create

| File | Purpose |
|---|---|
| `src/yoke/ingestion/models.py` | Pydantic data models: `PageText`, `Chunk`, `EnrichedChunk`, `DocumentRecord`, `IngestResult` |
| `src/yoke/ingestion/chunking.py` | Recursive character text splitting with page tracking |
| `src/yoke/ingestion/enrichment.py` | Contextual Retrieval: LLM-generated chunk summaries |
| `src/yoke/ingestion/embedding.py` | OpenAI embedding generation with batching |
| `src/yoke/ingestion/store.py` | SQLite storage + BM25 index building |
| `src/yoke/ingestion/pipeline.py` | End-to-end orchestrator + CLI entry point |
| `tests/test_chunking.py` | Unit tests for chunking logic |
| `tests/test_store.py` | Unit tests for SQLite storage and BM25 index |
| `evals/phase1_ingestion_eval.py` | Ingestion quality eval (see Evals section) |

## Files to Modify

| File | Change |
|---|---|
| `src/yoke/extract.py` | Add `extract_pdf_by_pages()` returning `list[PageText]` |
| `src/yoke/ingestion/__init__.py` | Export `ingest_directory` |
| `pyproject.toml` | Add `openai`, `numpy`, `rank-bm25` dependencies; add `yoke-ingest` script |
| `.env.example` | Add `OPENAI_API_KEY` |

## Evals Before Implementation

Per project convention: evals first, then implement.

### Eval 1: Chunking quality (`evals/phase1_ingestion_eval.py`)

Chunking is the foundation — bad chunks cascade into bad retrieval. The eval
verifies chunking quality using the Lee Ch.1 fixture (already available at
`tests/fixtures/docs-math/lee_ch1.txt`).

**Metrics:**

1. **Chunk size distribution** — Are chunks within the target range?
   - Acceptance: ≥90% of chunks between 1500-2500 characters
   - Report: min, max, mean, std, histogram

2. **Boundary quality** — Do chunks break at sensible points?
   - Measure: % of chunks that start at a paragraph boundary
   - Acceptance: ≥70% start at paragraph or sentence boundaries
   - Anti-pattern: chunks that split mid-sentence

3. **Page attribution accuracy** — Do `page_numbers` reflect actual content?
   - Manually verify 5 chunks: does the text actually appear on those pages?
   - (This is a spot-check, not automatable)

4. **Overlap correctness** — Does overlap work as specified?
   - Verify: last N characters of chunk[i] == first N characters of chunk[i+1]
   - Acceptance: 100% of adjacent chunk pairs have correct overlap

5. **Coverage** — Does chunking cover the full document?
   - Concatenate all chunks (minus overlap) and compare to original text
   - Acceptance: ≥99% character coverage (allowing for whitespace normalization)

### Eval 2: Enrichment quality (manual spot-check + LLM judge)

Enrichment quality is harder to measure automatically. After implementation:

1. Sample 10 chunks from Lee Ch.1
2. Run enrichment with the summary model
3. LLM-as-judge scores each summary on:
   - **Accuracy (1-5):** Does the summary correctly describe the chunk's content?
   - **Situating value (1-5):** Does it add context that wouldn't be obvious
     from the chunk alone?
   - **Conciseness (1-5):** Is it 2-3 sentences, no more?

This eval is deferred to after the enrichment module is built, because it
requires a running Ollama instance. The chunking eval can run without any
external services.

### Eval 3: Retrieval regression (Phase 0 parity)

After ingestion is complete, verify that the Phase 0 eval scores don't
regress. The ingestion pipeline is infrastructure — it should not change
answer quality for questions that already work with context stuffing.

This uses the existing `evals/phase0_math_eval.py` as-is.

## Acceptance Criteria

1. `uv run yoke-ingest --pdf-dir <dir> --db-path data/yoke.db` processes all
   PDFs in the directory and exits successfully
2. The SQLite database contains correct `documents` and `chunks` tables with
   all expected data
3. Each chunk has: `chunk_text`, `context_summary`, `enriched_text`,
   `embedding` (1536-dim float32 blob), `page_numbers` (JSON array)
4. A BM25 index file exists at `data/yoke.db.bm25.pkl` and can be loaded
5. `uv run pytest tests/test_chunking.py tests/test_store.py -v` — all pass
6. Chunking eval passes: ≥90% chunks in target size range, ≥70% at good
   boundaries, 100% overlap correctness, ≥99% coverage
7. Pipeline handles errors gracefully (corrupt PDF → skip + report, don't crash)
8. Re-running ingestion on the same directory is idempotent (upserts, not
   duplicates)

## Architectural Decisions

### 1. SQLite now vs. PostgreSQL + pgvector from the start

**Decision:** SQLite for Phase 1 ingestion; migrate to PostgreSQL when
building the retrieval layer.

**Why:** The ingestion pipeline's job is to produce chunks, embeddings, and
indices. Storage is an output format, not the core logic. SQLite lets us build
and test without infrastructure. The schema is designed for easy migration:
`embedding BLOB` → `embedding vector(1536)`, and the Python code uses raw SQL
(not an ORM), so switching backends means changing the connection string and
a few SQL dialect differences.

**Risk:** We'll need to do the migration work in Phase 1b. But the ingestion
code itself won't change — only the storage backend.

### 2. BM25 as a pickle file vs. in-database sparse vectors

**Decision:** Pickle file alongside the database.

**Why:** `rank-bm25` is a pure-Python library that operates on in-memory
token lists. It doesn't support incremental updates well — you rebuild the
index from scratch. For our use case (batch ingestion, then query), this is
fine. The index is small (~100KB for 1000 chunks) and loads instantly.

**Trade-off:** When we move to PostgreSQL, we can switch to `tsvector` for
native full-text search, which supports incremental updates and doesn't
require a sidecar file. The BM25 pickle is a Phase 1 expedient.

### 3. OpenAI embeddings vs. local embeddings

**Decision:** OpenAI `text-embedding-3-small` via API.

**Why:** It's cheap ($0.02/1M tokens), high quality (MTEB top-10), and
requires no GPU. A 30-page chapter produces ~40 chunks × ~600 tokens =
~24K tokens = $0.0005. The entire textbook would cost ~$0.01.

**Trade-off:** Requires an API key and internet connection during ingestion.
For a self-hosted alternative, we could use `sentence-transformers` with a
local model, but that adds a heavy dependency (~2GB model download) and is
slower without a GPU. OpenAI embeddings are the pragmatic choice for Phase 1.

### 4. Full document in every enrichment call

**Decision:** Send the full document text with every chunk for contextual
enrichment.

**Why:** This is the Anthropic Contextual Retrieval pattern — the model needs
the full document to situate the chunk. For a 30-page chapter with 40 chunks,
this is 40 × 19K = ~760K input tokens to the local Ollama model. Since it's
local, the cost is zero dollars; the cost is time (~5-10 minutes).

**Trade-off:** For very large documents (>100 pages), we may need to truncate
or use a sliding window of surrounding context instead of the full document.
This is a future optimization — the interface (`enrich_chunks(full_text, ...)`)
doesn't need to change.

### 5. Embedding enriched text vs. raw text

**Decision:** Embed the `enriched_text` (summary + chunk). BM25 indexes the
raw `chunk_text`.

**Why:** This is the core insight of Contextual Retrieval. The summary adds
semantic context that improves embedding quality for retrieval. But BM25
should index raw text because the summary introduces LLM-generated words
that could create false keyword matches.

### 6. Custom chunking vs. LangChain text splitter

**Decision:** Custom implementation.

**Why:** CLAUDE.md says "Do not use LangChain's high-level chains." While
the text splitter isn't technically a chain, our chunking has a specific
requirement (page-number tracking) that `RecursiveCharacterTextSplitter`
doesn't support natively. The algorithm is ~50 lines and well-understood.
Building our own avoids a dependency and gives full control.

### 7. Idempotent ingestion

**Decision:** Use `INSERT OR REPLACE` (upsert) keyed on `filename` for
documents. Re-ingesting a file replaces all its chunks.

**Why:** Users will re-run ingestion as they add PDFs to the directory or
after code changes improve extraction/chunking. Duplicate rows would corrupt
retrieval. Upsert semantics mean "the database always reflects the latest
ingestion run."

**Implementation:** Delete existing chunks for a document before inserting
new ones (cascading from `doc_id`). This is simpler than diffing chunks.

## Implementation Order

1. **Write chunking eval** (`evals/phase1_ingestion_eval.py`) — tests chunking
   quality against Lee Ch.1 fixture. Run with empty stubs to verify eval
   structure.

2. **Write unit tests** (`tests/test_chunking.py`, `tests/test_store.py`) —
   test chunking algorithm and SQLite storage in isolation.

3. **Implement data models** (`src/yoke/ingestion/models.py`)

4. **Implement chunking** (`src/yoke/ingestion/chunking.py`) — pure logic,
   no external dependencies. Run chunking eval + unit tests.

5. **Implement storage** (`src/yoke/ingestion/store.py`) — SQLite schema
   creation, document/chunk insertion, BM25 index building. Run storage tests.

6. **Modify `extract.py`** — add `extract_pdf_by_pages()`.

7. **Implement enrichment** (`src/yoke/ingestion/enrichment.py`) — requires
   running Ollama. Test with a small subset first.

8. **Implement embedding** (`src/yoke/ingestion/embedding.py`) — requires
   OpenAI API key.

9. **Implement pipeline** (`src/yoke/ingestion/pipeline.py`) — wire
   everything together, add CLI entry point.

10. **End-to-end test** — ingest Lee Ch.1 PDF, inspect database, verify
    all fields populated correctly.

## Cost Estimate

Per ingestion run (one 30-page chapter, ~40 chunks):

| Step | Model | Tokens | Cost |
|---|---|---|---|
| Enrichment | ollama/gemma4:e2b (local) | ~760K input | $0.00 |
| Embedding | text-embedding-3-small | ~24K | $0.0005 |
| **Total** | | | **~$0.001** |

Per full textbook (726 pages, ~1000 chunks):

| Step | Model | Tokens | Cost |
|---|---|---|---|
| Enrichment | ollama/gemma4:e2b (local) | ~19M input | $0.00 |
| Embedding | text-embedding-3-small | ~600K | $0.012 |
| **Total** | | | **~$0.01** |

Enrichment time is the bottleneck: ~2-5 hours for a full textbook on modest
GPU hardware. This is a batch operation run once per document.

## Future Work (Out of Scope)

- **Hybrid retrieval layer** — querying the database with RRF (Phase 1b)
- **PostgreSQL + pgvector migration** — when retrieval needs vector search
- **Incremental ingestion** — only re-process changed/new files
- **LaTeX-aware chunking** — split at theorem/definition/proof boundaries
- **Multi-modal chunks** — include figure references or extracted images
- **Embedding model selection** — compare against `nomic-embed-text` or
  other local alternatives
