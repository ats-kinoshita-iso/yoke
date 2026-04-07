# Plan 003: Real Corpus — PDF Ingestion + Math-Domain Evals

**Status:** Proposed
**Date:** 2026-04-06

## Goal

Extend the Phase 0 baseline to work with real mathematical textbooks (PDFs),
starting with Chapter 1 of Lee's *Introduction to Smooth Manifolds* (pages
19–49, ~19K tokens extracted). This validates the system against dense,
technical, real-world content and exposes the limitations that motivate
Phase 1 (chunking + retrieval).

## Context

Phase 0 achieved perfect scores (faithfulness 5.0, relevance 5.0) on ~500
lines of synthetic markdown. That was the point — establish the methodology.
Now we stress-test it against content that is:

- **PDF, not markdown** — requires extraction with loss (figures, equations)
- **Dense mathematical prose** — definitions, theorems, proofs with formal notation
- **Cross-referential** — "by Proposition 1.16..." or "see Chapter 4"
- **Larger** — Chapter 1 alone is ~19K tokens; the full book is 726 pages

The user's corpus lives at:
```
C:\Users\akino\OneDrive\Documents\Topics - Groups & Manifolds\
├── lee_smooth_manifolds.pdf          (726 pages, 4.6 MB)
├── Tu_AnIntroductionToManifolds.pdf
├── humphreys.pdf
├── David_S_Dummit_Richard_M_Foote_Abstract_Algeb_230928_225848.pdf
```

**Scope for this plan:** Lee Chapter 1 only (pages 19–49). Full-book and
multi-book work is Phase 1+.

## Extraction Quality Assessment

PyMuPDF (`pymupdf`) extracts readable text from Lee Ch.1. Key observations:

- **Prose**: Clean extraction. Paragraphs, section headers, and narrative flow intact.
- **Inline math**: Partial. Symbols like `∩`, `⊂`, `∈` survive. Ligatures (`ﬁ` → fi) need normalization.
- **Display equations**: Flattened to Unicode text. Readable but not LaTeX — e.g., `f ı φ⁻¹` not `f \circ \varphi^{-1}`.
- **Figures/diagrams**: Lost entirely (Fig 1.1, 1.2, etc.). Captions survive.
- **Cross-references**: Inline text like "Proposition 1.16" survives. Hyperlinks do not.

This quality level is sufficient for definition/theorem/proof lookup and
conceptual questions. It is NOT sufficient for questions about diagrams or
exact typeset formulas.

## Design

### Step 1: PDF extraction module (`src/yoke/extract.py`)

```python
def extract_pdf_pages(pdf_path: Path, start: int, end: int) -> str:
    """Extract text from a range of pages (1-indexed, inclusive)."""

def extract_pdf_chapter(pdf_path: Path, chapter: int) -> str:
    """Extract a chapter using the PDF's table of contents."""
```

- Uses `pymupdf` (already added as dev dependency)
- Normalizes Unicode ligatures (ﬁ→fi, ﬂ→fl, etc.)
- Strips page headers/footers (repeated "1 Smooth Manifolds" + page number)
- Preserves paragraph breaks
- Returns plain text (no markdown conversion — math notation is already
  degraded, don't pretend otherwise)

### Step 2: Extend baseline to accept extracted text

Current `baseline.ask()` reads `.md`/`.txt` from a directory. Rather than
changing its interface, we'll:

1. Add a **pre-extraction step** that dumps PDF text to a `.txt` file in a
   temp/fixture directory
2. Feed that directory to the existing `ask()` function

This keeps the baseline unchanged and testable with both synthetic and real
docs.

Concretely, add a helper:

```python
def prepare_pdf_fixture(
    pdf_path: Path, pages: tuple[int, int], output_dir: Path
) -> Path:
    """Extract pages from PDF and write to output_dir as .txt. Returns the path."""
```

### Step 3: Math-domain eval suite (`evals/phase0_math_eval.py`)

**QA pairs for Lee Chapter 1** (pages 19–49), organized by category:

#### Direct Lookup (4 pairs)
Questions whose answers are stated explicitly in the text:

1. "What three properties must a topological space have to be a topological manifold?"
   → Hausdorff, second-countable, locally Euclidean

2. "What is the definition of a smooth atlas on a topological manifold?"
   → A collection of charts whose domains cover M, where any two charts are
     smoothly compatible (transition maps are diffeomorphisms)

3. "What is the dimension of real projective space RPⁿ as a topological manifold?"
   → n (it is an n-dimensional topological manifold)

4. "What does it mean for two charts to be smoothly compatible?"
   → Either their domains don't intersect, or the transition map between them
     is a diffeomorphism (smooth with smooth inverse)

#### Cross-Reference / Synthesis (3 pairs)
Questions requiring combining information from different sections:

5. "How does the concept of a smooth structure relate to the idea of transition maps between charts?"
   → A smooth structure is a maximal smooth atlas. A smooth atlas is one where
     all transition maps (ψ ∘ φ⁻¹) between overlapping charts are diffeomorphisms.
     So the smooth structure is defined by requiring all chart-to-chart transitions
     to be smooth.

6. "Explain the relationship between coordinate charts, atlases, and smooth structures — how do they build on each other?"
   → A coordinate chart is a homeomorphism from an open subset of M to an open
     subset of Rⁿ. An atlas is a collection of charts covering M. A smooth atlas
     requires all transition maps to be smooth. A smooth structure is a maximal
     smooth atlas — one that contains every chart smoothly compatible with it.

7. "What is the connection between stereographic projection for Sⁿ and the smooth structure on Sⁿ?"
   → Stereographic projection from the north and south poles gives two charts
     covering Sⁿ. The transition map between them is smooth, so they form a
     smooth atlas, giving Sⁿ a smooth structure.

#### Reasoning / Inference (2 pairs)
Questions requiring mathematical reasoning from stated facts:

8. "If M is a topological manifold and φ: U → φ(U) is a chart, why must φ(U) be an open subset of Rⁿ?"
   → Because φ is a homeomorphism from U (open in M) to φ(U). Since M is
     locally Euclidean of dimension n, each point has a neighborhood homeomorphic
     to an open subset of Rⁿ. Homeomorphisms preserve openness, so φ(U) is open
     in Rⁿ.

9. "Why does the text require manifolds to be Hausdorff and second-countable, rather than just locally Euclidean?"
   → Locally Euclidean alone admits pathological spaces. Hausdorff ensures
     unique limits (rules out the "line with two origins"). Second-countability
     ensures paracompactness, which is needed for partitions of unity — a
     fundamental tool used throughout the book.

#### Unanswerable from Chapter 1 (2 pairs)
Questions about content NOT in Chapter 1:

10. "What is the definition of a tangent vector to a smooth manifold?"
    → Not covered in Chapter 1 (this is Chapter 3 material)

11. "State the inverse function theorem for smooth manifolds."
    → Not covered in Chapter 1 (this is Chapter 4 material)

#### New eval dimension: Mathematical Precision (1-5)

In addition to faithfulness and relevance, add a third scoring axis:

- **Precision (1-5):** Are mathematical statements correct and properly
  qualified? 5 = definitions match the source, conditions are complete.
  3 = roughly correct but missing conditions. 1 = mathematically wrong.

This matters because an LLM can produce "faithful" but imprecise math — e.g.,
saying "a manifold is a space that looks like Rⁿ" is relevant but omits
Hausdorff and second-countable.

Updated judge model:

```python
class MathJudgeScore(BaseModel):
    faithfulness: int = Field(ge=1, le=5)
    relevance: int = Field(ge=1, le=5)
    precision: int = Field(ge=1, le=5)
    reasoning: str
```

### Step 4: Judge calibration for math domain

Two calibration controls with intentionally wrong math:

1. Q: "What three properties define a topological manifold?"
   Bad answer: "Compact, connected, and orientable"
   → Should score precision < 3, faithfulness < 3

2. Q: "What is a smooth atlas?"
   Bad answer: "A collection of charts where the transition maps are continuous"
   → Should score precision < 3 (continuous ≠ smooth; the distinction is the
     whole point)

## Files to Create

| File | Purpose |
|---|---|
| `src/yoke/extract.py` | PDF text extraction with Unicode normalization |
| `tests/fixtures/docs-math/` | Directory for extracted PDF text fixtures |
| `evals/phase0_math_eval.py` | Math-domain eval with 11 QA pairs + calibration |

## Files to Modify

| File | Change |
|---|---|
| `pyproject.toml` | Move `pymupdf` from dev to main dependency (ingestion needs it) |
| `src/yoke/baseline.py` | No changes needed — it already reads `.txt` files |

## Acceptance Criteria

1. `extract_pdf_pages()` produces clean, readable text from Lee Ch.1
2. `uv run yoke-baseline --docs-dir tests/fixtures/docs-math "What is a topological manifold?"` returns a coherent answer grounded in Lee's definitions
3. `uv run pytest evals/phase0_math_eval.py -v` runs all 11 QA pairs + 2 calibration controls
4. Calibration controls pass (bad math answers score precision < 3)
5. Baseline scores are recorded in `evals/results/phase0_math_baseline.json`
6. Results establish a numeric baseline that future phases must beat

## Expected Outcomes & What We Learn

### What should work well (~19K tokens fits in context)
- Direct lookup questions (definitions stated verbatim in the text)
- The model knows this material from training — faithfulness scoring will
  distinguish "answered from context" vs "answered from parametric knowledge"

### What will likely struggle
- **Precision on formal definitions** — the model may paraphrase imprecisely
  or include conditions not in the extracted text
- **Cross-reference questions** — Ch.1 says "see Chapter 4" but we don't
  have Chapter 4 in context
- **Notation-dependent questions** — if the extracted text garbles a key
  formula, the model may hallucinate the correct version from training data
  (high precision but low faithfulness — knows the math, didn't get it from
  context)

### The key tension: faithfulness vs. training knowledge

The model has certainly seen Lee's textbook in its training data. When the
extracted text is garbled or incomplete, the model may "fill in" correct
mathematics from parametric memory. This is **useful** (correct answer) but
**unfaithful** (not grounded in context). The eval must catch this:

- If the model says something mathematically correct that ISN'T in the
  extracted text → faithfulness should be < 5
- This is the opposite of the synthetic eval, where faithfulness and
  correctness were aligned

This tension is exactly what motivates better extraction and retrieval.

## Architectural Decisions

### 1. Extract-then-feed vs. native PDF in prompt

**Decision:** Extract to `.txt`, then feed to existing baseline.

**Trade-off:** We could use the Anthropic API's PDF support (base64-encoded
PDF in messages). This would preserve layout and potentially handle math
better. However:
- It's expensive (vision tokens for every page)
- It couples the baseline to a specific API feature
- We need text extraction anyway for chunking in Phase 1
- Starting with text extraction reveals extraction quality issues early

We can add a "PDF-native" comparison later as a data point.

### 2. Chapter 1 only vs. full book

**Decision:** Chapter 1 only (pages 19–49, ~19K tokens).

**Trade-off:** The full book (726 pages, ~300K+ tokens) would exceed the
context window and force us to build chunking/retrieval immediately. Chapter 1
fits in context, so we can:
- Isolate extraction quality as a variable (no retrieval noise)
- Get baseline scores before adding retrieval complexity
- Use full-book work as the Phase 1 motivator

### 3. Separate eval file vs. extending phase0_eval.py

**Decision:** New file `evals/phase0_math_eval.py`.

**Trade-off:** The math eval has different fixtures (PDF-extracted text, not
synthetic markdown), different QA pairs, and an additional scoring dimension
(precision). Keeping it separate avoids bloating the original eval and makes
it clear which baseline we're comparing against.

### 4. Adding "precision" as a third scoring axis

**Decision:** Add mathematical precision scoring alongside faithfulness and
relevance.

**Trade-off:** More dimensions mean more judge calls (or a more complex judge
prompt). But for mathematical content, the distinction between "faithful to
context" and "mathematically precise" is critical. A model can be faithful to
garbled extraction (low precision) or precise from training data (low
faithfulness). We need to measure both.

### 5. Fixture strategy: checked-in extracted text vs. live extraction

**Decision:** Check in the extracted `.txt` file as a test fixture.

**Trade-off:** Live extraction from the PDF would ensure the fixture stays
in sync, but:
- The PDF lives on the user's OneDrive, not in the repo
- Extraction is deterministic for a given pymupdf version
- Checking in the fixture makes the eval reproducible without the PDF
- We note the source PDF, page range, and pymupdf version in the fixture
  metadata

## Evals Before Implementation

Per project convention, evals come first:

1. **Write `evals/phase0_math_eval.py`** with QA pairs, judge, and calibration
2. **Manually extract** Ch.1 text (quick script) and save as fixture
3. **Run eval** — this establishes the Phase 0 math baseline numbers
4. **Then** formalize `src/yoke/extract.py` with normalization/cleanup
5. **Re-run eval** — measure whether extraction cleanup improves scores
6. Commit both baseline and improved scores

## Implementation Order

1. Write extraction script, dump Lee Ch.1 to `tests/fixtures/docs-math/lee_ch1.txt`
2. Write `evals/phase0_math_eval.py` with 11 QA pairs + 2 calibration controls
3. Run eval, record baseline in `evals/results/phase0_math_baseline.json`
4. Formalize `src/yoke/extract.py` with Unicode normalization and header stripping
5. Re-extract with cleanup, re-run eval, compare scores
6. Commit: `feat: add PDF extraction and math-domain eval (Lee Ch.1)`

## Cost Estimate

Per eval run (13 questions including 2 calibration controls):
- 13 baseline calls: ~19K context + ~500 output each ≈ 253K tokens
- 13 judge calls: ~20K context + ~300 output each ≈ 264K tokens
- Total: ~517K tokens per run ≈ $1.60 with Sonnet pricing

~3x more expensive than the synthetic eval due to larger context, but still
cheap enough to iterate on.

## Future Work (Out of Scope)

- **Full-book ingestion** — needs chunking + retrieval (Phase 1)
- **Multi-book queries** — "Compare Lee's definition with Tu's" (Phase 2+)
- **LaTeX extraction** — using Nougat/Mathpix for better math (later optimization)
- **Figure understanding** — using vision models on embedded images (later)
- **Notation index** — mapping symbols to definitions across the text (later)
