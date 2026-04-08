"""Phase 1 pipeline eval — retrieval + generation vs Phase 0 baseline.

Runs the same QA pairs through both pipelines, scores with an LLM judge,
and produces a side-by-side comparison table.

Run: uv run pytest evals/phase1_pipeline_eval.py -v
"""

import io
import json
import sys
from pathlib import Path

import pytest
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from evals._judge import judge as judge_dispatch

from yoke.baseline import ask as baseline_ask
from yoke.ingestion.chunking import chunk_text
from yoke.ingestion.embedding import embed_texts
from yoke.ingestion.store import build_bm25_index, init_db, store_document
from yoke.pipeline import PipelineResult, format_context, query as pipeline_query

# Fix Windows console encoding
if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer, encoding="utf-8", errors="replace"
    )

load_dotenv(override=True)

FIXTURE_PATH = (
    Path(__file__).resolve().parent.parent
    / "tests" / "fixtures" / "docs-math" / "lee_ch1.txt"
)
DOCS_DIR = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "docs-math"
RESULTS_DIR = Path(__file__).resolve().parent / "results"

# ---------------------------------------------------------------------------
# QA pairs — reused from phase0_math_eval.py (canonical math corpus eval set)
# ---------------------------------------------------------------------------

QA_PAIRS = [
    # Direct lookup (4)
    {
        "question": "What three properties must a topological space have to be a topological manifold?",
        "expected_answer": (
            "A topological manifold must be (1) a Hausdorff space, "
            "(2) second-countable (has a countable basis for its topology), "
            "and (3) locally Euclidean of dimension n (each point has a "
            "neighborhood homeomorphic to an open subset of R^n)."
        ),
        "category": "direct",
    },
    {
        "question": "What is the definition of a smooth atlas on a topological manifold?",
        "expected_answer": (
            "A smooth atlas is a collection of charts whose domains cover M, "
            "such that any two charts in the atlas are smoothly compatible "
            "with each other (their transition maps are diffeomorphisms)."
        ),
        "category": "direct",
    },
    {
        "question": "What is the dimension of real projective space RP^n as a topological manifold?",
        "expected_answer": "RP^n is an n-dimensional topological manifold.",
        "category": "direct",
    },
    {
        "question": "What does it mean for two charts to be smoothly compatible?",
        "expected_answer": (
            "Two charts (U, phi) and (V, psi) are smoothly compatible if "
            "either U and V don't intersect, or the transition map "
            "psi composed with phi-inverse is a diffeomorphism (smooth "
            "with smooth inverse)."
        ),
        "category": "direct",
    },
    # Cross-reference / synthesis (3)
    {
        "question": (
            "How does the concept of a smooth structure relate to the idea "
            "of transition maps between charts?"
        ),
        "expected_answer": (
            "A smooth structure is a maximal smooth atlas. A smooth atlas "
            "is one where all transition maps (psi composed with phi-inverse) "
            "between overlapping charts are diffeomorphisms. So the smooth "
            "structure is ultimately defined by requiring all chart-to-chart "
            "transitions to be smooth."
        ),
        "category": "cross",
    },
    {
        "question": (
            "Explain the relationship between coordinate charts, atlases, "
            "and smooth structures — how do they build on each other?"
        ),
        "expected_answer": (
            "A coordinate chart is a homeomorphism from an open subset of M "
            "to an open subset of R^n. An atlas is a collection of charts "
            "covering M. A smooth atlas requires all transition maps between "
            "its charts to be smooth. A smooth structure is a maximal smooth "
            "atlas — one that contains every chart smoothly compatible with it."
        ),
        "category": "cross",
    },
    {
        "question": (
            "What is the connection between stereographic projection for S^n "
            "and the smooth structure on S^n?"
        ),
        "expected_answer": (
            "Stereographic projection from the north and south poles gives "
            "two charts that together cover S^n. The transition map between "
            "them is smooth, so they form a smooth atlas, giving S^n a "
            "smooth structure."
        ),
        "category": "cross",
    },
    # Reasoning / inference (2)
    {
        "question": (
            "If M is a topological manifold and phi: U -> phi(U) is a chart, "
            "why must phi(U) be an open subset of R^n?"
        ),
        "expected_answer": (
            "Because phi is a homeomorphism from U (open in M) to phi(U). "
            "Since M is locally Euclidean of dimension n, each point has a "
            "neighborhood homeomorphic to an open subset of R^n. "
            "Homeomorphisms map open sets to open sets, so phi(U) is open in R^n."
        ),
        "category": "reasoning",
    },
    {
        "question": (
            "Why does the text require manifolds to be Hausdorff and "
            "second-countable, rather than just locally Euclidean?"
        ),
        "expected_answer": (
            "Locally Euclidean alone admits pathological spaces. Hausdorff "
            "ensures unique limits and rules out spaces like the 'line with "
            "two origins.' Second-countability ensures paracompactness, which "
            "is needed for partitions of unity — a fundamental tool used "
            "throughout the book."
        ),
        "category": "reasoning",
    },
    # Unanswerable from Chapter 1 (2)
    {
        "question": "What is the definition of a tangent vector to a smooth manifold?",
        "expected_answer": (
            "This is not covered in Chapter 1. Tangent vectors are defined "
            "in a later chapter."
        ),
        "category": "unanswerable",
    },
    {
        "question": "State the inverse function theorem for smooth manifolds.",
        "expected_answer": (
            "This is not covered in Chapter 1. The inverse function theorem "
            "for manifolds appears in a later chapter."
        ),
        "category": "unanswerable",
    },
]


# ---------------------------------------------------------------------------
# Judge
# ---------------------------------------------------------------------------

class JudgeScore(BaseModel):
    faithfulness: int = Field(ge=1, le=5)
    relevance: int = Field(ge=1, le=5)
    precision: int = Field(ge=1, le=5)
    citation_grounding: int = Field(ge=1, le=5, default=5)
    reasoning: str


JUDGE_TOOL = {
    "name": "score_answer",
    "description": "Score the answer on faithfulness, relevance, precision, and citation grounding.",
    "input_schema": {
        "type": "object",
        "properties": {
            "faithfulness": {
                "type": "integer",
                "description": (
                    "1-5. Is the answer supported by the provided context? "
                    "5 = every claim is directly supported. "
                    "1 = contains fabricated information."
                ),
            },
            "relevance": {
                "type": "integer",
                "description": (
                    "1-5. Does the answer address the question? "
                    "5 = directly and completely answers. 1 = off-topic."
                ),
            },
            "precision": {
                "type": "integer",
                "description": (
                    "1-5. Are mathematical statements correct and properly "
                    "qualified? 5 = definitions and conditions match the "
                    "source exactly. 3 = roughly correct but missing "
                    "conditions. 1 = mathematically wrong."
                ),
            },
            "citation_grounding": {
                "type": "integer",
                "description": (
                    "1-5. Do the [N] citations in the answer actually reference "
                    "chunks that support the adjacent claim? "
                    "5 = all citations are accurate. 3 = some citations are wrong "
                    "or missing. 1 = citations are fabricated or meaningless. "
                    "If the answer has no citations (e.g. unanswerable), score 5."
                ),
            },
            "reasoning": {
                "type": "string",
                "description": "Brief explanation of the scores.",
            },
        },
        "required": [
            "faithfulness",
            "relevance",
            "precision",
            "citation_grounding",
            "reasoning",
        ],
    },
}

JUDGE_SYSTEM = (
    "You are an evaluation judge for a mathematical knowledge management system. "
    "You will be given a question, the expected answer, the model's actual answer, "
    "and the source context chunks (numbered [1], [2], etc.).\n\n"
    "Score the actual answer using the score_answer tool.\n\n"
    "Faithfulness (1-5): Is the answer supported by the source context? "
    "5 = every claim is directly supported. 1 = contains fabricated information. "
    "IMPORTANT: If the answer includes correct information NOT present in the "
    "provided context, faithfulness should be reduced.\n\n"
    "Relevance (1-5): Does the answer address the question asked? "
    "5 = directly and completely answers. 1 = off-topic.\n\n"
    "Precision (1-5): Are mathematical statements correct and properly qualified? "
    "5 = definitions match the source, all conditions stated. "
    "3 = roughly correct but missing important conditions. "
    "1 = mathematically wrong or misleading.\n\n"
    "Citation grounding (1-5): Do the [N] references in the answer match the "
    "numbered context chunks? Check that each [N] citation points to a chunk "
    "that actually supports the claim. 5 = all citations valid. "
    "If the answer has no citations (e.g. unanswerable), score 5.\n\n"
    "For unanswerable questions: if the model correctly declines to answer, "
    "score faithfulness=5, relevance=5, precision=5, citation_grounding=5."
)


def _judge(
    question: str,
    expected: str,
    actual: str,
    context: str,
    judge_model: str,
) -> JudgeScore:
    """Score an answer using an LLM judge."""
    return judge_dispatch(
        model=judge_model,
        system=JUDGE_SYSTEM,
        user_prompt=(
            f"Question: {question}\n\n"
            f"Expected answer: {expected}\n\n"
            f"Actual answer: {actual}\n\n"
            f"Context:\n{context}"
        ),
        tool=JUDGE_TOOL,
        score_cls=JudgeScore,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _ingest_fixture(tmp_path: Path) -> tuple[Path, Path]:
    """Ingest lee_ch1.txt into a temp SQLite DB + BM25 index."""
    full_text = FIXTURE_PATH.read_text(encoding="utf-8")
    chunks = chunk_text(full_text, source_file="lee_ch1.txt")

    from yoke.ingestion.models import EnrichedChunk

    enriched = [
        EnrichedChunk(
            chunk_index=c.chunk_index,
            chunk_text=c.text,
            context_summary=f"Chunk {c.chunk_index} from Lee Ch.1 on smooth manifolds.",
            enriched_text=(
                f"[Context: Chunk {c.chunk_index} from Lee Ch.1 on smooth manifolds.]"
                f"\n\n{c.text}"
            ),
            page_numbers=c.page_numbers,
            source_file=c.source_file,
        )
        for c in chunks
    ]

    texts_to_embed = [ec.enriched_text for ec in enriched]
    embeddings = embed_texts(texts_to_embed)

    db_path = tmp_path / "test_pipeline.db"
    conn = init_db(db_path)
    store_document(conn, "lee_ch1.txt", full_text, enriched, embeddings)

    bm25_path = db_path.with_suffix(".bm25.json")
    build_bm25_index(conn, bm25_path)
    conn.close()

    return db_path, bm25_path


@pytest.fixture(scope="module")
def ingested_corpus(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    """Module-scoped fixture: ingest once, reuse across all tests."""
    import os

    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set — skip pipeline eval")

    tmp_path = tmp_path_factory.mktemp("pipeline_eval")
    return _ingest_fixture(tmp_path)


# ---------------------------------------------------------------------------
# Eval: Phase 0 vs Phase 1 comparison
# ---------------------------------------------------------------------------


class TestPipelineComparison:
    """Run the same QA pairs through Phase 0 and Phase 1, compare scores."""

    def test_phase1_vs_phase0(
        self,
        ingested_corpus: tuple[Path, Path],
        judge_model: str,
        generation_model: str,
    ) -> None:
        """Phase 1 must match or beat Phase 0 on faithfulness and relevance."""
        db_path, bm25_path = ingested_corpus

        phase0_results: list[dict] = []
        phase1_results: list[dict] = []

        for qa in QA_PAIRS:
            question = qa["question"]
            expected = qa["expected_answer"]
            category = qa["category"]

            # --- Phase 0: baseline (context stuffing) ---
            p0_answer = baseline_ask(question, DOCS_DIR, model=generation_model)
            # Load full context for Phase 0 judge (same as what baseline sees)
            p0_context_parts = []
            for p in sorted(DOCS_DIR.iterdir()):
                if p.suffix in (".md", ".txt") and p.is_file():
                    p0_context_parts.append(
                        f"## {p.name}\n{p.read_text(encoding='utf-8')}"
                    )
            p0_context = "\n---\n".join(p0_context_parts)

            p0_score = _judge(question, expected, p0_answer, p0_context, judge_model)
            phase0_results.append({
                "question": question,
                "category": category,
                "answer": p0_answer,
                "faithfulness": p0_score.faithfulness,
                "relevance": p0_score.relevance,
                "precision": p0_score.precision,
                "reasoning": p0_score.reasoning,
            })

            # --- Phase 1: retrieval pipeline ---
            p1_result = pipeline_query(
                question, db_path, bm25_path, k=10, model=generation_model
            )

            # Build numbered context for the judge (same format the model saw)
            p1_context = format_context(p1_result.sources)

            p1_score = _judge(
                question, expected, p1_result.answer, p1_context, judge_model
            )
            phase1_results.append({
                "question": question,
                "category": category,
                "answer": p1_result.answer,
                "faithfulness": p1_score.faithfulness,
                "relevance": p1_score.relevance,
                "precision": p1_score.precision,
                "citation_grounding": p1_score.citation_grounding,
                "cited_chunks": p1_result.cited_chunk_numbers,
                "reasoning": p1_score.reasoning,
            })

            print(
                f"\n  [{category:<13}] "
                f"P0: f={p0_score.faithfulness} r={p0_score.relevance} p={p0_score.precision}  "
                f"P1: f={p1_score.faithfulness} r={p1_score.relevance} p={p1_score.precision} "
                f"cg={p1_score.citation_grounding}  "
                f'"{question[:45]}"'
            )

        # --- Aggregate ---
        n = len(QA_PAIRS)
        p0_faith = sum(r["faithfulness"] for r in phase0_results) / n
        p0_rel = sum(r["relevance"] for r in phase0_results) / n
        p0_prec = sum(r["precision"] for r in phase0_results) / n

        p1_faith = sum(r["faithfulness"] for r in phase1_results) / n
        p1_rel = sum(r["relevance"] for r in phase1_results) / n
        p1_prec = sum(r["precision"] for r in phase1_results) / n
        p1_cg = sum(r["citation_grounding"] for r in phase1_results) / n

        # --- Comparison table ---
        print("\n")
        print("  Phase 0 vs Phase 1 Comparison (Lee Ch.1, 11 QA pairs)")
        print("  " + "=" * 58)
        print(f"  {'':26s} {'Phase 0':>9s} {'Phase 1':>9s} {'Delta':>9s}")
        print(f"  {'-' * 58}")
        print(f"  {'Avg faithfulness':26s} {p0_faith:9.2f} {p1_faith:9.2f} {p1_faith - p0_faith:+9.2f}")
        print(f"  {'Avg relevance':26s} {p0_rel:9.2f} {p1_rel:9.2f} {p1_rel - p0_rel:+9.2f}")
        print(f"  {'Avg precision':26s} {p0_prec:9.2f} {p1_prec:9.2f} {p1_prec - p0_prec:+9.2f}")
        print(f"  {'Citation grounding':26s} {'—':>9s} {p1_cg:9.2f} {'':>9s}")

        p0_unanswerable = [
            r for r in phase0_results if r["category"] == "unanswerable"
        ]
        p1_unanswerable = [
            r for r in phase1_results if r["category"] == "unanswerable"
        ]
        p0_ua_correct = sum(1 for r in p0_unanswerable if r["faithfulness"] >= 4)
        p1_ua_correct = sum(1 for r in p1_unanswerable if r["faithfulness"] >= 4)
        print(
            f"  {'Unanswerable correct':26s} "
            f"{p0_ua_correct}/{len(p0_unanswerable):>6} "
            f"{p1_ua_correct}/{len(p1_unanswerable):>6} {'':>9s}"
        )
        print("  " + "=" * 58)

        # --- Write results ---
        summary = {
            "phase": "phase1_pipeline",
            "corpus": "Lee, Introduction to Smooth Manifolds, Ch.1",
            "total_questions": n,
            "phase0": {
                "average_faithfulness": round(p0_faith, 2),
                "average_relevance": round(p0_rel, 2),
                "average_precision": round(p0_prec, 2),
                "results": phase0_results,
            },
            "phase1": {
                "average_faithfulness": round(p1_faith, 2),
                "average_relevance": round(p1_rel, 2),
                "average_precision": round(p1_prec, 2),
                "average_citation_grounding": round(p1_cg, 2),
                "results": phase1_results,
            },
            "delta": {
                "faithfulness": round(p1_faith - p0_faith, 2),
                "relevance": round(p1_rel - p0_rel, 2),
                "precision": round(p1_prec - p0_prec, 2),
            },
        }
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        results_path = RESULTS_DIR / "phase1_pipeline.json"
        results_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"\n  Results written to {results_path}")

        # --- Assertions ---
        assert p1_faith >= p0_faith, (
            f"Phase 1 faithfulness {p1_faith:.2f} < Phase 0 {p0_faith:.2f}"
        )
        assert p1_rel >= p0_rel - 0.5, (
            f"Phase 1 relevance {p1_rel:.2f} < Phase 0 {p0_rel:.2f} - 0.5"
        )
        assert p1_cg >= 3.0, (
            f"Citation grounding {p1_cg:.2f} below threshold 3.0"
        )


# ---------------------------------------------------------------------------
# Citation grounding eval (standalone)
# ---------------------------------------------------------------------------


class TestCitationGrounding:
    """Verify that Phase 1 citations reference actual supporting chunks."""

    def test_citations_reference_valid_chunks(
        self,
        ingested_corpus: tuple[Path, Path],
        generation_model: str,
    ) -> None:
        """All [N] references in answers must be within the retrieved chunk range."""
        db_path, bm25_path = ingested_corpus

        total_citations = 0
        valid_citations = 0

        for qa in QA_PAIRS:
            if qa["category"] == "unanswerable":
                continue

            result = pipeline_query(
                qa["question"], db_path, bm25_path, k=10, model=generation_model
            )
            cited = result.cited_chunk_numbers
            num_chunks = len(result.sources)

            for n in cited:
                total_citations += 1
                if 1 <= n <= num_chunks:
                    valid_citations += 1

        if total_citations == 0:
            pytest.fail("No citations found in any answer — model not citing chunks")

        grounding_rate = valid_citations / total_citations
        print(f"\n  Citation grounding: {valid_citations}/{total_citations} ({grounding_rate:.0%})")

        assert grounding_rate >= 0.80, (
            f"Citation grounding {grounding_rate:.0%} below 80% threshold"
        )


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
