"""Phase 2 retrieval evals — hybrid retrieval quality on math corpus.

Tests hybrid (dense + sparse + RRF) retrieval against the Lee Ch.1 fixture.
Defines success criteria before implementation exists.

Run: uv run pytest evals/phase2_retrieval_eval.py -v
"""

import io
import json
import sqlite3
import sys
from pathlib import Path

import pytest
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from evals._judge import generate as llm_generate
from evals._judge import judge as judge_dispatch

# Fix Windows console encoding for Unicode math symbols
if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer, encoding="utf-8", errors="replace"
    )

from yoke.ingestion.chunking import chunk_text
from yoke.ingestion.embedding import embed_texts
from yoke.ingestion.store import build_bm25_index, init_db, store_document
from yoke.retrieval import retrieve, RetrievalResult

load_dotenv(override=True)

FIXTURE_PATH = (
    Path(__file__).resolve().parent.parent
    / "tests" / "fixtures" / "docs-math" / "lee_ch1.txt"
)
RESULTS_DIR = Path(__file__).resolve().parent / "results"

# ---------------------------------------------------------------------------
# Eval queries with known answer substrings
# ---------------------------------------------------------------------------

# Each query has a list of substrings; at least one must appear in a top-k chunk.
# Substrings are lowercased for matching.
KNOWN_ANSWER_QUERIES = [
    # 1. Direct term match (BM25-friendly)
    {
        "query": "What is a topological manifold?",
        "expected_substrings": [
            "hausdorff",
            "second-countable",
            "locally euclidean",
        ],
        "category": "direct",
    },
    # 2. Semantic paraphrase (dense-friendly)
    {
        "query": "What makes a set open in the context of coordinate charts?",
        "expected_substrings": [
            "coordinate chart",
            "homeomorphism",
            "open subset",
        ],
        "category": "semantic",
    },
    # 3. Specific definition lookup
    {
        "query": "definition of a smooth atlas",
        "expected_substrings": [
            "smooth atlas",
            "smoothly compatible",
        ],
        "category": "direct",
    },
    # 4. Notation-heavy query
    {
        "query": "transition map psi composed with phi inverse",
        "expected_substrings": [
            "transition map",
            "diffeomorphism",
        ],
        "category": "notation",
    },
    # 5. Specific example lookup
    {
        "query": "How is the n-sphere S^n shown to be a topological manifold?",
        "expected_substrings": [
            "sphere",
            "locally euclidean",
        ],
        "category": "direct",
    },
    # 6. Paraphrased concept
    {
        "query": "Why do we need manifolds to have unique limits of sequences?",
        "expected_substrings": [
            "hausdorff",
            "limits of convergent sequences are unique",
        ],
        "category": "semantic",
    },
    # 7. Cross-referencing concept
    {
        "query": "How does second-countability relate to partitions of unity?",
        "expected_substrings": [
            "second-countab",
            "partitions of unity",
        ],
        "category": "cross",
    },
    # 8. Specific theorem
    {
        "query": "Are topological manifolds paracompact?",
        "expected_substrings": [
            "paracompact",
            "locally finite",
        ],
        "category": "direct",
    },
    # 9. Product manifolds
    {
        "query": "How do you construct a manifold from a product of manifolds?",
        "expected_substrings": [
            "product",
            "manifold",
            "dimension",
        ],
        "category": "semantic",
    },
    # 10. Maximal atlas / smooth structure
    {
        "query": "What is a maximal smooth atlas and how does it define smooth structure?",
        "expected_substrings": [
            "maximal smooth atlas",
            "smooth structure",
        ],
        "category": "direct",
    },
]

# Queries with NO answer in the corpus
IRRELEVANT_QUERIES = [
    "Explain the Riemann hypothesis and its implications for prime numbers",
    "What is quantum entanglement and how does it relate to Bell's theorem?",
    "Describe the architecture of a transformer neural network",
]


# ---------------------------------------------------------------------------
# Shared fixtures — ingest the math corpus into a temp DB
# ---------------------------------------------------------------------------

def _ingest_fixture(tmp_path: Path) -> tuple[Path, Path]:
    """Ingest lee_ch1.txt into a temp SQLite DB + BM25 index.

    Returns (db_path, bm25_path).
    """
    full_text = FIXTURE_PATH.read_text(encoding="utf-8")
    chunks = chunk_text(full_text, source_file="lee_ch1.txt")

    # Enrich with dummy summaries (skip LLM call — enrichment is tested elsewhere)
    from yoke.ingestion.models import EnrichedChunk

    enriched = [
        EnrichedChunk(
            chunk_index=c.chunk_index,
            chunk_text=c.text,
            context_summary=f"Chunk {c.chunk_index} from Lee Ch.1 on smooth manifolds.",
            enriched_text=f"[Context: Chunk {c.chunk_index} from Lee Ch.1 on smooth manifolds.]\n\n{c.text}",
            page_numbers=c.page_numbers,
            source_file=c.source_file,
        )
        for c in chunks
    ]

    # Embed (real embeddings — this is what retrieval depends on)
    texts_to_embed = [ec.enriched_text for ec in enriched]
    embeddings = embed_texts(texts_to_embed)

    # Store
    db_path = tmp_path / "test_retrieval.db"
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
        pytest.skip("OPENAI_API_KEY not set — skip retrieval eval")

    tmp_path = tmp_path_factory.mktemp("retrieval_eval")
    return _ingest_fixture(tmp_path)


# ---------------------------------------------------------------------------
# Eval 1: Known-answer retrieval — recall@5 and recall@10
# ---------------------------------------------------------------------------


class TestKnownAnswerRetrieval:
    """For 10 known-answer queries, verify correct chunks appear in top results."""

    def test_recall_at_5_and_10(
        self, ingested_corpus: tuple[Path, Path]
    ) -> None:
        """Each query should retrieve a chunk containing expected substrings."""
        db_path, bm25_path = ingested_corpus

        hits_at_5 = 0
        hits_at_10 = 0
        total = len(KNOWN_ANSWER_QUERIES)

        for qa in KNOWN_ANSWER_QUERIES:
            results = retrieve(qa["query"], db_path, bm25_path, k=10)
            assert isinstance(results, list)
            assert all(isinstance(r, RetrievalResult) for r in results)

            # Check top-5
            found_5 = _check_hit(results[:5], qa["expected_substrings"])
            # Check top-10
            found_10 = _check_hit(results[:10], qa["expected_substrings"])

            if found_5:
                hits_at_5 += 1
            if found_10:
                hits_at_10 += 1

            status_5 = "Y" if found_5 else "N"
            status_10 = "Y" if found_10 else "N"
            print(
                f"\n  [{qa['category']:<9}] @5={status_5} @10={status_10}  "
                f'"{qa["query"][:55]}"'
            )

        recall_5 = hits_at_5 / total
        recall_10 = hits_at_10 / total

        print(f"\n  Recall@5:  {hits_at_5}/{total} ({recall_5:.0%})")
        print(f"  Recall@10: {hits_at_10}/{total} ({recall_10:.0%})")

        assert recall_5 >= 0.70, (
            f"Recall@5 = {recall_5:.0%} ({hits_at_5}/{total}), need >= 70%"
        )
        assert recall_10 >= 0.80, (
            f"Recall@10 = {recall_10:.0%} ({hits_at_10}/{total}), need >= 80%"
        )


def _check_hit(results: list[RetrievalResult], expected: list[str]) -> bool:
    """Return True if expected substrings are covered across the top-k results.

    Each expected substring must appear in at least one result, but they don't
    all need to be in the same chunk (cross-referencing queries may span chunks).
    """
    combined_lower = "\n".join(r.chunk_text.lower() for r in results)
    return all(sub.lower() in combined_lower for sub in expected)


# ---------------------------------------------------------------------------
# Eval 2: Hybrid vs dense-only — hybrid should match or beat dense on >= 8/10
# ---------------------------------------------------------------------------


class TestHybridVsDense:
    """Hybrid retrieval must match or beat dense-only recall@5 on >= 8/10 queries."""

    def test_hybrid_beats_dense(
        self, ingested_corpus: tuple[Path, Path]
    ) -> None:
        """Run both hybrid and dense-only, compare per-query recall@5."""
        db_path, bm25_path = ingested_corpus

        from yoke.retrieval.dense import dense_search
        from yoke.retrieval.models import RetrievalResult as _RR

        hybrid_wins_or_ties = 0
        total = len(KNOWN_ANSWER_QUERIES)

        for qa in KNOWN_ANSWER_QUERIES:
            # Hybrid (full pipeline)
            hybrid_results = retrieve(qa["query"], db_path, bm25_path, k=5)
            hybrid_hit = _check_hit(hybrid_results, qa["expected_substrings"])

            # Dense-only: embed query, get top-5 by cosine similarity
            query_emb = embed_texts([qa["query"]])[0]
            dense_candidates = dense_search(query_emb, db_path, top_n=5)

            # Fetch chunk texts for dense candidates
            conn = sqlite3.connect(str(db_path))
            dense_hit = False
            for chunk_id, _score in dense_candidates:
                row = conn.execute(
                    "SELECT chunk_text FROM chunks WHERE id = ?", (chunk_id,)
                ).fetchone()
                if row:
                    text_lower = row[0].lower()
                    if all(
                        sub.lower() in text_lower
                        for sub in qa["expected_substrings"]
                    ):
                        dense_hit = True
                        break
            conn.close()

            if hybrid_hit or not dense_hit:
                # Hybrid wins or ties (both hit, both miss, or only hybrid hits)
                hybrid_wins_or_ties += 1

            h = "Y" if hybrid_hit else "N"
            d = "Y" if dense_hit else "N"
            print(
                f"\n  hybrid={h} dense={d}  "
                f'"{qa["query"][:55]}"'
            )

        print(
            f"\n  Hybrid wins or ties: {hybrid_wins_or_ties}/{total}"
        )

        assert hybrid_wins_or_ties >= 8, (
            f"Hybrid wins/ties on {hybrid_wins_or_ties}/{total} queries, need >= 8"
        )


# ---------------------------------------------------------------------------
# Eval 3: Irrelevance rejection — top RRF score should be low for OOS queries
# ---------------------------------------------------------------------------


class TestIrrelevanceRejection:
    """Out-of-scope queries should produce lower dense similarity than in-scope.

    RRF scores are rank-based (1/(k+rank)) and cannot distinguish relevant from
    irrelevant queries — every query produces the same score structure. Instead,
    we compare raw dense cosine similarity of the top-1 result, which *does*
    reflect semantic distance.
    """

    def test_irrelevant_queries_low_similarity(
        self, ingested_corpus: tuple[Path, Path]
    ) -> None:
        """Top-1 dense cosine similarity for OOS queries should be lower than in-scope."""
        db_path, bm25_path = ingested_corpus

        from yoke.retrieval.dense import dense_search

        # Collect in-scope top-1 dense similarity scores
        in_scope_sims: list[float] = []
        for qa in KNOWN_ANSWER_QUERIES[:5]:
            query_emb = embed_texts([qa["query"]])[0]
            dense_results = dense_search(query_emb, db_path, top_n=1)
            if dense_results:
                in_scope_sims.append(dense_results[0][1])

        avg_in_scope = sum(in_scope_sims) / len(in_scope_sims) if in_scope_sims else 0.0
        # OOS queries should have top-1 similarity below 90% of in-scope average
        threshold = 0.90 * avg_in_scope

        print(f"\n  In-scope avg top-1 cosine sim: {avg_in_scope:.4f}")
        print(f"  Threshold (90%):               {threshold:.4f}")

        oos_sims: list[float] = []
        violations = 0
        for query in IRRELEVANT_QUERIES:
            query_emb = embed_texts([query])[0]
            dense_results = dense_search(query_emb, db_path, top_n=1)
            sim = dense_results[0][1] if dense_results else 0.0
            oos_sims.append(sim)
            status = "PASS" if sim < threshold else "FAIL"
            if sim >= threshold:
                violations += 1
            print(f"  [{status}] sim={sim:.4f}  \"{query[:55]}\"")

        assert violations <= 1, (
            f"{violations}/3 OOS queries exceeded similarity threshold {threshold:.4f}. "
            f"Sims: {oos_sims}"
        )


# ---------------------------------------------------------------------------
# Eval 4: End-to-end comparison — retrieval + generation vs Phase 0 baseline
# ---------------------------------------------------------------------------

PHASE0_RESULTS_PATH = RESULTS_DIR / "phase0_math_baseline.json"

# QA pairs shared with Phase 0 eval (subset for cost efficiency)
E2E_QA_PAIRS = [
    {
        "question": "What three properties must a topological space have to be a topological manifold?",
        "expected_answer": (
            "A topological manifold must be (1) a Hausdorff space, "
            "(2) second-countable, and (3) locally Euclidean of dimension n."
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
        "question": "What does it mean for two charts to be smoothly compatible?",
        "expected_answer": (
            "Two charts are smoothly compatible if either their domains don't "
            "intersect, or the transition map is a diffeomorphism."
        ),
        "category": "direct",
    },
    {
        "question": (
            "How does the concept of a smooth structure relate to the idea "
            "of transition maps between charts?"
        ),
        "expected_answer": (
            "A smooth structure is a maximal smooth atlas, defined by requiring "
            "all chart-to-chart transitions to be smooth diffeomorphisms."
        ),
        "category": "cross",
    },
    {
        "question": (
            "Why does the text require manifolds to be Hausdorff and "
            "second-countable, rather than just locally Euclidean?"
        ),
        "expected_answer": (
            "Hausdorff ensures unique limits; second-countability ensures "
            "paracompactness and partitions of unity."
        ),
        "category": "reasoning",
    },
    {
        "question": "What is the definition of a tangent vector to a smooth manifold?",
        "expected_answer": "Not covered in Chapter 1.",
        "category": "unanswerable",
    },
]


class E2EJudgeScore(BaseModel):
    faithfulness: int = Field(ge=1, le=5)
    relevance: int = Field(ge=1, le=5)
    reasoning: str


E2E_JUDGE_TOOL = {
    "name": "score_answer",
    "description": "Score the answer on faithfulness and relevance.",
    "input_schema": {
        "type": "object",
        "properties": {
            "faithfulness": {
                "type": "integer",
                "description": (
                    "1-5. Is the answer supported by the retrieved context? "
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
            "reasoning": {
                "type": "string",
                "description": "Brief explanation of the scores.",
            },
        },
        "required": ["faithfulness", "relevance", "reasoning"],
    },
}

E2E_JUDGE_SYSTEM = (
    "You are an evaluation judge. You will be given a question, the expected answer, "
    "the model's actual answer, and the retrieved context chunks that were used.\n\n"
    "Score the actual answer:\n"
    "Faithfulness (1-5): Is the answer supported by the retrieved chunks? "
    "5 = every claim is directly supported. 1 = fabricated.\n"
    "Relevance (1-5): Does the answer address the question? "
    "5 = directly and completely. 1 = off-topic.\n\n"
    "For unanswerable questions: if the model correctly says it lacks information, "
    "score faithfulness=5, relevance=5."
)

GENERATION_MODEL = "claude-sonnet-4-20250514"
GENERATION_SYSTEM = (
    "Answer the question using ONLY the provided context chunks. "
    "If the context does not contain enough information, say so. "
    "Be concise and specific."
)


def _generate_answer(
    question: str,
    chunks: list[RetrievalResult],
    model: str = GENERATION_MODEL,
) -> str:
    """Generate an answer using retrieved chunks as context."""
    context = "\n\n---\n\n".join(
        f"[Chunk from {c.source_file}, pages {c.page_numbers}]\n{c.chunk_text}"
        for c in chunks
    )
    return llm_generate(
        model,
        f"Context:\n{context}\n\nQuestion: {question}",
        system=GENERATION_SYSTEM,
        max_tokens=512,
    )


def _judge_answer(
    question: str, expected: str, actual: str, context: str, judge_model: str
) -> E2EJudgeScore:
    """Score an answer using an LLM judge."""
    return judge_dispatch(
        model=judge_model,
        system=E2E_JUDGE_SYSTEM,
        user_prompt=(
            f"Question: {question}\n\n"
            f"Expected answer: {expected}\n\n"
            f"Actual answer: {actual}\n\n"
            f"Retrieved context:\n{context}"
        ),
        tool=E2E_JUDGE_TOOL,
        score_cls=E2EJudgeScore,
    )


class TestEndToEndComparison:
    """Retrieval + generation must match or beat Phase 0 baseline scores."""

    def test_retrieval_generation_vs_baseline(
        self,
        ingested_corpus: tuple[Path, Path],
        judge_model: str,
        generation_model: str,
    ) -> None:
        """Average faithfulness and relevance >= Phase 0 baseline averages."""
        db_path, bm25_path = ingested_corpus

        # Load Phase 0 baseline scores
        if not PHASE0_RESULTS_PATH.exists():
            pytest.skip("Phase 0 baseline results not found — run phase0_math_eval first")

        phase0 = json.loads(PHASE0_RESULTS_PATH.read_text(encoding="utf-8"))
        baseline_faith = phase0["average_faithfulness"]
        baseline_rel = phase0["average_relevance"]

        results: list[dict] = []

        for qa in E2E_QA_PAIRS:
            # Retrieve
            chunks = retrieve(qa["question"], db_path, bm25_path, k=5)

            # Generate
            answer = _generate_answer(qa["question"], chunks, model=generation_model)

            # Build context string for judge
            context_str = "\n\n".join(c.chunk_text for c in chunks)

            # Judge
            score = _judge_answer(
                qa["question"],
                qa["expected_answer"],
                answer,
                context_str,
                judge_model or "claude-haiku-4-5-20251001",
            )

            results.append({
                "question": qa["question"],
                "category": qa["category"],
                "faithfulness": score.faithfulness,
                "relevance": score.relevance,
                "reasoning": score.reasoning,
            })

            print(
                f"\n  [{qa['category']:<13}] faith={score.faithfulness} "
                f"rel={score.relevance}  \"{qa['question'][:50]}\""
            )

        avg_faith = sum(r["faithfulness"] for r in results) / len(results)
        avg_rel = sum(r["relevance"] for r in results) / len(results)

        print(f"\n  Phase 2 average faithfulness: {avg_faith:.2f} (baseline: {baseline_faith:.2f})")
        print(f"  Phase 2 average relevance:    {avg_rel:.2f} (baseline: {baseline_rel:.2f})")

        # Write results
        summary = {
            "phase": "phase2_retrieval_e2e",
            "total_questions": len(results),
            "average_faithfulness": round(avg_faith, 2),
            "average_relevance": round(avg_rel, 2),
            "baseline_faithfulness": baseline_faith,
            "baseline_relevance": baseline_rel,
            "results": results,
        }
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        (RESULTS_DIR / "phase2_retrieval_e2e.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )

        # Phase 2 must match or beat Phase 0 (allow 0.5 margin for eval noise)
        assert avg_faith >= baseline_faith - 0.5, (
            f"Faithfulness {avg_faith:.2f} below baseline {baseline_faith:.2f} - 0.5"
        )
        assert avg_rel >= baseline_rel - 0.5, (
            f"Relevance {avg_rel:.2f} below baseline {baseline_rel:.2f} - 0.5"
        )


# ---------------------------------------------------------------------------
# Summary & results
# ---------------------------------------------------------------------------


class TestRetrievalSummary:
    """Write a summary of retrieval eval metrics."""

    def test_write_summary(
        self, ingested_corpus: tuple[Path, Path]
    ) -> None:
        """Run core metrics and write results JSON."""
        db_path, bm25_path = ingested_corpus

        hits_at_5 = 0
        hits_at_10 = 0
        reciprocal_ranks: list[float] = []
        total = len(KNOWN_ANSWER_QUERIES)

        for qa in KNOWN_ANSWER_QUERIES:
            results = retrieve(qa["query"], db_path, bm25_path, k=10)
            if _check_hit(results[:5], qa["expected_substrings"]):
                hits_at_5 += 1
            if _check_hit(results[:10], qa["expected_substrings"]):
                hits_at_10 += 1

            # MRR: find the rank of first chunk covering all expected substrings
            rr = 0.0
            for rank, r in enumerate(results, 1):
                text_lower = r.chunk_text.lower()
                if all(sub.lower() in text_lower for sub in qa["expected_substrings"]):
                    rr = 1.0 / rank
                    break
            reciprocal_ranks.append(rr)

        mrr = sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0

        summary = {
            "phase": "phase2_retrieval",
            "corpus": "Lee, Introduction to Smooth Manifolds, Ch.1",
            "total_queries": total,
            "recall_at_5": round(hits_at_5 / total, 2),
            "recall_at_10": round(hits_at_10 / total, 2),
            "mrr_at_10": round(mrr, 3),
        }

        print(f"\n  Phase 2 Retrieval Eval Summary")
        print(f"  {'=' * 40}")
        for k, v in summary.items():
            print(f"    {k}: {v}")

        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        results_path = RESULTS_DIR / "phase2_retrieval.json"
        results_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"\n  Results written to {results_path}")


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
