"""Phase 0 math-domain eval — LLM-as-judge scoring on real PDF content.

Corpus: Lee, Introduction to Smooth Manifolds, 2nd ed., Chapter 1 (pp. 19-49).
Source: pymupdf extraction with Unicode ligature normalization.
"""

import json
import time
from pathlib import Path

import anthropic
import pytest
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from yoke.baseline import ask

load_dotenv(override=True)

DOCS_DIR = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "docs-math"
RESULTS_DIR = Path(__file__).resolve().parent / "results"


# ---------- Pydantic model for structured judge output ----------

class MathJudgeScore(BaseModel):
    faithfulness: int = Field(ge=1, le=5)
    relevance: int = Field(ge=1, le=5)
    precision: int = Field(ge=1, le=5)
    reasoning: str


# ---------- QA pairs ----------

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
        "expected_answer": (
            "RP^n is an n-dimensional topological manifold."
        ),
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

# ---------- Judge ----------

JUDGE_TOOL = {
    "name": "score_answer",
    "description": "Score the answer on faithfulness, relevance, and mathematical precision.",
    "input_schema": {
        "type": "object",
        "properties": {
            "faithfulness": {
                "type": "integer",
                "description": (
                    "1-5. Is the answer supported by the provided context? "
                    "5 = every claim is directly supported by the text. "
                    "1 = contains fabricated information not in the context. "
                    "Note: if the model gives a correct mathematical answer "
                    "that goes BEYOND what the context states, that lowers "
                    "faithfulness even if the math is correct."
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
                    "conditions or qualifications. 1 = mathematically wrong."
                ),
            },
            "reasoning": {
                "type": "string",
                "description": "Brief explanation of the scores.",
            },
        },
        "required": ["faithfulness", "relevance", "precision", "reasoning"],
    },
}

JUDGE_SYSTEM = (
    "You are an evaluation judge for a mathematical knowledge management system. "
    "You will be given a question about differential geometry / smooth manifolds, "
    "the expected answer, the model's actual answer, and the source context "
    "(extracted from a PDF textbook).\n\n"
    "Score the actual answer using the score_answer tool.\n\n"
    "Faithfulness (1-5): Is the answer supported by the source context? "
    "5 = every claim is directly supported. 1 = contains fabricated information. "
    "IMPORTANT: The model may know correct mathematics from its training data. "
    "If the answer includes correct information that is NOT present in the "
    "provided context, faithfulness should be reduced — the system must answer "
    "from the context, not from parametric knowledge.\n\n"
    "Relevance (1-5): Does the answer address the question asked? "
    "5 = directly and completely answers. 1 = off-topic.\n\n"
    "Precision (1-5): Are mathematical statements correct and properly qualified? "
    "5 = definitions match the source, all conditions stated. "
    "3 = roughly correct but missing important conditions. "
    "1 = mathematically wrong or misleading.\n\n"
    "For unanswerable questions: if the model correctly declines to answer "
    "(says it doesn't have enough information or that the topic isn't covered), "
    "score faithfulness=5, relevance=5, precision=5. "
    "If it fabricates an answer using knowledge not in the context, "
    "score faithfulness=1."
)


_last_api_call: float = 0.0
_API_DELAY: float = 45.0  # seconds between calls (30K input tokens/min limit)


def _throttle() -> None:
    """Wait if needed to respect API rate limits."""
    global _last_api_call
    now = time.time()
    elapsed = now - _last_api_call
    if _last_api_call > 0 and elapsed < _API_DELAY:
        wait = _API_DELAY - elapsed
        print(f"    (rate limit: waiting {wait:.0f}s)", flush=True)
        time.sleep(wait)
    _last_api_call = time.time()


def judge(
    question: str, expected: str, actual: str, context: str
) -> MathJudgeScore:
    _throttle()
    client = anthropic.Anthropic()
    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        temperature=0,
        max_tokens=512,
        system=JUDGE_SYSTEM,
        tools=[JUDGE_TOOL],
        tool_choice={"type": "tool", "name": "score_answer"},
        messages=[
            {
                "role": "user",
                "content": (
                    f"Question: {question}\n\n"
                    f"Expected answer: {expected}\n\n"
                    f"Actual answer: {actual}\n\n"
                    f"Source context:\n{context}"
                ),
            }
        ],
    )
    for block in message.content:
        if block.type == "tool_use":
            return MathJudgeScore(**block.input)
    raise ValueError("Judge did not return a tool_use block")


# ---------- Helpers ----------

def _load_context() -> str:
    paths = sorted(p for p in DOCS_DIR.iterdir() if p.suffix in (".md", ".txt"))
    sections = []
    for p in paths:
        sections.append(f"## {p.name}\n{p.read_text(encoding='utf-8')}")
    return "\n---\n".join(sections)


# ---------- Calibration controls ----------

CALIBRATION_CONTROLS = [
    {
        "question": "What three properties define a topological manifold?",
        "expected_answer": (
            "Hausdorff, second-countable, and locally Euclidean."
        ),
        "bad_answer": (
            "A topological manifold must be compact, connected, and orientable."
        ),
        "category": "calibration",
    },
    {
        "question": "What is a smooth atlas?",
        "expected_answer": (
            "A smooth atlas is a collection of charts covering M where all "
            "transition maps are diffeomorphisms."
        ),
        "bad_answer": (
            "A smooth atlas is a collection of charts where the transition "
            "maps are continuous."
        ),
        "category": "calibration",
    },
]


# ---------- Pytest tests ----------

class TestMathCalibration:
    """Verify the judge scores intentionally wrong math answers below 3."""

    def test_judge_catches_wrong_math(self) -> None:
        context = _load_context()
        for control in CALIBRATION_CONTROLS:
            score = judge(
                control["question"],
                control["expected_answer"],
                control["bad_answer"],
                context,
            )
            assert score.precision < 3, (
                f"Judge failed to catch wrong math: {control['question']} "
                f"scored precision={score.precision}"
            )
            assert score.faithfulness < 3, (
                f"Judge failed to catch unfaithful answer: {control['question']} "
                f"scored faithfulness={score.faithfulness}"
            )


class TestPhase0MathBaseline:
    """Run all 11 QA pairs through the baseline on Lee Ch.1."""

    def test_math_baseline_eval(self) -> None:
        context = _load_context()
        results = []

        for pair in QA_PAIRS:
            actual = ask(pair["question"], DOCS_DIR)
            score = judge(pair["question"], pair["expected_answer"], actual, context)
            results.append({
                "question": pair["question"],
                "category": pair["category"],
                "expected_answer": pair["expected_answer"],
                "actual_answer": actual,
                "faithfulness": score.faithfulness,
                "relevance": score.relevance,
                "precision": score.precision,
                "reasoning": score.reasoning,
            })

        # Compute summary
        avg_faith = sum(r["faithfulness"] for r in results) / len(results)
        avg_rel = sum(r["relevance"] for r in results) / len(results)
        avg_prec = sum(r["precision"] for r in results) / len(results)
        unanswerable = [r for r in results if r["category"] == "unanswerable"]
        unanswerable_correct = sum(
            1 for r in unanswerable if r["faithfulness"] >= 4
        )

        summary = {
            "phase": "phase0_math_baseline",
            "corpus": "Lee, Introduction to Smooth Manifolds, Ch.1 (pp.19-49)",
            "total_questions": len(results),
            "average_faithfulness": round(avg_faith, 2),
            "average_relevance": round(avg_rel, 2),
            "average_precision": round(avg_prec, 2),
            "unanswerable_correct": unanswerable_correct,
            "unanswerable_total": len(unanswerable),
            "results": results,
        }

        # Print summary
        print("\n")
        print("Phase 0 Math Baseline Eval Results")
        print("Corpus: Lee, Smooth Manifolds, Ch.1")
        print("=" * 60)
        for r in results:
            cat = r["category"]
            f = r["faithfulness"]
            rel = r["relevance"]
            p = r["precision"]
            q = r["question"][:55]
            print(
                f"  [{cat:<13}] faith={f} rel={rel} prec={p}  \"{q}\""
            )
        print("-" * 60)
        print(f"  Average faithfulness: {avg_faith:.1f}")
        print(f"  Average relevance:    {avg_rel:.1f}")
        print(f"  Average precision:    {avg_prec:.1f}")
        print(f"  Unanswerable correct: {unanswerable_correct}/{len(unanswerable)}")
        print()

        # Write results JSON
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        results_path = RESULTS_DIR / "phase0_math_baseline.json"
        results_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        # Assertions — baseline thresholds
        assert avg_faith >= 3.0, f"Average faithfulness {avg_faith} below threshold 3.0"
        assert avg_rel >= 3.0, f"Average relevance {avg_rel} below threshold 3.0"
        assert avg_prec >= 3.0, f"Average precision {avg_prec} below threshold 3.0"


# ---------- Standalone runner ----------

if __name__ == "__main__":
    import io
    import sys

    # Windows console encoding fix
    if sys.stdout.encoding != "utf-8":
        sys.stdout = io.TextIOWrapper(
            sys.stdout.buffer, encoding="utf-8", errors="replace"
        )

    context = _load_context()

    print("=== Judge Calibration ===\n")
    cal_pass = True
    for control in CALIBRATION_CONTROLS:
        score = judge(
            control["question"],
            control["expected_answer"],
            control["bad_answer"],
            context,
        )
        status = "PASS" if score.precision < 3 and score.faithfulness < 3 else "FAIL"
        if status == "FAIL":
            cal_pass = False
        print(f"  [{status}] faith={score.faithfulness} prec={score.precision}  \"{control['question'][:50]}\"")
        print(f"         {score.reasoning}\n")

    if not cal_pass:
        print("CALIBRATION FAILED — judge is not reliable. Aborting.\n")
        sys.exit(1)

    print("=== Math Baseline Eval ===\n")
    results = []
    for pair in QA_PAIRS:
        _throttle()
        actual = ask(pair["question"], DOCS_DIR)
        score = judge(pair["question"], pair["expected_answer"], actual, context)
        results.append({
            "question": pair["question"],
            "category": pair["category"],
            "expected_answer": pair["expected_answer"],
            "actual_answer": actual,
            "faithfulness": score.faithfulness,
            "relevance": score.relevance,
            "precision": score.precision,
            "reasoning": score.reasoning,
        })
        cat = pair["category"]
        print(
            f"  [{cat:<13}] faith={score.faithfulness} rel={score.relevance} "
            f"prec={score.precision}  \"{pair['question'][:55]}\""
        )

    avg_faith = sum(r["faithfulness"] for r in results) / len(results)
    avg_rel = sum(r["relevance"] for r in results) / len(results)
    avg_prec = sum(r["precision"] for r in results) / len(results)
    unanswerable = [r for r in results if r["category"] == "unanswerable"]
    unanswerable_correct = sum(1 for r in unanswerable if r["faithfulness"] >= 4)

    print()
    print("-" * 60)
    print(f"  Average faithfulness: {avg_faith:.1f}")
    print(f"  Average relevance:    {avg_rel:.1f}")
    print(f"  Average precision:    {avg_prec:.1f}")
    print(f"  Unanswerable correct: {unanswerable_correct}/{len(unanswerable)}")

    summary = {
        "phase": "phase0_math_baseline",
        "corpus": "Lee, Introduction to Smooth Manifolds, Ch.1 (pp.19-49)",
        "total_questions": len(results),
        "average_faithfulness": round(avg_faith, 2),
        "average_relevance": round(avg_rel, 2),
        "average_precision": round(avg_prec, 2),
        "unanswerable_correct": unanswerable_correct,
        "unanswerable_total": len(unanswerable),
        "results": results,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_path = RESULTS_DIR / "phase0_math_baseline.json"
    results_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n  Results written to {results_path}")

    if avg_faith < 3.0 or avg_rel < 3.0 or avg_prec < 3.0:
        sys.exit(1)
