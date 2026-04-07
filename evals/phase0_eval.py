"""Phase 0 baseline eval — LLM-as-judge scoring for naive context stuffing."""

import json
from pathlib import Path

import anthropic
import pytest
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from yoke.baseline import ask

load_dotenv(override=True)

DOCS_DIR = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "docs"
RESULTS_DIR = Path(__file__).resolve().parent / "results"

# ---------- Pydantic model for structured judge output ----------

class JudgeScore(BaseModel):
    faithfulness: int = Field(ge=1, le=5)
    relevance: int = Field(ge=1, le=5)
    reasoning: str


# ---------- QA pairs ----------

QA_PAIRS = [
    # Direct lookup (3)
    {
        "question": "What port does the API Gateway run on?",
        "expected_answer": "The API Gateway runs on port 8090.",
        "category": "direct",
    },
    {
        "question": "What is the default value of YOKE_EMBEDDING_BATCH_SIZE?",
        "expected_answer": "The default value is 64.",
        "category": "direct",
    },
    {
        "question": "What error code corresponds to 'Redis connection timeout'?",
        "expected_answer": "Error code YK-005.",
        "category": "direct",
    },
    # Cross-document synthesis (3)
    {
        "question": "If I get error YK-002, what embedding model and dimension count should I switch to?",
        "expected_answer": "You should use the voyage-3 model which produces 1024-dimension vectors.",
        "category": "cross",
    },
    {
        "question": "Which endpoint does not require the Authorization: Bearer header, and what does it return?",
        "expected_answer": "GET /health does not require authentication. It returns {\"status\": \"ok\", \"version\": \"0.1.0\"}.",
        "category": "cross",
    },
    {
        "question": "What are the three required environment variables, and which service uses Redis?",
        "expected_answer": "ANTHROPIC_API_KEY, DATABASE_URL, and REDIS_URL are required. Redis is used for async job processing.",
        "category": "cross",
    },
    # Reasoning (2)
    {
        "question": "If hybrid retrieval is falling back to dense-only, what is the most likely cause and how would you diagnose it?",
        "expected_answer": "The BM25 index is likely unavailable. Enable DEBUG logging with YOKE_LOG_LEVEL=DEBUG and check the retrieval logs.",
        "category": "reasoning",
    },
    {
        "question": "A user is hitting error YK-004. Their document is 200,000 tokens. What two steps should they take?",
        "expected_answer": "Split the document into chunks under 50,000 tokens and re-ingest with yoke-ingest --chunk-size 50000. Also reduce top_k to retrieve fewer documents.",
        "category": "reasoning",
    },
    # Unanswerable (2)
    {
        "question": "What is the pricing per API call for the Yoke platform?",
        "expected_answer": "This information is not in the documentation.",
        "category": "unanswerable",
    },
    {
        "question": "How do I configure SSO/SAML authentication for Yoke?",
        "expected_answer": "This information is not in the documentation.",
        "category": "unanswerable",
    },
]

# ---------- Judge ----------

JUDGE_TOOL = {
    "name": "score_answer",
    "description": "Score the answer on faithfulness and relevance.",
    "input_schema": {
        "type": "object",
        "properties": {
            "faithfulness": {
                "type": "integer",
                "description": "1-5. Is the answer supported by the context? 5=fully grounded, 1=hallucinated.",
            },
            "relevance": {
                "type": "integer",
                "description": "1-5. Does the answer address the question? 5=directly answers, 1=off-topic.",
            },
            "reasoning": {
                "type": "string",
                "description": "Brief explanation of the scores.",
            },
        },
        "required": ["faithfulness", "relevance", "reasoning"],
    },
}

JUDGE_SYSTEM = (
    "You are an evaluation judge. You will be given a question, the expected answer, "
    "the model's actual answer, and the source context. Score the actual answer using "
    "the score_answer tool.\n\n"
    "Faithfulness (1-5): Is the answer supported by the source context? "
    "5 = every claim is directly supported, 1 = contains fabricated information.\n\n"
    "Relevance (1-5): Does the answer address the question asked? "
    "5 = directly and completely answers, 1 = completely off-topic.\n\n"
    "For unanswerable questions: if the model correctly declines to answer "
    "(says it doesn't have enough information), score faithfulness=5 and relevance=5. "
    "If it fabricates an answer, score faithfulness=1."
)


def judge(question: str, expected: str, actual: str, context: str) -> JudgeScore:
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
            return JudgeScore(**block.input)
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
        "question": "What port does the API Gateway run on?",
        "expected_answer": "The API Gateway runs on port 8090.",
        "bad_answer": "The API Gateway runs on port 443 and uses NGINX as a reverse proxy with WebSocket support.",
        "category": "calibration",
    },
    {
        "question": "What embedding model does Yoke use?",
        "expected_answer": "Yoke uses the voyage-3 model with 1024 dimensions.",
        "bad_answer": "Yoke uses OpenAI's text-embedding-3-large model with 3072 dimensions and cosine similarity.",
        "category": "calibration",
    },
]


# ---------- Pytest tests ----------

class TestPhase0Calibration:
    """Verify the judge scores intentionally wrong answers below 3."""

    def test_judge_catches_hallucination(self) -> None:
        context = _load_context()
        for control in CALIBRATION_CONTROLS:
            score = judge(
                control["question"],
                control["expected_answer"],
                control["bad_answer"],
                context,
            )
            assert score.faithfulness < 3, (
                f"Judge failed to catch hallucination: {control['question']} "
                f"scored faithfulness={score.faithfulness}"
            )


class TestPhase0Baseline:
    """Run all 10 QA pairs through the baseline and score them."""

    def test_baseline_eval(self) -> None:
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
                "reasoning": score.reasoning,
            })

        # Compute summary
        avg_faith = sum(r["faithfulness"] for r in results) / len(results)
        avg_rel = sum(r["relevance"] for r in results) / len(results)
        unanswerable = [r for r in results if r["category"] == "unanswerable"]
        unanswerable_correct = sum(
            1 for r in unanswerable if r["faithfulness"] >= 4
        )

        summary = {
            "phase": "phase0_baseline",
            "total_questions": len(results),
            "average_faithfulness": round(avg_faith, 2),
            "average_relevance": round(avg_rel, 2),
            "unanswerable_correct": unanswerable_correct,
            "unanswerable_total": len(unanswerable),
            "results": results,
        }

        # Print summary
        print("\n")
        print("Phase 0 Baseline Eval Results")
        print("=" * 50)
        for r in results:
            cat = r["category"]
            f = r["faithfulness"]
            rel = r["relevance"]
            q = r["question"][:60]
            print(f"  [{cat:<13}] faithfulness={f} relevance={rel}  \"{q}\"")
        print("-" * 50)
        print(f"  Average faithfulness: {avg_faith:.1f}")
        print(f"  Average relevance:    {avg_rel:.1f}")
        print(f"  Unanswerable correct: {unanswerable_correct}/{len(unanswerable)}")
        print()

        # Write results JSON
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        results_path = RESULTS_DIR / "phase0_baseline.json"
        results_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        # Assertions
        assert avg_faith >= 3.0, f"Average faithfulness {avg_faith} below threshold 3.0"
        assert avg_rel >= 3.0, f"Average relevance {avg_rel} below threshold 3.0"


# ---------- Standalone runner ----------

if __name__ == "__main__":
    import sys

    context = _load_context()

    print("=== Judge Calibration ===\n")
    for control in CALIBRATION_CONTROLS:
        score = judge(
            control["question"],
            control["expected_answer"],
            control["bad_answer"],
            context,
        )
        status = "PASS" if score.faithfulness < 3 else "FAIL"
        print(f"  [{status}] faithfulness={score.faithfulness}  \"{control['question'][:50]}\"")
        print(f"         {score.reasoning}\n")

    print("=== Baseline Eval ===\n")
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
            "reasoning": score.reasoning,
        })
        cat = pair["category"]
        print(f"  [{cat:<13}] faithfulness={score.faithfulness} relevance={score.relevance}  \"{pair['question'][:60]}\"")

    avg_faith = sum(r["faithfulness"] for r in results) / len(results)
    avg_rel = sum(r["relevance"] for r in results) / len(results)
    unanswerable = [r for r in results if r["category"] == "unanswerable"]
    unanswerable_correct = sum(1 for r in unanswerable if r["faithfulness"] >= 4)

    print()
    print("-" * 50)
    print(f"  Average faithfulness: {avg_faith:.1f}")
    print(f"  Average relevance:    {avg_rel:.1f}")
    print(f"  Unanswerable correct: {unanswerable_correct}/{len(unanswerable)}")

    summary = {
        "phase": "phase0_baseline",
        "total_questions": len(results),
        "average_faithfulness": round(avg_faith, 2),
        "average_relevance": round(avg_rel, 2),
        "unanswerable_correct": unanswerable_correct,
        "unanswerable_total": len(unanswerable),
        "results": results,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_path = RESULTS_DIR / "phase0_baseline.json"
    results_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n  Results written to {results_path}")

    if avg_faith < 3.0 or avg_rel < 3.0:
        sys.exit(1)
