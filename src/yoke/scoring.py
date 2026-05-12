"""Runtime quality scoring — faithfulness and relevance via LLM-as-judge."""

from __future__ import annotations

import anthropic
from pydantic import BaseModel, Field


class QualityScore(BaseModel):
    """Faithfulness + relevance score for an agent response."""

    faithfulness: int = Field(ge=1, le=5)
    relevance: int = Field(ge=1, le=5)
    reasoning: str


# ---------- Judge tool schema (Anthropic tool_use) ----------

_JUDGE_TOOL = {
    "name": "score_answer",
    "description": "Score the answer on faithfulness and relevance.",
    "input_schema": {
        "type": "object",
        "properties": {
            "faithfulness": {
                "type": "integer",
                "description": (
                    "1-5. Is the answer supported by the context? "
                    "5 = fully grounded, 1 = hallucinated."
                ),
            },
            "relevance": {
                "type": "integer",
                "description": (
                    "1-5. Does the answer address the question? "
                    "5 = directly answers, 1 = off-topic."
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

_JUDGE_SYSTEM = (
    "You are an evaluation judge. You will be given a question, the model's "
    "actual answer, and the source context. Score the actual answer using the "
    "score_answer tool.\n\n"
    "Faithfulness (1-5): Is the answer supported by the source context? "
    "5 = every claim is directly supported, 1 = contains fabricated information.\n\n"
    "Relevance (1-5): Does the answer address the question asked? "
    "5 = directly and completely answers, 1 = completely off-topic.\n\n"
    "For unanswerable questions: if the model correctly declines to answer "
    "(says it doesn't have enough information), score faithfulness=5 and "
    "relevance=5. If it fabricates an answer, score faithfulness=1."
)

DEFAULT_JUDGE_MODEL = "claude-haiku-4-5-20251001"


async def score_response(
    question: str,
    answer: str,
    context: str,
    *,
    model: str = DEFAULT_JUDGE_MODEL,
) -> QualityScore:
    """Score an answer for faithfulness and relevance using an LLM judge.

    Args:
        question: The original user question.
        answer: The model's response to score.
        context: The source documents / retrieved context.
        model: Judge model to use (default: claude-haiku).

    Returns:
        QualityScore with faithfulness, relevance, and reasoning.
    """
    client = anthropic.AsyncAnthropic()
    message = await client.messages.create(
        model=model,
        temperature=0,
        max_tokens=512,
        system=_JUDGE_SYSTEM,
        tools=[_JUDGE_TOOL],
        tool_choice={"type": "tool", "name": "score_answer"},
        messages=[
            {
                "role": "user",
                "content": (
                    f"Question: {question}\n\n"
                    f"Actual answer: {answer}\n\n"
                    f"Source context:\n{context}"
                ),
            }
        ],
    )

    for block in message.content:
        if block.type == "tool_use":
            return QualityScore(**block.input)

    raise ValueError("Judge did not return a tool_use block")
