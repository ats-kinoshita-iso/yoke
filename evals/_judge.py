"""Shared LLM-as-judge dispatch — works with both Claude (tool_use) and Ollama (JSON).

Usage:
    score = judge(
        model="claude-haiku-4-5-20251001",   # or "ollama/gemma4:e4b"
        system=JUDGE_SYSTEM,
        user_prompt="Chunk text: ...\nSummary: ...",
        tool=JUDGE_TOOL,           # Anthropic tool schema dict
        score_cls=JudgeScore,      # Pydantic model to parse into
    )
"""

import json
import re
import time

import anthropic
import httpx
from pydantic import BaseModel

from yoke.config import parse_model_spec

# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------

_last_api_call: float = 0.0
_API_DELAY: float = 2.0


def _throttle() -> None:
    global _last_api_call
    now = time.time()
    elapsed = now - _last_api_call
    if _last_api_call > 0 and elapsed < _API_DELAY:
        time.sleep(_API_DELAY - elapsed)
    _last_api_call = time.time()


# ---------------------------------------------------------------------------
# Claude judge (tool_use)
# ---------------------------------------------------------------------------


def _judge_claude(
    model: str,
    system: str,
    user_prompt: str,
    tool: dict,
    score_cls: type[BaseModel],
) -> BaseModel:
    """Score using Anthropic tool_use — structured, reliable."""
    _throttle()
    client = anthropic.Anthropic()
    message = client.messages.create(
        model=model,
        temperature=0,
        max_tokens=512,
        system=system,
        tools=[tool],
        tool_choice={"type": "tool", "name": tool["name"]},
        messages=[{"role": "user", "content": user_prompt}],
    )
    for block in message.content:
        if block.type == "tool_use":
            return score_cls(**block.input)
    raise ValueError("Claude judge did not return a tool_use block")


# ---------------------------------------------------------------------------
# Ollama judge (JSON prompt)
# ---------------------------------------------------------------------------

_JSON_SUFFIX = """

Respond with ONLY a JSON object (no markdown, no commentary) using this schema:
{schema}"""


def _extract_json(text: str) -> dict:
    """Extract a JSON object from potentially noisy LLM output."""
    # Try the whole text first
    text = text.strip()
    if text.startswith("{"):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

    # Find the first { ... } block
    match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    # Try nested braces
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not extract JSON from Ollama response:\n{text[:500]}")


def _build_json_schema(tool: dict) -> str:
    """Convert Anthropic tool schema into a compact JSON example for the prompt."""
    props = tool["input_schema"]["properties"]
    example = {}
    for name, prop in props.items():
        if prop.get("type") == "integer":
            example[name] = 3
        elif prop.get("type") == "string":
            example[name] = "brief explanation"
        else:
            example[name] = "..."
    return json.dumps(example, indent=2)


def _judge_ollama(
    model: str,
    system: str,
    user_prompt: str,
    tool: dict,
    score_cls: type[BaseModel],
    base_url: str = "http://localhost:11434",
) -> BaseModel:
    """Score using Ollama — prompt for JSON output, parse response."""
    schema_example = _build_json_schema(tool)

    # Build a system prompt that includes field descriptions
    field_descriptions = []
    for name, prop in tool["input_schema"]["properties"].items():
        desc = prop.get("description", "")
        field_descriptions.append(f"- {name}: {desc}")

    full_system = (
        f"{system}\n\n"
        f"Score fields:\n"
        + "\n".join(field_descriptions)
        + _JSON_SUFFIX.format(schema=schema_example)
    )

    payload = {
        "model": model,
        "system": full_system,
        "prompt": user_prompt,
        "stream": False,
    }
    resp = httpx.post(
        f"{base_url}/api/generate",
        json=payload,
        timeout=120.0,
    )
    resp.raise_for_status()
    raw = resp.json()["response"]

    data = _extract_json(raw)
    return score_cls(**data)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def judge(
    model: str,
    system: str,
    user_prompt: str,
    tool: dict,
    score_cls: type[BaseModel],
) -> BaseModel:
    """Score using an LLM judge — dispatches to Claude or Ollama.

    Args:
        model: Model spec, e.g. "claude-haiku-4-5-20251001" or "ollama/gemma4:e4b".
        system: System prompt for the judge.
        user_prompt: The user message containing chunk/summary/answer to judge.
        tool: Anthropic-style tool definition dict (used for both providers).
        score_cls: Pydantic model class to parse the result into.

    Returns:
        Instance of score_cls with the judge's scores.
    """
    provider, model_name = parse_model_spec(model)

    if provider == "ollama":
        return _judge_ollama(model_name, system, user_prompt, tool, score_cls)
    else:
        return _judge_claude(model_name, system, user_prompt, tool, score_cls)


def generate(
    model: str,
    prompt: str,
    *,
    system: str = "",
    max_tokens: int = 1024,
) -> str:
    """Generate a completion — dispatches to Claude or Ollama.

    Thin wrapper around yoke.config.generate that adds rate throttling
    for Claude API calls during eval runs.
    """
    from yoke.config import generate as _generate

    provider, _ = parse_model_spec(model)
    if provider != "ollama":
        _throttle()
    return _generate(model, prompt, system=system, max_tokens=max_tokens)
