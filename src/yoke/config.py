"""Application settings and shared model dispatch."""

import anthropic
import httpx
from pydantic_settings import BaseSettings, SettingsConfigDict


class YokeSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="YOKE_")

    generation_model: str = "claude-sonnet-4-20250514"
    judge_model: str = "claude-haiku-4-5-20251001"
    summary_model: str = "claude-haiku-4-5-20251001"
    embedding_model: str = "text-embedding-3-small"

    # Langfuse observability
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"
    tracing_enabled: bool = True
    quality_scoring_enabled: bool = True


def parse_model_spec(spec: str) -> tuple[str, str]:
    """Parse 'provider/model' into (provider, model).

    Examples:
        'ollama/gemma4:e4b' → ('ollama', 'gemma4:e4b')
        'claude-sonnet-4-20250514' → ('claude', 'claude-sonnet-4-20250514')
    """
    if "/" in spec:
        provider, model = spec.split("/", 1)
        return provider, model
    return "claude", spec


def generate(
    model: str,
    prompt: str,
    *,
    system: str = "",
    max_tokens: int = 1024,
) -> str:
    """Generate a completion — dispatches to Claude API or Ollama.

    Args:
        model: Model spec, e.g. "claude-sonnet-4-20250514" or "ollama/gemma4:e4b".
        prompt: The user message / prompt text.
        system: Optional system prompt.
        max_tokens: Maximum tokens in the response.
    """
    provider, model_name = parse_model_spec(model)

    if provider == "ollama":
        payload: dict[str, object] = {
            "model": model_name,
            "prompt": prompt,
            "stream": False,
        }
        if system:
            payload["system"] = system
        resp = httpx.post(
            "http://localhost:11434/api/generate", json=payload, timeout=120.0
        )
        resp.raise_for_status()
        return resp.json()["response"]

    client = anthropic.Anthropic()
    message = client.messages.create(
        model=model_name,
        temperature=0,
        max_tokens=max_tokens,
        system=system or "",
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text
