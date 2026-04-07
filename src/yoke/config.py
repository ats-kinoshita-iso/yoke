"""Application settings read from environment variables."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class YokeSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="YOKE_")

    generation_model: str = "claude-sonnet-4-20250514"
    judge_model: str = "claude-haiku-4-5-20251001"
    summary_model: str = "ollama/gemma4:e2b"
    embedding_model: str = "text-embedding-3-small"


def parse_model_spec(spec: str) -> tuple[str, str]:
    """Parse 'provider/model' into (provider, model). Bare names default to 'claude'."""
    if "/" in spec:
        provider, model = spec.split("/", 1)
        return provider, model
    return "claude", spec
