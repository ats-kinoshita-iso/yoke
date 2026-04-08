"""Tests for model client abstraction — factory, config parsing, and OllamaClient."""

import json

import httpx
import pytest

from yoke.config import YokeSettings, parse_model_spec
from yoke.models import ClaudeClient, ModelClient, OllamaClient, get_model_client


# ---------- parse_model_spec ----------


class TestParseModelSpec:
    def test_with_provider(self) -> None:
        assert parse_model_spec("ollama/gemma4:e2b") == ("ollama", "gemma4:e2b")

    def test_default_provider_is_claude(self) -> None:
        assert parse_model_spec("claude-sonnet-4-20250514") == (
            "claude",
            "claude-sonnet-4-20250514",
        )

    def test_openai_provider(self) -> None:
        assert parse_model_spec("openai/text-embedding-3-small") == (
            "openai",
            "text-embedding-3-small",
        )


# ---------- get_model_client factory ----------


class TestGetModelClient:
    def test_claude_returns_claude_client(self) -> None:
        client = get_model_client("claude", "claude-sonnet-4-20250514")
        assert isinstance(client, ClaudeClient)

    def test_ollama_returns_ollama_client(self) -> None:
        client = get_model_client("ollama", "gemma4:e2b")
        assert isinstance(client, OllamaClient)

    def test_unknown_provider_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown provider"):
            get_model_client("fake", "fake-model")


# ---------- ClaudeClient satisfies protocol ----------


class TestClaudeClient:
    def test_satisfies_protocol(self) -> None:
        client = ClaudeClient("claude-sonnet-4-20250514")
        assert isinstance(client, ModelClient)


# ---------- OllamaClient ----------


class TestOllamaClient:
    async def test_complete_with_system(self, httpx_mock) -> None:
        httpx_mock.add_response(json={"response": "The answer is 42."})

        client = OllamaClient("gemma4:e2b")
        result = await client.complete("What is the answer?", system="Be concise.")

        assert result == "The answer is 42."
        request = httpx_mock.get_requests()[0]
        body = json.loads(request.content)
        assert body["model"] == "gemma4:e2b"
        assert body["prompt"] == "What is the answer?"
        assert body["system"] == "Be concise."
        assert body["stream"] is False

    async def test_complete_without_system(self, httpx_mock) -> None:
        httpx_mock.add_response(json={"response": "Hello."})

        client = OllamaClient("gemma4:e2b")
        result = await client.complete("Hi")

        assert result == "Hello."
        request = httpx_mock.get_requests()[0]
        body = json.loads(request.content)
        assert "system" not in body

    async def test_custom_base_url(self, httpx_mock) -> None:
        httpx_mock.add_response(json={"response": "OK"})

        client = OllamaClient("gemma4:e2b", base_url="http://remote:11434")
        await client.complete("test")

        request = httpx_mock.get_requests()[0]
        assert str(request.url) == "http://remote:11434/api/generate"

    async def test_http_error_propagates(self, httpx_mock) -> None:
        httpx_mock.add_response(status_code=500)

        client = OllamaClient("gemma4:e2b")
        with pytest.raises(httpx.HTTPStatusError):
            await client.complete("test")

    def test_satisfies_protocol(self) -> None:
        client = OllamaClient("gemma4:e2b")
        assert isinstance(client, ModelClient)


# ---------- YokeSettings defaults ----------


class TestYokeSettings:
    def test_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Clear any YOKE_ env vars that might be set
        for key in (
            "YOKE_GENERATION_MODEL",
            "YOKE_JUDGE_MODEL",
            "YOKE_SUMMARY_MODEL",
            "YOKE_EMBEDDING_MODEL",
        ):
            monkeypatch.delenv(key, raising=False)

        settings = YokeSettings()
        assert settings.generation_model == "claude-sonnet-4-20250514"
        assert settings.judge_model == "claude-haiku-4-5-20251001"
        assert settings.summary_model == "claude-haiku-4-5-20251001"
        assert settings.embedding_model == "text-embedding-3-small"

    def test_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("YOKE_GENERATION_MODEL", "ollama/llama3:8b")
        settings = YokeSettings()
        assert settings.generation_model == "ollama/llama3:8b"
