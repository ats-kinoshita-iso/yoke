"""Shared pytest configuration for evals."""

import pytest

DEFAULT_JUDGE_MODEL = "claude-haiku-4-5-20251001"
DEFAULT_GENERATION_MODEL = "claude-sonnet-4-20250514"
LOCAL_MODEL = "ollama/gemma4:e4b"


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--judge-model",
        default=DEFAULT_JUDGE_MODEL,
        help="Model to use for LLM-as-judge scoring (default: %(default)s)",
    )
    parser.addoption(
        "--generation-model",
        default=DEFAULT_GENERATION_MODEL,
        help="Model to use for answer generation (default: %(default)s)",
    )
    parser.addoption(
        "--all-local",
        action="store_true",
        default=False,
        help=(
            f"Use {LOCAL_MODEL} for both generation and judging. "
            "Fast/free for dev iteration; overrides --judge-model and "
            "--generation-model."
        ),
    )


@pytest.fixture
def judge_model(request: pytest.FixtureRequest) -> str:
    if request.config.getoption("--all-local"):
        return LOCAL_MODEL
    return request.config.getoption("--judge-model")


@pytest.fixture
def generation_model(request: pytest.FixtureRequest) -> str:
    if request.config.getoption("--all-local"):
        return LOCAL_MODEL
    return request.config.getoption("--generation-model")


@pytest.fixture
def all_local(request: pytest.FixtureRequest) -> bool:
    return request.config.getoption("--all-local")
