"""Shared pytest configuration for evals."""

import pytest

DEFAULT_JUDGE_MODEL = "claude-haiku-4-5-20251001"


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--judge-model",
        default=DEFAULT_JUDGE_MODEL,
        help="Model to use for LLM-as-judge scoring (default: %(default)s)",
    )


@pytest.fixture
def judge_model(request: pytest.FixtureRequest) -> str:
    return request.config.getoption("--judge-model")
