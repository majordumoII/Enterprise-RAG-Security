"""Shared fixtures for all rag_guardrails tests."""

import os
from collections.abc import Generator

import pytest

from src.rag_guardrails.auth.models import ClearanceLevel, UserContext
from src.rag_guardrails.config import RAGConfig


@pytest.fixture
def mock_env() -> Generator[None, None, None]:
    """Set common test env vars and restore after."""
    env_vars = {
        "GOOGLE_CLOUD_PROJECT": "test-project",
        "EMBEDDING_MODEL": "text-embedding-005",
        "EMBEDDING_DIMENSIONS": "768",
        "PG_CONNECTION_STRING": "postgresql://user:pass@localhost:5432/testdb",
        "VECTOR_TABLE": "test_chunks",
        "LLM_PROVIDER": "openai",
        "OPENAI_API_KEY": "sk-test",
        "OPENAI_MODEL": "gpt-4.1-mini",
        "DEEPSEEK_API_KEY": "ds-test",
        "DEEPSEEK_MODEL": "deepseek-chat",
        "DEEPSEEK_BASE_URL": "https://api.deepseek.com",
        "GUARDRAILS_CONFIG_PATH": "config/rails",
        "TOP_K": "5",
    }
    saved = {k: os.environ.get(k) for k in env_vars}
    os.environ.update(env_vars)
    yield
    for k in env_vars:
        if saved[k] is not None:
            os.environ[k] = saved[k]
        else:
            os.environ.pop(k, None)


@pytest.fixture
def config(mock_env: None) -> RAGConfig:
    return RAGConfig.from_env()


@pytest.fixture
def public_user() -> UserContext:
    return UserContext(user_id="alice", clearance=ClearanceLevel.PUBLIC, roles=[])


@pytest.fixture
def internal_user() -> UserContext:
    return UserContext(user_id="bob", clearance=ClearanceLevel.INTERNAL, roles=["engineering"])


@pytest.fixture
def confidential_finance_user() -> UserContext:
    return UserContext(
        user_id="carol", clearance=ClearanceLevel.CONFIDENTIAL, roles=["finance"]
    )


@pytest.fixture
def restricted_user() -> UserContext:
    return UserContext(user_id="dave", clearance=ClearanceLevel.RESTRICTED, roles=["security"])
