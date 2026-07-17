from unittest.mock import AsyncMock, MagicMock

import pytest

from src.rag_guardrails.auth.models import UserContext
from src.rag_guardrails.config import RAGConfig
from src.rag_guardrails.pipeline import RAGPipeline


@pytest.fixture
def pipeline(config: RAGConfig, mocker) -> RAGPipeline:
    mocker.patch("src.rag_guardrails.pipeline.QueryEmbedder")
    mocker.patch("src.rag_guardrails.pipeline.PermissionAwareVectorStore")
    mocker.patch("src.rag_guardrails.pipeline.get_chat_llm")
    mocker.patch("src.rag_guardrails.pipeline.GuardrailsEngine")

    p = RAGPipeline(config)
    p.embedder.embed = MagicMock(return_value=[0.1, 0.2])
    p.vector_store.search = MagicMock(
        return_value=[
            {"filename": "doc.pdf", "chunk_index": 0, "content": "hello", "similarity": 0.9}
        ]
    )
    p.guardrails.generate = AsyncMock(return_value="the answer")
    return p


class TestRagPipelineQuery:
    async def test_query_orchestrates_retrieve_then_generate(
        self, pipeline: RAGPipeline, internal_user: UserContext
    ):
        result = await pipeline.query("what is the policy?", internal_user, top_k=3)

        pipeline.embedder.embed.assert_called_once_with("what is the policy?")
        pipeline.vector_store.search.assert_called_once_with(
            [0.1, 0.2], internal_user, 3
        )
        assert result.answer == "the answer"
        assert result.sources[0]["filename"] == "doc.pdf"

    async def test_query_passes_context_into_system_prompt(
        self, pipeline: RAGPipeline, internal_user: UserContext
    ):
        await pipeline.query("q", internal_user)

        messages = pipeline.guardrails.generate.call_args.kwargs["messages"]
        assert messages[0]["role"] == "system"
        assert "doc.pdf" in messages[0]["content"]
        assert messages[1] == {"role": "user", "content": "q"}

    async def test_query_with_no_chunks_notes_absence_in_context(
        self, pipeline: RAGPipeline, internal_user: UserContext
    ):
        pipeline.vector_store.search.return_value = []

        await pipeline.query("q", internal_user)

        messages = pipeline.guardrails.generate.call_args.kwargs["messages"]
        assert "no matching documents" in messages[0]["content"]
