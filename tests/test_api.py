from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from src.rag_guardrails.api.app import create_app
from src.rag_guardrails.pipeline import RAGResponse


@pytest.fixture
def client(config, mocker):
    mock_pipeline_cls = mocker.patch("src.rag_guardrails.api.app.RAGPipeline")
    mock_pipeline = mock_pipeline_cls.return_value
    mock_pipeline.query = AsyncMock(
        return_value=RAGResponse(
            answer="the answer",
            sources=[
                {
                    "filename": "doc.pdf",
                    "chunk_index": 0,
                    "content": "hello",
                    "similarity": 0.9,
                }
            ],
        )
    )

    app = create_app(config)
    with TestClient(app) as c:
        c.mock_pipeline = mock_pipeline
        yield c


class TestHealth:
    def test_health_ok(self, client: TestClient):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


class TestQuery:
    def test_query_returns_answer_and_sources(self, client: TestClient):
        response = client.post(
            "/query",
            json={
                "question": "what is the policy?",
                "user_id": "bob",
                "clearance": "internal",
                "roles": ["engineering"],
            },
        )

        assert response.status_code == 200
        body = response.json()
        assert body["answer"] == "the answer"
        assert body["sources"][0]["filename"] == "doc.pdf"

    def test_query_builds_user_context_from_body(self, client: TestClient):
        client.post(
            "/query",
            json={
                "question": "q",
                "user_id": "bob",
                "clearance": "confidential",
                "roles": ["finance"],
                "top_k": 2,
            },
        )

        args, kwargs = client.mock_pipeline.query.call_args
        assert args[0] == "q"
        user = args[1]
        assert user.user_id == "bob"
        assert user.roles == ["finance"]
        assert kwargs["top_k"] == 2

    def test_query_rejects_unknown_clearance(self, client: TestClient):
        response = client.post(
            "/query",
            json={"question": "q", "user_id": "bob", "clearance": "nonsense"},
        )
        assert response.status_code == 400
