"""Tests for PermissionAwareVectorStore (with mocked pg8000 and pgvector)."""

from unittest.mock import MagicMock

import pytest

from src.rag_guardrails.auth.models import UserContext
from src.rag_guardrails.retrieval.vector_store import PermissionAwareVectorStore


@pytest.fixture
def mock_conn():
    conn = MagicMock()
    cursor = MagicMock()
    cursor.fetchone.return_value = [42]
    cursor.fetchall.return_value = []
    cursor.rowcount = 0
    conn.cursor.return_value = cursor
    return conn


class TestSearch:
    def test_search_applies_acl_predicate_and_params(
        self, config, mocker, mock_conn, internal_user: UserContext
    ):
        mocker.patch("pg8000.dbapi.connect", return_value=mock_conn)
        mocker.patch(
            "src.rag_guardrails.retrieval.vector_store._register_vector_dbapi"
        )
        cursor = mock_conn.cursor.return_value
        cursor.fetchall.return_value = [
            (1, "gs://b/doc.pdf", "doc.pdf", 0, "Content 1", {"sensitivity": "public"}, 0.95),
        ]

        store = PermissionAwareVectorStore(config)
        results = store.search(query_embedding=[0.1, 0.2], user=internal_user, top_k=3)

        sql, params = cursor.execute.call_args[0]
        assert "allowed_roles" in sql
        assert "WHERE" in sql
        # params: query vector, clearance rank, roles list, query vector again, top_k
        assert params[1] == int(internal_user.clearance)
        assert params[2] == internal_user.roles
        assert params[-1] == 3

        assert len(results) == 1
        assert results[0]["filename"] == "doc.pdf"

    def test_search_defense_in_depth_drops_leaked_row(
        self, config, mocker, mock_conn, internal_user: UserContext
    ):
        """Even if the SQL predicate somehow let a disallowed row through,
        the Python-side re-check must still remove it before it reaches the LLM."""
        mocker.patch("pg8000.dbapi.connect", return_value=mock_conn)
        mocker.patch(
            "src.rag_guardrails.retrieval.vector_store._register_vector_dbapi"
        )
        cursor = mock_conn.cursor.return_value
        cursor.fetchall.return_value = [
            (1, "gs://b/pub.pdf", "pub.pdf", 0, "public content", {"sensitivity": "public"}, 0.9),
            (
                2,
                "gs://b/secret.pdf",
                "secret.pdf",
                0,
                "restricted content",
                {"sensitivity": "restricted"},
                0.99,
            ),
        ]

        store = PermissionAwareVectorStore(config)
        results = store.search(query_embedding=[0.1], user=internal_user)

        assert [r["filename"] for r in results] == ["pub.pdf"]

    def test_search_parses_string_jsonb_metadata(
        self, config, mocker, mock_conn, internal_user: UserContext
    ):
        mocker.patch("pg8000.dbapi.connect", return_value=mock_conn)
        mocker.patch(
            "src.rag_guardrails.retrieval.vector_store._register_vector_dbapi"
        )
        cursor = mock_conn.cursor.return_value
        cursor.fetchall.return_value = [
            (1, "gs://b/doc.pdf", "doc.pdf", 0, "Content", '{"sensitivity": "public"}', 0.9),
        ]

        store = PermissionAwareVectorStore(config)
        results = store.search(query_embedding=[0.1], user=internal_user)

        assert results[0]["metadata"] == {"sensitivity": "public"}


class TestUpdatePermissions:
    def test_merges_sensitivity_and_roles(self, config, mocker, mock_conn):
        mocker.patch("pg8000.dbapi.connect", return_value=mock_conn)
        mocker.patch(
            "src.rag_guardrails.retrieval.vector_store._register_vector_dbapi"
        )
        cursor = mock_conn.cursor.return_value
        cursor.rowcount = 3

        store = PermissionAwareVectorStore(config)
        updated = store.update_permissions(
            "doc.pdf", sensitivity="confidential", allowed_roles=["finance", "legal"]
        )

        assert updated == 3
        sql, params = cursor.execute.call_args[0]
        assert "UPDATE" in sql
        assert "metadata || %s::jsonb" in sql
        assert params[1] == "doc.pdf"
        assert mock_conn.commit.called

    def test_no_fields_returns_zero_without_query(self, config, mocker, mock_conn):
        mocker.patch("pg8000.dbapi.connect", return_value=mock_conn)
        mocker.patch(
            "src.rag_guardrails.retrieval.vector_store._register_vector_dbapi"
        )

        store = PermissionAwareVectorStore(config)
        updated = store.update_permissions("doc.pdf")

        assert updated == 0
        mock_conn.cursor.return_value.execute.assert_not_called()


class TestClose:
    def test_close_idempotent_when_never_opened(self, config):
        store = PermissionAwareVectorStore(config)
        store.close()
