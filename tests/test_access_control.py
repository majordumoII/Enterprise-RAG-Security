import pytest

from src.rag_guardrails.auth.models import ClearanceLevel, UserContext
from src.rag_guardrails.security.access_control import (
    build_acl_predicate,
    filter_chunks_by_permission,
    is_visible,
)


class TestBuildAclPredicate:
    def test_returns_sql_and_params(self, internal_user: UserContext):
        sql, params = build_acl_predicate(internal_user)

        assert "metadata->>'sensitivity'" in sql
        assert "allowed_roles" in sql
        assert params == [int(ClearanceLevel.INTERNAL), ["engineering"]]

    def test_custom_metadata_column(self, internal_user: UserContext):
        sql, _ = build_acl_predicate(internal_user, metadata_column="m")
        assert "m->>'sensitivity'" in sql
        assert "m->'allowed_roles'" in sql


class TestIsVisible:
    @pytest.mark.parametrize(
        "metadata,user_fixture,expected",
        [
            # public chunk visible to everyone
            ({"sensitivity": "public"}, "public_user", True),
            ({"sensitivity": "public"}, "restricted_user", True),
            # missing metadata fails closed to RESTRICTED -> blocked for lower clearance
            ({}, "public_user", False),
            ({}, "restricted_user", True),
            # unrecognized sensitivity string fails closed
            ({"sensitivity": "top-secret"}, "confidential_finance_user", False),
            ({"sensitivity": "top-secret"}, "restricted_user", True),
            # under-cleared user blocked regardless of role
            ({"sensitivity": "confidential", "allowed_roles": ["finance"]}, "internal_user", False),
            # cleared but no role overlap blocked
            ({"sensitivity": "internal", "allowed_roles": ["hr"]}, "internal_user", False),
            # cleared and role overlap allowed
            (
                {"sensitivity": "confidential", "allowed_roles": ["finance"]},
                "confidential_finance_user",
                True,
            ),
            # empty allowed_roles list means no role restriction
            ({"sensitivity": "internal", "allowed_roles": []}, "internal_user", True),
            # allowed_roles present but malformed (not a list) treated as no match
            ({"sensitivity": "public", "allowed_roles": "finance"}, "public_user", False),
        ],
    )
    def test_visibility_cases(self, metadata, user_fixture, expected, request):
        user = request.getfixturevalue(user_fixture)
        assert is_visible(metadata, user) is expected


class TestFilterChunksByPermission:
    def test_drops_disallowed_chunks(self, internal_user: UserContext):
        chunks = [
            {"id": 1, "metadata": {"sensitivity": "public"}},
            {"id": 2, "metadata": {"sensitivity": "confidential"}},
            {"id": 3, "metadata": {"sensitivity": "internal", "allowed_roles": ["engineering"]}},
            {"id": 4, "metadata": {"sensitivity": "internal", "allowed_roles": ["hr"]}},
        ]

        visible = filter_chunks_by_permission(chunks, internal_user)

        assert [c["id"] for c in visible] == [1, 3]

    def test_missing_metadata_key_defaults_to_restricted(self, public_user: UserContext):
        chunks = [{"id": 1}]
        assert filter_chunks_by_permission(chunks, public_user) == []
