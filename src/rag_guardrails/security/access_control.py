"""Permission model for chunk-level access control.

Each chunk's existing `metadata` JSONB column (from Project 1) may carry:
    - "sensitivity": one of "public" | "internal" | "confidential" | "restricted"
    - "allowed_roles": list[str], empty/absent = visible to any authenticated user

Access is granted iff the user's clearance >= the chunk's sensitivity AND
(the chunk has no role restriction OR the user has an overlapping role).
Chunks with missing or unrecognized sensitivity fail closed to RESTRICTED.

Filtering happens twice: once as a SQL predicate pushed into the similarity
query (`build_acl_predicate`), and again in Python against the returned rows
(`filter_chunks_by_permission`) as a defense-in-depth check — if the SQL
predicate is ever misapplied or bypassed, the second gate still stops
disallowed content from reaching the LLM prompt.
"""

import logging

from ..auth.models import ClearanceLevel, UserContext

logger = logging.getLogger(__name__)

_SENSITIVITY_RANK: dict[str, ClearanceLevel] = {
    "public": ClearanceLevel.PUBLIC,
    "internal": ClearanceLevel.INTERNAL,
    "confidential": ClearanceLevel.CONFIDENTIAL,
    "restricted": ClearanceLevel.RESTRICTED,
}

_SENSITIVITY_RANK_SQL = """CASE lower(coalesce({column}->>'sensitivity', 'restricted'))
        WHEN 'public' THEN 0
        WHEN 'internal' THEN 1
        WHEN 'confidential' THEN 2
        WHEN 'restricted' THEN 3
        ELSE 3
    END"""


def build_acl_predicate(
    user: UserContext, metadata_column: str = "metadata"
) -> tuple[str, list]:
    """Build a SQL WHERE-clause fragment (and its params) enforcing the ACL.

    Intended to be interpolated into a query alongside other %s placeholders,
    e.g. `... WHERE {predicate} ORDER BY embedding <=> %s ...`.
    """
    predicate = f"""(
        {_SENSITIVITY_RANK_SQL.format(column=metadata_column)} <= %s
        AND (
            NOT ({metadata_column} ? 'allowed_roles')
            OR jsonb_array_length({metadata_column}->'allowed_roles') = 0
            OR {metadata_column}->'allowed_roles' ?| %s
        )
    )"""
    params = [int(user.clearance), user.roles]
    return predicate, params


def _chunk_sensitivity(metadata: dict) -> ClearanceLevel:
    raw = str(metadata.get("sensitivity", "restricted")).strip().lower()
    return _SENSITIVITY_RANK.get(raw, ClearanceLevel.RESTRICTED)


def _chunk_allowed_roles(metadata: dict) -> list[str] | None:
    """None means "no role restriction"; a list (possibly empty) means the
    chunk is role-gated. Malformed (non-list) values fail closed to an empty
    list, so they gate the chunk rather than silently opening it to everyone.
    """
    if "allowed_roles" not in metadata:
        return None
    roles = metadata["allowed_roles"]
    if not isinstance(roles, list):
        return []
    if not roles:
        return None
    return [str(r) for r in roles]


def is_visible(metadata: dict, user: UserContext) -> bool:
    """Pure re-implementation of the SQL ACL predicate, for defense-in-depth."""
    if _chunk_sensitivity(metadata) > user.clearance:
        return False
    allowed_roles = _chunk_allowed_roles(metadata)
    if allowed_roles is None:
        return True
    return bool(set(allowed_roles) & set(user.roles))


def filter_chunks_by_permission(chunks: list[dict], user: UserContext) -> list[dict]:
    """Re-check permissions on already-fetched rows before they reach the LLM."""
    visible = []
    dropped = 0
    for chunk in chunks:
        metadata = chunk.get("metadata") or {}
        if is_visible(metadata, user):
            visible.append(chunk)
        else:
            dropped += 1
    if dropped:
        logger.warning(
            "Defense-in-depth filter dropped %d chunk(s) that the SQL ACL "
            "predicate should already have excluded for user %s",
            dropped,
            user.user_id,
        )
    return visible
