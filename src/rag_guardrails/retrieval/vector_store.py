"""Permission-aware similarity search over Project 1's pgvector table."""

import json
import logging
from typing import Any

from google.cloud.sql.connector import Connector
from pgvector import Vector

from ..auth.models import UserContext
from ..config import RAGConfig
from ..security.access_control import build_acl_predicate, filter_chunks_by_permission

logger = logging.getLogger(__name__)


def _register_vector_dbapi(conn: Any) -> None:
    """Register pgvector in/out adapters on a pg8000 DBAPI connection.

    Mirrors Project 1's vector_store.py: the Cloud SQL Connector and local
    dbapi.connect() both produce pg8000.dbapi.Connection, which lacks the
    .run() helper pgvector.pg8000.register_vector expects, but supports the
    same register_in/out_adapter calls directly.
    """
    cur = conn.cursor()
    cur.execute(
        "SELECT typname, oid FROM pg_type WHERE oid IN "
        "(to_regtype('vector'), to_regtype('halfvec'), to_regtype('sparsevec'))"
    )
    type_info = dict(cur.fetchall())
    cur.close()

    if "vector" not in type_info:
        raise RuntimeError("vector type not found in the database")

    conn.register_out_adapter(Vector, lambda v: v.to_text())
    conn.register_in_adapter(type_info["vector"], Vector.from_text)


class PermissionAwareVectorStore:
    """Reads Project 1's document_chunks table with an ACL filter baked into the query."""

    def __init__(self, config: RAGConfig):
        self.config = config
        self._conn: Any = None
        self._closed = True
        self._connector: Connector | None = None

    @property
    def conn(self):
        if self._conn is None or self._closed:
            if self.config.db_instance_connection_name:
                self._connector = Connector()
                self._conn = self._connector.connect(
                    self.config.db_instance_connection_name,
                    "pg8000",
                    user=self.config.db_user,
                    password=self.config.db_password,
                    db=self.config.db_name,
                )
            else:
                from urllib.parse import urlparse

                import pg8000.dbapi as dbapi

                parsed = urlparse(self.config.pg_connection_string)
                self._conn = dbapi.connect(
                    user=parsed.username,
                    password=parsed.password,
                    host=parsed.hostname or "localhost",
                    port=parsed.port or 5432,
                    database=parsed.path.lstrip("/"),
                )
            _register_vector_dbapi(self._conn)
            self._closed = False
        return self._conn

    def search(
        self,
        query_embedding: list[float],
        user: UserContext,
        top_k: int | None = None,
    ) -> list[dict]:
        """Find the top_k most similar chunks visible to `user`.

        The ACL predicate is applied inside the SQL WHERE clause (so
        disallowed rows are never fetched), and the results are re-checked in
        Python via filter_chunks_by_permission as a defense-in-depth gate.
        """
        top_k = top_k or self.config.top_k
        acl_predicate, acl_params = build_acl_predicate(user)

        cur = self.conn.cursor()
        cur.execute(
            f"""
            SELECT id, source, filename, chunk_index, content, metadata,
                   1 - (embedding <=> %s::vector) AS similarity
            FROM {self.config.vector_table}
            WHERE {acl_predicate}
            ORDER BY embedding <=> %s::vector
            LIMIT %s
            """,
            (Vector(query_embedding), *acl_params, Vector(query_embedding), top_k),
        )
        rows = cur.fetchall()
        cur.close()

        chunks = [
            {
                "id": r[0],
                "source": r[1],
                "filename": r[2],
                "chunk_index": r[3],
                "content": r[4],
                "metadata": _as_dict(r[5]),
                "similarity": r[6],
            }
            for r in rows
        ]
        return filter_chunks_by_permission(chunks, user)

    def update_permissions(
        self,
        filename: str,
        sensitivity: str | None = None,
        allowed_roles: list[str] | None = None,
    ) -> int:
        """Merge ACL fields into the metadata of every chunk for `filename`.

        Seeding/demo utility for tagging Project 1's already-ingested chunks
        with the permission fields this project's ACL filter reads.
        """
        patch: dict[str, Any] = {}
        if sensitivity is not None:
            patch["sensitivity"] = sensitivity
        if allowed_roles is not None:
            patch["allowed_roles"] = allowed_roles
        if not patch:
            return 0

        cur = self.conn.cursor()
        cur.execute(
            f"""
            UPDATE {self.config.vector_table}
            SET metadata = metadata || %s::jsonb
            WHERE filename = %s
            """,
            (json.dumps(patch), filename),
        )
        updated = cur.rowcount
        self.conn.commit()
        cur.close()
        logger.info("Tagged %d chunk(s) for %s with %s", updated, filename, patch)
        return updated

    def close(self) -> None:
        if self._conn and not self._closed:
            self._conn.close()
            self._closed = True
        if self._connector is not None:
            self._connector.close()
            self._connector = None


def _as_dict(metadata: Any) -> dict:
    if isinstance(metadata, str):
        return json.loads(metadata) if metadata else {}
    return metadata or {}
