"""
vector_db/pgvector_store.py
============================
pgvector PostgreSQL extension — ACID transactions, no new infrastructure.

Best choice if your stack already includes PostgreSQL.
Provides full SQL filtering, ACID guarantees, and JOINs with other tables.

Scaling options
---------------
pgvector       : < 10 M vectors  (default, built-in HNSW index)
pgvectorscale  : < 100 M+ vectors (DiskANN-based, Timescale extension)

Re-index prevention
-------------------
Queries ``langchain_pg_embedding`` table row count for this collection.

Scale     : < 50 M vectors (pgvector) / 100 M+ (pgvectorscale)
Use when  : already on PostgreSQL, need ACID, want SQL filtering.

Env var: DATABASE_URL  (postgresql+psycopg://user:pass@host:5432/db)
"""

from __future__ import annotations

import logging
import os

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore

from vector_db.base import BaseVectorStore

logger = logging.getLogger(__name__)


class PGVectorStore(BaseVectorStore):
    """
    Parameters
    ----------
    collection_name       : Table identifier inside PostgreSQL.
    connection_string     : PostgreSQL DSN. Falls back to DATABASE_URL env var.
    distance_strategy     : "cosine" | "euclidean" | "inner_product"
    force_reindex         : Drop and recreate the collection table.
    """

    def __init__(
        self,
        collection_name:   str  = "rag",
        connection_string: str | None = None,
        distance_strategy: str  = "cosine",
        force_reindex:     bool = False,
    ):
        super().__init__(collection_name, force_reindex)
        self.connection_string = connection_string or os.environ.get("DATABASE_URL", "")
        self.distance_strategy = distance_strategy

    def get_or_create(self, chunks: list[Document], embedder) -> VectorStore:
        from langchain_postgres import PGVector

        lc_embedder = self._langchain_embedder(embedder)

        # Check existing row count
        if not self.force_reindex:
            count = self._row_count()
            if count > 0:
                logger.info("pgvector: collection '%s' has %d rows — skipping.", self.collection_name, count)
                return PGVector(
                    connection=self.connection_string,
                    embeddings=lc_embedder,
                    collection_name=self.collection_name,
                )

        logger.info("pgvector: indexing %d chunks into '%s'.", len(chunks), self.collection_name)
        return PGVector.from_documents(
            documents=self.sanitize_metadata(chunks),
            embedding=lc_embedder,
            connection=self.connection_string,
            collection_name=self.collection_name,
            pre_delete_collection=self.force_reindex,
        )

    def _row_count(self) -> int:
        """Return existing row count for this collection, or 0 on error."""
        import psycopg
        # Normalise URI scheme for psycopg
        conn_str = self.connection_string.replace("postgresql+psycopg://", "postgresql://")
        try:
            with psycopg.connect(conn_str) as db, db.cursor() as cur:
                cur.execute(
                    "SELECT COUNT(*) FROM langchain_pg_embedding "
                    "WHERE collection_id = ("
                    "  SELECT uuid FROM langchain_pg_collection WHERE name = %s"
                    ")",
                    (self.collection_name,),
                )
                row = cur.fetchone()
                return row[0] if row else 0
        except Exception:
            return 0
