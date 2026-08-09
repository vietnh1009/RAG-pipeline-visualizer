"""
vector_db/pgvector_store.py
============================
pgvector — extension của PostgreSQL, có transaction ACID, không cần dựng thêm
hạ tầng mới.

Lựa chọn tốt nhất nếu hệ thống đã dùng PostgreSQL: lọc bằng SQL đầy đủ, bảo đảm
ACID, JOIN được với các bảng khác.

Mức mở rộng
-----------
pgvector      : < 10 triệu vector (mặc định, index HNSW sẵn có)
pgvectorscale : 100 triệu+ vector (dựa trên DiskANN, extension của Timescale)

Tránh index lại: đếm số dòng của collection trong bảng ``langchain_pg_embedding``.

Biến môi trường: DATABASE_URL (postgresql+psycopg://user:pass@host:5432/db)
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
    Vector store pgvector.

    Tham số
    -------
    collection_name   : Định danh bảng trong PostgreSQL.
    connection_string : DSN PostgreSQL; bỏ trống thì lấy từ DATABASE_URL.
    distance_strategy : "cosine" | "euclidean" | "inner_product"
    force_reindex     : Xoá bảng collection và tạo lại.
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

        # Đếm số dòng sẵn có
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
        """Số dòng hiện có của collection; lỗi thì trả 0."""
        import psycopg
        # Chuẩn hoá scheme URI cho psycopg
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
