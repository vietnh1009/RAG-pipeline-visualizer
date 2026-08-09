"""
vector_db/lancedb_store.py
===========================
LanceDB — định dạng cột, chạy nhúng, không cần server.

Dùng định dạng cột Lance (gần giống Parquet) nên đọc rất nhanh khi tìm ANN.
Ghép thẳng được với pandas, Arrow và DuckDB.

Hai chế độ
----------
Nhúng cục bộ : persist_dir trỏ tới đường dẫn trên máy (mặc định).
Cloud        : persist_dir là URI LanceDB Cloud ("db://your-slug").

Tránh index lại: kiểm tra số dòng trong bảng sẵn có.

Quy mô  : ~ 1 tỷ vector
Dùng khi: quy trình thuần Python, chạy nhúng, ưu tiên serverless / tiết kiệm.
"""

from __future__ import annotations

import logging

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore

from vector_db.base import BaseVectorStore
from vector_db.utils import corpus_changed, corpus_fingerprint, save_fingerprint

logger = logging.getLogger(__name__)


class LanceDBVectorStore(BaseVectorStore):
    """
    Vector store LanceDB.

    Tham số
    -------
    collection_name : Tên bảng trong LanceDB.
    persist_dir     : Đường dẫn cục bộ hoặc URI LanceDB Cloud.
    distance        : "cosine" | "l2" | "dot"
    force_reindex   : Xoá bảng và dựng lại từ đầu.
    """

    def __init__(
        self,
        collection_name: str  = "rag",
        persist_dir:     str  = "./storage/lancedb_rag",
        distance:        str  = "cosine",
        force_reindex:   bool = False,
    ):
        super().__init__(collection_name, force_reindex)
        self.persist_dir = persist_dir
        self.distance    = distance

    def get_or_create(self, chunks: list[Document], embedder) -> VectorStore:
        import lancedb
        from langchain_community.vectorstores import LanceDB

        lc_embedder = self._langchain_embedder(embedder)
        db          = lancedb.connect(self.persist_dir)
        fp_path     = f"{self.persist_dir}/{self.collection_name}_fp.json"

        # Corpus chưa đổi thì trả về bảng sẵn có
        if self.collection_name in db.table_names() and not self.force_reindex:
            if not corpus_changed(chunks, fp_path):
                table = db.open_table(self.collection_name)
                if table.count_rows() > 0:
                    logger.info("LanceDB: table '%s' loaded (%d rows).", self.collection_name, table.count_rows())
                    return LanceDB(connection=table, embedding=lc_embedder)

        # Xoá bảng cũ và tạo lại
        if self.collection_name in db.table_names():
            db.drop_table(self.collection_name)

        logger.info("LanceDB: indexing %d chunks into '%s'.", len(chunks), self.collection_name)
        store = LanceDB.from_documents(
            documents=self.sanitize_metadata(chunks),
            embedding=lc_embedder,
            connection=db,
            table_name=self.collection_name,
        )
        save_fingerprint(corpus_fingerprint(chunks), fp_path)
        return store
