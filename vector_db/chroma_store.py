"""
vector_db/chroma_store.py
==========================
ChromaDB — lưu vector cục bộ, dữ liệu nằm ở ``persist_dir`` và tồn tại qua các
lần chạy; lần sau dùng lại collection cũ.

Tránh index lại: nếu collection đã có dữ liệu VÀ vân tay corpus khớp thì không
embed lại, trả về collection sẵn có.

Có sẵn lọc metadata cơ bản.

Quy mô  : < 10 triệu vector
Dùng khi: phát triển cục bộ, không cần cấu hình, dự án thuần Python.
"""

from __future__ import annotations

import logging
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore

from vector_db.base import BaseVectorStore
from vector_db.utils import corpus_changed, corpus_fingerprint, save_fingerprint

logger = logging.getLogger(__name__)


import functools

@functools.lru_cache(maxsize=8)
def _get_or_create_chroma_client(persist_dir: str):
    """
    PersistentClient của Chroma, cache theo persist_dir ở mức module.

    lru_cache bảo đảm mỗi đường dẫn chỉ tạo client một lần trong suốt vòng đời
    process — tránh xung đột settings và khỏi khởi tạo lặp.
    """
    import chromadb
    from chromadb.config import Settings
    return chromadb.PersistentClient(
        path     = persist_dir,
        settings = Settings(anonymized_telemetry=False, allow_reset=True),
    )


class ChromaVectorStore(BaseVectorStore):
    """
    Vector store ChromaDB.

    Tham số
    -------
    collection_name : Tên collection trong Chroma.
    persist_dir     : Thư mục Chroma lưu SQLite và embedding.
    force_reindex   : Xoá collection và dựng lại từ đầu.
    """

    def __init__(
        self,
        collection_name: str  = "rag",
        persist_dir:     str  = "./storage/chroma_rag",
        force_reindex:   bool = False,
    ):
        super().__init__(collection_name, force_reindex)
        self.persist_dir = persist_dir

    # Không đặt lru_cache thẳng lên method — dùng hàm cache ở mức module
    @staticmethod
    def _make_client(persist_dir: str):
        return _get_or_create_chroma_client(persist_dir)

    def get_or_create(self, chunks: list[Document], embedder) -> VectorStore:
        from langchain_chroma import Chroma
        import shutil

        lc_embedder = self._langchain_embedder(embedder)
        fp_path     = str(Path(self.persist_dir) / f"{self.collection_name}_fp.json")

        # force_reindex: xoá sạch trước khi tạo client
        if self.force_reindex and Path(self.persist_dir).exists():
            shutil.rmtree(self.persist_dir, ignore_errors=True)

        # Dùng PersistentClient trực tiếp để tránh conflict settings
        client = self._make_client(self.persist_dir)

        # Corpus chưa đổi thì nạp lại collection cũ
        if not self.force_reindex and Path(self.persist_dir).exists():
            if not corpus_changed(chunks, fp_path):
                store = Chroma(
                    client             = client,
                    collection_name    = self.collection_name,
                    embedding_function = lc_embedder,
                )
                try:
                    if store._collection.count() > 0:
                        logger.info("Chroma: loaded existing collection '%s'.", self.collection_name)
                        return store
                except Exception:
                    pass

        logger.info("Chroma: creating collection '%s' with %d chunks.", self.collection_name, len(chunks))
        store = Chroma.from_documents(
            documents         = self.sanitize_metadata(chunks),
            embedding         = lc_embedder,
            client            = client,
            collection_name   = self.collection_name,
            collection_metadata = {"hnsw:space": "cosine"},
        )
        save_fingerprint(corpus_fingerprint(chunks), fp_path)
        return store
