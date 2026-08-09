"""
vector_db/faiss_store.py
=========================
FAISS (Meta AI) — chạy hoàn toàn trong process, không server, không API key,
không cần mạng.

Index lưu thành hai file: ``<persist_dir>/index.faiss`` và ``index.pkl``
(persist_dir đã bao gồm tên collection: ./storage/faiss_{collection_name}).

Tránh index lại: file ``fingerprint.json`` giữ MD5 của corpus; corpus không đổi
thì nạp index cũ, khỏi embed lại.

Hạn chế
-------
- Không lọc metadata sẵn có — phải lọc sau khi truy hồi.
- Không hybrid search — muốn thì ghép thêm SparseEmbedder BM25.
- Không hợp cho hơn ~10 triệu vector trên một máy.

Quy mô  : < 10 triệu vector
Dùng khi: làm prototype, pipeline offline, không có server.
"""

from __future__ import annotations

import logging
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore

from vector_db.base import BaseVectorStore
from vector_db.utils import corpus_changed, corpus_fingerprint, save_fingerprint

logger = logging.getLogger(__name__)


class FAISSVectorStore(BaseVectorStore):
    """
    Vector store FAISS.

    Tham số
    -------
    collection_name : Tên thư mục con bên trong persist_dir.
    persist_dir     : Thư mục gốc chứa file index FAISS.
    force_reindex   : Xoá và dựng lại kể cả khi corpus không đổi.
    """

    def __init__(
        self,
        collection_name: str  = "rag",
        persist_dir:     str  = "./storage/faiss",
        force_reindex:   bool = False,
    ):
        super().__init__(collection_name, force_reindex)
        self.persist_dir = persist_dir

    def get_or_create(self, chunks: list[Document], embedder) -> VectorStore:
        from langchain_community.vectorstores import FAISS

        lc_embedder = self._langchain_embedder(embedder)
        idx_dir     = Path(self.persist_dir)   # persist_dir chính là thư mục index
        fp_path     = str(idx_dir / "fingerprint.json")

        # Corpus chưa đổi thì nạp lại index cũ
        if idx_dir.exists() and not self.force_reindex and not corpus_changed(chunks, fp_path):
            logger.info("FAISS: loading existing index from '%s'.", idx_dir)
            return FAISS.load_local(
                folder_path=str(idx_dir),
                embeddings=lc_embedder,
                allow_dangerous_deserialization=True,
            )

        # Dựng index mới
        logger.info("FAISS: building index for %d chunks.", len(chunks))
        store = FAISS.from_documents(documents=self.sanitize_metadata(chunks), embedding=lc_embedder)

        idx_dir.mkdir(parents=True, exist_ok=True)
        store.save_local(str(idx_dir))
        save_fingerprint(corpus_fingerprint(chunks), fp_path)
        logger.info("FAISS: index saved to '%s'.", idx_dir)
        return store
