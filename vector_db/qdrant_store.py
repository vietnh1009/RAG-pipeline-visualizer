"""
vector_db/qdrant_store.py
==========================
Qdrant — lọc metadata tốt nhất (ACORN) và có hybrid search.

Ba cách chạy:
- Trong bộ nhớ    (url=":memory:", không lưu lại)
- Server cục bộ   (url="http://localhost:6333", cần tự dựng server)
- Qdrant Cloud    (truyền url + api_key)

ACORN lọc ngay trong lúc duyệt đồ thị HNSW thay vì lọc sau, nên recall gần như
không giảm dù filter metadata phức tạp hay chọn lọc gắt tới đâu.

Tránh index lại: kiểm tra ``collection_info.vectors_count`` trước khi nạp.

Quy mô  : ~ 1 tỷ+ vector
Dùng khi: cần hybrid search, lọc metadata phức tạp, tự dựng nhưng vẫn có đường
          lên cloud.

Biến môi trường (tuỳ chọn): QDRANT_URL, QDRANT_API_KEY
"""

from __future__ import annotations

import logging
import os

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore

from vector_db.base import BaseVectorStore

logger = logging.getLogger(__name__)


class QdrantVectorStore(BaseVectorStore):
    """
    Vector store Qdrant.

    Tham số
    -------
    collection_name : Tên collection trong Qdrant.
    url             : URL server, hoặc ":memory:" để chạy trong process.
    api_key         : API key Qdrant Cloud; None nếu tự dựng.
    dimension       : Số chiều vector dense.
    distance        : "Cosine" | "Dot" | "Euclid"
    on_disk         : Lưu vector xuống đĩa để đỡ tốn RAM.
    force_reindex   : Xoá collection và tạo lại.
    """

    def __init__(
        self,
        collection_name: str  = "rag",
        url:             str  = "http://localhost:6333",
        api_key:         str | None = None,
        dimension:       int  = 1536,
        distance:        str  = "Cosine",
        on_disk:         bool = False,
        force_reindex:   bool = False,
    ):
        super().__init__(collection_name, force_reindex)
        self.url      = os.environ.get("QDRANT_URL",     url)
        self.api_key  = os.environ.get("QDRANT_API_KEY", api_key)
        self.dimension = dimension
        self.distance  = distance
        self.on_disk   = on_disk

    def get_or_create(self, chunks: list[Document], embedder) -> VectorStore:
        from qdrant_client import QdrantClient
        from qdrant_client.http.models import Distance, VectorParams, OptimizersConfigDiff
        from langchain_qdrant import QdrantVectorStore as _QVS

        lc_embedder = self._langchain_embedder(embedder)
        client      = QdrantClient(url=self.url, api_key=self.api_key)
        dist_map    = {"Cosine": Distance.COSINE, "Dot": Distance.DOT, "Euclid": Distance.EUCLID}
        existing    = [c.name for c in client.get_collections().collections]

        # Collection đã có dữ liệu thì dùng lại
        if self.collection_name in existing and not self.force_reindex:
            info  = client.get_collection(self.collection_name)
            count = info.vectors_count or 0
            if count > 0:
                logger.info("Qdrant: collection '%s' has %d vectors — skipping.", self.collection_name, count)
                return _QVS(client=client, collection_name=self.collection_name, embedding=lc_embedder)

        # force_reindex: xoá collection cũ
        if self.collection_name in existing:
            client.delete_collection(self.collection_name)

        client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=self.dimension,
                distance=dist_map.get(self.distance, Distance.COSINE),
                on_disk=self.on_disk,
            ),
            optimizers_config=OptimizersConfigDiff(indexing_threshold=20_000),
        )

        logger.info("Qdrant: indexing %d chunks into '%s'.", len(chunks), self.collection_name)
        return _QVS.from_documents(
            documents=self.sanitize_metadata(chunks),
            embedding=lc_embedder,
            collection_name=self.collection_name,
            url=self.url,
            api_key=self.api_key,
        )
