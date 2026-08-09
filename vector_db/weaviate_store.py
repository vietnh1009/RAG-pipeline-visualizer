"""
vector_db/weaviate_store.py
============================
Weaviate — hybrid search sẵn có, API GraphQL.

Weaviate lưu đồng thời vector dense và tần suất từ BM25, nên hybrid retrieval
dùng được ngay không cần ghép thêm gì.

Lưu ý: tên collection của Weaviate BẮT BUỘC bắt đầu bằng chữ hoa — lớp bọc này
tự viết hoa chữ cái đầu của ``collection_name``.

Tránh index lại: đếm số object hiện có trước khi nạp.

Quy mô  : ~ 1 tỷ vector
Dùng khi: cần hybrid search, schema phong phú, hệ sinh thái GraphQL.

Biến môi trường (tuỳ chọn): WEAVIATE_URL, WEAVIATE_API_KEY
"""

from __future__ import annotations

import logging
import os

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore

from vector_db.base import BaseVectorStore

logger = logging.getLogger(__name__)


class WeaviateVectorStore(BaseVectorStore):
    """
    Vector store Weaviate.

    Tham số
    -------
    collection_name : Tên class Weaviate, tự viết hoa chữ đầu.
    url             : URL server Weaviate.
    api_key         : API key Weaviate Cloud; None nếu tự dựng.
    force_reindex   : Xoá class và dựng lại từ đầu.
    """

    def __init__(
        self,
        collection_name: str  = "Rag",
        url:             str  = "http://localhost:8080",
        api_key:         str | None = None,
        force_reindex:   bool = False,
    ):
        # Weaviate bắt buộc tên class viết hoa chữ đầu
        name = collection_name[0].upper() + collection_name[1:]
        super().__init__(name, force_reindex)
        self.url     = os.environ.get("WEAVIATE_URL",     url)
        self.api_key = os.environ.get("WEAVIATE_API_KEY", api_key)

    def get_or_create(self, chunks: list[Document], embedder) -> VectorStore:
        import weaviate
        from langchain_weaviate import WeaviateVectorStore as _WVS

        lc_embedder = self._langchain_embedder(embedder)
        auth        = weaviate.auth.AuthApiKey(self.api_key) if self.api_key else None
        host, port  = self._parse_url(self.url)

        client = weaviate.connect_to_custom(
            http_host=host,
            http_port=port,
            http_secure=self.url.startswith("https"),
            auth_credentials=auth,
        )

        # Đếm object sẵn có
        if not self.force_reindex:
            try:
                result = client.collections.get(self.collection_name).aggregate.over_all(total_count=True)
                if (result.total_count or 0) > 0:
                    logger.info("Weaviate: class '%s' has %d objects — skipping.", self.collection_name, result.total_count)
                    return _WVS(client=client, index_name=self.collection_name, text_key="text", embedding=lc_embedder)
            except Exception:
                pass  # class chưa tồn tại

        # force_reindex: xoá class cũ
        try:
            client.collections.delete(self.collection_name)
        except Exception:
            pass

        logger.info("Weaviate: indexing %d chunks into '%s'.", len(chunks), self.collection_name)
        return _WVS.from_documents(
            documents=self.sanitize_metadata(chunks),
            embedding=lc_embedder,
            client=client,
            index_name=self.collection_name,
        )

    @staticmethod
    def _parse_url(url: str) -> tuple[str, int]:
        """Tách host và port từ chuỗi URL."""
        stripped = url.replace("https://", "").replace("http://", "")
        if ":" in stripped:
            host, port_str = stripped.rsplit(":", 1)
            return host, int(port_str)
        return stripped, 8080
