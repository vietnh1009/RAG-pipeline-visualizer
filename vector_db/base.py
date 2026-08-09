"""
vector_db/base.py
=================
Abstract base class cho mọi vector store provider.

Giao kèo:
    store = get_vector_store(provider, chunks, embedder, **cfg)
    results = store.similarity_search(query, k=5)
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore


class _EmbedderAdapter(Embeddings):
    """
    Bọc một embedder của project thành đối tượng ``Embeddings`` thật của LangChain.

    Vì sao cần
    ----------
    ``BaseEmbedder`` của project có đủ ``embed_documents`` / ``embed_query``
    nhưng KHÔNG kế thừa ``langchain_core.embeddings.Embeddings``.

    Các store duck-typing (Chroma) vẫn chạy được. Nhưng ``langchain_community``
    FAISS lại kiểm tra ``isinstance(embedding, Embeddings)``; không khớp thì nó
    coi tham số là một *hàm* và gọi ``embedding(text)`` → ném
    ``TypeError: 'XEmbedder' object is not callable`` ngay lúc dựng index.

    Adapter này để mọi provider store đều nhận đúng kiểu, không phụ thuộc vào
    việc store đó có duck-typing hay không.

    Ví dụ
    -----
    >>> lc = _EmbedderAdapter(FastEmbedEmbedder())
    >>> FAISS.from_documents(documents=docs, embedding=lc)
    """

    def __init__(self, inner):
        self._inner = inner

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._inner.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._inner.embed_query(text)

    def __getattr__(self, item):
        # Cho phép truy cập các thuộc tính khác của embedder gốc
        # (model_name, dimensions, ...) mà không phải khai lại từng cái.
        return getattr(self._inner, item)


class BaseVectorStore(ABC):
    """
    Lớp bọc trừu tượng quanh VectorStore của LangChain.

    Tham số
    -------
    collection_name : Tên collection / index / bảng.
    force_reindex   : Xoá dữ liệu cũ và dựng lại từ đầu.
    """

    def __init__(self, collection_name: str = "rag", force_reindex: bool = False):
        self.collection_name = collection_name
        self.force_reindex   = force_reindex

    @abstractmethod
    def get_or_create(
        self,
        chunks:   list[Document],
        embedder,                   # EmbeddingPipeline
    ) -> VectorStore:
        """
        Trả về VectorStore đã có dữ liệu — tạo mới hoặc nạp lại tuỳ tình huống.

        Tham số
        -------
        chunks   : Document chunk từ bước chunking.
        embedder : ``EmbeddingPipeline`` từ bước embedding.
        """

    # ── Tiện ích dùng chung ───────────────────────────────────────────────────

    def _langchain_embedder(self, embedder):
        """
        Trả về một đối tượng ``Embeddings`` hợp lệ của LangChain.

        Thứ tự ưu tiên:
          1. ``EmbeddingPipeline`` có sẵn ``.langchain_embedder`` → dùng luôn.
          2. Đã là ``Embeddings`` thật → trả nguyên.
          3. Còn lại (embedder của project, chỉ duck-typing) → bọc bằng
             ``_EmbedderAdapter``. Bắt buộc cho FAISS, xem docstring của adapter.
        """
        if hasattr(embedder, "langchain_embedder"):
            inner = embedder.langchain_embedder
            return inner if isinstance(inner, Embeddings) else _EmbedderAdapter(inner)
        if isinstance(embedder, Embeddings):
            return embedder
        return _EmbedderAdapter(embedder)

    @staticmethod
    def sanitize_metadata(docs: list[Document]) -> list[Document]:
        """
        Chuẩn hoá metadata sao cho mọi giá trị là str | int | float | bool.

        Phần lớn vector DB (Chroma, pgvector, Qdrant, Pinecone…) từ chối
        metadata chứa list/dict/None. Quy tắc chuyển:
          - None        → bỏ hẳn key
          - list / dict → chuỗi JSON qua json.dumps()
          - còn lại (Path, datetime, …) → str()

        Trả về list Document mới, không sửa bản gốc.
        """
        _SCALAR = (str, int, float, bool)
        result: list[Document] = []
        for doc in docs:
            clean: dict = {}
            for k, v in doc.metadata.items():
                if v is None:
                    continue
                if isinstance(v, _SCALAR):
                    clean[k] = v
                elif isinstance(v, (list, dict)):
                    clean[k] = json.dumps(v, ensure_ascii=False, default=str)
                else:
                    clean[k] = str(v)
            result.append(Document(page_content=doc.page_content, metadata=clean))
        return result
