"""
embedding/base.py
=================
Lớp ABC dùng chung cho mọi embedder dense.

Mỗi embedder bọc một đối tượng Embeddings của LangChain và phơi ra qua thuộc
tính ``embedder`` để vector store dùng trực tiếp.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from langchain_core.embeddings import Embeddings


class BaseEmbedder(ABC):
    """
    Lớp bọc mỏng quanh đối tượng Embeddings của LangChain.

    Lớp con chỉ cần cài ``_build()`` để trả về Embeddings của provider tương
    ứng; phần còn lại (embed_documents, embed_query, thuộc tính ``embedder``)
    kế thừa từ đây.
    """

    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name
        self.kwargs     = kwargs
        self._embedder: Embeddings | None = None

    @abstractmethod
    def _build(self) -> Embeddings:
        """Tạo và trả về đối tượng Embeddings của provider tương ứng."""

    @property
    def embedder(self) -> Embeddings:
        """Đối tượng Embeddings, khởi tạo trễ ở lần dùng đầu tiên."""
        if self._embedder is None:
            self._embedder = self._build()
        return self._embedder

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed một danh sách text tài liệu."""
        return self.embedder.embed_documents(texts)

    def embed_query(self, query: str) -> list[float]:
        """Embed một chuỗi query."""
        return self.embedder.embed_query(query)
