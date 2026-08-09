"""
post_retrieval/base.py
======================
Lớp ABC dùng chung cho mọi bộ xử lý sau truy hồi.

Hợp đồng chung:
    docs = SomeProcessor(**options).process(query, docs) -> list[Document]

Query luôn được truyền vào để reranker / compressor / filter dùng trực tiếp,
không cần giữ thêm trạng thái.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from langchain_core.documents import Document


class BasePostProcessor(ABC):
    """Lớp cơ sở cho mọi bước xử lý sau truy hồi."""

    @abstractmethod
    def process(self, query: str, docs: list[Document]) -> list[Document]:
        """
        Tinh lọc danh sách document đã truy hồi.

        Tham số
        -------
        query : Query chính của người dùng (hoặc query đầu trong TransformResult).
        docs  : Document thô từ bước retrieval.

        Trả về
        ------
        Danh sách Document có thể ngắn hơn, đổi thứ tự hoặc đã nén.
        """