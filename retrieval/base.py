"""
retrieval/base.py
=================
Lớp ABC và helper dùng chung cho mọi chiến lược truy hồi.

Hợp đồng chung:
    docs = SomeRetriever(vector_store, **options).retrieve(result) -> list[Document]

Đầu vào luôn là TransformResult từ bước pre-retrieval — mang một hoặc nhiều
query kèm metadata filter tuỳ chọn.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore


class BaseRetriever(ABC):
    """
    Lớp cơ sở cho mọi chiến lược truy hồi.

    Tham số
    -------
    vector_store : VectorStore của LangChain đã nạp dữ liệu.
    top_k        : Số document tối đa trả về.
    """

    def __init__(self, vector_store: VectorStore, top_k: int = 5):
        self.vector_store = vector_store
        self.top_k        = top_k

    @abstractmethod
    def retrieve(self, result) -> list[Document]:
        """
        Truy hồi document cho một TransformResult.

        Tham số
        -------
        result : TransformResult từ bước pre_retrieval. Dùng ``queries``,
                 ``metadata_filter`` và ``extra``.
        """

    def _search(
        self,
        query:           str,
        k:               int,
        filter:          dict | None = None,
        search_type:     str = "similarity",
        score_threshold: float = 0.0,
    ) -> list[Document]:
        """
        Chạy một lần tìm vector.

        Bọc API similarity search của LangChain VectorStore để mọi provider
        dùng chung một cách gọi.
        """
        search_kwargs: dict[str, Any] = {"k": k}
        if filter:
            search_kwargs["filter"] = filter

        if search_type == "mmr":
            return self.vector_store.max_marginal_relevance_search(
                query, k=k, filter=filter
            )

        if search_type == "similarity_score_threshold":
            search_kwargs["score_threshold"] = score_threshold
            pairs = self.vector_store.similarity_search_with_relevance_scores(
                query, **search_kwargs
            )
            return [doc for doc, score in pairs if score >= score_threshold][:k]

        return self.vector_store.similarity_search(query, **search_kwargs)