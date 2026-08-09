"""
retrieval/multi_query.py
=========================
Truy hồi nhiều query — chạy N query rồi gộp bằng RRF.

Thiết kế để đi cùng MultiQueryTransformer hoặc QueryDecompositionTransformer ở
bước pre-retrieval, những transformer sinh nhiều query trong
``TransformResult.queries``.

Chỉ có một query thì hành vi giống hệt DenseRetriever.
"""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore

from retrieval.base import BaseRetriever
from retrieval.utils import reciprocal_rank_fusion
from utils.documents import deduplicate


class MultiQueryRetriever(BaseRetriever):
    """
    Truy hồi cho từng query trong TransformResult.queries rồi gộp bằng RRF.

    Tham số
    -------
    vector_store : VectorStore của LangChain đã nạp dữ liệu.
    top_k        : Số kết quả cuối sau khi gộp.
    candidate_k  : Số kết quả mỗi query trước khi gộp.
    rrf_k        : Hằng số RRF, mặc định 60.
    search_type  : Kiểu tìm áp dụng cho từng lần tìm.
    """

    def __init__(
        self,
        vector_store: VectorStore,
        top_k:        int = 5,
        candidate_k:  int = 10,
        rrf_k:        int = 60,
        search_type:  str = "similarity",
    ):
        super().__init__(vector_store, top_k)
        self.candidate_k = candidate_k
        self.rrf_k       = rrf_k
        self.search_type = search_type

    def retrieve(self, result) -> list[Document]:
        queries = result.all_queries()
        filter  = result.metadata_filter

        ranked_lists = []
        for q in queries:
            docs = self._search(query=q, k=self.candidate_k, filter=filter, search_type=self.search_type)
            if docs:
                ranked_lists.append(docs)

        if not ranked_lists:
            return []

        fused = reciprocal_rank_fusion(ranked_lists, k=self.rrf_k)
        return deduplicate(fused)[:self.top_k]
