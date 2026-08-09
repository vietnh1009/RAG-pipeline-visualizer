"""
retrieval/dense.py
==================
Dense retrieval — tìm ANN theo độ tương đồng vector.

Embed query rồi tìm trong index k chunk gần nhất. Đây là nền của mọi hệ RAG.

Kiểu tìm
--------
similarity                 : Cosine / dot-product, luôn trả về đủ top-k.
similarity_score_threshold : Chỉ trả document vượt ngưỡng điểm, tối đa k —
                             tránh trả rác khi query lệch khỏi miền dữ liệu.
mmr                        : Maximal Marginal Relevance, đánh đổi liên quan lấy
                             đa dạng để context bớt trùng lặp.

Dùng khi: câu hỏi ngữ nghĩa, ý nghĩa quan trọng hơn khớp từ khoá.
"""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore

from retrieval.base import BaseRetriever


class DenseRetriever(BaseRetriever):
    """
    Truy hồi dense theo độ tương đồng vector.

    Tham số
    -------
    vector_store     : VectorStore của LangChain đã nạp dữ liệu.
    top_k            : Số chunk trả về.
    search_type      : "similarity" | "similarity_score_threshold" | "mmr"
    score_threshold  : Ngưỡng điểm tối thiểu, chỉ dùng ở chế độ score_threshold.
    """

    def __init__(
        self,
        vector_store:    VectorStore,
        top_k:           int   = 5,
        search_type:     str   = "similarity",
        score_threshold: float = 0.0,
    ):
        super().__init__(vector_store, top_k)
        self.search_type     = search_type
        self.score_threshold = score_threshold

    def retrieve(self, result) -> list[Document]:
        query  = result.queries[0] if result.queries else result.original_query
        return self._search(
            query=query,
            k=self.top_k,
            filter=result.metadata_filter,
            search_type=self.search_type,
            score_threshold=self.score_threshold,
        )
