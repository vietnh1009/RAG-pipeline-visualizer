"""
retrieval/contextual.py
========================
Contextual Retrieval — dense + ngưỡng điểm + đa dạng hoá MMR.

Một retriever đơn nhưng gắn sẵn hai lớp kiểm soát chất lượng:
  1. Tìm dense có ngưỡng điểm, loại bỏ kết quả lạc đề.
  2. Xếp lại bằng MMR để context không chứa nhiều chunk trùng nội dung.

Dùng khi: muốn một chiến lược duy nhất, chắc chắn, không cần dựng tổ hợp nhiều
retriever.
"""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore

from retrieval.base import BaseRetriever


class ContextualRetriever(BaseRetriever):
    """
    Truy hồi dense kèm ngưỡng điểm và đa dạng hoá MMR.

    Tham số
    -------
    vector_store     : VectorStore của LangChain đã nạp dữ liệu.
    top_k            : Số document cuối, sau MMR.
    candidate_k      : Số ứng viên lấy trước khi lọc, khoảng 2–5× top_k.
    score_threshold  : Điểm tương đồng tối thiểu để giữ document.
    mmr_lambda       : Cân bằng liên quan / đa dạng của MMR.
                       1.0 = thuần liên quan; 0.0 = đa dạng tối đa.
    """

    def __init__(
        self,
        vector_store:    VectorStore,
        top_k:           int   = 5,
        candidate_k:     int   = 20,
        score_threshold: float = 0.3,
        mmr_lambda:      float = 0.7,
    ):
        super().__init__(vector_store, top_k)
        self.candidate_k     = candidate_k
        self.score_threshold = score_threshold
        self.mmr_lambda      = mmr_lambda

    def retrieve(self, result) -> list[Document]:
        query  = result.queries[0] if result.queries else result.original_query
        filter = result.metadata_filter

        # Tìm dense có lọc theo ngưỡng điểm
        candidates = self._search(
            query=query, k=self.candidate_k, filter=filter,
            search_type="similarity_score_threshold",
            score_threshold=self.score_threshold,
        )
        if not candidates:
            candidates = self._search(query=query, k=self.candidate_k, filter=filter)

        # Xếp lại bằng MMR để tăng đa dạng
        try:
            return self.vector_store.max_marginal_relevance_search(
                query,
                k=self.top_k,
                fetch_k=min(len(candidates), self.candidate_k),
                lambda_mult=self.mmr_lambda,
                filter=filter,
            )
        except Exception:
            return candidates[:self.top_k]
