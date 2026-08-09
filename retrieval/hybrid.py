"""
retrieval/hybrid.py
===================
Truy hồi lai — gộp dense ANN và sparse BM25 thành một bảng xếp hạng.

Hybrid gần như luôn tốt hơn từng cách riêng lẻ: vector dense nắm ngữ nghĩa,
sparse lo phần khớp từ khoá chính xác. Đây là lựa chọn mặc định khuyến nghị.

Cách gộp
--------
rrf      : Reciprocal Rank Fusion (mặc định). Dựa trên thứ hạng nên không bị
           ảnh hưởng bởi thang điểm khác nhau giữa dense và sparse.
weighted : Cộng tuyến tính điểm đã chuẩn hoá, điều chỉnh bằng ``alpha``.
dbsf     : Distribution-Based Score Fusion, chuẩn hoá z-score, ít nhạy với
           điểm ngoại lai.
"""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore

from retrieval.base import BaseRetriever
from retrieval.dense import DenseRetriever
from retrieval.sparse import SparseRetriever
from retrieval.utils import reciprocal_rank_fusion, weighted_fusion, distribution_based_fusion
from utils.documents import deduplicate


class HybridRetriever(BaseRetriever):
    """
    Gộp kết quả dense và BM25 thành một danh sách xếp hạng duy nhất.

    Tham số
    -------
    vector_store   : VectorStore của LangChain đã nạp dữ liệu.
    documents      : Toàn bộ corpus để dựng index BM25.
    top_k          : Số kết quả cuối sau khi gộp.
    fusion_method  : "rrf" | "weighted" | "dbsf"
    alpha          : Trọng số dense trong weighted (0–1); 0.5 là ngang nhau.
    rrf_k          : Hằng số k của RRF, mặc định 60 theo bài báo gốc.
    candidate_k    : Số ứng viên lấy từ mỗi retriever con trước khi gộp,
                     thường 3–4× top_k.
    score_threshold: Ngưỡng điểm áp cho nhánh dense.
    """

    def __init__(
        self,
        vector_store:    VectorStore,
        documents:       list[Document],
        top_k:           int   = 5,
        fusion_method:   str   = "rrf",
        alpha:           float = 0.5,
        rrf_k:           int   = 60,
        candidate_k:     int   = 20,
        score_threshold: float = 0.0,
    ):
        super().__init__(vector_store, top_k)
        self.fusion_method = fusion_method
        self.rrf_k         = rrf_k
        self.alpha         = alpha

        self._dense  = DenseRetriever(vector_store, top_k=candidate_k, score_threshold=score_threshold)
        self._sparse = SparseRetriever(vector_store, documents, top_k=candidate_k)

    def retrieve(self, result) -> list[Document]:
        dense_docs  = self._dense.retrieve(result)
        sparse_docs = self._sparse.retrieve(result)

        if self.fusion_method == "weighted":
            fused = weighted_fusion(dense_docs, sparse_docs, alpha=self.alpha)
        elif self.fusion_method == "dbsf":
            fused = distribution_based_fusion([dense_docs, sparse_docs])
        else:  # rrf — mặc định
            fused = reciprocal_rank_fusion([dense_docs, sparse_docs], k=self.rrf_k)

        return deduplicate(fused)[:self.top_k]
