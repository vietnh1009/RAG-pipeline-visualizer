"""
post_retrieval/cross_encoder_reranker.py
=========================================
Cross-Encoder Reranker — chấm điểm liên quan bằng transformer chạy cục bộ.

Cross-encoder đưa cả cặp (query, document) qua transformer cùng lúc, khác
bi-encoder mã hoá riêng từng bên. Độ chính xác cao hơn hẳn, đổi lại chi phí
O(N) — mỗi document một lượt forward.

Model khuyến nghị cho Việt + Anh
--------------------------------
  BAAI/bge-reranker-v2-m3               Đa ngôn ngữ, tốt nhất cho VI+EN ★★★★★
  BAAI/bge-reranker-large               Đa ngôn ngữ, chất lượng cao     ★★★★
  cross-encoder/ms-marco-MiniLM-L-6-v2  Nhanh, chỉ tiếng Anh            ★★★★★
  cross-encoder/ms-marco-MiniLM-L-12-v2 Tốt hơn, chỉ tiếng Anh          ★★★★
  mixedbread-ai/mxbai-rerank-large-v1   Tiếng Anh, mạnh                 ★★★★

Dùng khi: retrieval đã lấy đúng ứng viên nhưng thứ hạng chưa tốt. Không cần GPU
nếu batch nhỏ — CPU đủ cho top_n ≤ 20.
"""

from __future__ import annotations

from langchain_core.documents import Document

from post_retrieval.base import BasePostProcessor


class CrossEncoderReranker(BasePostProcessor):
    """
    Rerank document bằng model cross-encoder chạy cục bộ.

    Tham số
    -------
    model_name  : Tên model cross-encoder trên HuggingFace.
    top_n       : Số document giữ lại sau rerank.
    batch_size  : Số cặp mỗi batch suy luận.
    device      : "cpu" | "cuda" | "mps"
    max_length  : Độ dài token tối đa cho mỗi cặp (query, doc).
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        top_n:      int = 5,
        batch_size: int = 32,
        device:     str = "cpu",
        max_length: int = 512,
    ):
        self.model_name = model_name
        self.top_n      = top_n
        self.batch_size = batch_size
        self.device     = device
        self.max_length = max_length
        self._model     = None   # nạp trễ, chỉ khi thật sự dùng

    def _load(self):
        if self._model is not None:
            return
        from sentence_transformers import CrossEncoder
        self._model = CrossEncoder(self.model_name, device=self.device, max_length=self.max_length)

    def process(self, query: str, docs: list[Document]) -> list[Document]:
        if not docs:
            return docs
        self._load()

        pairs  = [(query, doc.page_content) for doc in docs]
        scores = self._model.predict(pairs, batch_size=self.batch_size)

        for doc, score in zip(docs, scores):
            doc.metadata["rerank_score"] = float(score)

        ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in ranked][:self.top_n]
