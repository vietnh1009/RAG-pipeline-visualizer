"""
post_retrieval/cohere_reranker.py
==================================
Cohere Rerank API — model huấn luyện riêng cho việc rerank, hỗ trợ 100+ ngôn ngữ
kể cả tiếng Việt.

Model
-----
  rerank-v3.5              Mới nhất, chất lượng tốt nhất, đa ngôn ngữ
  rerank-multilingual-v3.0 Bản đa ngôn ngữ đời trước
  rerank-english-v3.0      Chỉ tiếng Anh, nhanh hơn chút

Chất lượng tiếng Việt: ⭐⭐⭐⭐⭐ — thuộc nhóm tốt nhất trong các lựa chọn qua API.

Biến môi trường: COHERE_API_KEY
"""

from __future__ import annotations

from langchain_core.documents import Document

from post_retrieval.base import BasePostProcessor


class CohereReranker(BasePostProcessor):
    """
    Rerank bằng Cohere Rerank API.

    Tham số
    -------
    model_name : Tên model rerank của Cohere.
    top_n      : Số document giữ lại sau rerank.
    """

    def __init__(self, model_name: str = "rerank-v3.5", top_n: int = 5):
        self.model_name = model_name
        self.top_n      = top_n

    def process(self, query: str, docs: list[Document]) -> list[Document]:
        if not docs:
            return docs
        import cohere

        results = cohere.Client().rerank(
            query=query,
            documents=[d.page_content for d in docs],
            model=self.model_name,
            top_n=self.top_n,
        ).results

        reranked: list[Document] = []
        for hit in results:
            doc = docs[hit.index]
            doc.metadata["rerank_score"]    = hit.relevance_score
            doc.metadata["rerank_provider"] = "cohere"
            reranked.append(doc)
        return reranked
