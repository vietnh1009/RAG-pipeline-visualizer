"""
retrieval/sparse.py
===================
Truy hồi từ khoá thưa bằng BM25.

BM25 khớp đúng từ nên bổ trợ tốt cho dense. Mạnh ở:
  - Danh từ riêng (tên người, mã sản phẩm, từ viết tắt)
  - Thuật ngữ chuyên ngành hiếm trong dữ liệu huấn luyện của model embedding
  - Query lệch miền, nơi embedding dense không đáng tin

Index BM25 dựng trong bộ nhớ từ corpus ``documents``.
"""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore

from retrieval.base import BaseRetriever


class SparseRetriever(BaseRetriever):
    """
    Truy hồi từ khoá BM25 bằng thư viện rank-bm25.

    Tham số
    -------
    vector_store : Lớp cơ sở đòi hỏi, nhưng BM25 không dùng đến.
    documents    : Toàn bộ corpus để dựng index BM25.
    top_k        : Số kết quả trả về.
    """

    def __init__(
        self,
        vector_store: VectorStore,
        documents:    list[Document],
        top_k:        int = 5,
    ):
        super().__init__(vector_store, top_k)
        self.documents = documents
        self._bm25     = self._build_index(documents)

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return text.lower().split()

    def _build_index(self, docs: list[Document]):
        from rank_bm25 import BM25Okapi
        return BM25Okapi([self._tokenize(d.page_content) for d in docs])

    def retrieve(self, result) -> list[Document]:
        query  = result.queries[0] if result.queries else result.original_query
        tokens = self._tokenize(query)
        scores = self._bm25.get_scores(tokens)

        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:self.top_k]

        docs: list[Document] = []
        for idx in ranked:
            if scores[idx] > 0:
                doc = self.documents[idx]
                doc.metadata["bm25_score"] = float(scores[idx])
                docs.append(doc)
        return docs
