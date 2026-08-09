"""
post_retrieval/redundancy_filter.py
=====================================
Lọc đoạn gần trùng về ngữ nghĩa sau khi truy hồi.

Khác khử trùng chính xác (MD5), lớp này dùng cosine similarity nên bắt được cả
những đoạn diễn đạt lại hoặc chỉ đổi cách trình bày.

Dùng khi: chạy nhiều query (multi_query, hybrid) và kết quả gộp lại có nhiều
đoạn na ná nhau.
"""

from __future__ import annotations

from langchain_core.documents import Document

from post_retrieval.base import BasePostProcessor


class RedundancyFilter(BasePostProcessor):
    """
    Bỏ document có độ tương đồng ngữ nghĩa vượt ngưỡng so với document đã giữ.

    Tham số
    -------
    top_n           : Số document tối đa giữ lại.
    threshold       : Ngưỡng cosine để coi là gần trùng.
    embedding_model : Model sentence-transformers để tính tương đồng.
                      None thì lùi về Jaccard.
    """

    def __init__(
        self,
        top_n:           int   = 5,
        threshold:       float = 0.92,
        embedding_model: str | None = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        self.top_n           = top_n
        self.threshold       = threshold
        self.embedding_model = embedding_model
        self._encoder        = None

    def _load_encoder(self):
        if self._encoder or not self.embedding_model:
            return
        from sentence_transformers import SentenceTransformer
        self._encoder = SentenceTransformer(self.embedding_model)

    def process(self, query: str, docs: list[Document]) -> list[Document]:
        if not docs:
            return docs
        self._load_encoder()
        return self._filter_embed(docs) if self._encoder else self._filter_jaccard(docs)

    def _filter_embed(self, docs: list[Document]) -> list[Document]:
        import numpy as np
        embs  = self._encoder.encode([d.page_content for d in docs], normalize_embeddings=True)
        kept:      list[Document] = []
        kept_embs: list          = []
        for doc, emb in zip(docs, embs):
            if kept_embs and max(float(np.dot(emb, ke)) for ke in kept_embs) >= self.threshold:
                continue
            kept.append(doc)
            kept_embs.append(emb)
            if len(kept) >= self.top_n:
                break
        return kept

    def _filter_jaccard(self, docs: list[Document]) -> list[Document]:
        def jaccard(a: str, b: str) -> float:
            sa, sb = set(a.lower().split()), set(b.lower().split())
            return len(sa & sb) / len(sa | sb) if (sa | sb) else 0.0

        kept: list[Document] = []
        for doc in docs:
            if any(jaccard(doc.page_content, k.page_content) >= self.threshold for k in kept):
                continue
            kept.append(doc)
            if len(kept) >= self.top_n:
                break
        return kept
