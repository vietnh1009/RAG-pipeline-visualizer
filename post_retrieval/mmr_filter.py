"""
post_retrieval/mmr_filter.py
=============================
Bộ lọc đa dạng MMR (Maximal Marginal Relevance).

MMR chọn document vừa liên quan tới query, vừa khác biệt với những document đã
chọn — tránh việc LLM nhận năm chunk gần như giống hệt nhau, vừa phí context
vừa làm câu trả lời thiên về nội dung lặp.

Công thức (Carbonell & Goldstein, 1998):
    MMR = argmax [ λ · Sim(q, d) − (1−λ) · max_{s∈S} Sim(d, s) ]

Đo tương đồng bằng embedding sentence-transformers nếu có; không có thì lùi về
Jaccard trên token (không cần thư viện thêm).
"""

from __future__ import annotations

from langchain_core.documents import Document

from post_retrieval.base import BasePostProcessor


class MMRFilter(BasePostProcessor):
    """
    Chọn tập con đa dạng bằng Maximal Marginal Relevance.

    Tham số
    -------
    top_n           : Số document cần chọn.
    mmr_lambda      : Cân bằng liên quan / đa dạng.
                      1.0 = thuần liên quan; 0.0 = thuần đa dạng.
    embedding_model : Model sentence-transformers để tính cosine.
                      Truyền None để dùng Jaccard.
    """

    def __init__(
        self,
        top_n:           int   = 5,
        mmr_lambda:      float = 0.5,
        embedding_model: str | None = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        self.top_n           = top_n
        self.mmr_lambda      = mmr_lambda
        self.embedding_model = embedding_model
        self._encoder        = None

    def _load_encoder(self):
        if self._encoder or not self.embedding_model:
            return
        from sentence_transformers import SentenceTransformer
        self._encoder = SentenceTransformer(self.embedding_model)

    def process(self, query: str, docs: list[Document]) -> list[Document]:
        if not docs or len(docs) <= self.top_n:
            return docs[:self.top_n]
        self._load_encoder()
        return self._mmr_embed(query, docs) if self._encoder else self._mmr_jaccard(query, docs)

    def _mmr_embed(self, query: str, docs: list[Document]) -> list[Document]:
        import numpy as np
        texts = [query] + [d.page_content for d in docs]
        embs  = self._encoder.encode(texts, normalize_embeddings=True)
        q_emb = embs[0]
        d_embs = embs[1:]
        q_sims = [float(np.dot(q_emb, e)) for e in d_embs]

        selected:  list[int] = []
        remaining: list[int] = list(range(len(docs)))

        while remaining and len(selected) < self.top_n:
            if not selected:
                best = max(remaining, key=lambda i: q_sims[i])
            else:
                def score(i: int) -> float:
                    rel = self.mmr_lambda * q_sims[i]
                    red = (1 - self.mmr_lambda) * max(
                        float(np.dot(d_embs[i], d_embs[s])) for s in selected
                    )
                    return rel - red
                best = max(remaining, key=score)
            selected.append(best)
            remaining.remove(best)

        return [docs[i] for i in selected]

    def _mmr_jaccard(self, query: str, docs: list[Document]) -> list[Document]:
        def jaccard(a: str, b: str) -> float:
            sa, sb = set(a.lower().split()), set(b.lower().split())
            inter  = len(sa & sb)
            return inter / len(sa | sb) if (sa | sb) else 0.0

        q_sims    = [jaccard(query, d.page_content) for d in docs]
        selected:  list[int] = []
        remaining: list[int] = list(range(len(docs)))

        while remaining and len(selected) < self.top_n:
            if not selected:
                best = max(remaining, key=lambda i: q_sims[i])
            else:
                def score(i: int) -> float:
                    rel = self.mmr_lambda * q_sims[i]
                    red = (1 - self.mmr_lambda) * max(
                        jaccard(docs[i].page_content, docs[s].page_content) for s in selected
                    )
                    return rel - red
                best = max(remaining, key=score)
            selected.append(best)
            remaining.remove(best)

        return [docs[i] for i in selected]
