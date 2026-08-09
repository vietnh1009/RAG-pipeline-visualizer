"""
post_retrieval/mmr_filter.py
=============================
Maximal Marginal Relevance (MMR) diversity filter.

MMR selects documents that are simultaneously:
  - Relevant to the query.
  - Diverse from already-selected documents.

This prevents the LLM from receiving five nearly-identical chunks that
waste context window space and bias the answer toward repeated content.

Algorithm (Carbonell & Goldstein, 1998):
    MMR = argmax [ λ · Sim(q, d) − (1−λ) · max_{s∈S} Sim(d, s) ]

Similarity backend
------------------
Uses sentence-transformers embeddings when available; falls back to
token-level Jaccard similarity (no extra dependencies).

Use when: multiple retrieval queries (multi_query, hybrid) produce many
          near-duplicate passages about the same fact.
"""

from __future__ import annotations

from langchain_core.documents import Document

from post_retrieval.base import BasePostProcessor


class MMRFilter(BasePostProcessor):
    """
    Select a diverse subset via Maximal Marginal Relevance.

    Parameters
    ----------
    top_n           : Number of documents to select.
    mmr_lambda      : Relevance vs diversity trade-off.
                      1.0 = pure relevance; 0.0 = pure diversity.
    embedding_model : sentence-transformers model for cosine similarity.
                      Pass None to use Jaccard fallback.
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
