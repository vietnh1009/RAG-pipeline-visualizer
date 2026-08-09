"""
post_retrieval/redundancy_filter.py
=====================================
Semantic Near-Duplicate Filter — drop near-identical chunks after retrieval.

Unlike exact deduplication (MD5), this catches paraphrased or lightly
reformatted versions of the same content using cosine similarity.

Use when: multiple retrieval queries (multi_query, hybrid) are used and the
          merged results contain many near-duplicate passages.
"""

from __future__ import annotations

from langchain_core.documents import Document

from post_retrieval.base import BasePostProcessor


class RedundancyFilter(BasePostProcessor):
    """
    Drop documents whose semantic similarity to an already-kept document
    exceeds the threshold.

    Parameters
    ----------
    top_n           : Max documents to keep after filtering.
    threshold       : Cosine similarity above which a doc is near-duplicate.
    embedding_model : sentence-transformers model for similarity computation.
                      Falls back to Jaccard when None.
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
