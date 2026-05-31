"""
retrieval/contextual.py
========================
Contextual Retrieval — dense search with score threshold + MMR diversity.

A production-grade single retriever that applies multiple quality controls:
  1. Dense similarity search with a score threshold (removes off-topic results).
  2. MMR re-ranking for diversity (avoids redundant chunks in context window).

This is a good default when you want a reliable, well-tuned single strategy
without the complexity of a multi-strategy ensemble.

Use when: you need one solid retriever with noise filtering and diversity.
"""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore

from retrieval.base import BaseRetriever


class ContextualRetriever(BaseRetriever):
    """
    Dense retrieval with score threshold and MMR diversity.

    Parameters
    ----------
    vector_store     : Populated LangChain VectorStore.
    top_k            : Final documents after MMR.
    candidate_k      : Candidates fetched before filtering (2–5× top_k).
    score_threshold  : Minimum similarity score to include a document.
    mmr_lambda       : MMR diversity trade-off.
                       1.0 = pure relevance; 0.0 = maximum diversity.
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

        # Score-threshold filtered dense search
        candidates = self._search(
            query=query, k=self.candidate_k, filter=filter,
            search_type="similarity_score_threshold",
            score_threshold=self.score_threshold,
        )
        if not candidates:
            candidates = self._search(query=query, k=self.candidate_k, filter=filter)

        # MMR re-ranking for diversity
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
