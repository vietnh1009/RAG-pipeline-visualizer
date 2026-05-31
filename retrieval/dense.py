"""
retrieval/dense.py
==================
Dense retrieval — standard ANN vector similarity search.

Embeds the query and searches the vector index for the top-k most
similar document chunks. This is the foundation of all RAG systems.

Search types
------------
similarity                : Cosine / dot-product. Returns top-k unconditionally.
similarity_score_threshold: Returns only documents above a min score, up to k.
                            Prevents returning irrelevant content on OOD queries.
mmr                       : Maximal Marginal Relevance. Trades off relevance vs
                            diversity to avoid redundant chunks in the context.

Use when: semantic queries where meaning matters more than exact keywords.
"""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore

from retrieval.base import BaseRetriever


class DenseRetriever(BaseRetriever):
    """
    Standard dense vector similarity retrieval.

    Parameters
    ----------
    vector_store     : Populated LangChain VectorStore.
    top_k            : Number of chunks to return.
    search_type      : "similarity" | "similarity_score_threshold" | "mmr"
    score_threshold  : Min similarity score (similarity_score_threshold mode).
    """

    def __init__(
        self,
        vector_store:    VectorStore,
        top_k:           int   = 5,
        search_type:     str   = "similarity",
        score_threshold: float = 0.0,
    ):
        super().__init__(vector_store, top_k)
        self.search_type     = search_type
        self.score_threshold = score_threshold

    def retrieve(self, result) -> list[Document]:
        query  = result.queries[0] if result.queries else result.original_query
        return self._search(
            query=query,
            k=self.top_k,
            filter=result.metadata_filter,
            search_type=self.search_type,
            score_threshold=self.score_threshold,
        )
