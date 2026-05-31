"""
post_retrieval/cohere_reranker.py
==================================
Cohere Rerank API.

Cohere's reranker is trained specifically for retrieval reranking and
supports 100+ languages including Vietnamese.

Models
------
  rerank-v3.5              Latest, best quality, multilingual
  rerank-multilingual-v3.0 Previous multilingual model
  rerank-english-v3.0      English only, slightly faster

Vietnamese quality: ⭐⭐⭐⭐⭐ — one of the best API options.

Env var: COHERE_API_KEY
"""

from __future__ import annotations

from langchain_core.documents import Document

from post_retrieval.base import BasePostProcessor


class CohereReranker(BasePostProcessor):
    """
    Parameters
    ----------
    model_name : Cohere rerank model identifier.
    top_n      : Documents to keep after reranking.
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
