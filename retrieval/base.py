"""
retrieval/base.py
=================
Abstract base class and shared helpers for all retrieval strategies.

Every retriever follows the same contract:
    retriever = SomeRetriever(vector_store, **options)
    docs      = retriever.retrieve(transform_result) -> list[Document]

The input is always a TransformResult from the pre-retrieval stage,
which carries one or more query strings plus optional metadata filters.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore


class BaseRetriever(ABC):
    """
    Abstract base for all retrieval strategies.

    Parameters
    ----------
    vector_store : Populated LangChain VectorStore.
    top_k        : Maximum number of documents to return.
    """

    def __init__(self, vector_store: VectorStore, top_k: int = 5):
        self.vector_store = vector_store
        self.top_k        = top_k

    @abstractmethod
    def retrieve(self, result) -> list[Document]:
        """
        Retrieve documents for the given TransformResult.

        Parameters
        ----------
        result : TransformResult from pre_retrieval stage.
                 Uses result.queries, result.metadata_filter, result.extra.
        """

    def _search(
        self,
        query:           str,
        k:               int,
        filter:          dict | None = None,
        search_type:     str = "similarity",
        score_threshold: float = 0.0,
    ) -> list[Document]:
        """
        Execute a single vector search.

        Wraps the LangChain VectorStore similarity search API uniformly
        across all provider backends.
        """
        search_kwargs: dict[str, Any] = {"k": k}
        if filter:
            search_kwargs["filter"] = filter

        if search_type == "mmr":
            return self.vector_store.max_marginal_relevance_search(
                query, k=k, filter=filter
            )

        if search_type == "similarity_score_threshold":
            search_kwargs["score_threshold"] = score_threshold
            pairs = self.vector_store.similarity_search_with_relevance_scores(
                query, **search_kwargs
            )
            return [doc for doc, score in pairs if score >= score_threshold][:k]

        return self.vector_store.similarity_search(query, **search_kwargs)