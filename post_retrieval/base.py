"""
post_retrieval/base.py
======================
Abstract base class shared by all post-retrieval processors.

Every processor follows the same contract:
    processor = SomeProcessor(**options)
    docs      = processor.process(query, docs) -> list[Document]

The query string is always available so processors that need it
(rerankers, compressors, filters) can use it without extra state.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from langchain_core.documents import Document


class BasePostProcessor(ABC):
    """Abstract base for all post-retrieval processing steps."""

    @abstractmethod
    def process(self, query: str, docs: list[Document]) -> list[Document]:
        """
        Process a list of retrieved documents and return a refined list.

        Parameters
        ----------
        query : The primary user query (or first query from TransformResult).
        docs  : Raw retrieved documents from the retrieval stage.

        Returns
        -------
        Possibly shorter, reordered, or compressed list of Documents.
        """