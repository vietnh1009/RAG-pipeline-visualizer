"""
embedding/base.py
=================
Abstract base class shared by all dense embedding wrappers.

Every embedder wraps a LangChain Embeddings object and exposes it
via the ``embedder`` property so vector stores can use it directly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from langchain_core.embeddings import Embeddings


class BaseEmbedder(ABC):
    """
    Thin wrapper around a LangChain Embeddings object.

    Subclasses implement ``_build()`` which returns the provider-specific
    LangChain Embeddings instance.  Everything else (embed_documents,
    embed_query, the ``embedder`` property) is inherited from here.
    """

    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name
        self.kwargs     = kwargs
        self._embedder: Embeddings | None = None

    @abstractmethod
    def _build(self) -> Embeddings:
        """Construct and return the provider-specific LangChain Embeddings object."""

    @property
    def embedder(self) -> Embeddings:
        """Lazy-initialised LangChain Embeddings instance."""
        if self._embedder is None:
            self._embedder = self._build()
        return self._embedder

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of document texts."""
        return self.embedder.embed_documents(texts)

    def embed_query(self, query: str) -> list[float]:
        """Embed a single query string."""
        return self.embedder.embed_query(query)
