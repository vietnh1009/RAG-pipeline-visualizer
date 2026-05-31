"""
vector_db/base.py
=================
Abstract base class cho mọi vector store provider.

Giao kèo:
    store = get_vector_store(provider, chunks, embedder, **cfg)
    results = store.similarity_search(query, k=5)
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore


class BaseVectorStore(ABC):
    """
    Abstract wrapper around a LangChain VectorStore.

    Parameters
    ----------
    collection_name : Name of the collection / index / table.
    force_reindex   : Wipe existing data and rebuild from scratch.
    """

    def __init__(self, collection_name: str = "rag", force_reindex: bool = False):
        self.collection_name = collection_name
        self.force_reindex   = force_reindex

    @abstractmethod
    def get_or_create(
        self,
        chunks:   list[Document],
        embedder,                   # EmbeddingPipeline
    ) -> VectorStore:
        """
        Return a populated VectorStore, creating or loading as needed.

        Parameters
        ----------
        chunks   : Chunk Documents from the chunking stage.
        embedder : ``EmbeddingPipeline`` from the embedding stage.
        """

    # ── Shared utilities ──────────────────────────────────────────────────────

    def _langchain_embedder(self, embedder):
        """Extract the LangChain Embeddings object from an EmbeddingPipeline."""
        # EmbeddingPipeline exposes .langchain_embedder
        if hasattr(embedder, "langchain_embedder"):
            return embedder.langchain_embedder
        # Plain LangChain Embeddings object passed directly
        return embedder

    @staticmethod
    def sanitize_metadata(docs: list[Document]) -> list[Document]:
        """
        Normalize Document metadata so every value is str | int | float | bool.

        Most vector DBs (Chroma, pgvector, Qdrant, Pinecone…) reject metadata
        that contains list/dict/None values.  This method converts them:
          - None        → key dropped entirely
          - list / dict → JSON string via json.dumps()
          - anything else (e.g. Path, datetime) → str()

        Returns a new list of Documents; originals are not mutated.
        """
        _SCALAR = (str, int, float, bool)
        result: list[Document] = []
        for doc in docs:
            clean: dict = {}
            for k, v in doc.metadata.items():
                if v is None:
                    continue
                if isinstance(v, _SCALAR):
                    clean[k] = v
                elif isinstance(v, (list, dict)):
                    clean[k] = json.dumps(v, ensure_ascii=False, default=str)
                else:
                    clean[k] = str(v)
            result.append(Document(page_content=doc.page_content, metadata=clean))
        return result
