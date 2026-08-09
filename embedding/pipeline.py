"""
embedding/pipeline.py
=====================
EmbeddingPipeline — combines a dense embedder with an optional sparse embedder.

This is the object that vector_db.py and retrieval.py use.
It produces both dense and sparse vectors in a single interface.

Usage
-----
    pipeline = EmbeddingPipeline(
        dense_provider="huggingface",
        dense_model="BAAI/bge-m3",
        enable_sparse=True,
        sparse_method="bm25",
    )
    pipeline.fit_sparse(corpus_texts)        # required for BM25
    result = pipeline.embed_documents(texts)
    # result["dense"]  -> list[list[float]]
    # result["sparse"] -> list[dict[str, float]] or None
"""

from __future__ import annotations

from typing import Any

from langchain_core.embeddings import Embeddings

from embedding.sparse_embedder import SparseMethod, get_sparse_embedder


class EmbeddingPipeline:
    """
    Unified dense + optional sparse embedding interface.

    Parameters
    ----------
    dense_provider : Provider name (see factory.py for valid values).
    dense_model    : Model identifier for the dense embedder.
    dense_kwargs   : Extra kwargs forwarded to the dense embedder constructor.
    enable_sparse  : Also compute sparse (BM25 / SPLADE) vectors.
    sparse_method  : "bm25" | "splade"
    sparse_model   : SPLADE model name (ignored for BM25).
    """

    def __init__(
        self,
        dense_provider: str          = "openai",
        dense_model:    str          = "text-embedding-3-small",
        dense_kwargs:   dict | None  = None,
        enable_sparse:  bool         = False,
        sparse_method:  SparseMethod = "bm25",
        sparse_model:   str          = "naver/splade-cocondenser-ensembledistil",
    ):
        from embedding.factory import get_embedder

        self.enable_sparse  = enable_sparse
        self._dense         = get_embedder(dense_provider, dense_model, **(dense_kwargs or {}))
        self._sparse        = get_sparse_embedder(sparse_method, sparse_model) if enable_sparse else None

    # ------------------------------------------------------------------
    # Corpus fitting (BM25 requires this before embedding)
    # ------------------------------------------------------------------

    def fit_sparse(self, corpus_texts: list[str]) -> "EmbeddingPipeline":
        """Fit BM25 IDF statistics on the corpus. No-op for SPLADE."""
        if self._sparse:
            self._sparse.fit(corpus_texts)
        return self

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

    def embed_documents(self, texts: list[str]) -> dict[str, Any]:
        """
        Embed a list of document texts.

        Returns
        -------
        {
            "dense":  list[list[float]],           # always present
            "sparse": list[dict[str, float]] | None # only if enable_sparse
        }
        """
        return {
            "dense":  self._dense.embed_documents(texts),
            "sparse": self._sparse.embed_documents(texts) if self._sparse else None,
        }

    def embed_query(self, query: str) -> dict[str, Any]:
        """
        Embed a single query string.

        Returns
        -------
        {
            "dense":  list[float],
            "sparse": dict[str, float] | None
        }
        """
        return {
            "dense":  self._dense.embed_query(query),
            "sparse": self._sparse.embed_query(query) if self._sparse else None,
        }

    @property
    def langchain_embedder(self) -> Embeddings:
        """
        The underlying LangChain Embeddings object.

        Pass this to LangChain vector stores that require a standard
        Embeddings object (e.g. Chroma, FAISS, pgvector).
        """
        return self._dense.embedder
