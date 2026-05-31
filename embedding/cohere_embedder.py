"""
embedding/cohere_embedder.py
=============================
Cohere Embed models.

Models (May 2025)
-----------------
  embed-multilingual-v3.0   1024-dim  108 languages  $0.10/1M  ⭐⭐⭐⭐ VI
  embed-english-v3.0        1024-dim  English only   $0.10/1M
  embed-v4.0                1536-dim  128K ctx (!)   $0.10/1M  image support

Asymmetric embedding
--------------------
Cohere requires different ``input_type`` for documents vs queries.
This MUST be consistent between index time and query time.
  "search_document"  → use when embedding corpus chunks
  "search_query"     → use when embedding user queries

Vietnamese quality: ⭐⭐⭐⭐ — one of the best API options.

Env var: COHERE_API_KEY
"""

from __future__ import annotations

from langchain_core.embeddings import Embeddings

from embedding.base import BaseEmbedder


class CohereEmbedder(BaseEmbedder):
    """
    Parameters
    ----------
    model_name  : Cohere embedding model identifier.
    input_type  : "search_document" for corpus | "search_query" for queries.
    """

    def __init__(
        self,
        model_name: str = "embed-multilingual-v3.0",
        input_type: str = "search_document",
        **kwargs,
    ):
        super().__init__(model_name, **kwargs)
        self.input_type = input_type

    def _build(self) -> Embeddings:
        from langchain_cohere import CohereEmbeddings

        return CohereEmbeddings(
            model=self.model_name,
            input_type=self.input_type,
            **self.kwargs,
        )
