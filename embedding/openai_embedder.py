"""
embedding/openai_embedder.py
============================
OpenAI text-embedding models.

Models (May 2025)
-----------------
  text-embedding-3-small  1536-dim  $0.02/1M tokens  Good default
  text-embedding-3-large  3072-dim  $0.13/1M tokens  Higher quality
  text-embedding-ada-002  1536-dim  Legacy — use 3-small instead

Matryoshka (MRL) support
------------------------
Both text-embedding-3-* models support truncating to fewer dimensions
via the ``dimensions`` parameter, with only ~5–10 % quality loss.
DO NOT use MRL with ada-002.

Vietnamese quality: ⭐⭐⭐ — adequate but multilingual models score higher.

Env var: OPENAI_API_KEY
"""

from __future__ import annotations

from langchain_core.embeddings import Embeddings

from embedding.base import BaseEmbedder


class OpenAIEmbedder(BaseEmbedder):
    """
    Parameters
    ----------
    model_name  : OpenAI embedding model identifier.
    dimensions  : Optional MRL truncation (text-embedding-3-* only).
                  e.g. 256 → 6× RAM reduction, ~5 % quality loss.
    """

    def __init__(
        self,
        model_name:  str = "text-embedding-3-small",
        dimensions:  int | None = None,
        **kwargs,
    ):
        super().__init__(model_name, **kwargs)
        self.dimensions = dimensions

    def _build(self) -> Embeddings:
        from langchain_openai import OpenAIEmbeddings

        init_kwargs = {"model": self.model_name, **self.kwargs}
        if self.dimensions and "3" in self.model_name:
            init_kwargs["dimensions"] = self.dimensions
        return OpenAIEmbeddings(**init_kwargs)
