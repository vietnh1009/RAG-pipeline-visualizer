"""
embedding/fastembed_embedder.py
================================
FastEmbed by Qdrant — CPU-optimised ONNX inference.

Significantly faster than sentence-transformers on CPU because models
are exported to ONNX with quantisation.  No GPU required.

Popular models
--------------
  BAAI/bge-small-en-v1.5                           384-dim   Fast, English
  BAAI/bge-base-en-v1.5                            768-dim   Better quality
  intfloat/multilingual-e5-small                   384-dim   Multilingual (VI ok)
  sentence-transformers/paraphrase-multilingual-    384-dim   Multilingual
      MiniLM-L12-v2

Use when: CPU-only deployment, edge devices, quick prototyping without GPU.
"""

from __future__ import annotations

from langchain_core.embeddings import Embeddings

from embedding.base import BaseEmbedder


class FastEmbedEmbedder(BaseEmbedder):
    """
    Parameters
    ----------
    model_name : FastEmbed model name (auto-downloaded on first use).
    max_length : Maximum token length per text.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        max_length: int = 512,
        **kwargs,
    ):
        super().__init__(model_name, **kwargs)
        self.max_length = max_length

    def _build(self) -> Embeddings:
        from langchain_community.embeddings import FastEmbedEmbeddings

        return FastEmbedEmbeddings(
            model_name=self.model_name,
            max_length=self.max_length,
            **self.kwargs,
        )
