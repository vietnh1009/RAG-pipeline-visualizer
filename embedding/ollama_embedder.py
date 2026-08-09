"""
embedding/ollama_embedder.py
=============================
Embedding via a locally running Ollama server.

Ideal for:
- Privacy-sensitive data (fully local, no API calls).
- Development / testing without API keys.
- Edge deployments with a GPU.

Pull a model first:
    ollama pull nomic-embed-text
    ollama pull bge-m3

Popular models
--------------
  nomic-embed-text      768-dim   Fast, good English
  mxbai-embed-large     1024-dim  Strong general-purpose
  bge-m3                1024-dim  Multilingual (best for Vietnamese)
  snowflake-arctic-embed  —       Strong English retrieval
"""

from __future__ import annotations

from langchain_core.embeddings import Embeddings

from embedding.base import BaseEmbedder


class OllamaEmbedder(BaseEmbedder):
    """
    Parameters
    ----------
    model_name : Ollama model name (must already be pulled).
    base_url   : Ollama server URL (default: http://localhost:11434).
    """

    def __init__(
        self,
        model_name: str = "nomic-embed-text",
        base_url:   str = "http://localhost:11434",
        **kwargs,
    ):
        super().__init__(model_name, **kwargs)
        self.base_url = base_url

    def _build(self) -> Embeddings:
        from langchain_ollama import OllamaEmbeddings

        return OllamaEmbeddings(
            model=self.model_name,
            base_url=self.base_url,
            **self.kwargs,
        )
