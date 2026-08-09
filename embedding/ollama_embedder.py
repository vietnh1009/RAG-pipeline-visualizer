"""
embedding/ollama_embedder.py
=============================
Embedding qua server Ollama chạy cục bộ.

Hợp với: dữ liệu nhạy cảm cần chạy hoàn toàn nội bộ, phát triển / kiểm thử
không có API key, và triển khai biên có GPU.

Phải pull model trước: ``ollama pull bge-m3``.

Model phổ biến
--------------
  nomic-embed-text        768 chiều   nhanh, tốt với tiếng Anh
  mxbai-embed-large      1024 chiều   mạnh, dùng chung nhiều việc
  bge-m3                 1024 chiều   đa ngôn ngữ, tốt nhất cho tiếng Việt
  snowflake-arctic-embed      —       truy hồi tiếng Anh mạnh
"""

from __future__ import annotations

from langchain_core.embeddings import Embeddings

from embedding.base import BaseEmbedder


class OllamaEmbedder(BaseEmbedder):
    """
    Embedder dùng Ollama.

    Tham số
    -------
    model_name : Tên model Ollama, phải pull sẵn.
    base_url   : URL server Ollama, mặc định http://localhost:11434.
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
