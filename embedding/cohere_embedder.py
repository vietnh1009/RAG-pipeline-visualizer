"""
embedding/cohere_embedder.py
=============================
Các model Cohere Embed.

Model (05/2025)
---------------
  embed-multilingual-v3.0  1024 chiều  108 ngôn ngữ  $0.10/1M  ⭐⭐⭐⭐ tiếng Việt
  embed-english-v3.0       1024 chiều  chỉ tiếng Anh $0.10/1M
  embed-v4.0               1536 chiều  context 128K  $0.10/1M  hỗ trợ ảnh

Embedding bất đối xứng: Cohere đòi ``input_type`` khác nhau cho tài liệu và
query, và giá trị này PHẢI thống nhất giữa lúc index và lúc truy vấn.
  "search_document" → khi embed chunk của corpus
  "search_query"    → khi embed câu hỏi người dùng

Chất lượng tiếng Việt: ⭐⭐⭐⭐ — thuộc nhóm tốt nhất trong các lựa chọn qua API.

Biến môi trường: COHERE_API_KEY
"""

from __future__ import annotations

from langchain_core.embeddings import Embeddings

from embedding.base import BaseEmbedder


class CohereEmbedder(BaseEmbedder):
    """
    Embedder dùng Cohere.

    Tham số
    -------
    model_name : Tên model embedding của Cohere.
    input_type : "search_document" cho corpus | "search_query" cho câu hỏi.
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
