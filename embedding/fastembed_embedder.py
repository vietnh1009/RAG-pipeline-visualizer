"""
embedding/fastembed_embedder.py
================================
FastEmbed của Qdrant — suy luận ONNX tối ưu cho CPU.

Nhanh hơn hẳn sentence-transformers trên CPU vì model được xuất sang ONNX kèm
lượng tử hoá. Không cần GPU.

Model phổ biến
--------------
  BAAI/bge-small-en-v1.5          384 chiều  nhanh, tiếng Anh
  BAAI/bge-base-en-v1.5           768 chiều  chất lượng tốt hơn
  intfloat/multilingual-e5-small  384 chiều  đa ngôn ngữ, tiếng Việt tạm ổn

Dùng khi: triển khai chỉ có CPU, thiết bị biên, dựng thử nhanh không cần GPU.
"""

from __future__ import annotations

from langchain_core.embeddings import Embeddings

from embedding.base import BaseEmbedder


class FastEmbedEmbedder(BaseEmbedder):
    """
    Embedder dùng FastEmbed.

    Tham số
    -------
    model_name : Tên model FastEmbed, tự tải về ở lần dùng đầu.
    max_length : Số token tối đa cho mỗi đoạn text.
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
