"""
embedding/openai_embedder.py
============================
Các model text-embedding của OpenAI.

Model (05/2025)
---------------
  text-embedding-3-small  1536 chiều  $0.02/1M token  mặc định hợp lý
  text-embedding-3-large  3072 chiều  $0.13/1M token  chất lượng cao hơn
  text-embedding-ada-002  1536 chiều  cũ — nên dùng 3-small thay thế

Matryoshka (MRL): hai model text-embedding-3-* cho phép cắt bớt số chiều qua
tham số ``dimensions``, chất lượng chỉ giảm ~5–10 %. KHÔNG dùng MRL với ada-002.

Chất lượng tiếng Việt: ⭐⭐⭐ — tạm ổn, nhưng model đa ngôn ngữ tốt hơn.

Biến môi trường: OPENAI_API_KEY
"""

from __future__ import annotations

from langchain_core.embeddings import Embeddings

from embedding.base import BaseEmbedder


class OpenAIEmbedder(BaseEmbedder):
    """
    Embedder dùng OpenAI.

    Tham số
    -------
    model_name : Tên model embedding của OpenAI.
    dimensions : Cắt chiều theo MRL, chỉ áp cho text-embedding-3-*.
                 Ví dụ 256 → RAM giảm 6 lần, chất lượng giảm ~5 %.
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
