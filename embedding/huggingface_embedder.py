"""
embedding/huggingface_embedder.py
==================================
sentence-transformers của HuggingFace — chạy model bất kỳ ngay trên máy.

Model khuyến nghị cho RAG tiếng Việt (05/2025), cột: chiều · context · điểm
---------------------------------------------------------------------------
  BAAI/bge-m3                     1024  8192  ⭐⭐⭐⭐   dense+sparse+multivec
  Qwen/Qwen3-Embedding            2048  32K   ⭐⭐⭐⭐⭐  đứng đầu MTEB, có instruction
  intfloat/multilingual-e5-large  1024  512   ⭐⭐⭐     100 ngôn ngữ
  intfloat/e5-mistral-7b-instruct 4096  32K   ⭐⭐⭐     tinh chỉnh theo instruction
  nomic-ai/nomic-embed-text-v1.5   768  8192  ⭐⭐⭐     hỗ trợ MRL
  Alibaba-NLP/gte-Qwen2-7B        3584  131K  ⭐⭐⭐⭐   context rất dài
  VinAI/phobert-large              768   256  ⭐⭐⭐⭐   chuyên tiếng Việt

Model theo instruction: Qwen3-Embedding, E5-mistral, GTE-Qwen2 nhận tiền tố
instruction riêng cho query và tài liệu. Truyền ``query_instruction`` và
``document_instruction`` để bật, và giữ nguyên giá trị giữa lúc index và lúc
truy vấn, ví dụ:

    query_instruction    = "Retrieve relevant passages for this medical question:"
    document_instruction = "Represent this Vietnamese medical document:"
"""

from __future__ import annotations

from langchain_core.embeddings import Embeddings

from embedding.base import BaseEmbedder


class HuggingFaceEmbedder(BaseEmbedder):
    """
    Embedder chạy model HuggingFace cục bộ.

    Tham số
    -------
    model_name           : Tên model trên HuggingFace.
    device               : "cpu" | "cuda" | "mps"
    normalize_embeddings : Chuẩn hoá L2, nên bật khi dùng cosine similarity.
    query_instruction    : Tiền tố instruction cho query.
    document_instruction : Tiền tố instruction cho tài liệu.
    encode_kwargs        : Tham số phụ cho model.encode(), vd batch_size=64.
    """

    def __init__(
        self,
        model_name:           str  = "BAAI/bge-m3",
        device:               str  = "cpu",
        normalize_embeddings: bool = True,
        query_instruction:    str | None = None,
        document_instruction: str | None = None,
        encode_kwargs:        dict | None = None,
        **kwargs,
    ):
        super().__init__(model_name, **kwargs)
        self.device               = device
        self.normalize_embeddings = normalize_embeddings
        self.query_instruction    = query_instruction
        self.document_instruction = document_instruction
        self.encode_kwargs        = encode_kwargs or {}

    def _build(self) -> Embeddings:
        from langchain_huggingface import HuggingFaceEmbeddings

        _encode_kwargs = {"normalize_embeddings": self.normalize_embeddings}
        _encode_kwargs.update(self.encode_kwargs)

        init_kwargs: dict = {
            "model_name":    self.model_name,
            "model_kwargs":  {"device": self.device},
            "encode_kwargs": _encode_kwargs,
        }

        if self.query_instruction:
            init_kwargs["query_instruction"] = self.query_instruction
        if self.document_instruction:
            init_kwargs["embed_instruction"] = self.document_instruction

        return HuggingFaceEmbeddings(**init_kwargs)
