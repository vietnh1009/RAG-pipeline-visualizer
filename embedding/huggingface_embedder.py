"""
embedding/huggingface_embedder.py
==================================
HuggingFace sentence-transformers — run any model locally.

Recommended models for Vietnamese RAG (May 2025)
-------------------------------------------------
  BAAI/bge-m3                     1024  8192   ⭐⭐⭐⭐   Dense+sparse+multivec
  Qwen/Qwen3-Embedding            2048  32K    ⭐⭐⭐⭐⭐  MTEB #1, instruction
  intfloat/multilingual-e5-large  1024  512    ⭐⭐⭐    100 languages
  intfloat/e5-mistral-7b-instruct 4096  32K    ⭐⭐⭐    Instruction-tuned
  nomic-ai/nomic-embed-text-v1.5  768   8192   ⭐⭐⭐    MRL support
  Alibaba-NLP/gte-Qwen2-7B        3584  131K   ⭐⭐⭐⭐   Very long context
  VinAI/phobert-large             768   256    ⭐⭐⭐⭐   Vietnamese-specific

Instruction-following models
-----------------------------
Qwen3-Embedding, E5-mistral, GTE-Qwen2 accept asymmetric instruction prefixes.
Pass ``query_instruction`` and ``document_instruction`` to activate them.
These MUST be consistent between index time and query time.

  Example for medical Vietnamese:
    query_instruction    = "Retrieve relevant passages for this medical question:"
    document_instruction = "Represent this Vietnamese medical document:"
"""

from __future__ import annotations

from langchain_core.embeddings import Embeddings

from embedding.base import BaseEmbedder


class HuggingFaceEmbedder(BaseEmbedder):
    """
    Parameters
    ----------
    model_name           : HuggingFace model identifier.
    device               : "cpu" | "cuda" | "mps"
    normalize_embeddings : L2-normalise vectors (recommended for cosine similarity).
    query_instruction    : Instruction prefix for queries (instruction-following models).
    document_instruction : Instruction prefix for documents.
    encode_kwargs        : Extra kwargs for model.encode() (e.g. batch_size=64).
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
