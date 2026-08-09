"""
embedding/factory.py
====================
Factory functions — điểm vào duy nhất cho phần còn lại của pipeline.

    get_embedder(provider, model_name, **kwargs) -> BaseEmbedder
    get_embedder_from_config(cfg)               -> EmbeddingPipeline

Providers
---------
  openai        text-embedding-3-small / large  API, MRL
  cohere        embed-multilingual-v3.0         Chất lượng VI tốt nhất qua API
  huggingface   BAAI/bge-m3, Qwen3-Embedding   Local, instruction-tuned
  fastembed     multilingual-e5-small           CPU ONNX, không cần GPU
  ollama        bge-m3, nomic-embed-text        Local server
"""

from __future__ import annotations

import logging
from typing import Any

from embedding.base import BaseEmbedder

logger = logging.getLogger(__name__)

_REGISTRY: dict[str, tuple[str, str]] = {
    "openai":      ("embedding.openai_embedder",      "OpenAIEmbedder"),
    "cohere":      ("embedding.cohere_embedder",      "CohereEmbedder"),
    "huggingface": ("embedding.huggingface_embedder", "HuggingFaceEmbedder"),
    "fastembed":   ("embedding.fastembed_embedder",   "FastEmbedEmbedder"),
    "ollama":      ("embedding.ollama_embedder",      "OllamaEmbedder"),
}


def get_embedder(provider: str, model_name: str, **kwargs: Any) -> BaseEmbedder:
    """
    Khởi tạo embedder dense theo tên provider.

    Tham số
    -------
    provider   : "openai" | "cohere" | "huggingface" | "fastembed" | "ollama".
    model_name : Tên model của provider tương ứng.
    **kwargs   : Tham số khởi tạo chuyển thẳng cho lớp embedder.

    Ví dụ
    -----
    >>> get_embedder("openai",      "text-embedding-3-large", dimensions=512)
    >>> get_embedder("cohere",      "embed-multilingual-v3.0", input_type="search_document")
    >>> get_embedder("huggingface", "BAAI/bge-m3", device="cuda")
    >>> get_embedder("fastembed",   "intfloat/multilingual-e5-small")
    >>> get_embedder("ollama",      "bge-m3", base_url="http://localhost:11434")
    """
    entry = _REGISTRY.get(provider)
    if entry is None:
        valid = ", ".join(sorted(_REGISTRY))
        raise ValueError(f"Unknown embedding provider '{provider}'. Valid: {valid}")

    import importlib
    module_path, class_name = entry
    cls = getattr(importlib.import_module(module_path), class_name)

    logger.info("Embedding: provider=%s  model=%s", provider, model_name)
    return cls(model_name=model_name, **kwargs)


def get_embedder_from_config(cfg: dict) -> "EmbeddingPipeline":
    """
    Dựng EmbeddingPipeline từ mục ``indexing.embedding`` của config.yaml.

    Đọc ``provider``, ``model_name``, ``enable_sparse``, ``sparse_method``, cùng
    vài key chỉ áp cho một số provider: ``dimensions`` (openai), ``input_type``
    (cohere), ``device`` / ``query_instruction`` / ``document_instruction``
    (huggingface, fastembed), ``ollama_base_url`` (ollama).
    """
    from embedding.pipeline import EmbeddingPipeline

    emb_cfg  = cfg["indexing"]["embedding"]
    provider = emb_cfg.get("provider",   "openai")
    model    = emb_cfg.get("model_name", "text-embedding-3-small")

    dense_kwargs: dict[str, Any] = {}

    # Chỉ truyền đúng những key mà từng provider hiểu
    if provider == "openai":
        if emb_cfg.get("dimensions"):
            dense_kwargs["dimensions"] = emb_cfg["dimensions"]

    elif provider == "cohere":
        dense_kwargs["input_type"] = emb_cfg.get("input_type", "search_document")

    elif provider in ("huggingface", "fastembed"):
        dense_kwargs["device"] = emb_cfg.get("device", "cpu")
        if emb_cfg.get("query_instruction"):
            dense_kwargs["query_instruction"] = emb_cfg["query_instruction"]
        if emb_cfg.get("document_instruction"):
            dense_kwargs["document_instruction"] = emb_cfg["document_instruction"]

    elif provider == "ollama":
        dense_kwargs["base_url"] = emb_cfg.get(
            "ollama_base_url", "http://localhost:11434"
        )

    return EmbeddingPipeline(
        dense_provider=provider,
        dense_model=model,
        dense_kwargs=dense_kwargs,
        enable_sparse=emb_cfg.get("enable_sparse", False),
        sparse_method=emb_cfg.get("sparse_method",  "bm25"),
    )
