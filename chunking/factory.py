"""
chunking/factory.py
===================
Factory function và config builder cho chunking strategy.

Strategies
----------
  recursive      Cắt theo đoạn → dòng → câu → ký tự. Mặc định tốt nhất.
  token_based    Đếm token (BPE). Quan trọng với tiếng Việt & embedding limit.
  format_aware   Cắt theo cấu trúc tài liệu (Markdown heading, code block, HTML).
  sentence_aware Cắt theo ranh giới câu. Tốt cho Q&A, FAQ.
  semantic       Cắt theo cosine similarity. Tốt cho PDF nhiều chủ đề.
  hierarchical   Parent (section lớn) + child (đoạn nhỏ). Giảm hallucination.
  contextual     Recursive + LLM thêm context prefix mỗi chunk (Anthropic method).
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.documents import Document

from chunking.base import BaseChunker

logger = logging.getLogger(__name__)

_REGISTRY: dict[str, tuple[str, str]] = {
    "recursive":      ("chunking.recursive",      "RecursiveChunker"),
    "token_based":    ("chunking.token_based",     "TokenChunker"),
    "format_aware":   ("chunking.format_aware",    "FormatAwareChunker"),
    "sentence_aware": ("chunking.sentence_aware",  "SentenceChunker"),
    "semantic":       ("chunking.semantic",         "SemanticChunker"),
    "hierarchical":   ("chunking.hierarchical",    "HierarchicalChunker"),
    "contextual":     ("chunking.contextual",      "ContextualChunker"),
}


def get_chunker(strategy: str, **kwargs: Any) -> BaseChunker:
    """
    Instantiate a chunker by strategy name.

    Parameters
    ----------
    strategy : One of the strategy keys listed in ``_REGISTRY``.
    **kwargs : Constructor arguments forwarded to the chosen class.

    Examples
    --------
    >>> get_chunker("recursive",    chunk_size=500, chunk_overlap=100)
    >>> get_chunker("format_aware", chunk_size=500, chunk_overlap=100, format_type="markdown")
    >>> get_chunker("hierarchical", parent_chunk_size=2000, chunk_size=400)
    >>> get_chunker("contextual",   base_strategy="recursive", llm_model="claude-haiku-4-5-20251001")
    """
    entry = _REGISTRY.get(strategy)
    if entry is None:
        valid = ", ".join(sorted(_REGISTRY))
        raise ValueError(f"Unknown chunking strategy '{strategy}'. Valid: {valid}")

    module_path, class_name = entry
    import importlib
    cls = getattr(importlib.import_module(module_path), class_name)
    return cls(**kwargs)


def chunk_documents_from_config(docs: list[Document], cfg: dict) -> list[Document]:
    """
    Build and run a chunker from the ``indexing.chunking`` section of config.yaml.
    """
    chunk_cfg = cfg["indexing"]["chunking"]
    strategy  = chunk_cfg.get("strategy", "recursive")

    kwargs: dict[str, Any] = {
        "chunk_size":    chunk_cfg.get("chunk_size",    500),
        "chunk_overlap": chunk_cfg.get("chunk_overlap", 100),
    }

    if strategy == "token_based":
        kwargs["encoding_name"] = chunk_cfg.get("token_encoding", "cl100k_base")

    elif strategy == "semantic":
        kwargs["embedding_model_name"] = chunk_cfg.get(
            "semantic_embedding_model", "sentence-transformers/all-MiniLM-L6-v2"
        )

    elif strategy == "contextual":
        kwargs.update({
            "base_strategy": chunk_cfg.get("base_strategy",  "recursive"),
            "llm_model":     chunk_cfg.get("contextual_llm", "claude-haiku-4-5-20251001"),
            "llm_provider":  chunk_cfg.get("llm_provider",   "anthropic"),
        })

    elif strategy == "hierarchical":
        kwargs["parent_chunk_size"] = chunk_cfg.get("parent_chunk_size", 2000)
        kwargs["chunk_size"]        = chunk_cfg.get("chunk_size", 400)

    elif strategy == "format_aware":
        kwargs["format_type"] = chunk_cfg.get("format_type", "auto")

    chunker = get_chunker(strategy, **kwargs)
    chunks  = chunker.split(docs)

    logger.info(
        "Chunking ('%s'): %d docs → %d chunks.", strategy, len(docs), len(chunks)
    )
    return chunks
