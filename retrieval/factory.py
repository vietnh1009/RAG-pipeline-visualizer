"""
retrieval/factory.py
====================
Hai điểm vào duy nhất để tạo retriever.

    get_retriever(strategy, vector_store, **kwargs) -> BaseRetriever
    build_retriever_from_config(cfg, vector_store)  -> BaseRetriever

Chiến lược
----------
  dense           Tìm ANN theo độ tương đồng vector
  sparse          Tìm từ khoá bằng BM25
  hybrid          Dense + BM25, gộp bằng RRF / weighted / DBSF ← khuyến nghị
  multi_query     Chạy các query trong TransformResult rồi gộp bằng RRF
  parent_document Tìm chunk con, trả về chunk cha
  sentence_window Tìm theo câu, mở rộng ±window câu lân cận
  multi_hop       Lặp truy hồi rồi suy luận, LLM quyết định dừng hay đi tiếp
  contextual      Dense + ngưỡng điểm + đa dạng hoá MMR
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore

from retrieval.base import BaseRetriever

logger = logging.getLogger(__name__)

_REGISTRY: dict[str, tuple[str, str]] = {
    "dense":            ("retrieval.dense",            "DenseRetriever"),
    "sparse":           ("retrieval.sparse",           "SparseRetriever"),
    "hybrid":           ("retrieval.hybrid",           "HybridRetriever"),
    "multi_query":      ("retrieval.multi_query",      "MultiQueryRetriever"),
    "parent_document":  ("retrieval.parent_document",  "ParentDocumentRetriever"),
    "sentence_window":  ("retrieval.sentence_window",  "SentenceWindowRetriever"),
    "multi_hop":        ("retrieval.multi_hop",        "MultiHopRetriever"),
    "contextual":       ("retrieval.contextual",       "ContextualRetriever"),
}


def get_retriever(
    strategy:     str,
    vector_store: VectorStore,
    documents:    list[Document] | None = None,
    **kwargs: Any,
) -> BaseRetriever:
    """
    Khởi tạo retriever theo tên chiến lược.

    Tham số
    -------
    strategy     : Một trong các key của ``_REGISTRY``.
    vector_store : VectorStore của LangChain đã nạp dữ liệu.
    documents    : Toàn bộ corpus — bắt buộc cho sparse, hybrid,
                   parent_document, sentence_window (BM25 / mở rộng cửa sổ).
    **kwargs     : Tham số khởi tạo chuyển thẳng cho lớp retriever.

    Ví dụ
    -----
    >>> get_retriever("dense",   store, top_k=10)
    >>> get_retriever("hybrid",  store, documents=chunks, fusion_method="rrf")
    >>> get_retriever("multi_hop", store, max_hops=3, llm_model="gpt-4.1-mini")
    """
    entry = _REGISTRY.get(strategy)
    if entry is None:
        valid = ", ".join(sorted(_REGISTRY))
        raise ValueError(f"Unknown retrieval strategy '{strategy}'. Valid: {valid}")

    import importlib
    module_path, class_name = entry
    cls = getattr(importlib.import_module(module_path), class_name)

    # Tiêm corpus vào các chiến lược cần truy cập toàn bộ tài liệu
    _docs_strategies = {"sparse", "hybrid", "sentence_window", "parent_document"}
    if strategy in _docs_strategies and documents is not None:
        kwargs["documents"] = documents

    logger.info("Retriever: strategy=%s  top_k=%s", strategy, kwargs.get("top_k", 5))
    return cls(vector_store=vector_store, **kwargs)


def build_retriever_from_config(
    cfg:          dict,
    vector_store: VectorStore,
    documents:    list[Document] | None = None,
) -> BaseRetriever:
    """
    Dựng retriever từ mục ``query_pipeline.retrieval`` của config.yaml.

    Tuỳ chiến lược mà đọc thêm: ``fusion_method`` / ``rrf_k`` / ``hybrid_alpha``
    (hybrid), ``score_threshold`` (dense, contextual), ``sentence_window_size``
    (sentence_window). Chiến lược multi_hop lấy LLM từ
    ``query_pipeline.generation``, nên phải truyền config đầy đủ.
    """
    ret_cfg  = cfg["query_pipeline"]["retrieval"]
    gen_cfg  = cfg["query_pipeline"]["generation"]
    strategy = ret_cfg.get("strategy", "hybrid")
    top_k    = ret_cfg.get("top_k", 10)
    kwargs: dict[str, Any] = {"top_k": top_k}

    if strategy == "hybrid":
        kwargs.update({
            "fusion_method": ret_cfg.get("fusion_method", "rrf"),
            "alpha":         ret_cfg.get("hybrid_alpha",  0.5),
            "rrf_k":         ret_cfg.get("rrf_k",         60),
            "candidate_k":   top_k * 3,
        })

    elif strategy in ("dense", "contextual"):
        kwargs["score_threshold"] = ret_cfg.get("score_threshold", 0.3)
        if strategy == "contextual":
            kwargs["candidate_k"] = top_k * 2

    elif strategy == "sentence_window":
        kwargs["window_size"] = ret_cfg.get("sentence_window_size", 2)

    elif strategy == "multi_hop":
        kwargs.update({
            "llm_model":    gen_cfg.get("model_name", "gpt-4.1-mini"),
            "llm_provider": gen_cfg.get("provider",   "openai"),
        })

    elif strategy == "multi_query":
        kwargs["candidate_k"] = top_k * 2

    return get_retriever(
        strategy=strategy,
        vector_store=vector_store,
        documents=documents,
        **kwargs,
    )