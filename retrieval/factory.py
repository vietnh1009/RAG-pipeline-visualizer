"""
retrieval/factory.py
====================
Factory function and config-driven builder — single entry points.

    get_retriever(strategy, vector_store, **kwargs) -> BaseRetriever
    build_retriever_from_config(cfg, vector_store)  -> BaseRetriever

Strategies
----------
  dense           Standard ANN vector similarity search
  sparse          BM25 keyword search
  hybrid          Dense + BM25 fused (RRF / weighted / DBSF) ← recommended
  multi_query     Run N queries from TransformResult, merge via RRF
  parent_document Search child chunks, return parent context
  sentence_window Search sentences, expand to ±window neighbours
  multi_hop       Iterative retrieve-then-reason (LLM decides to continue)
  contextual      Dense + score threshold + MMR diversity
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
    Instantiate a retriever by strategy name.

    Parameters
    ----------
    strategy     : One of the strategy keys in _REGISTRY.
    vector_store : Populated LangChain VectorStore.
    documents    : Full corpus list — required for sparse, hybrid,
                   parent_document, sentence_window (BM25 / window expansion).
    **kwargs     : Constructor arguments forwarded to the retriever.

    Examples
    --------
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

    # Inject documents for strategies that need corpus access
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
    Build a retriever from the ``query_pipeline.retrieval`` section of config.yaml.

    Config keys used
    ----------------
    query_pipeline.retrieval.strategy           retrieval strategy name
    query_pipeline.retrieval.top_k              number of results
    query_pipeline.retrieval.fusion_method      "rrf" | "weighted" | "dbsf"
    query_pipeline.retrieval.rrf_k              RRF constant (default 60)
    query_pipeline.retrieval.hybrid_alpha       dense weight 0–1 (weighted fusion)
    query_pipeline.retrieval.score_threshold    min similarity score (dense/contextual)
    query_pipeline.retrieval.sentence_window_size  expansion window (sentence_window)
    query_pipeline.generation.provider         LLM provider (multi_hop)
    query_pipeline.generation.model_name       LLM model (multi_hop)
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