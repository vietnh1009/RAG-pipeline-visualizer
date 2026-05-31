"""
post_retrieval/factory.py
==========================
Factory function and config-driven builder — single entry points.

    build_pipeline(**kwargs)               -> PostRetrievalPipeline
    build_pipeline_from_config(cfg)        -> PostRetrievalPipeline
"""

from __future__ import annotations

from post_retrieval.pipeline import PostRetrievalPipeline


def build_pipeline(**kwargs) -> PostRetrievalPipeline:
    """
    Build a PostRetrievalPipeline with explicit parameters.

    All keyword arguments are forwarded directly to PostRetrievalPipeline.

    Examples
    --------
    >>> build_pipeline(reranker="cross_encoder", top_n=5)
    >>> build_pipeline(reranker="cohere", apply_mmr=True, context_ordering="sandwich")
    >>> build_pipeline(reranker="none", apply_compression=True, compression_mode="extract")
    """
    return PostRetrievalPipeline(**kwargs)


def build_pipeline_from_config(cfg: dict) -> PostRetrievalPipeline:
    """
    Build a PostRetrievalPipeline from the ``query_pipeline.post_retrieval``
    section of config.yaml.

    Parameters
    ----------
    cfg : Full config dict from ``utils.config.load_config()``.

    Config keys used
    ----------------
    query_pipeline.post_retrieval.reranker
    query_pipeline.post_retrieval.top_n
    query_pipeline.post_retrieval.cross_encoder_model
    query_pipeline.post_retrieval.apply_mmr
    query_pipeline.post_retrieval.mmr_lambda
    query_pipeline.post_retrieval.apply_compression
    query_pipeline.post_retrieval.compression_mode
    query_pipeline.post_retrieval.apply_llm_filter
    query_pipeline.post_retrieval.apply_redundancy
    query_pipeline.post_retrieval.redundancy_threshold
    query_pipeline.post_retrieval.context_ordering
    query_pipeline.post_retrieval.metadata_conditions
    query_pipeline.generation.provider
    query_pipeline.generation.model_name
    data.language
    """
    post_cfg = cfg["query_pipeline"]["post_retrieval"]
    gen_cfg  = cfg["query_pipeline"]["generation"]

    return PostRetrievalPipeline(
        reranker             = post_cfg.get("reranker",              "cross_encoder"),
        top_n                = post_cfg.get("top_n",                 5),
        apply_mmr            = post_cfg.get("apply_mmr",             False),
        mmr_lambda           = post_cfg.get("mmr_lambda",            0.5),
        apply_compression    = post_cfg.get("apply_compression",     False),
        compression_mode     = post_cfg.get("compression_mode",      "extract"),
        apply_llm_filter     = post_cfg.get("apply_llm_filter",      False),
        apply_redundancy     = post_cfg.get("apply_redundancy",      True),
        redundancy_threshold = post_cfg.get("redundancy_threshold",  0.92),
        context_ordering     = post_cfg.get("context_ordering",      "sandwich"),
        metadata_conditions  = post_cfg.get("metadata_conditions",   None),
        llm_model            = gen_cfg.get("model_name",             "gpt-4.1-mini"),
        llm_provider         = gen_cfg.get("provider",               "openai"),
        cross_encoder_model  = post_cfg.get("cross_encoder_model",   "BAAI/bge-reranker-v2-m3"),
        cohere_rerank_model  = post_cfg.get("cohere_rerank_model",   "rerank-v3.5"),
        language             = cfg["data"].get("language",           "both"),
    )