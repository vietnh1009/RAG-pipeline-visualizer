"""
post_retrieval/factory.py
==========================
Hai điểm vào duy nhất để dựng pipeline xử lý sau truy hồi.

    build_pipeline(**kwargs)        -> PostRetrievalPipeline
    build_pipeline_from_config(cfg) -> PostRetrievalPipeline
"""

from __future__ import annotations

from post_retrieval.pipeline import PostRetrievalPipeline


def build_pipeline(**kwargs) -> PostRetrievalPipeline:
    """
    Dựng PostRetrievalPipeline từ tham số truyền thẳng.

    Mọi keyword argument được chuyển nguyên vẹn xuống PostRetrievalPipeline.

    Ví dụ
    -----
    >>> build_pipeline(reranker="cross_encoder", top_n=5)
    >>> build_pipeline(reranker="cohere", apply_mmr=True, context_ordering="sandwich")
    >>> build_pipeline(reranker="none", apply_compression=True)
    """
    return PostRetrievalPipeline(**kwargs)


def build_pipeline_from_config(cfg: dict) -> PostRetrievalPipeline:
    """
    Dựng PostRetrievalPipeline từ mục ``query_pipeline.post_retrieval`` của config.yaml.

    Tham số
    -------
    cfg : Dict config đầy đủ từ ``utils.config.load_config()``.

    Đọc thêm ``query_pipeline.generation`` (provider, model_name) và
    ``data.language``, nên phải truyền config đầy đủ chứ không phải một mẩu —
    truyền thiếu sẽ KeyError.
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