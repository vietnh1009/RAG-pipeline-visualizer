"""
pre_retrieval/factory.py
========================
Hai điểm vào duy nhất để dựng pipeline biến đổi query.

    build_pipeline(**kwargs)        -> PreRetrievalPipeline
    build_pipeline_from_config(cfg) -> PreRetrievalPipeline
"""

from __future__ import annotations

from typing import Any

from pre_retrieval.pipeline import PreRetrievalPipeline


def build_pipeline(
    transformations: list[str] | None = None,
    llm_model:       str = "gpt-4.1-mini",
    llm_provider:    str = "openai",
    language:        str = "both",
    **kwargs: Any,
) -> PreRetrievalPipeline:
    """
    Dựng PreRetrievalPipeline từ tham số truyền thẳng.

    Tham số
    -------
    transformations : Tên các chiến lược, theo thứ tự áp dụng.
                      Mặc định ["none"] (không biến đổi).
    llm_model       : LLM cho mọi transformer dùng LLM.
    llm_provider    : "openai" | "anthropic" | "google"
    language        : "vi" | "en" | "both"
    **kwargs        : Tuỳ chọn riêng từng chiến lược — multi_query_count,
                      num_expansions, self_query_schema, routes, ...

    Ví dụ
    -----
    >>> build_pipeline(["rewrite", "multi_query"], multi_query_count=4)
    """
    return PreRetrievalPipeline(
        transformations=transformations or ["none"],
        llm_model=llm_model,
        llm_provider=llm_provider,
        language=language,
        **kwargs,
    )


def build_pipeline_from_config(cfg: dict) -> PreRetrievalPipeline:
    """
    Dựng PreRetrievalPipeline từ mục ``query_pipeline.pre_retrieval`` của config.yaml.

    Tham số
    -------
    cfg : Dict config đầy đủ từ ``utils.config.load_config()``.

    Đọc thêm ``query_pipeline.generation`` (provider, model_name làm mặc định)
    và ``data.language``, nên phải truyền config đầy đủ chứ không phải một mẩu.
    """
    pre_cfg = cfg["query_pipeline"]["pre_retrieval"]
    gen_cfg = cfg["query_pipeline"]["generation"]

    return PreRetrievalPipeline(
        transformations  = pre_cfg.get("transformations", ["none"]),
        llm_model        = pre_cfg.get("transformation_llm",
                           gen_cfg.get("model_name", "gpt-4.1-mini")),
        llm_provider     = gen_cfg.get("provider", "openai"),
        language         = cfg["data"].get("language", "both"),
        # Per-strategy overrides
        multi_query_count            = pre_cfg.get("multi_query_count", 3),
        multi_query_include_original = True,
expansion_mode               = pre_cfg.get("expansion_mode", "llm"),
        num_expansions               = pre_cfg.get("num_expansions", 3),
        self_query_schema            = pre_cfg.get("self_query_schema", {}),
        routes                       = pre_cfg.get("routes", {}),
        routing_mode                 = pre_cfg.get("routing_mode", "llm"),
        route_rules                  = pre_cfg.get("route_rules"),
        default_route                = pre_cfg.get("default_route", "general"),
decomposition_mode           = pre_cfg.get("decomposition_mode", "parallel"),
        max_sub_questions            = pre_cfg.get("max_sub_questions", 4),
    )