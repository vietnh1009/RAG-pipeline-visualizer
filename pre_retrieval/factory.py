"""
pre_retrieval/factory.py
========================
Factory function and config-driven builder — single entry points.

    build_pipeline(**kwargs)             -> PreRetrievalPipeline
    build_pipeline_from_config(cfg)      -> PreRetrievalPipeline
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
    Build a PreRetrievalPipeline with explicit parameters.

    Parameters
    ----------
    transformations : List of strategy names in order of application.
                      Defaults to ["none"] (passthrough).
    llm_model       : LLM for all LLM-based transformers.
    llm_provider    : "openai" | "anthropic" | "google"
    language        : "vi" | "en" | "both"
    **kwargs        : Per-strategy overrides:
                        multi_query_count, num_expansions,
                        self_query_schema, routes, routing_mode, ...

    Examples
    --------
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
    Build a PreRetrievalPipeline from the ``query_pipeline.pre_retrieval``
    section of config.yaml.

    Parameters
    ----------
    cfg : Full config dict from ``utils.config.load_config()``.

    Config keys used
    ----------------
    query_pipeline.pre_retrieval.transformations   list of strategy names
    query_pipeline.pre_retrieval.transformation_llm LLM model name
    query_pipeline.pre_retrieval.multi_query_count  int
    query_pipeline.pre_retrieval.hyde_doc_length    int
    query_pipeline.pre_retrieval.expansion_mode     "llm" | "wordnet"
    query_pipeline.pre_retrieval.num_expansions     int
    query_pipeline.pre_retrieval.self_query_schema  dict
    query_pipeline.pre_retrieval.routes             dict
    query_pipeline.pre_retrieval.routing_mode       "llm" | "keyword" | "semantic"
    query_pipeline.pre_retrieval.route_rules        list of [pattern, route] pairs
    query_pipeline.pre_retrieval.default_route      str
    query_pipeline.generation.provider              LLM provider
    query_pipeline.generation.model_name            fallback LLM model
    data.language                                   corpus language
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