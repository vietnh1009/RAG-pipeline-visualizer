"""
pre_retrieval/pipeline.py
==========================
PreRetrievalPipeline — orchestrate multiple transformers in sequence.

Each transformer in ``transformations`` is applied to the original query.
Their outputs are merged into a single TransformResult:
  - All generated queries are accumulated and deduplicated.
  - metadata_filter, intent, retrieval_path — last writer wins.
  - extra dicts are merged (later keys overwrite earlier ones).

The original query is always guaranteed to be in the final query list.

Usage
-----
    pipeline = PreRetrievalPipeline(
        transformations=["rewrite", "multi_query"],
        llm_model="gpt-4.1-mini",
        llm_provider="openai",
        language="both",
        multi_query_count=3,
    )
    result = pipeline.transform("how does rag work?")
    # result.queries -> ["How does RAG work?", "sub-q1", "sub-q2", "sub-q3"]
"""

from __future__ import annotations

import logging
from typing import Any

from pre_retrieval.base import BaseTransformer, TransformResult

logger = logging.getLogger(__name__)


class PreRetrievalPipeline:
    """
    Chain multiple pre-retrieval transformers and merge their outputs.

    Parameters
    ----------
    transformations : Ordered list of strategy names.
                      Valid names: none, rewrite, expand, hyde, step_back,
                      multi_query, decompose, self_query, route.
    llm_model       : LLM used by all LLM-based transformers.
    llm_provider    : "openai" | "anthropic" | "google"
    language        : "vi" | "en" | "both"
    **kwargs        : Per-strategy overrides (see _build for supported keys).
    """

    def __init__(
        self,
        transformations: list[str],
        llm_model:       str = "gpt-4.1-mini",
        llm_provider:    str = "openai",
        language:        str = "both",
        **kwargs: Any,
    ):
        self.llm_model       = llm_model
        self.llm_provider    = llm_provider
        self.language        = language
        self.strategy_kwargs = kwargs

        self._transformers: list[BaseTransformer] = [
            self._build(name) for name in (transformations or ["none"])
        ]

    def transform(self, query: str) -> TransformResult:
        """Apply all transformers to the query and return a merged result."""
        merged        = TransformResult(original_query=query, queries=[])
        current_query = query  # updated by transformers that rewrite (single replacement)

        for transformer in self._transformers:
            result = transformer.transform(current_query)

            # Accumulate queries (deduplicated)
            for q in result.queries:
                if q and q not in merged.queries:
                    merged.queries.append(q)

            # Last-writer-wins for structured fields
            if result.metadata_filter is not None:
                merged.metadata_filter = result.metadata_filter
            if result.intent is not None:
                merged.intent = result.intent
            if result.retrieval_path is not None:
                merged.retrieval_path = result.retrieval_path

            merged.extra.update(result.extra)

            # If transformer replaced the query (single non-original result)
            # update current_query so subsequent transformers chain correctly
            if (len(result.queries) == 1
                    and result.queries[0] != query
                    and result.queries[0] != current_query):
                current_query = result.queries[0]

        # Guarantee original query is always present
        if query not in merged.queries:
            merged.queries.insert(0, query)

        logger.info(
            "PreRetrievalPipeline: '%s' → %d queries, filter=%s, intent=%s",
            query[:60], len(merged.queries), merged.metadata_filter, merged.intent,
        )
        return merged

    # ------------------------------------------------------------------
    # Builder
    # ------------------------------------------------------------------

    def _build(self, name: str) -> BaseTransformer:
        """Instantiate a transformer from its strategy name."""
        kw = self.strategy_kwargs
        base = dict(
            llm_model=self.llm_model,
            llm_provider=self.llm_provider,
            language=self.language,
        )

        if name == "none":
            from pre_retrieval.passthrough import PassthroughTransformer
            return PassthroughTransformer(**base)

        if name == "rewrite":
            from pre_retrieval.query_rewriter import QueryRewriter
            return QueryRewriter(**base)

        if name == "expand":
            from pre_retrieval.query_expander import QueryExpander
            return QueryExpander(
                mode=kw.get("expansion_mode", "llm"),
                num_expansions=kw.get("num_expansions", 3),
                **base,
            )

        if name == "hyde":
            from pre_retrieval.hyde import HyDETransformer
            return HyDETransformer(
                doc_length=kw.get("hyde_doc_length", 100),
                **base,
            )

        if name == "step_back":
            from pre_retrieval.step_back import StepBackTransformer
            return StepBackTransformer(
                include_original=kw.get("step_back_include_original", True),
                **base,
            )

        if name == "multi_query":
            from pre_retrieval.multi_query import MultiQueryTransformer
            return MultiQueryTransformer(
                n_queries=kw.get("multi_query_count", 3),
                include_original=kw.get("multi_query_include_original", True),
                **base,
            )

        if name == "decompose":
            from pre_retrieval.query_decomposition import QueryDecompositionTransformer
            return QueryDecompositionTransformer(
                mode=kw.get("decomposition_mode", "parallel"),
                max_sub_questions=kw.get("max_sub_questions", 4),
                include_original=kw.get("decompose_include_original", False),
                **base,
            )

        if name == "self_query":
            from pre_retrieval.self_query import SelfQueryTransformer
            return SelfQueryTransformer(
                schema=kw.get("self_query_schema", {}),
                **base,
            )


        if name == "route":
            from pre_retrieval.query_router import QueryRouter
            return QueryRouter(
                routes=kw.get("routes", {"general": "all questions"}),
                mode=kw.get("routing_mode", "llm"),
                route_rules=kw.get("route_rules"),
                default_route=kw.get("default_route", "general"),
                **base,
            )

        raise ValueError(
            f"Unknown pre-retrieval transformation: '{name}'. "
            "Valid: none, rewrite, expand, hyde, step_back, multi_query, "
            "decompose, self_query, route"
        )