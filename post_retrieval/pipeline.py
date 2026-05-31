"""
post_retrieval/pipeline.py
===========================
PostRetrievalPipeline — orchestrate multiple processors in a fixed sequence.

Standard processing order (each step is optional, enabled via constructor):
  1. MetadataFilter    — hard filter (if conditions provided)
  2. RedundancyFilter  — semantic near-duplicate removal
  3. Reranker          — relevance scoring (cross_encoder / API / LLM)
  4. LLMFilter         — binary relevance filter (optional)
  5. MMRFilter         — diversity selection (optional)
  6. ContextCompressor — content trimming (optional)
  7. ContextOrderer    — lost-in-the-middle mitigation (always last)

Each step is only executed if enabled.  The order is fixed and
intentional: dedup before rerank (saves rerank API calls), rerank
before MMR (MMR needs scores), ordering always last.

Usage
-----
    pipeline = PostRetrievalPipeline(reranker="cross_encoder", top_n=5)
    docs = pipeline.process(query="What is RAG?", docs=retrieved_docs)
"""

from __future__ import annotations

import logging

from langchain_core.documents import Document

from post_retrieval.base import BasePostProcessor
from post_retrieval.utils import deduplicate

logger = logging.getLogger(__name__)


class PostRetrievalPipeline:
    """
    Chain multiple post-retrieval processors.

    Parameters
    ----------
    reranker             : "none" | "cross_encoder" | "cohere" | "llm"
    top_n                : Documents kept after reranking.
    apply_mmr            : Enable MMR diversity filter.
    mmr_lambda           : MMR relevance vs diversity trade-off (0–1).
    apply_compression    : Enable LLM contextual compression.
    compression_mode     : "extract" | "summarise"
    apply_llm_filter     : Enable LLM binary relevance filter.
    apply_redundancy     : Enable semantic near-duplicate filter.
    redundancy_threshold : Cosine similarity threshold for near-dup.
    context_ordering     : "relevance" | "reverse" | "sandwich" | "original"
    metadata_conditions  : Hard filter conditions (list of dicts).
    llm_model            : LLM for reranker / compressor / filter.
    llm_provider         : "openai" | "anthropic" | "google"
    cross_encoder_model  : Model for CrossEncoderReranker.
    cohere_rerank_model  : Model for CohereReranker.
    language             : "vi" | "en" | "both"
    """

    def __init__(
        self,
        reranker:             str   = "cross_encoder",
        top_n:                int   = 5,
        apply_mmr:            bool  = False,
        mmr_lambda:           float = 0.5,
        apply_compression:    bool  = False,
        compression_mode:     str   = "extract",
        apply_llm_filter:     bool  = False,
        apply_redundancy:     bool  = True,
        redundancy_threshold: float = 0.92,
        context_ordering:     str   = "sandwich",
        metadata_conditions:  list[dict] | None = None,
        llm_model:            str   = "gpt-4.1-mini",
        llm_provider:         str   = "openai",
        cross_encoder_model:  str   = "BAAI/bge-reranker-v2-m3",
        cohere_rerank_model:  str   = "rerank-v3.5",
                    language:             str   = "both",
    ):
        self.top_n = top_n
        self._steps: list[BasePostProcessor] = []

        # 1. Hard metadata filter
        if metadata_conditions:
            from post_retrieval.metadata_filter import MetadataFilter
            self._steps.append(MetadataFilter(conditions=metadata_conditions))

        # 2. Near-duplicate removal (run before reranking to save API calls)
        if apply_redundancy:
            from post_retrieval.redundancy_filter import RedundancyFilter
            self._steps.append(RedundancyFilter(
                top_n=top_n * 3, threshold=redundancy_threshold
            ))

        # 3. Reranker
        if reranker and reranker != "none":
            self._steps.append(self._build_reranker(
                reranker, top_n, llm_model, llm_provider,
                cross_encoder_model, cohere_rerank_model, language,
            ))

        # 4. LLM binary filter
        if apply_llm_filter:
            from post_retrieval.llm_filter import LLMFilter
            self._steps.append(LLMFilter(llm_model=llm_model, llm_provider=llm_provider, language=language))

        # 5. MMR diversity
        if apply_mmr:
            from post_retrieval.mmr_filter import MMRFilter
            self._steps.append(MMRFilter(top_n=top_n, mmr_lambda=mmr_lambda))

        # 6. Contextual compression
        if apply_compression:
            from post_retrieval.context_compressor import ContextCompressor
            self._steps.append(ContextCompressor(
                llm_model=llm_model, llm_provider=llm_provider,
                mode=compression_mode, language=language,
            ))

        # 7. Context ordering — always applied last
        from post_retrieval.context_orderer import ContextOrderer
        self._steps.append(ContextOrderer(ordering=context_ordering))

    def process(self, query: str, docs: list[Document]) -> list[Document]:
        """
        Run all configured processors in sequence.

        Parameters
        ----------
        query : Primary user query.
        docs  : Raw retrieved documents from the retrieval stage.

        Returns
        -------
        Refined list of documents ready for prompt construction.
        """
        current = deduplicate(docs)

        for step in self._steps:
            if not current:
                break
            current = step.process(query, current)
            logger.debug("%s → %d docs", type(step).__name__, len(current))

        result = current[:self.top_n]
        logger.info("PostRetrievalPipeline: %d → %d docs.", len(docs), len(result))
        return result

    @staticmethod
    def _build_reranker(
        reranker:     str, top_n: int,
        llm_model:    str, llm_provider: str,
        ce_model:     str, cohere_model: str, language: str,
    ) -> BasePostProcessor:
        if reranker == "cross_encoder":
            from post_retrieval.cross_encoder_reranker import CrossEncoderReranker
            return CrossEncoderReranker(model_name=ce_model, top_n=top_n)
        if reranker == "cohere":
            from post_retrieval.cohere_reranker import CohereReranker
            return CohereReranker(model_name=cohere_model, top_n=top_n)
        if reranker == "llm":
            from post_retrieval.llm_reranker import LLMReranker
            return LLMReranker(llm_model=llm_model, llm_provider=llm_provider,
                               top_n=top_n, language=language)
        raise ValueError(f"Unknown reranker: '{reranker}'")