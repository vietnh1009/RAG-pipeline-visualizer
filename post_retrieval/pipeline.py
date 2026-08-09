"""
post_retrieval/pipeline.py
===========================
PostRetrievalPipeline — chạy các bộ xử lý theo một thứ tự cố định.

  1. MetadataFilter    — lọc cứng theo metadata (nếu có điều kiện)
  2. RedundancyFilter  — bỏ đoạn gần trùng ngữ nghĩa
  3. Reranker          — chấm lại độ liên quan (cross_encoder / API / LLM)
  4. LLMFilter         — lọc nhị phân bằng LLM (tuỳ chọn)
  5. MMRFilter         — chọn theo độ đa dạng (tuỳ chọn)
  6. ContextCompressor — cắt gọn nội dung (tuỳ chọn)
  7. ContextOrderer    — chống "lost in the middle" (luôn chạy cuối)

Bước nào không bật thì bỏ qua. Thứ tự là cố ý: khử trùng trước rerank để tiết
kiệm lời gọi API, rerank trước MMR vì MMR cần điểm, sắp thứ tự luôn cuối cùng.

Ví dụ
-----
    pipeline = PostRetrievalPipeline(reranker="cross_encoder", top_n=5)
    docs = pipeline.process(query="RAG là gì?", docs=retrieved_docs)
"""

from __future__ import annotations

import logging

from langchain_core.documents import Document

from post_retrieval.base import BasePostProcessor
from utils.documents import deduplicate

logger = logging.getLogger(__name__)


class PostRetrievalPipeline:
    """
    Nối nhiều bộ xử lý sau truy hồi.

    Tham số
    -------
    reranker             : "none" | "cross_encoder" | "cohere" | "llm"
    top_n                : Số document giữ lại sau rerank.
    apply_mmr            : Bật bộ lọc đa dạng MMR.
    mmr_lambda           : Cân bằng liên quan / đa dạng của MMR (0–1).
    apply_compression    : Bật nén ngữ cảnh bằng LLM.
    compression_mode     : "extract" | "summarise"
    apply_llm_filter     : Bật lọc nhị phân bằng LLM.
    apply_redundancy     : Bật lọc đoạn gần trùng.
    redundancy_threshold : Ngưỡng cosine coi là gần trùng.
    context_ordering     : "relevance" | "reverse" | "sandwich" | "original"
    metadata_conditions  : Điều kiện lọc cứng, list các dict.
    llm_model            : LLM cho reranker / compressor / filter.
    llm_provider         : "openai" | "anthropic" | "google"
    cross_encoder_model  : Model cho CrossEncoderReranker.
    cohere_rerank_model  : Model cho CohereReranker.
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

        # 1. Lọc cứng theo metadata
        if metadata_conditions:
            from post_retrieval.metadata_filter import MetadataFilter
            self._steps.append(MetadataFilter(conditions=metadata_conditions))

        # 2. Bỏ đoạn gần trùng — chạy trước rerank để tiết kiệm lời gọi API
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

        # 4. Lọc nhị phân bằng LLM
        if apply_llm_filter:
            from post_retrieval.llm_filter import LLMFilter
            self._steps.append(LLMFilter(llm_model=llm_model, llm_provider=llm_provider, language=language))

        # 5. Đa dạng hoá bằng MMR
        if apply_mmr:
            from post_retrieval.mmr_filter import MMRFilter
            self._steps.append(MMRFilter(top_n=top_n, mmr_lambda=mmr_lambda))

        # 6. Nén ngữ cảnh
        if apply_compression:
            from post_retrieval.context_compressor import ContextCompressor
            self._steps.append(ContextCompressor(
                llm_model=llm_model, llm_provider=llm_provider,
                mode=compression_mode, language=language,
            ))

        # 7. Sắp thứ tự ngữ cảnh — luôn chạy cuối cùng
        from post_retrieval.context_orderer import ContextOrderer
        self._steps.append(ContextOrderer(ordering=context_ordering))

    def process(self, query: str, docs: list[Document]) -> list[Document]:
        """
        Chạy lần lượt toàn bộ bộ xử lý đã bật.

        Tham số
        -------
        query : Query chính của người dùng.
        docs  : Document thô từ bước retrieval.

        Trả về
        ------
        Danh sách document đã tinh lọc, sẵn sàng dựng prompt.
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