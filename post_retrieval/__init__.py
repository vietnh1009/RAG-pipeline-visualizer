"""
post_retrieval/
===============
Xử lý sau truy hồi — Stage 7 của pipeline RAG.

API công khai
-------------
    from post_retrieval import build_pipeline, build_pipeline_from_config

    pipeline = build_pipeline(
        reranker="cross_encoder",
        top_n=5,
        apply_redundancy=True,
        context_ordering="sandwich",
    )
    docs = pipeline.process(query="RAG là gì?", docs=retrieved_docs)

Thứ tự xử lý (cố định)
----------------------
  1. MetadataFilter     lọc cứng theo metadata (nếu có điều kiện)
  2. RedundancyFilter   bỏ đoạn gần trùng về ngữ nghĩa
  3. Reranker           cross_encoder / cohere / llm
  4. LLMFilter          LLM trả lời YES/NO về độ liên quan (tuỳ chọn)
  5. MMRFilter          chọn theo độ đa dạng (tuỳ chọn)
  6. ContextCompressor  trích hoặc tóm tắt phần liên quan (tuỳ chọn)
  7. ContextOrderer     sắp lại chống "lost in the middle" (luôn chạy)
"""

from post_retrieval.factory  import build_pipeline, build_pipeline_from_config
from post_retrieval.pipeline import PostRetrievalPipeline

__all__ = ["build_pipeline", "build_pipeline_from_config", "PostRetrievalPipeline"]
