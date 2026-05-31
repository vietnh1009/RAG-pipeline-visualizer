"""
post_retrieval/
===============
Post-retrieval processing package — Stage 7 of the full RAG pipeline.

Public API
----------
    from post_retrieval import build_pipeline, build_pipeline_from_config
    from post_retrieval import PostRetrievalPipeline

    pipeline = build_pipeline(
        reranker="cross_encoder",
        cross_encoder_model="BAAI/bge-reranker-v2-m3",
        top_n=5,
        apply_redundancy=True,
        context_ordering="sandwich",
    )
    docs = pipeline.process(query="What is RAG?", docs=retrieved_docs)

Processing pipeline (fixed order)
-----------------------------------
  1. MetadataFilter     hard filter on metadata fields (if conditions set)
  2. RedundancyFilter   semantic near-duplicate removal
  3. Reranker           cross_encoder / cohere / llm
  4. LLMFilter          binary YES/NO relevance check (optional)
  5. MMRFilter          diversity selection (optional)
  6. ContextCompressor  extract or summarise relevant content (optional)
  7. ContextOrderer     reorder for lost-in-the-middle mitigation (always)
"""

from post_retrieval.factory  import build_pipeline, build_pipeline_from_config
from post_retrieval.pipeline import PostRetrievalPipeline

__all__ = ["build_pipeline", "build_pipeline_from_config", "PostRetrievalPipeline"]
