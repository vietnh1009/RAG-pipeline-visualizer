"""
pre_retrieval/
==============
Query transformation package — Stage 5 of the full RAG pipeline.

Public API
----------
    from pre_retrieval import build_pipeline, build_pipeline_from_config
    from pre_retrieval import PreRetrievalPipeline, TransformResult

    pipeline = build_pipeline(
        transformations=["rewrite", "multi_query"],
        llm_model="claude-haiku-4-5-20251001",
        multi_query_count=3,
    )
    result = pipeline.transform("how does rag work?")
    print(result.queries)          # list of query strings for retrieval
    print(result.metadata_filter)  # populated by self_query if used
    print(result.retrieval_path)   # populated by route if used

Available transformations
-------------------------
  none       Passthrough — no transformation
  rewrite    Fix grammar/spelling, resolve pronouns, normalise
  expand     Add synonyms / related terms (LLM or WordNet)
  hyde       Generate hypothetical answer document to embed
  step_back  Abstract to a broader background question
  multi_query  Decompose into N retrieval perspectives
  decompose  Break compound question into logical sub-questions
  self_query Extract metadata filters from natural language
  route      Direct query to the most appropriate retrieval path
"""

from pre_retrieval.factory  import build_pipeline, build_pipeline_from_config
from pre_retrieval.pipeline import PreRetrievalPipeline
from pre_retrieval.base     import TransformResult

__all__ = [
    "build_pipeline",
    "build_pipeline_from_config",
    "PreRetrievalPipeline",
    "TransformResult",
]
