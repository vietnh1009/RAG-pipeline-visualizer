"""
pipeline/
=========
Hai pipeline chính của RAG-pipeline-visualizer.

Stage 1 — Indexing (offline, chạy một lần):
    from pipeline.indexing_pipeline import IndexingPipeline

Stage 2 — Generation (online, mỗi lần query):
    from pipeline.generation_pipeline import GenerationPipeline, GenerationPipelineResult
"""

from pipeline.indexing_pipeline  import IndexingPipeline,  IndexingResult
from pipeline.generation_pipeline import GenerationPipeline, GenerationPipelineResult

__all__ = [
    "IndexingPipeline",  "IndexingResult",
    "GenerationPipeline", "GenerationPipelineResult",
]
