"""
embedding/
==========
Text embedding package for the RAG indexing pipeline.

Providers
---------
  openai       text-embedding-3-small / large  — API, MRL support
  cohere       embed-multilingual-v3.0         — best API choice for Vietnamese
  huggingface  BAAI/bge-m3, Qwen3-Embedding   — local, no API key, instruction-tuned
  fastembed    multilingual-e5-small           — CPU ONNX, no GPU needed
  ollama       bge-m3, nomic-embed-text        — local server, privacy-first

Public API
----------
    from embedding import get_embedder, get_embedder_from_config, EmbeddingPipeline

    # Dense only
    embedder = get_embedder("huggingface", "BAAI/bge-m3", device="cuda")
    vectors  = embedder.embed_documents(["Hello", "Xin chào"])

    # Dense + sparse (hybrid retrieval)
    pipeline = EmbeddingPipeline(
        dense_provider="cohere",
        dense_model="embed-multilingual-v3.0",
        enable_sparse=True,
        sparse_method="bm25",
    )
    pipeline.fit_sparse(corpus_texts)
    result = pipeline.embed_documents(texts)
    # result["dense"]  -> list[list[float]]
    # result["sparse"] -> list[dict[str, float]]

    # Config-driven
    pipeline = get_embedder_from_config(cfg)
"""

from embedding.factory  import get_embedder, get_embedder_from_config
from embedding.pipeline import EmbeddingPipeline

__all__ = ["get_embedder", "get_embedder_from_config", "EmbeddingPipeline"]
