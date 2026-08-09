"""
embedding/
==========
Chuyển text thành vector cho pipeline indexing.

Provider
--------
  openai       text-embedding-3-small / large — API, hỗ trợ MRL
  cohere       embed-multilingual-v3.0        — lựa chọn API tốt nhất cho tiếng Việt
  huggingface  BAAI/bge-m3, Qwen3-Embedding   — chạy cục bộ, không cần API key
  fastembed    multilingual-e5-small          — ONNX trên CPU, không cần GPU
  ollama       bge-m3, nomic-embed-text       — server cục bộ, ưu tiên riêng tư

API công khai
-------------
    from embedding import get_embedder, get_embedder_from_config, EmbeddingPipeline

    # chỉ dense
    embedder = get_embedder("huggingface", "BAAI/bge-m3", device="cuda")
    vectors  = embedder.embed_documents(["Hello", "Xin chào"])

    # dense + sparse cho hybrid retrieval
    pipeline = EmbeddingPipeline(
        dense_provider="cohere",
        dense_model="embed-multilingual-v3.0",
        enable_sparse=True,
        sparse_method="bm25",
    )
    pipeline.fit_sparse(corpus_texts)
    result = pipeline.embed_documents(texts)   # {"dense": ..., "sparse": ...}

    # hoặc lấy cấu hình từ config.yaml
    pipeline = get_embedder_from_config(cfg)
"""

from embedding.factory  import get_embedder, get_embedder_from_config
from embedding.pipeline import EmbeddingPipeline

__all__ = ["get_embedder", "get_embedder_from_config", "EmbeddingPipeline"]
