"""
vector_db/factory.py
====================
Factory function tạo vector store theo provider.

Providers: faiss, chroma, lancedb, qdrant, weaviate, pgvector, pinecone.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore

logger = logging.getLogger(__name__)

# Registry nạp trễ: tên provider -> (đường dẫn module, tên lớp)
_REGISTRY: dict[str, tuple[str, str]] = {
    "faiss":            ("vector_db.faiss_store",           "FAISSVectorStore"),
    "chroma":           ("vector_db.chroma_store",          "ChromaVectorStore"),
    "pinecone":         ("vector_db.pinecone_store",        "PineconeVectorStore"),
    "qdrant":           ("vector_db.qdrant_store",          "QdrantVectorStore"),
    "weaviate":         ("vector_db.weaviate_store",        "WeaviateVectorStore"),
    "pgvector":         ("vector_db.pgvector_store",        "PGVectorStore"),
    "lancedb":          ("vector_db.lancedb_store",         "LanceDBVectorStore"),
}


def get_vector_store(
    provider:    str,
    chunks:      list[Document],
    embedder,
    force_reindex: bool = False,
    **kwargs: Any,
) -> VectorStore:
    """
    Dựng (hoặc nạp lại) vector store cho provider đã chọn.

    Tham số
    -------
    provider      : Một trong các key của ``_REGISTRY``.
    chunks        : Document chunk từ bước chunking.
    embedder      : ``EmbeddingPipeline`` hoặc một Embeddings thuần của LangChain.
    force_reindex : Xoá dữ liệu cũ và dựng lại từ đầu.
    **kwargs      : Tuỳ chọn riêng từng provider, chuyển thẳng cho lớp store
                    (persist_dir, url, dimension, distance, …).

    Trả về
    ------
    VectorStore của LangChain đã nạp dữ liệu, sẵn sàng gọi ``.as_retriever()``.

    Ví dụ
    -----
    >>> get_vector_store("chroma", chunks, pipeline, persist_dir="./storage/chroma")
    >>> get_vector_store("qdrant", chunks, pipeline,
    ...                  url="http://localhost:6333", dimension=1024)
    >>> get_vector_store("faiss",  chunks, pipeline, force_reindex=True)
    """
    entry = _REGISTRY.get(provider)
    if entry is None:
        valid = ", ".join(sorted(_REGISTRY))
        raise ValueError(f"Unknown vector DB provider '{provider}'. Valid: {valid}")

    import importlib
    module_path, class_name = entry
    cls  = getattr(importlib.import_module(module_path), class_name)
    inst = cls(force_reindex=force_reindex, **kwargs)

    # BM25 bắt buộc fit trên corpus trước khi nạp dữ liệu
    if hasattr(embedder, "enable_sparse") and embedder.enable_sparse:
        logger.info("Fitting sparse embedder on %d documents …", len(chunks))
        embedder.fit_sparse([c.page_content for c in chunks])

    logger.info("VectorStore: provider=%s, collection=%s, chunks=%d",
                provider, kwargs.get("collection_name", "rag"), len(chunks))
    return inst.get_or_create(chunks, embedder)


def get_vector_store_from_config(
    chunks:       list[Document],
    embedder,
    cfg:          dict,
    force_reindex: bool = False,
) -> VectorStore:
    """
    Dựng vector store từ mục ``indexing.vector_db`` của config.yaml.

    Tham số
    -------
    chunks        : Chunk từ bước chunking.
    embedder      : EmbeddingPipeline từ bước embedding.
    cfg           : Dict config đầy đủ từ ``utils.config.load_config()``.
    force_reindex : Ép index lại toàn bộ, bất kể config.

    Provider cục bộ đọc ``persist_dir``; provider có server lấy URL và API key
    từ biến môi trường (QDRANT_URL, WEAVIATE_URL, DATABASE_URL, …).
    """
    db_cfg   = cfg["indexing"]["vector_db"]
    provider = db_cfg.get("provider", "chroma")
    kwargs: dict[str, Any] = {
        "collection_name": db_cfg.get("collection_name", "rag"),
    }

    # Provider lưu file cục bộ cần persist_dir
    if provider in ("faiss", "chroma", "lancedb"):
        kwargs["persist_dir"] = db_cfg.get("persist_dir", "./storage")

    # Provider tự dựng / cloud lấy thông tin kết nối từ biến môi trường
    if provider == "qdrant":
        kwargs["url"]      = os.environ.get("QDRANT_URL",     "http://localhost:6333")
        kwargs["api_key"]  = os.environ.get("QDRANT_API_KEY", None)


    elif provider == "weaviate":
        kwargs["url"]     = os.environ.get("WEAVIATE_URL",     "http://localhost:8080")
        kwargs["api_key"] = os.environ.get("WEAVIATE_API_KEY", None)


    elif provider == "pgvector":
        kwargs["connection_string"] = os.environ.get("DATABASE_URL", "")

    elif provider == "pinecone":
        kwargs["cloud"]  = db_cfg.get("pinecone_cloud",  "aws")
        kwargs["region"] = db_cfg.get("pinecone_region", "us-east-1")

    return get_vector_store(
        provider=provider,
        chunks=chunks,
        embedder=embedder,
        force_reindex=force_reindex,
        **kwargs,
    )
