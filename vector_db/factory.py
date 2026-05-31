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

# Lazy registry: provider name -> (module_path, class_name)
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
    Build (or load) a vector store for the given provider.

    Parameters
    ----------
    provider      : One of the keys in _REGISTRY.
    chunks        : Chunk Documents from the chunking stage.
    embedder      : ``EmbeddingPipeline`` (or a plain LangChain Embeddings object).
    force_reindex : Wipe existing data and rebuild from scratch.
    **kwargs      : Provider-specific options forwarded to the store class
                    (e.g. persist_dir, url, dimension, distance).

    Returns
    -------
    A populated LangChain VectorStore ready for ``.as_retriever()``.

    Examples
    --------
    >>> store = get_vector_store("chroma", chunks, pipeline,
    ...                          persist_dir="./storage/chroma")
    >>> store = get_vector_store("qdrant", chunks, pipeline,
    ...                          url="http://localhost:6333", dimension=1024)
    >>> store = get_vector_store("faiss",  chunks, pipeline,
    ...                          force_reindex=True)
    """
    entry = _REGISTRY.get(provider)
    if entry is None:
        valid = ", ".join(sorted(_REGISTRY))
        raise ValueError(f"Unknown vector DB provider '{provider}'. Valid: {valid}")

    import importlib
    module_path, class_name = entry
    cls  = getattr(importlib.import_module(module_path), class_name)
    inst = cls(force_reindex=force_reindex, **kwargs)

    # Fit sparse embedder on corpus texts before insertion (required for BM25)
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
    Build a vector store from the ``indexing.vector_db`` section of config.yaml.

    Parameters
    ----------
    chunks        : Chunks from the chunking stage.
    embedder      : EmbeddingPipeline from the embedding stage.
    cfg           : Full config dict from ``utils.config.load_config()``.
    force_reindex : Override config to force a full re-index.

    Config keys used
    ----------------
    indexing.vector_db.provider        e.g. "chroma"
    indexing.vector_db.collection_name e.g. "rag"
    indexing.vector_db.persist_dir     local path (faiss, chroma, lancedb)
    """
    db_cfg   = cfg["indexing"]["vector_db"]
    provider = db_cfg.get("provider", "chroma")
    kwargs: dict[str, Any] = {
        "collection_name": db_cfg.get("collection_name", "rag"),
    }

    # Local-file providers need a persist_dir
    if provider in ("faiss", "chroma", "lancedb"):
        kwargs["persist_dir"] = db_cfg.get("persist_dir", "./storage")

    # Self-hosted / cloud providers read credentials from env vars
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
