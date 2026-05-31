"""
vector_db/
==========
Vector storage package for the RAG indexing pipeline.

Public API
----------
    from vector_db import get_vector_store, get_vector_store_from_config

    # Direct usage
    store = get_vector_store(
        provider="chroma",
        chunks=chunks,
        embedder=pipeline,
        persist_dir="./storage/chroma",
    )
    retriever = store.as_retriever(search_kwargs={"k": 10})

    # Config-driven (used by main.py)
    store = get_vector_store_from_config(chunks, pipeline, cfg)

Individual store classes can also be imported directly:
    from vector_db.qdrant_store  import QdrantVectorStore
    from vector_db.chroma_store  import ChromaVectorStore
    from vector_db.faiss_store   import FAISSVectorStore

Providers
---------
  faiss              Local file, no server, < 10 M vectors
  chroma             Local persistent, dev-friendly, < 10 M vectors
  pinecone           Managed cloud, auto-scale, ~ 1 B vectors
  qdrant             Self/cloud, best filtering (ACORN), ~ 1 B+ vectors
  weaviate           Self/cloud, native hybrid search, ~ 1 B vectors
  pgvector           PostgreSQL extension, ACID, < 50 M vectors
  lancedb            Serverless columnar, embedded mode, ~ 1 B vectors
"""

from vector_db.factory import get_vector_store, get_vector_store_from_config

__all__ = ["get_vector_store", "get_vector_store_from_config"]
