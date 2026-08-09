"""
vector_db/
==========
Lưu trữ vector cho pipeline indexing.

API công khai
-------------
    from vector_db import get_vector_store, get_vector_store_from_config

    store = get_vector_store(
        provider="chroma",
        chunks=chunks,
        embedder=pipeline,
        persist_dir="./storage/chroma",
    )
    retriever = store.as_retriever(search_kwargs={"k": 10})

    # hoặc lấy cấu hình từ config.yaml
    store = get_vector_store_from_config(chunks, pipeline, cfg)

Cũng có thể import thẳng từng lớp store, ví dụ
``from vector_db.faiss_store import FAISSVectorStore``.

Provider
--------
  faiss     File cục bộ, không cần server, < 10 triệu vector
  chroma    Lưu cục bộ, tiện khi phát triển, < 10 triệu vector
  pinecone  Cloud có quản lý, tự co giãn, ~ 1 tỷ vector
  qdrant    Tự dựng hoặc cloud, lọc tốt nhất (ACORN), ~ 1 tỷ+ vector
  weaviate  Tự dựng hoặc cloud, hybrid search sẵn có, ~ 1 tỷ vector
  pgvector  Extension của PostgreSQL, có ACID, < 50 triệu vector
  lancedb   Cột hoá, chạy nhúng không cần server, ~ 1 tỷ vector
"""

from vector_db.factory import get_vector_store, get_vector_store_from_config

__all__ = ["get_vector_store", "get_vector_store_from_config"]
