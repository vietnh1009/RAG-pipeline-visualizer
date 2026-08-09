"""
chunking/
=========
Cắt tài liệu thành chunk cho pipeline indexing.

API công khai
-------------
    from chunking import get_chunker, deduplicate_chunks

    chunker = get_chunker("recursive", chunk_size=500, chunk_overlap=100)
    chunks  = chunker.split(docs)
    chunks  = deduplicate_chunks(chunks, method="minhash")

Cũng có thể import thẳng từng chunker, ví dụ
``from chunking.hierarchical import HierarchicalChunker``.
"""

from chunking.factory       import get_chunker, chunk_documents_from_config
from chunking.deduplication import deduplicate_chunks

__all__ = ["get_chunker", "chunk_documents_from_config", "deduplicate_chunks"]
