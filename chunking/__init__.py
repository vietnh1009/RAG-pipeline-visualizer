"""
chunking/
=========
Document chunking package for the RAG indexing pipeline.

Public API
----------
    from chunking import get_chunker
    from chunking import deduplicate_chunks

    chunker = get_chunker("recursive", chunk_size=500, chunk_overlap=100)
    chunks  = chunker.split(docs)

    chunks = deduplicate_chunks(chunks, method="minhash")

Individual chunkers can also be imported directly:

    from chunking.recursive     import RecursiveChunker
    from chunking.format_aware  import FormatAwareChunker
    from chunking.hierarchical  import HierarchicalChunker
    from chunking.contextual    import ContextualChunker
"""

from chunking.factory       import get_chunker, chunk_documents_from_config
from chunking.deduplication import deduplicate_chunks

__all__ = ["get_chunker", "chunk_documents_from_config", "deduplicate_chunks"]
