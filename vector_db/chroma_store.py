"""
vector_db/chroma_store.py
==========================
ChromaDB local persistent vector store.

Data is stored on disk at ``persist_dir``.  The collection persists
between runs; on subsequent runs the existing collection is re-used.

Re-index prevention
-------------------
If the target collection already has documents AND the corpus fingerprint
matches, no embedding is done and the existing collection is returned.

Supports basic metadata filtering natively.

Scale   : < 10 M vectors
Use when: local development, zero-config, Python-native projects.
"""

from __future__ import annotations

import logging
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore

from vector_db.base import BaseVectorStore
from vector_db.utils import corpus_changed, corpus_fingerprint, save_fingerprint

logger = logging.getLogger(__name__)


class ChromaVectorStore(BaseVectorStore):
    """
    Parameters
    ----------
    collection_name : Chroma collection name.
    persist_dir     : Directory where Chroma stores its SQLite + embeddings.
    force_reindex   : Delete the collection and rebuild from scratch.
    """

    def __init__(
        self,
        collection_name: str  = "rag",
        persist_dir:     str  = "./storage/chroma",
        force_reindex:   bool = False,
    ):
        super().__init__(collection_name, force_reindex)
        self.persist_dir = persist_dir

    def get_or_create(self, chunks: list[Document], embedder) -> VectorStore:
        from langchain_chroma import Chroma

        lc_embedder = self._langchain_embedder(embedder)
        fp_path     = str(Path(self.persist_dir) / f"{self.collection_name}_fp.json")

        # Load existing collection if corpus is unchanged
        if Path(self.persist_dir).exists() and not self.force_reindex:
            if not corpus_changed(chunks, fp_path):
                store = Chroma(
                    collection_name=self.collection_name,
                    embedding_function=lc_embedder,
                    persist_directory=self.persist_dir,
                )
                if store._collection.count() > 0:
                    logger.info("Chroma: loaded existing collection '%s'.", self.collection_name)
                    return store

        # Wipe persist_dir on force_reindex to avoid duplicates
        if self.force_reindex and Path(self.persist_dir).exists():
            import shutil
            shutil.rmtree(self.persist_dir, ignore_errors=True)

        logger.info("Chroma: creating collection '%s' with %d chunks.", self.collection_name, len(chunks))
        store = Chroma.from_documents(
            documents=self.sanitize_metadata(chunks),
            embedding=lc_embedder,
            collection_name=self.collection_name,
            persist_directory=self.persist_dir,
            collection_metadata={"hnsw:space": "cosine"},
        )
        save_fingerprint(corpus_fingerprint(chunks), fp_path)
        return store
