"""
vector_db/faiss_store.py
=========================
FAISS local vector store (Meta AI).

Runs entirely in-process — no server, no API key, no network.
The index is persisted to two files on disk:
    <persist_dir>/index.faiss
    <persist_dir>/index.pkl
(persist_dir already includes collection name: ./storage/faiss_{collection_name})

Re-index prevention
-------------------
A ``fingerprint.json`` file tracks the MD5 of the corpus.
If the corpus has not changed since the last run, the existing index
is loaded without re-embedding.

Limitations
-----------
- No native metadata filtering (filter after retrieval).
- No hybrid search (pair with a BM25 SparseEmbedder separately).
- Not suitable for > ~10 M vectors on a single machine.

Scale   : < 10 M vectors
Use when: prototyping, offline pipelines, no server available.
"""

from __future__ import annotations

import logging
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore

from vector_db.base import BaseVectorStore
from vector_db.utils import corpus_changed, corpus_fingerprint, save_fingerprint

logger = logging.getLogger(__name__)


class FAISSVectorStore(BaseVectorStore):
    """
    Parameters
    ----------
    collection_name : Sub-directory name inside persist_dir.
    persist_dir     : Root directory for FAISS index files.
    force_reindex   : Wipe and rebuild even if corpus is unchanged.
    """

    def __init__(
        self,
        collection_name: str  = "rag",
        persist_dir:     str  = "./storage/faiss",
        force_reindex:   bool = False,
    ):
        super().__init__(collection_name, force_reindex)
        self.persist_dir = persist_dir

    def get_or_create(self, chunks: list[Document], embedder) -> VectorStore:
        from langchain_community.vectorstores import FAISS

        lc_embedder = self._langchain_embedder(embedder)
        idx_dir     = Path(self.persist_dir)   # persist_dir already is the index dir
        fp_path     = str(idx_dir / "fingerprint.json")

        # Load existing index if corpus is unchanged
        if idx_dir.exists() and not self.force_reindex and not corpus_changed(chunks, fp_path):
            logger.info("FAISS: loading existing index from '%s'.", idx_dir)
            return FAISS.load_local(
                folder_path=str(idx_dir),
                embeddings=lc_embedder,
                allow_dangerous_deserialization=True,
            )

        # Build new index
        logger.info("FAISS: building index for %d chunks.", len(chunks))
        store = FAISS.from_documents(documents=self.sanitize_metadata(chunks), embedding=lc_embedder)

        idx_dir.mkdir(parents=True, exist_ok=True)
        store.save_local(str(idx_dir))
        save_fingerprint(corpus_fingerprint(chunks), fp_path)
        logger.info("FAISS: index saved to '%s'.", idx_dir)
        return store
