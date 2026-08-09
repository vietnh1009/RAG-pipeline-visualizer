"""
vector_db/lancedb_store.py
===========================
LanceDB — columnar format, embedded mode, no server needed.

Uses the Lance columnar format (similar to Parquet), giving excellent
read performance for ANN search.  Integrates natively with pandas,
Arrow, and DuckDB.

Modes
-----
Local embedded : set persist_dir to a local path (default).
Managed cloud  : set persist_dir to a LanceDB Cloud URI ("db://your-slug").

Re-index prevention
-------------------
Checks the row count in the existing table.

Scale     : ~ 1 B vectors
Use when  : Python workflows, embedded use-cases, serverless / cost-sensitive.
"""

from __future__ import annotations

import logging

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore

from vector_db.base import BaseVectorStore
from vector_db.utils import corpus_changed, corpus_fingerprint, save_fingerprint

logger = logging.getLogger(__name__)


class LanceDBVectorStore(BaseVectorStore):
    """
    Parameters
    ----------
    collection_name : LanceDB table name.
    persist_dir     : Local directory path or LanceDB Cloud URI.
    distance        : "cosine" | "l2" | "dot"
    force_reindex   : Drop the table and rebuild from scratch.
    """

    def __init__(
        self,
        collection_name: str  = "rag",
        persist_dir:     str  = "./storage/lancedb_rag",
        distance:        str  = "cosine",
        force_reindex:   bool = False,
    ):
        super().__init__(collection_name, force_reindex)
        self.persist_dir = persist_dir
        self.distance    = distance

    def get_or_create(self, chunks: list[Document], embedder) -> VectorStore:
        import lancedb
        from langchain_community.vectorstores import LanceDB

        lc_embedder = self._langchain_embedder(embedder)
        db          = lancedb.connect(self.persist_dir)
        fp_path     = f"{self.persist_dir}/{self.collection_name}_fp.json"

        # Return existing table if corpus is unchanged
        if self.collection_name in db.table_names() and not self.force_reindex:
            if not corpus_changed(chunks, fp_path):
                table = db.open_table(self.collection_name)
                if table.count_rows() > 0:
                    logger.info("LanceDB: table '%s' loaded (%d rows).", self.collection_name, table.count_rows())
                    return LanceDB(connection=table, embedding=lc_embedder)

        # Drop and recreate
        if self.collection_name in db.table_names():
            db.drop_table(self.collection_name)

        logger.info("LanceDB: indexing %d chunks into '%s'.", len(chunks), self.collection_name)
        store = LanceDB.from_documents(
            documents=self.sanitize_metadata(chunks),
            embedding=lc_embedder,
            connection=db,
            table_name=self.collection_name,
        )
        save_fingerprint(corpus_fingerprint(chunks), fp_path)
        return store
