"""
vector_db/pinecone_store.py
============================
Pinecone managed serverless vector database.

Zero-ops, auto-scaling, stable API.
Best for startups that want speed-to-market without infra management.

Re-index prevention
-------------------
Checks ``index.describe_index_stats().total_vector_count``.
If > 0 and force_reindex=False, the existing index is returned as-is.

Scale     : ~ 1 B vectors (Serverless)
Use when  : managed cloud, no infra ops, auto-scale.
Limitation: vendor lock-in, pricing grows with scale.

Env var: PINECONE_API_KEY
"""

from __future__ import annotations

import logging
import os

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore

from vector_db.base import BaseVectorStore

logger = logging.getLogger(__name__)


class PineconeVectorStore(BaseVectorStore):
    """
    Parameters
    ----------
    collection_name : Pinecone index name.
    dimension       : Embedding vector dimension (must match the model).
    metric          : "cosine" | "euclidean" | "dotproduct"
    cloud           : Cloud provider for serverless ("aws" | "gcp" | "azure").
    region          : Cloud region (e.g. "us-east-1").
    force_reindex   : Delete and recreate the index.
    """

    def __init__(
        self,
        collection_name: str  = "rag",
        dimension:       int  = 1536,
        metric:          str  = "cosine",
        cloud:           str  = "aws",
        region:          str  = "us-east-1",
        force_reindex:   bool = False,
    ):
        super().__init__(collection_name, force_reindex)
        self.dimension = dimension
        self.metric    = metric
        self.cloud     = cloud
        self.region    = region

    def get_or_create(self, chunks: list[Document], embedder) -> VectorStore:
        from pinecone import Pinecone, ServerlessSpec
        from langchain_pinecone import PineconeVectorStore as _PVS

        api_key     = os.environ["PINECONE_API_KEY"]
        pc          = Pinecone(api_key=api_key)
        lc_embedder = self._langchain_embedder(embedder)
        existing    = [idx.name for idx in pc.list_indexes()]

        # Return existing index if populated
        if self.collection_name in existing and not self.force_reindex:
            index = pc.Index(self.collection_name)
            count = index.describe_index_stats().total_vector_count
            if count > 0:
                logger.info("Pinecone: index '%s' has %d vectors — skipping.", self.collection_name, count)
                return _PVS(index=index, embedding=lc_embedder)

        # Delete on force_reindex
        if self.collection_name in existing and self.force_reindex:
            pc.delete_index(self.collection_name)
            existing = []

        # Create serverless index
        if self.collection_name not in existing:
            logger.info("Pinecone: creating index '%s'.", self.collection_name)
            pc.create_index(
                name=self.collection_name,
                dimension=self.dimension,
                metric=self.metric,
                spec=ServerlessSpec(cloud=self.cloud, region=self.region),
            )

        logger.info("Pinecone: upserting %d chunks.", len(chunks))
        return _PVS.from_documents(
            documents=self.sanitize_metadata(chunks),
            embedding=lc_embedder,
            index_name=self.collection_name,
        )
