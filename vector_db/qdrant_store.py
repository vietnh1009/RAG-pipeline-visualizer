"""
vector_db/qdrant_store.py
==========================
Qdrant vector database — best filtering (ACORN) and hybrid search.

Can run:
- Locally in-memory (url=":memory:" — no persistence)
- Locally on disk   (url="http://localhost:6333", requires Docker)
- Qdrant Cloud      (provide url + api_key)

ACORN filtering
---------------
Qdrant's ACORN algorithm filters during HNSW graph traversal rather
than as a post-processing step, giving near-identical recall with
arbitrary metadata filters at any selectivity level.

Re-index prevention
-------------------
Checks ``collection_info.vectors_count`` before inserting.

Scale     : ~ 1 B+ vectors
Use when  : hybrid search, complex metadata filtering, self-host with cloud option.

Env vars (optional): QDRANT_URL, QDRANT_API_KEY
"""

from __future__ import annotations

import logging
import os

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore

from vector_db.base import BaseVectorStore

logger = logging.getLogger(__name__)


class QdrantVectorStore(BaseVectorStore):
    """
    Parameters
    ----------
    collection_name : Qdrant collection name.
    url             : Server URL or ":memory:" for in-process mode.
    api_key         : Qdrant Cloud API key (None for self-hosted).
    dimension       : Dense embedding dimension.
    distance        : "Cosine" | "Dot" | "Euclid"
    on_disk         : Store vectors on disk to reduce RAM usage.
    force_reindex   : Delete and recreate the collection.
    """

    def __init__(
        self,
        collection_name: str  = "rag",
        url:             str  = "http://localhost:6333",
        api_key:         str | None = None,
        dimension:       int  = 1536,
        distance:        str  = "Cosine",
        on_disk:         bool = False,
        force_reindex:   bool = False,
    ):
        super().__init__(collection_name, force_reindex)
        self.url      = os.environ.get("QDRANT_URL",     url)
        self.api_key  = os.environ.get("QDRANT_API_KEY", api_key)
        self.dimension = dimension
        self.distance  = distance
        self.on_disk   = on_disk

    def get_or_create(self, chunks: list[Document], embedder) -> VectorStore:
        from qdrant_client import QdrantClient
        from qdrant_client.http.models import Distance, VectorParams, OptimizersConfigDiff
        from langchain_qdrant import QdrantVectorStore as _QVS

        lc_embedder = self._langchain_embedder(embedder)
        client      = QdrantClient(url=self.url, api_key=self.api_key)
        dist_map    = {"Cosine": Distance.COSINE, "Dot": Distance.DOT, "Euclid": Distance.EUCLID}
        existing    = [c.name for c in client.get_collections().collections]

        # Return existing collection if populated
        if self.collection_name in existing and not self.force_reindex:
            info  = client.get_collection(self.collection_name)
            count = info.vectors_count or 0
            if count > 0:
                logger.info("Qdrant: collection '%s' has %d vectors — skipping.", self.collection_name, count)
                return _QVS(client=client, collection_name=self.collection_name, embedding=lc_embedder)

        # Delete on force_reindex
        if self.collection_name in existing:
            client.delete_collection(self.collection_name)

        client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=self.dimension,
                distance=dist_map.get(self.distance, Distance.COSINE),
                on_disk=self.on_disk,
            ),
            optimizers_config=OptimizersConfigDiff(indexing_threshold=20_000),
        )

        logger.info("Qdrant: indexing %d chunks into '%s'.", len(chunks), self.collection_name)
        return _QVS.from_documents(
            documents=self.sanitize_metadata(chunks),
            embedding=lc_embedder,
            collection_name=self.collection_name,
            url=self.url,
            api_key=self.api_key,
        )
