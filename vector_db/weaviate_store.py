"""
vector_db/weaviate_store.py
============================
Weaviate vector database — native hybrid search, GraphQL API.

Weaviate stores both dense vectors and BM25 term frequencies natively,
making it one of the best choices for hybrid retrieval out of the box.

Note: Weaviate collection names MUST start with an uppercase letter.
      This wrapper auto-capitalises the collection_name.

Re-index prevention
-------------------
Queries the aggregate object count before inserting.

Scale     : ~ 1 B vectors
Use when  : hybrid search, rich schema, GraphQL ecosystem.

Env vars (optional): WEAVIATE_URL, WEAVIATE_API_KEY
"""

from __future__ import annotations

import logging
import os

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore

from vector_db.base import BaseVectorStore

logger = logging.getLogger(__name__)


class WeaviateVectorStore(BaseVectorStore):
    """
    Parameters
    ----------
    collection_name : Weaviate class name (auto-capitalised).
    url             : Weaviate server URL.
    api_key         : Weaviate Cloud Services API key (None for self-hosted).
    force_reindex   : Delete the class and rebuild from scratch.
    """

    def __init__(
        self,
        collection_name: str  = "Rag",
        url:             str  = "http://localhost:8080",
        api_key:         str | None = None,
        force_reindex:   bool = False,
    ):
        # Weaviate requires capitalised class names
        name = collection_name[0].upper() + collection_name[1:]
        super().__init__(name, force_reindex)
        self.url     = os.environ.get("WEAVIATE_URL",     url)
        self.api_key = os.environ.get("WEAVIATE_API_KEY", api_key)

    def get_or_create(self, chunks: list[Document], embedder) -> VectorStore:
        import weaviate
        from langchain_weaviate import WeaviateVectorStore as _WVS

        lc_embedder = self._langchain_embedder(embedder)
        auth        = weaviate.auth.AuthApiKey(self.api_key) if self.api_key else None
        host, port  = self._parse_url(self.url)

        client = weaviate.connect_to_custom(
            http_host=host,
            http_port=port,
            http_secure=self.url.startswith("https"),
            auth_credentials=auth,
        )

        # Check existing object count
        if not self.force_reindex:
            try:
                result = client.collections.get(self.collection_name).aggregate.over_all(total_count=True)
                if (result.total_count or 0) > 0:
                    logger.info("Weaviate: class '%s' has %d objects — skipping.", self.collection_name, result.total_count)
                    return _WVS(client=client, index_name=self.collection_name, text_key="text", embedding=lc_embedder)
            except Exception:
                pass  # class does not exist yet

        # Delete existing class if force_reindex
        try:
            client.collections.delete(self.collection_name)
        except Exception:
            pass

        logger.info("Weaviate: indexing %d chunks into '%s'.", len(chunks), self.collection_name)
        return _WVS.from_documents(
            documents=self.sanitize_metadata(chunks),
            embedding=lc_embedder,
            client=client,
            index_name=self.collection_name,
        )

    @staticmethod
    def _parse_url(url: str) -> tuple[str, int]:
        """Extract host and port from a URL string."""
        stripped = url.replace("https://", "").replace("http://", "")
        if ":" in stripped:
            host, port_str = stripped.rsplit(":", 1)
            return host, int(port_str)
        return stripped, 8080
