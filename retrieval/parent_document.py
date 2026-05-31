"""
retrieval/parent_document.py
=============================
Parent Document Retrieval — search small child chunks, return large parents.

Works with HierarchicalChunker which stores both levels with metadata:
  child : chunk_level="child",  parent_id=<pid>
  parent: chunk_level="parent", parent_id=<pid>

Why this improves generation quality
--------------------------------------
- Small child chunks → precise embeddings → high retrieval recall.
- Large parent chunks → rich context for the LLM to generate a full answer.
- Without this, the LLM sees fragments that are missing key surrounding context.

Use when: corpus was indexed with HierarchicalChunker.
"""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore

from retrieval.base import BaseRetriever
from retrieval.utils import deduplicate


class ParentDocumentRetriever(BaseRetriever):
    """
    Retrieve child chunks, then fetch and return their parent chunks.

    Parameters
    ----------
    vector_store : Populated VectorStore containing BOTH parent and child docs.
    top_k        : Number of parent chunks to return.
    candidate_k  : Child chunks to retrieve before parent lookup.
    """

    def __init__(
        self,
        vector_store: VectorStore,
        top_k:        int = 5,
        candidate_k:  int = 20,
    ):
        super().__init__(vector_store, top_k)
        self.candidate_k = candidate_k

    def retrieve(self, result) -> list[Document]:
        query  = result.queries[0] if result.queries else result.original_query
        filter = result.metadata_filter or {}

        # Step 1: retrieve child chunks only
        child_docs = self._search(
            query=query, k=self.candidate_k,
            filter={**filter, "chunk_level": "child"},
        )
        if not child_docs:
            child_docs = self._search(query=query, k=self.candidate_k, filter=filter)

        # Step 2: collect unique parent_ids in ranked order
        parent_ids: list[str] = []
        seen: set[str]        = set()
        for doc in child_docs:
            pid = doc.metadata.get("parent_id")
            if pid and pid not in seen:
                seen.add(pid)
                parent_ids.append(pid)
            if len(parent_ids) >= self.top_k:
                break

        # Step 3: fetch parent chunks by parent_id
        parent_docs: list[Document] = []
        for pid in parent_ids:
            parents = self._search(
                query=query, k=1,
                filter={"parent_id": pid, "chunk_level": "parent"},
            )
            if parents:
                parent_docs.append(parents[0])

        # Fallback to child chunks if parent lookup fails
        if not parent_docs:
            return deduplicate(child_docs)[:self.top_k]

        return parent_docs[:self.top_k]
