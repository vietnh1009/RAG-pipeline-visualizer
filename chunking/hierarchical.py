"""
chunking/hierarchical.py
========================
Hierarchical (Parent-Child) Chunking.

Creates two levels of chunks from the same document:
  - Child chunks (small)  → embedded and indexed for precise retrieval.
  - Parent chunks (large) → fetched at query time for full context.

Metadata links:
  parent: chunk_level="parent", parent_id=<pid>, children_ids=[...]
  child:  chunk_level="child",  parent_id=<pid>, child_id=<cid>

The retrieval stage searches child chunks but returns parent chunks,
giving the LLM rich context while keeping embedding precision high.

Use when : long, structured documents where search precision and
           generation context both matter.
"""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from chunking.base import BaseChunker
from chunking.recursive import _SEPARATORS   # shared separator priority list


class HierarchicalChunker(BaseChunker):
    """
    Build a two-level parent-child chunk hierarchy.

    Returns a flat list containing BOTH parent and child Documents.
    Distinguish them via ``metadata["chunk_level"]``.

    Both levels use the same separator priority list as RecursiveChunker
    so chunks always end at a natural boundary (heading → paragraph →
    sentence → word → character).

    Parameters
    ----------
    parent_chunk_size  : Characters per parent chunk.
    child_chunk_size   : Characters per child chunk (must be < parent).
    parent_overlap     : Overlap between consecutive parent chunks.
    child_overlap      : Overlap between child chunks within one parent.
                         Usually kept at 0 — sibling children are already
                         linked via the shared parent context.
    """

    def __init__(
        self,
        parent_chunk_size: int = 1500,
        child_chunk_size:  int = 300,
        parent_overlap:    int = 100,
        child_overlap:     int = 0,
        # BaseChunker compat — map chunk_size to child level
        chunk_size:        int = 300,
        chunk_overlap:     int = 0,
    ):
        super().__init__(child_chunk_size, child_overlap)
        self.parent_chunk_size = parent_chunk_size
        self.parent_overlap    = parent_overlap

    def split(self, docs: list[Document]) -> list[Document]:
        parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.parent_chunk_size,
            chunk_overlap=self.parent_overlap,
            separators=_SEPARATORS,
            add_start_index=True,   # offset trong document gốc — hữu ích để debug/tracing
            strip_whitespace=True,
        )
        child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=_SEPARATORS,
            # add_start_index: bỏ qua ở child — offset sẽ tính từ parent text,
            #                  không phải document gốc, dễ gây nhầm lẫn.
            #                  Quan hệ child→parent đã được ghi qua parent_id.
            strip_whitespace=True,
        )

        all_docs: list[Document] = []
        pid_counter = 0

        for doc in docs:
            parents = parent_splitter.split_documents([doc])
            for parent in parents:
                pid = f"parent_{pid_counter}"
                pid_counter += 1

                children      = child_splitter.split_documents([parent])
                children_ids  = [f"{pid}_child_{j}" for j in range(len(children))]

                parent.metadata.update({
                    "chunk_level":   "parent",
                    "parent_id":     pid,
                    "children_ids":  children_ids,
                })
                all_docs.append(parent)

                for j, child in enumerate(children):
                    child.metadata.update({
                        "chunk_level": "child",
                        "parent_id":   pid,
                        "child_id":    children_ids[j],
                    })
                    all_docs.append(child)

        return self._enrich(all_docs)
