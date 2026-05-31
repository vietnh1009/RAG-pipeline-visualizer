"""
retrieval/sentence_window.py
=============================
Sentence Window Retrieval — search sentences, expand to surrounding context.

Works best when chunks are very small (single sentences). After finding
the most relevant sentence, the retriever expands to include ±window_size
neighbouring sentences so the LLM receives sufficient context.

Requirements
------------
Chunks must have ``chunk_index`` and ``source`` in their metadata
(added automatically by all BaseChunker subclasses).

Use when: corpus was chunked at sentence granularity; individual sentences
          lack context without their neighbours.
"""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore

from retrieval.base import BaseRetriever
from retrieval.utils import deduplicate


class SentenceWindowRetriever(BaseRetriever):
    """
    Search sentences, then expand to a ±window_size context window.

    Parameters
    ----------
    vector_store  : Populated VectorStore.
    all_documents : Full list of documents from the index (for window expansion).
    top_k         : Number of expanded windows to return.
    window_size   : Sentences to include on each side of the matched sentence.
    """

    def __init__(
        self,
        vector_store:  VectorStore,
        all_documents: list[Document],
        top_k:         int = 5,
        window_size:   int = 2,
    ):
        super().__init__(vector_store, top_k)
        self.window_size = window_size
        # Build lookup: (source, chunk_index) -> Document
        self._doc_map: dict[tuple[str, int], Document] = {
            (d.metadata.get("source", ""), d.metadata.get("chunk_index", -1)): d
            for d in all_documents
        }

    def retrieve(self, result) -> list[Document]:
        query  = result.queries[0] if result.queries else result.original_query
        filter = result.metadata_filter
        sentence_docs = self._search(query=query, k=self.top_k, filter=filter)

        expanded: list[Document] = []
        seen_keys: set[tuple[str, int]] = set()

        for doc in sentence_docs:
            source = doc.metadata.get("source", "")
            idx    = doc.metadata.get("chunk_index", -1)

            window: list[str] = []
            for offset in range(-self.window_size, self.window_size + 1):
                key = (source, idx + offset)
                if key in seen_keys:
                    continue
                neighbour = self._doc_map.get(key)
                if neighbour:
                    window.append(neighbour.page_content)
                    seen_keys.add(key)

            if window:
                expanded.append(Document(
                    page_content=" ".join(window),
                    metadata={
                        **doc.metadata,
                        "window_size":        self.window_size,
                        "retrieval_strategy": "sentence_window",
                    },
                ))
            else:
                expanded.append(doc)

        return deduplicate(expanded)[:self.top_k]
