"""
retrieval/sparse.py
===================
Sparse BM25 keyword retrieval.

BM25 matches exact terms, making it complementary to dense retrieval.
It excels at:
  - Proper nouns (person names, product codes, acronyms)
  - Technical terms rare in the embedding model's training data
  - Out-of-domain queries where dense embeddings are unreliable

BM25 index is built in-memory from the ``documents`` corpus.

Use when: queries contain product codes, exact names, or technical jargon.
"""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore

from retrieval.base import BaseRetriever


class SparseRetriever(BaseRetriever):
    """
    BM25 keyword retrieval using rank-bm25.

    Parameters
    ----------
    vector_store : Required by the base class; not used for BM25 search.
    documents    : Full corpus to build the BM25 index from.
    top_k        : Number of results to return.
    """

    def __init__(
        self,
        vector_store: VectorStore,
        documents:    list[Document],
        top_k:        int = 5,
    ):
        super().__init__(vector_store, top_k)
        self.documents = documents
        self._bm25     = self._build_index(documents)

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return text.lower().split()

    def _build_index(self, docs: list[Document]):
        from rank_bm25 import BM25Okapi
        return BM25Okapi([self._tokenize(d.page_content) for d in docs])

    def retrieve(self, result) -> list[Document]:
        query  = result.queries[0] if result.queries else result.original_query
        tokens = self._tokenize(query)
        scores = self._bm25.get_scores(tokens)

        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:self.top_k]

        docs: list[Document] = []
        for idx in ranked:
            if scores[idx] > 0:
                doc = self.documents[idx]
                doc.metadata["bm25_score"] = float(scores[idx])
                docs.append(doc)
        return docs
