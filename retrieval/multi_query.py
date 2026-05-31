"""
retrieval/multi_query.py
=========================
Multi-Query retrieval — run N queries, merge results via RRF.

Designed to work with MultiQueryTransformer or QueryDecompositionTransformer
in the pre-retrieval stage, which produce multiple queries in
``TransformResult.queries``.

If only one query is present, this behaves identically to DenseRetriever.

Use when: pre-retrieval generates multiple query variants (multi_query,
          step_back, decompose, expand strategies).
"""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore

from retrieval.base import BaseRetriever
from retrieval.utils import deduplicate, reciprocal_rank_fusion


class MultiQueryRetriever(BaseRetriever):
    """
    Retrieve for each query in TransformResult.queries, fuse via RRF.

    Parameters
    ----------
    vector_store : Populated LangChain VectorStore.
    top_k        : Final results after fusion.
    candidate_k  : Results per query before fusion.
    rrf_k        : RRF constant (default 60).
    search_type  : Search type for each individual search.
    """

    def __init__(
        self,
        vector_store: VectorStore,
        top_k:        int = 5,
        candidate_k:  int = 10,
        rrf_k:        int = 60,
        search_type:  str = "similarity",
    ):
        super().__init__(vector_store, top_k)
        self.candidate_k = candidate_k
        self.rrf_k       = rrf_k
        self.search_type = search_type

    def retrieve(self, result) -> list[Document]:
        queries = result.all_queries()
        filter  = result.metadata_filter

        ranked_lists = []
        for q in queries:
            docs = self._search(query=q, k=self.candidate_k, filter=filter, search_type=self.search_type)
            if docs:
                ranked_lists.append(docs)

        if not ranked_lists:
            return []

        fused = reciprocal_rank_fusion(ranked_lists, k=self.rrf_k)
        return deduplicate(fused)[:self.top_k]
