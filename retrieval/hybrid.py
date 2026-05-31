"""
retrieval/hybrid.py
===================
Hybrid retrieval — dense ANN + sparse BM25 fused into one ranking.

Hybrid consistently outperforms either approach alone because dense
vectors capture semantic meaning while sparse handles exact keywords.
This is the recommended default for production RAG systems.

Fusion methods
--------------
rrf      : Reciprocal Rank Fusion (default). Position-based; robust to
           score scale differences between dense and sparse.
weighted : Linear combination of normalised scores. Control via ``alpha``.
dbsf     : Distribution-Based Score Fusion. Z-score normalised; robust
           to outlier scores in either list.

Use when: production systems where recall and precision both matter.
"""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore

from retrieval.base import BaseRetriever
from retrieval.dense import DenseRetriever
from retrieval.sparse import SparseRetriever
from retrieval.utils import deduplicate, reciprocal_rank_fusion, weighted_fusion, distribution_based_fusion


class HybridRetriever(BaseRetriever):
    """
    Fuse dense and BM25 sparse results into a single ranked list.

    Parameters
    ----------
    vector_store   : Populated LangChain VectorStore.
    documents      : Full corpus for the BM25 index.
    top_k          : Final number of results after fusion.
    fusion_method  : "rrf" | "weighted" | "dbsf"
    alpha          : Dense weight in weighted fusion (0–1). 0.5 = equal.
    rrf_k          : RRF constant k (default 60 from the original paper).
    candidate_k    : Candidates fetched from each sub-retriever before fusion.
                     Typically 3–4× top_k.
    score_threshold: Min score filter applied to dense results.
    """

    def __init__(
        self,
        vector_store:    VectorStore,
        documents:       list[Document],
        top_k:           int   = 5,
        fusion_method:   str   = "rrf",
        alpha:           float = 0.5,
        rrf_k:           int   = 60,
        candidate_k:     int   = 20,
        score_threshold: float = 0.0,
    ):
        super().__init__(vector_store, top_k)
        self.fusion_method = fusion_method
        self.rrf_k         = rrf_k
        self.alpha         = alpha

        self._dense  = DenseRetriever(vector_store, top_k=candidate_k, score_threshold=score_threshold)
        self._sparse = SparseRetriever(vector_store, documents, top_k=candidate_k)

    def retrieve(self, result) -> list[Document]:
        dense_docs  = self._dense.retrieve(result)
        sparse_docs = self._sparse.retrieve(result)

        if self.fusion_method == "weighted":
            fused = weighted_fusion(dense_docs, sparse_docs, alpha=self.alpha)
        elif self.fusion_method == "dbsf":
            fused = distribution_based_fusion([dense_docs, sparse_docs])
        else:  # rrf (default)
            fused = reciprocal_rank_fusion([dense_docs, sparse_docs], k=self.rrf_k)

        return deduplicate(fused)[:self.top_k]
