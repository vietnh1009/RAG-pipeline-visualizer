"""
post_retrieval/cross_encoder_reranker.py
=========================================
Cross-Encoder Reranker — local transformer-based relevance scoring.

A cross-encoder scores each (query, document) pair jointly through a
transformer — unlike bi-encoders that encode query and doc separately.
This gives significantly higher relevance accuracy at the cost of O(N)
inference (one forward pass per document).

Recommended models for Vietnamese + English
--------------------------------------------
  BAAI/bge-reranker-v2-m3          Multilingual, best for VI+EN  ★★★★★
  BAAI/bge-reranker-large          Multilingual, higher quality  ★★★★
  cross-encoder/ms-marco-MiniLM-L-6-v2  Fast, English only      ★★★★★
  cross-encoder/ms-marco-MiniLM-L-12-v2 Better, English only    ★★★★
  mixedbread-ai/mxbai-rerank-large-v1   English, strong         ★★★★

Use when: retrieval recalls good candidates but ranking needs improvement;
          no GPU required for small batches (CPU is fine for top_n ≤ 20).
"""

from __future__ import annotations

from langchain_core.documents import Document

from post_retrieval.base import BasePostProcessor


class CrossEncoderReranker(BasePostProcessor):
    """
    Rerank documents using a local cross-encoder model.

    Parameters
    ----------
    model_name  : HuggingFace cross-encoder model identifier.
    top_n       : Documents to keep after reranking.
    batch_size  : Pairs per inference batch.
    device      : "cpu" | "cuda" | "mps"
    max_length  : Max token length per (query, doc) pair.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        top_n:      int = 5,
        batch_size: int = 32,
        device:     str = "cpu",
        max_length: int = 512,
    ):
        self.model_name = model_name
        self.top_n      = top_n
        self.batch_size = batch_size
        self.device     = device
        self.max_length = max_length
        self._model     = None   # lazy-loaded

    def _load(self):
        if self._model is not None:
            return
        from sentence_transformers import CrossEncoder
        self._model = CrossEncoder(self.model_name, device=self.device, max_length=self.max_length)

    def process(self, query: str, docs: list[Document]) -> list[Document]:
        if not docs:
            return docs
        self._load()

        pairs  = [(query, doc.page_content) for doc in docs]
        scores = self._model.predict(pairs, batch_size=self.batch_size)

        for doc, score in zip(docs, scores):
            doc.metadata["rerank_score"] = float(score)

        ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in ranked][:self.top_n]
