"""
post_retrieval/context_orderer.py
===================================
Context Orderer — reorder chunks to mitigate "lost-in-the-middle".

Liu et al. (2023) showed that LLMs attend most strongly to content at
the *beginning* and *end* of their context window. Information placed in
the middle is frequently ignored during generation.

Ordering strategies
-------------------
relevance : Descending relevance score (most relevant first).
            Simple; assumes LLM reads top-to-bottom.
reverse   : Ascending relevance (most relevant LAST).
            Exploits recency bias in some LLM architectures.
sandwich  : Most relevant at START and END; least relevant in the middle.
            Maximises the chance that at least one highly-relevant chunk
            falls in a well-attended position. ← Recommended default.
original  : Keep the order returned by the retriever unchanged.

Use when: context window contains many chunks and generation quality
          is sensitive to ordering (most production RAG systems).
"""

from __future__ import annotations

from langchain_core.documents import Document

from post_retrieval.base import BasePostProcessor


class ContextOrderer(BasePostProcessor):
    """
    Reorder retrieved chunks to reduce lost-in-the-middle degradation.

    Parameters
    ----------
    ordering  : "relevance" | "reverse" | "sandwich" | "original"
    score_key : Metadata key holding the relevance score.
                Falls back through rerank_score → relevance_score → rrf_score.
    """

    def __init__(
        self,
        ordering:  str = "sandwich",
        score_key: str = "rerank_score",
    ):
        self.ordering  = ordering
        self.score_key = score_key

    def process(self, query: str, docs: list[Document]) -> list[Document]:
        if not docs or self.ordering == "original":
            return docs

        def get_score(doc: Document) -> float:
            for key in (self.score_key, "relevance_score", "rrf_score", "hybrid_score"):
                v = doc.metadata.get(key)
                if v is not None:
                    return float(v)
            return 0.0

        if self.ordering == "relevance":
            return sorted(docs, key=get_score, reverse=True)

        if self.ordering == "reverse":
            return sorted(docs, key=get_score)

        if self.ordering == "sandwich":
            sorted_docs = sorted(docs, key=get_score, reverse=True)
            n = len(sorted_docs)
            if n <= 2:
                return sorted_docs
            # Best chunk first, worst chunk last, rest in middle
            best  = sorted_docs[0]
            worst = sorted_docs[-1]
            mid   = sorted_docs[1:-1]
            half  = len(mid) // 2
            return [best] + mid[:half] + list(reversed(mid[half:])) + [worst]

        return docs
