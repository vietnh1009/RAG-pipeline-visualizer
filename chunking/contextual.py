"""
chunking/contextual.py
======================
Contextual Chunking (Anthropic, September 2024).

Problem: a chunk pulled out of a document loses surrounding context.
  Orphan chunk: "The prevalence is estimated at 25 %."
  Missing:       what "prevalence" refers to.

Solution: for each chunk produced by a base splitter, an LLM reads the
full document and writes 1–2 context sentences that are prepended to the
chunk before it is embedded.

Reported impact: + Contextual Retrieval + BM25 reduces retrieval failure
rate by up to 67 % compared to naive chunking (Anthropic, 2024).

Cost tip: use prompt caching on the document portion to cut costs ~80 %.

Use when : chunks are short, terminology is specialised, coreferences are
           common, and recall is more important than ingestion cost.
"""

from __future__ import annotations

from langchain_core.documents import Document

from chunking.base import BaseChunker
from chunking.utils import call_llm


class ContextualChunker(BaseChunker):
    """
    Prepend an LLM-generated context prefix to each chunk.

    Parameters
    ----------
    base_strategy : Inner chunking strategy applied first.
                    Accepts any key from the factory registry
                    (e.g. "recursive", "sentence_aware").
    base_kwargs   : Constructor kwargs forwarded to the base chunker.
    llm_model     : Fast, cheap model recommended (e.g. claude-haiku-4-5-20251001).
    llm_provider  : "anthropic" | "openai" | "google"
    n_sentences   : Number of context sentences to prepend.
    """

    _PROMPT = (
        "<document>\n{document}\n</document>\n\n"
        "Here is the chunk to situate within the document:\n"
        "<chunk>\n{chunk}\n</chunk>\n\n"
        "Write {n} concise sentence(s) placing this chunk in the context "
        "of the full document to improve retrieval. "
        "Output ONLY those sentences, nothing else."
    )

    def __init__(
        self,
        base_strategy: str  = "recursive",
        base_kwargs:   dict | None = None,
        llm_model:     str  = "claude-haiku-4-5-20251001",
        llm_provider:  str  = "anthropic",
        n_sentences:   int  = 2,
        chunk_size:    int  = 1000,
        chunk_overlap: int  = 150,
    ):
        super().__init__(chunk_size, chunk_overlap)
        self.base_strategy = base_strategy
        self.base_kwargs   = base_kwargs or {}
        self.llm_model     = llm_model
        self.llm_provider  = llm_provider
        self.n_sentences   = n_sentences

    def split(self, docs: list[Document]) -> list[Document]:
        from chunking.factory import get_chunker

        base_chunker = get_chunker(
            self.base_strategy,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            **self.base_kwargs,
        )
        base_chunks = base_chunker.split(docs)

        # Build source -> full text map for context generation
        source_text: dict[str, str] = {}
        for doc in docs:
            src = doc.metadata.get("source", "")
            source_text.setdefault(src, "")
            source_text[src] += doc.page_content + "\n\n"

        enriched: list[Document] = []
        for chunk in base_chunks:
            src      = chunk.metadata.get("source", "")
            full_doc = source_text.get(src, chunk.page_content)
            context  = self._generate_context(full_doc, chunk.page_content)
            content  = f"{context}\n\n{chunk.page_content}" if context else chunk.page_content

            enriched.append(Document(
                page_content=content,
                metadata={
                    **chunk.metadata,
                    "chunking_strategy": "contextual",
                    "context_prefix":    context,
                },
            ))
        return self._enrich(enriched)

    def _generate_context(self, document: str, chunk: str) -> str:
        prompt = self._PROMPT.format(
            document=document[:8000],
            chunk=chunk,
            n=self.n_sentences,
        )
        return call_llm(prompt, self.llm_provider, self.llm_model, max_tokens=200)
