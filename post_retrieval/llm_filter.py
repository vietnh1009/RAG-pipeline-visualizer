"""
post_retrieval/llm_filter.py
==============================
LLM Binary Relevance Filter — keep or drop each document with YES/NO.

Faster than ContextCompressor (binary decision, not extraction) but less
nuanced — the entire document is kept or removed, not trimmed.

Fail-open: if the LLM call fails, the document is kept (not discarded).

Use when: retrieval occasionally returns clearly off-topic documents;
          a quick yes/no pass is more cost-effective than full compression.
"""

from __future__ import annotations

from langchain_core.documents import Document

from post_retrieval.base import BasePostProcessor
from post_retrieval.utils import call_llm


class LLMFilter(BasePostProcessor):
    """
    Drop documents judged irrelevant by the LLM.

    Parameters
    ----------
    llm_model    : LLM for filtering.
    llm_provider : "openai" | "anthropic" | "google"
    language     : "vi" | "en" | "both"
    """

    _PROMPT_EN = (
        "Is the following passage relevant to answering the question?\n"
        "Respond with YES or NO only.\n\n"
        "Question: {query}\n\nPassage:\n{passage}\n\nRelevant (YES/NO):"
    )
    _PROMPT_VI = (
        "Đoạn văn sau có liên quan đến việc trả lời câu hỏi không?\n"
        "Chỉ trả lời CÓ hoặc KHÔNG.\n\n"
        "Câu hỏi: {query}\n\nĐoạn văn:\n{passage}\n\nLiên quan (CÓ/KHÔNG):"
    )

    def __init__(
        self,
        llm_model:    str = "gpt-4.1-mini",
        llm_provider: str = "openai",
        language:     str = "both",
    ):
        self.llm_model    = llm_model
        self.llm_provider = llm_provider
        self.language     = language

    def process(self, query: str, docs: list[Document]) -> list[Document]:
        tmpl = self._PROMPT_VI if self.language == "vi" else self._PROMPT_EN
        kept: list[Document] = []

        for doc in docs:
            prompt = tmpl.format(query=query, passage=doc.page_content[:1500])
            try:
                answer = call_llm(prompt, self.llm_provider, self.llm_model, max_tokens=5)
                if any(w in answer.upper() for w in ("YES", "CÓ", "Y")):
                    kept.append(doc)
            except Exception:
                kept.append(doc)   # fail-open: keep on error

        return kept
