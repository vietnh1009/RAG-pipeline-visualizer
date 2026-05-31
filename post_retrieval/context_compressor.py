"""
post_retrieval/context_compressor.py
======================================
LLM-based Contextual Compression (Gao et al., 2023).

For each retrieved document, asks an LLM to extract ONLY the sentences
relevant to the query — discarding surrounding irrelevant content.

Modes
-----
extract   : LLM copies the relevant sentences verbatim.
            Preserves exact wording; good for citation-heavy use cases.
summarise : LLM writes a concise paraphrase of the relevant parts.
            Shorter output; less token usage downstream.

Use when: chunks are large (500–1000 tokens) and only a fraction answers
          the query; context window is the bottleneck.
"""

from __future__ import annotations

from langchain_core.documents import Document

from post_retrieval.base import BasePostProcessor
from post_retrieval.utils import call_llm


_NOT_RELEVANT = {"NOT_RELEVANT", "KHÔNG_LIÊN_QUAN", ""}


class ContextCompressor(BasePostProcessor):
    """
    Extract or summarise relevant content from each document.

    Parameters
    ----------
    llm_model         : LLM for compression.
    llm_provider      : "openai" | "anthropic" | "google"
    mode              : "extract" | "summarise"
    max_output_tokens : Max tokens in the compressed output per document.
    min_chars         : Drop document if compressed result is shorter than this.
    language          : "vi" | "en" | "both"
    """

    _EXTRACT_EN = (
        "Extract ONLY the sentences from the document below that directly "
        "answer the question. Copy them verbatim.\n"
        "If nothing is relevant, respond with: NOT_RELEVANT\n\n"
        "Question: {query}\n\nDocument:\n{doc}\n\nRelevant sentences:"
    )
    _EXTRACT_VI = (
        "Trích xuất CHỈ những câu trong tài liệu sau trả lời trực tiếp câu hỏi. "
        "Sao chép nguyên văn.\n"
        "Nếu không có gì liên quan, trả lời: KHÔNG_LIÊN_QUAN\n\n"
        "Câu hỏi: {query}\n\nTài liệu:\n{doc}\n\nCác câu liên quan:"
    )
    _SUMM_EN = (
        "Summarise only the parts of the document relevant to the question. "
        "Be concise. If nothing is relevant: NOT_RELEVANT\n\n"
        "Question: {query}\n\nDocument:\n{doc}\n\nRelevant summary:"
    )
    _SUMM_VI = (
        "Tóm tắt chỉ những phần của tài liệu liên quan đến câu hỏi. "
        "Ngắn gọn. Nếu không có phần nào: KHÔNG_LIÊN_QUAN\n\n"
        "Câu hỏi: {query}\n\nTài liệu:\n{doc}\n\nTóm tắt liên quan:"
    )

    def __init__(
        self,
        llm_model:         str = "gpt-4.1-mini",
        llm_provider:      str = "openai",
        mode:              str = "extract",
        max_output_tokens: int = 300,
        min_chars:         int = 30,
        language:          str = "both",
    ):
        self.llm_model         = llm_model
        self.llm_provider      = llm_provider
        self.mode              = mode
        self.max_output_tokens = max_output_tokens
        self.min_chars         = min_chars
        self.language          = language

    def process(self, query: str, docs: list[Document]) -> list[Document]:
        result: list[Document] = []
        for doc in docs:
            compressed = self._compress(query, doc.page_content)
            if compressed and len(compressed) >= self.min_chars:
                result.append(Document(
                    page_content=compressed,
                    metadata={
                        **doc.metadata,
                        "compressed":          True,
                        "compression_mode":    self.mode,
                        "original_char_count": len(doc.page_content),
                    },
                ))
        return result

    def _compress(self, query: str, document: str) -> str:
        if self.mode == "summarise":
            tmpl = self._SUMM_VI if self.language == "vi" else self._SUMM_EN
        else:
            tmpl = self._EXTRACT_VI if self.language == "vi" else self._EXTRACT_EN

        raw = call_llm(
            tmpl.format(query=query, doc=document[:3000]),
            self.llm_provider, self.llm_model,
            max_tokens=self.max_output_tokens,
        )
        return "" if raw.strip().upper() in _NOT_RELEVANT else raw.strip()
