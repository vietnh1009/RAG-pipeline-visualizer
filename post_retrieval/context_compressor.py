"""
post_retrieval/context_compressor.py
======================================
Nén ngữ cảnh bằng LLM (Gao et al., 2023).

Với mỗi document, nhờ LLM giữ lại CHỈ những câu liên quan tới query và bỏ phần
còn lại.

Hai chế độ
----------
extract   : Chép nguyên văn các câu liên quan. Giữ đúng chữ, hợp khi cần trích dẫn.
summarise : Diễn giải ngắn gọn phần liên quan. Kết quả ngắn hơn, đỡ tốn token.

Dùng khi: chunk lớn (500–1000 token) mà chỉ một phần nhỏ trả lời được câu hỏi,
và context window là nút thắt.
"""

from __future__ import annotations

from langchain_core.documents import Document

from post_retrieval.base import BasePostProcessor
from utils.llm import call_llm


_NOT_RELEVANT = {"NOT_RELEVANT", "KHÔNG_LIÊN_QUAN", ""}


class ContextCompressor(BasePostProcessor):
    """
    Trích hoặc tóm tắt phần nội dung liên quan trong từng document.

    Tham số
    -------
    llm_model         : LLM dùng để nén.
    llm_provider      : "openai" | "anthropic" | "google"
    mode              : "extract" | "summarise"
    max_output_tokens : Số token tối đa cho kết quả nén mỗi document.
    min_chars         : Kết quả ngắn hơn ngưỡng này thì bỏ luôn document.
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
