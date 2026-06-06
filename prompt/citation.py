"""
prompt/citation.py
==================
Citation RAG prompt — yêu cầu LLM trích dẫn [NGUỒN N] inline.

Dùng khi: production cần truy xuất nguồn gốc; domain nhạy cảm;
người dùng cần verify từng claim.
"""

from __future__ import annotations

import re

from langchain_core.documents import Document

from prompt.base import BasePromptBuilder, PromptResult


class CitationPromptBuilder(BasePromptBuilder):
    """
    Prompt RAG yêu cầu trích dẫn [NGUỒN N] cho từng claim.

    Tham số
    -------
    validate_citations : Nếu True, PromptResult sẽ chứa thông tin để
                         generation stage có thể kiểm tra số nguồn hợp lệ.
    language           : \"vi\" | \"en\" | \"both\"
    max_context_chars  : Giới hạn ký tự context (0 = không giới hạn).
    """

    _SYSTEM_VI = (
        "Bạn là trợ lý nghiên cứu chính xác. "
        "Trả lời CHỈ dựa trên các nguồn được đánh số. "
        "Trích dẫn số nguồn ngay trong câu như: [NGUỒN 1] hoặc [NGUỒN 2, NGUỒN 3]. "
        "Không bịa thông tin, không dùng kiến thức bên ngoài."
    )

    _SYSTEM_EN = (
        "You are a precise, citation-focused research assistant. "
        "Answer using ONLY the numbered sources below. "
        "Cite sources inline as [SOURCE 1] or [SOURCE 2, SOURCE 3]. "
        "Never fabricate information or use outside knowledge."
    )

    _USER_VI = (
        "NGUỒN:\n{context}\n\n"
        "CÂU HỎI: {query}\n\n"
        "Trả lời có trích dẫn. Nếu câu trả lời không có trong nguồn, nói: "
        "\"Tôi không tìm thấy thông tin này trong các tài liệu được cung cấp.\"\n\n"
        "TRẢ LỜI (có trích dẫn):"
    )

    _USER_EN = (
        "SOURCES:\n{context}\n\n"
        "QUESTION: {query}\n\n"
        "Answer with inline citations. If the answer is not in the sources, say: "
        "\"I cannot find this in the provided documents.\"\n\n"
        "ANSWER (with citations):"
    )

    def __init__(self, validate_citations: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.validate_citations = validate_citations

    def build(
        self,
        query:   str,
        docs:    list[Document],
        history: list[dict] | None = None,
    ) -> PromptResult:
        context = self._format_context(docs)
        vi      = (self.language == "vi")

        # System instruction always in EN — LLM follows EN instructions more reliably
        # VI/EN setting only affects the user message template (question/context format)
        system_text = self._SYSTEM_EN
        user_text   = (self._USER_VI if vi else self._USER_EN).format(
            context=context, query=query
        )

        if self.system_instruction:
            # Nếu đã có system_instruction (domain role + rules), bỏ base text
            # để tránh "You are a precise research assistant" xuất hiện lại
            system_text = self.system_instruction.strip()

        messages = [
            {"role": "system", "content": system_text},
            {"role": "user",   "content": user_text},
        ]

        return PromptResult(
            messages      = messages,
            full_prompt   = self._messages_to_string(messages),
            context_docs  = docs,
            n_sources     = len(docs),
            template_name = "citation",
        )

    @staticmethod
    def extract_cited_indices(answer: str) -> list[int]:
        """
        Trích xuất số nguồn từ câu trả lời.
        Nhận dạng cả: [NGUỒN 1], [SOURCE 1], [1], [1,2], [1, 2, 3].

        Trả về list số nguyên (1-indexed) của các nguồn được trích dẫn.
        """
        pattern = re.compile(
            r'\[(?:NGUỒN|SOURCE)?\s*(\d+(?:\s*,\s*\d+)*)\s*\]',
            re.IGNORECASE
        )
        indices: list[int] = []
        for m in pattern.finditer(answer):
            for num in re.split(r'\s*,\s*', m.group(1)):
                if num.strip().isdigit():
                    indices.append(int(num.strip()))
        return sorted(set(indices))
