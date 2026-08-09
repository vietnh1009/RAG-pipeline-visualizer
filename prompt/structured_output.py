"""
prompt/structured_output.py
============================
Structured Output RAG prompt — yêu cầu JSON với claims + sources.

Dùng khi: downstream code cần parse câu trả lời; fact-checking pipeline;
audit trail; cần phân biệt rõ từng claim và nguồn tương ứng.
"""

from __future__ import annotations

import json

from langchain_core.documents import Document

from prompt.base import BasePromptBuilder, PromptResult


class StructuredOutputPromptBuilder(BasePromptBuilder):
    """
    Prompt yêu cầu JSON có cấu trúc: answer + claims + sources + confidence.

    Schema output:
    {
      "answer":     "Câu trả lời tổng hợp",
      "claims":     ["Sự kiện 1", "Sự kiện 2"],
      "sources":    [1, 2],
      "confidence": "high | medium | low",
      "unanswered": "Phần câu hỏi không được nguồn đề cập, hoặc null"
    }

    Tham số
    -------
    include_confidence : Yêu cầu LLM tự đánh giá độ tin cậy.
    language           : \"vi\" | \"en\" | \"both\"
    max_context_chars  : Giới hạn ký tự context (0 = không giới hạn).
    """

    _SYSTEM_VI = (
        "Bạn là trợ lý nghiên cứu chính xác. "
        "Chỉ trả lời JSON hợp lệ, không có text nào khác, không có markdown fence."
    )

    _SYSTEM_EN = (
        "You are a precise research assistant. "
        "Respond with ONLY valid JSON, no other text, no markdown fences."
    )

    _SCHEMA_VI = '''{
  "answer":     "<câu trả lời ngắn gọn chỉ dùng các nguồn>",
  "claims":     ["<sự kiện nguyên tử 1>", "<sự kiện nguyên tử 2>"],
  "sources":    [1, 2],
  "confidence": "high | medium | low",
  "unanswered": "<phần câu hỏi không được nguồn đề cập, hoặc null>"
}'''

    _SCHEMA_EN = '''{
  "answer":     "<concise answer using only the sources>",
  "claims":     ["<atomic fact 1>", "<atomic fact 2>"],
  "sources":    [1, 2],
  "confidence": "high | medium | low",
  "unanswered": "<what the sources do not cover, or null>"
}'''

    _USER_VI = (
        "NGỮ CẢNH (nguồn được đánh số):\n{context}\n\n"
        "CÂU HỎI: {query}\n\n"
        "Trả về JSON theo đúng schema sau, không thêm gì khác:\n{schema}\n\n"
        "JSON:"
    )

    _USER_EN = (
        "CONTEXT (numbered sources):\n{context}\n\n"
        "QUESTION: {query}\n\n"
        "Respond with ONLY a valid JSON object matching this schema:\n{schema}\n\n"
        "JSON:"
    )

    def __init__(self, include_confidence: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.include_confidence = include_confidence

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
            context=context,
            query=query,
            schema=self._SCHEMA_VI if vi else self._SCHEMA_EN,
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
            template_name = "structured",
        )

    @staticmethod
    def parse_response(response_text: str) -> dict:
        """
        Parse JSON từ LLM response.
        Tự động strip markdown fence nếu model quên không tuân thủ.
        Trả về dict rỗng nếu parse thất bại.
        """
        import re
        text = response_text.strip()
        # Strip ```json ... ``` fence nếu có
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        text = text.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {"answer": text, "claims": [], "sources": [], "confidence": "low", "unanswered": None}
