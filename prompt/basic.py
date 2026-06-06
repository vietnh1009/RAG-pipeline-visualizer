"""
prompt/basic.py
===============
Basic RAG prompt — tối giản, grounded, chặt chẽ.

Dùng khi: prototype nhanh, Q&A đơn giản, hoặc khi muốn grounding
tối đa với overhead prompt engineering thấp nhất.
"""

from __future__ import annotations

from langchain_core.documents import Document

from prompt.base import BasePromptBuilder, PromptResult


class BasicPromptBuilder(BasePromptBuilder):
    """
    Prompt RAG tối giản với ràng buộc chỉ dùng nguồn được cung cấp.

    Tham số
    -------
    system_instruction : Instruction bổ sung (tuỳ chọn).
    language           : \"vi\" | \"en\" | \"both\"
    max_context_chars  : Giới hạn ký tự context (0 = không giới hạn).
    """

    _SYSTEM_VI = (
        "Bạn là trợ lý thực tế và nghiêm ngặt cho cơ sở tri thức nội bộ. "
        "Chỉ trả lời dựa trên ngữ cảnh được cung cấp. "
        "Không bịa đặt, không dùng kiến thức bên ngoài."
    )

    _SYSTEM_EN = (
        "You are a strict, fact-grounded assistant for a private knowledge base. "
        "Answer ONLY from the provided context. "
        "Do not fabricate information or use outside knowledge."
    )

    _USER_VI = (
        "NGỮ CẢNH:\n{context}\n\n"
        "CÂU HỎI: {query}\n\n"
        "Trả lời dựa trên ngữ cảnh trên. "
        "Nếu câu trả lời không có trong ngữ cảnh, hãy nói: "
        "\"Tôi không tìm thấy thông tin này trong tài liệu được cung cấp.\"\n\n"
        "TRẢ LỜI:"
    )

    _USER_EN = (
        "CONTEXT:\n{context}\n\n"
        "QUESTION: {query}\n\n"
        "Answer based only on the context above. "
        "If the answer is not in the context, say: "
        "\"I cannot find this information in the provided documents.\"\n\n"
        "ANSWER:"
    )

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
            template_name = "basic",
        )
