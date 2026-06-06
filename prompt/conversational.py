"""
prompt/conversational.py
========================
Conversational RAG prompt — hội thoại nhiều lượt với lịch sử.

Dùng khi: chatbot, assistant tương tác; người dùng hỏi follow-up
tham chiếu đến câu trả lời trước ("điểm thứ hai bạn vừa đề cập...").
"""

from __future__ import annotations

from langchain_core.documents import Document

from prompt.base import BasePromptBuilder, PromptResult


class ConversationalPromptBuilder(BasePromptBuilder):
    """
    Prompt hội thoại nhiều lượt với injection lịch sử.

    Tham số
    -------
    max_history_turns  : Số lượt hội thoại tối đa đưa vào prompt.
    language           : \"vi\" | \"en\" | \"both\"
    max_context_chars  : Giới hạn ký tự context (0 = không giới hạn).
    """

    _SYSTEM_VI = (
        "Bạn là trợ lý hội thoại hữu ích và thực tế. "
        "Trả lời CHỈ dựa trên ngữ cảnh được cung cấp. "
        "Trích dẫn [NGUỒN N] khi phù hợp. "
        "Nếu không tìm thấy câu trả lời trong ngữ cảnh, hãy thành thật nói vậy. "
        "Giữ tính nhất quán với các câu trả lời trước trong lịch sử hội thoại."
    )

    _SYSTEM_EN = (
        "You are a helpful, grounded conversational assistant. "
        "Answer using ONLY the provided context. Cite [SOURCE N] when helpful. "
        "If you cannot find the answer in the context, say so honestly. "
        "Stay consistent with prior turns in the conversation history."
    )

    _USER_VI = (
        "NGỮ CẢNH:\n{context}\n\n"
        "{history_block}"
        "Người dùng: {query}\n\n"
        "Trợ lý:"
    )

    _USER_EN = (
        "CONTEXT:\n{context}\n\n"
        "{history_block}"
        "User: {query}\n\n"
        "Assistant:"
    )

    _HISTORY_HEADER_VI = "LỊCH SỬ HỘI THOẠI:\n{history}\n\n"
    _HISTORY_HEADER_EN = "CONVERSATION HISTORY:\n{history}\n\n"

    def __init__(self, max_history_turns: int = 5, **kwargs):
        super().__init__(**kwargs)
        self.max_history_turns = max_history_turns

    def build(
        self,
        query:   str,
        docs:    list[Document],
        history: list[dict] | None = None,
    ) -> PromptResult:
        context = self._format_context(docs)
        vi      = (self.language == "vi")

        # Xây history block
        if history:
            hist_str  = self._format_history(history, self.max_history_turns)
            hist_header = self._HISTORY_HEADER_VI if vi else self._HISTORY_HEADER_EN
            history_block = hist_header.format(history=hist_str)
        else:
            history_block = ""

        system_text = self._SYSTEM_EN  # always EN for reliability
        user_text   = (self._USER_VI if vi else self._USER_EN).format(
            context=context,
            history_block=history_block,
            query=query,
        )

        if self.system_instruction:
            # Nếu đã có system_instruction (domain role + rules), bỏ base text
            # để tránh "You are a precise research assistant" xuất hiện lại
            system_text = self.system_instruction.strip()

        # Xây messages: system + lịch sử dạng alternating turns + user hiện tại
        messages: list[dict] = [{"role": "system", "content": system_text}]

        # Thêm lịch sử dưới dạng alternating messages để tận dụng chat memory
        # của các API hỗ trợ (thay vì nhét vào user message)
        if history:
            recent = history[-(self.max_history_turns * 2):]
            # Chèn thêm context vào user message đầu tiên nếu lịch sử dài
            context_injected = False
            for turn in recent:
                role    = turn.get("role", "user")
                content = turn.get("content", "")
                if role in ("user", "assistant"):
                    if role == "user" and not context_injected:
                        # Chỉ inject context vào lượt user đầu tiên trong history
                        # để model biết ngữ cảnh nhưng không lặp lại nhiều lần
                        messages.append({"role": role, "content": content})
                        context_injected = True
                    else:
                        messages.append({"role": role, "content": content})
        else:
            # Không có lịch sử — đưa toàn bộ context vào user message
            messages.append({"role": "user", "content": user_text})
            return PromptResult(
                messages      = messages,
                full_prompt   = self._messages_to_string(messages),
                context_docs  = docs,
                n_sources     = len(docs),
                template_name = "conversational",
            )

        # User turn hiện tại: đính kèm context vào đây
        messages.append({"role": "user", "content": user_text})

        return PromptResult(
            messages      = messages,
            full_prompt   = self._messages_to_string(messages),
            context_docs  = docs,
            n_sources     = len(docs),
            template_name = "conversational",
        )
