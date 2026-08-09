"""
prompt/base.py
==============
Abstract base class cho tất cả prompt builder.

Giao kèo của mọi builder:
    builder = SomePromptBuilder(**options)
    result  = builder.build(query, docs, history) -> PromptResult
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

from langchain_core.documents import Document


@dataclass
class PromptResult:
    """
    Output của prompt builder — chứa cả prompt lẫn metadata cần thiết
    cho generation stage (system/user split, history, citation map...).

    Attributes
    ----------
    messages       : List[{"role": ..., "content": ...}] — định dạng chat API chuẩn.
                     Dùng trực tiếp với OpenAI / Anthropic / Ollama chat endpoints.
    full_prompt    : Prompt dạng chuỗi duy nhất — fallback cho completion API.
    context_docs   : Tài liệu đã dùng để xây context (dùng để validate citations).
    n_sources      : Số nguồn được đưa vào context.
    template_name  : Tên template đã dùng.
    """
    messages:      list[dict]      = field(default_factory=list)
    full_prompt:   str             = ""
    context_docs:  list[Document]  = field(default_factory=list)
    n_sources:     int             = 0
    template_name: str             = ""


class BasePromptBuilder(ABC):
    """
    Abstract base cho mọi prompt builder.

    Tham số
    -------
    system_instruction : Instruction bổ sung gắn vào đầu system message.
    language           : \"vi\" | \"en\" | \"both\" — chọn ngôn ngữ prompt.
    max_context_chars  : Giới hạn tổng ký tự context (0 = không giới hạn).
                         Ngăn prompt vượt context window của LLM.
    """

    def __init__(
        self,
        system_instruction: str = "",
        language:           str = "both",
        max_context_chars:  int = 0,
    ):
        self.system_instruction = system_instruction
        self.language           = language
        self.max_context_chars  = max_context_chars

    @abstractmethod
    def build(
        self,
        query:   str,
        docs:    list[Document],
        history: list[dict] | None = None,
    ) -> PromptResult:
        """
        Xây dựng prompt từ query, tài liệu retrieved, và lịch sử hội thoại.

        Tham số
        -------
        query   : Câu hỏi của người dùng.
        docs    : Tài liệu đã qua post-retrieval (đã rerank, filter, order).
        history : Lịch sử hội thoại — list[{"role": "user"|"assistant", "content": str}].

        Trả về
        ------
        PromptResult với messages (chat format) và full_prompt (string format).
        """

    # ── Helpers dùng chung ────────────────────────────────────────────────────

    def _format_context(self, docs: list[Document]) -> str:
        """
        Format tài liệu thành numbered context block.
        Tự động truncate nếu max_context_chars > 0.
        """
        parts: list[str] = []
        total_chars = 0

        for i, doc in enumerate(docs, start=1):
            source  = doc.metadata.get("source", "unknown")
            page    = doc.metadata.get("page", "")
            heading = doc.metadata.get("heading", "")

            # Xây label nguồn: file + trang + heading nếu có
            ref_parts = [Path(source).name if source else "unknown"]
            if page:
                ref_parts.append(f"p.{page}")
            if heading:
                ref_parts.append(f'"{heading}"')
            ref = " · ".join(ref_parts)

            text = doc.page_content.strip()

            # Truncate từng chunk nếu cần
            if self.max_context_chars > 0:
                remaining = self.max_context_chars - total_chars
                if remaining <= 0:
                    break
                if len(text) > remaining:
                    text = text[:remaining] + "…"

            block = f"[NGUỒN {i}] ({ref})\n{text}"
            parts.append(block)
            total_chars += len(block)

        return "\n\n".join(parts)

    @staticmethod
    def _format_history(history: list[dict], max_turns: int = 10) -> str:
        """Flatten lịch sử hội thoại thành chuỗi đọc được."""
        _ROLE_MAP = {"user": "Người dùng", "assistant": "Trợ lý", "system": "Hệ thống"}
        recent = history[-(max_turns * 2):]
        lines  = []
        for turn in recent:
            role    = _ROLE_MAP.get(turn.get("role", "user"), turn.get("role", "").capitalize())
            content = turn.get("content", "").strip()
            if content:
                lines.append(f"{role}: {content}")
        return "\n".join(lines)

    @staticmethod
    def _messages_to_string(messages: list[dict]) -> str:
        """Chuyển chat messages thành single string (fallback cho completion API)."""
        parts = []
        for m in messages:
            role    = m.get("role", "user")
            content = m.get("content", "")
            if role == "system":
                parts.append(content)
            elif role == "user":
                parts.append(f"Người dùng: {content}")
            elif role == "assistant":
                parts.append(f"Trợ lý: {content}")
        return "\n\n".join(parts)
