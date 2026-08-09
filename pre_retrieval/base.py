"""
pre_retrieval/base.py
=====================
Dataclass dùng chung và lớp ABC cho mọi query transformer.

Hợp đồng chung:
    result = SomeTransformer(**options).transform(query) -> TransformResult

``TransformResult`` là vật thể duy nhất đi từ pre-retrieval sang retrieval.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class TransformResult:
    """
    Kết quả của một pre-retrieval transformer.

    Tham số
    -------
    original_query  : Query gốc của người dùng, giữ nguyên.
    queries         : Các query đã biến đổi; retrieval chạy hết rồi gộp kết quả.
    metadata_filter : Filter cho vector DB, vd {"source": "policy_2024.pdf"}.
    intent          : Nhãn ý định, do IntentClassifier gán.
    retrieval_path  : Đích định tuyến, do QueryRouter gán.
    extra           : Metadata riêng của từng transformer.
    """
    original_query:  str
    queries:         list[str]      = field(default_factory=list)
    metadata_filter: dict | None    = None
    intent:          str | None     = None
    retrieval_path:  str | None     = None
    extra:           dict           = field(default_factory=dict)

    def all_queries(self) -> list[str]:
        """Trả về query gốc + query đã biến đổi, bỏ trùng, giữ nguyên thứ tự."""
        seen:   set[str]  = set()
        result: list[str] = []
        for q in [self.original_query] + self.queries:
            key = q.strip().lower()
            if key and key not in seen:
                seen.add(key)
                result.append(q.strip())
        return result


class BaseTransformer(ABC):
    """
    Lớp cơ sở cho mọi pre-retrieval transformer.

    Tham số
    -------
    llm_model    : Tên model cho các transformer dùng LLM.
    llm_provider : "openai" | "anthropic" | "google"
    language     : "vi" | "en" | "both" — ngôn ngữ prompt.
    """

    def __init__(
        self,
        llm_model:    str = "gpt-4.1-mini",
        llm_provider: str = "openai",
        language:     str = "both",
    ):
        self.llm_model    = llm_model
        self.llm_provider = llm_provider
        self.language     = language

    @abstractmethod
    def transform(self, query: str) -> TransformResult:
        """Biến đổi query thô. Lớp con bắt buộc cài đặt."""