"""
loader/base.py
==============
Abstract base class cho tất cả document loader.

Giao kèo (contract) của mọi loader:
    loader = SomeLoader(language="both", **options)
    docs   = loader.load(file_path)    -> list[Document]
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal

from langchain_core.documents import Document

Language = Literal["vi", "en", "both"]


class BaseLoader(ABC):
    """
    Base class cho mọi document loader.

    Tham số
    -------
    language : Ngôn ngữ tài liệu nguồn.
               \"vi\" = Tiếng Việt, \"en\" = Tiếng Anh, \"both\" = hỗn hợp.
               Được lưu vào metadata[\"language\"] và dùng bởi OCR engine.
    """

    def __init__(self, language: Language = "both"):
        self.language = language

    @abstractmethod
    def load(self, file_path: str) -> list[Document]:
        """
        Load một file và trả về danh sách Document.

        Tham số
        -------
        file_path : Đường dẫn tuyệt đối hoặc tương đối đến file nguồn.

        Trả về
        ------
        List[Document] — một file có thể cho ra nhiều document
        (ví dụ: 1 document/trang, 1 document/sheet bảng tính).
        """

    def _stamp(self, docs: list[Document]) -> list[Document]:
        """Gắn ngôn ngữ vào metadata của mỗi document (in-place)."""
        for doc in docs:
            doc.metadata.setdefault("language", self.language)
        return docs
