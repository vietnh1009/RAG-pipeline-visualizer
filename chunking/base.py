"""
chunking/base.py
================
Abstract base class dùng chung cho tất cả chunking strategy.

Giao kèo của mọi chunker:
    chunker = SomeChunker(chunk_size=1000, chunk_overlap=150, **options)
    chunks  = chunker.split(docs)   -> list[Document]
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from langchain_core.documents import Document


class BaseChunker(ABC):
    """
    Lớp cơ sở cho mọi chiến lược chunking.

    Tham số
    -------
    chunk_size    : Kích thước chunk mong muốn — đơn vị là ký tự, token hay câu
                    tuỳ chiến lược.
    chunk_overlap : Phần chồng lấn giữa hai chunk liên tiếp, cùng đơn vị với
                    chunk_size.
    """

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 150):
        self.chunk_size    = chunk_size
        self.chunk_overlap = chunk_overlap

    @abstractmethod
    def split(self, docs: list[Document]) -> list[Document]:
        """
        Cắt danh sách Document thành các chunk nhỏ hơn.

        Tham số
        -------
        docs : Document nguồn, kết quả của bước loader.

        Trả về
        ------
        List Document chunk. Mỗi chunk kế thừa metadata của document gốc, thêm
        ``chunk_index`` và ``char_count``.
        """

    @staticmethod
    def _enrich(chunks: list[Document]) -> list[Document]:
        """Gắn số thứ tự và số ký tự vào metadata của từng chunk."""
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = i
            chunk.metadata["char_count"]  = len(chunk.page_content)
        return chunks
