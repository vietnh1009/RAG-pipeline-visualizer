"""
chunking/sentence_aware.py
==========================
Cắt theo câu — ranh giới chunk luôn rơi vào chỗ hết câu.

Thứ tự ưu tiên khi tách câu:
    1. underthesea  — tốt nhất cho tiếng Việt
    2. NLTK punkt   — phương án đa ngôn ngữ
    3. Regex        — phương án cuối, không cần thư viện

Dùng khi: dữ liệu FAQ, cặp hỏi–đáp, văn bản tường thuật ngắn, hay bất kỳ nội
dung nào mà câu không được đứt quãng.
"""

from __future__ import annotations

import re

from langchain_core.documents import Document

from chunking.base import BaseChunker


def _split_sentences(text: str) -> list[str]:
    """Tách câu bằng thư viện tốt nhất đang có trên máy."""
    # 1. underthesea — tốt nhất cho tiếng Việt
    try:
        from underthesea import sent_tokenize
        return [s.strip() for s in sent_tokenize(text) if s.strip()]
    except ImportError:
        pass

    # 2. NLTK punkt
    try:
        import nltk
        try:
            return [s.strip() for s in nltk.sent_tokenize(text) if s.strip()]
        except LookupError:
            nltk.download("punkt_tab", quiet=True)
            return [s.strip() for s in nltk.sent_tokenize(text) if s.strip()]
    except ImportError:
        pass

    # 3. Phương án cuối: regex
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


class SentenceChunker(BaseChunker):
    """
    Gom các câu liên tiếp thành chunk không vượt ``chunk_size`` ký tự, có thể
    chồng lấn ở mức câu.

    Tham số
    -------
    chunk_size          : Kích thước mong muốn mỗi chunk, tính bằng ký tự.
    chunk_overlap       : Số ký tự chồng lấn, luôn cắt trọn câu.
    sentences_per_chunk : Đặt giá trị này để cố định số câu mỗi chunk thay vì
                          tính theo ngân sách ký tự.
    """

    def __init__(
        self,
        chunk_size:          int = 1000,
        chunk_overlap:       int = 0,
        sentences_per_chunk: int | None = None,
    ):
        super().__init__(chunk_size, chunk_overlap)
        self.sentences_per_chunk = sentences_per_chunk

    def split(self, docs: list[Document]) -> list[Document]:
        all_chunks: list[Document] = []
        for doc in docs:
            sentences = _split_sentences(doc.page_content)
            groups    = self._group(sentences)
            for group in groups:
                content = " ".join(group).strip()
                if content:
                    all_chunks.append(Document(
                        page_content=content,
                        metadata={**doc.metadata},
                    ))
        return self._enrich(all_chunks)

    def _group(self, sentences: list[str]) -> list[list[str]]:
        """Gom câu thành từng nhóm, tôn trọng giới hạn chunk_size."""
        if self.sentences_per_chunk:
            step = max(1, self.sentences_per_chunk - self.chunk_overlap)
            return [
                sentences[i: i + self.sentences_per_chunk]
                for i in range(0, len(sentences), step)
            ]

        groups:  list[list[str]] = []
        current: list[str]       = []
        current_len = 0

        for sentence in sentences:
            s_len = len(sentence)
            if current_len + s_len > self.chunk_size and current:
                groups.append(current)
                # Lấy phần chồng lấn từ cuối nhóm hiện tại
                overlap:     list[str] = []
                overlap_len: int       = 0
                for s in reversed(current):
                    if overlap_len + len(s) <= self.chunk_overlap:
                        overlap.insert(0, s)
                        overlap_len += len(s)
                    else:
                        break
                current     = overlap
                current_len = overlap_len
            current.append(sentence)
            current_len += s_len

        if current:
            groups.append(current)
        return groups
