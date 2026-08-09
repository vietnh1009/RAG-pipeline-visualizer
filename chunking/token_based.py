"""
chunking/token_based.py
=======================
Cắt theo token thay vì theo ký tự.

Rất quan trọng với corpus đa ngôn ngữ, nhất là tiếng Việt, vì tỉ lệ ký tự trên
token khác nhau nhiều giữa các ngôn ngữ: một chunk 500 ký tự tiếng Việt có thể
thành 600–700 token với tokenizer BPE, âm thầm vượt giới hạn của model embedding.

Dùng khi: pipeline đa ngôn ngữ, cần tuân thủ chặt giới hạn token, hoặc hệ thống
production không chấp nhận việc bị cắt cụt trong im lặng.
"""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_text_splitters import TokenTextSplitter

from chunking.base import BaseChunker


class TokenChunker(BaseChunker):
    """
    Cắt document theo số token, dùng tiktoken.

    Tham số
    -------
    chunk_size    : Số token mỗi chunk.
    chunk_overlap : Số token chồng lấn giữa hai chunk liên tiếp.
    encoding_name : Bộ mã tiktoken — "cl100k_base" cho model OpenAI đời mới
                    (GPT-4, text-embedding-3-*), "p50k_base" cho GPT-3 cũ.
    """

    def __init__(
        self,
        chunk_size:    int = 512,
        chunk_overlap: int = 64,
        encoding_name: str = "cl100k_base",
    ):
        super().__init__(chunk_size, chunk_overlap)
        self.encoding_name = encoding_name

    def split(self, docs: list[Document]) -> list[Document]:
        splitter = TokenTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            encoding_name=self.encoding_name,
        )
        return self._enrich(splitter.split_documents(docs))
