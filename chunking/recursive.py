"""
chunking/recursive.py
=====================
Cắt đệ quy theo ký tự — chiến lược mặc định khuyến nghị.

Thử lần lượt các dấu tách theo thứ tự ưu tiên (đoạn → dòng → câu → từ → ký tự)
nên chunk luôn kết thúc ở ranh giới tự nhiên khi có thể.

Danh sách dấu tách có thêm cú pháp Markdown để dòng heading (# / ##) và khối
code (```) trở thành điểm ngắt được ưu tiên.

Dùng khi: mặc định cho văn bản thuần và tài liệu Markdown.
"""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from chunking.base import BaseChunker

# Dấu tách xếp từ thô đến mịn
_SEPARATORS = [
    "\n#{1,6} ",   # heading Markdown
    "```\n",       # khối code
    "\n\n",        # hết đoạn
    "\n",          # xuống dòng
    ". ",          # hết câu
    "! ", "? ",
    "。", "，",    # dấu câu / mệnh đề CJK
    " ",           # ranh giới từ
    "",            # từng ký tự — phương án cuối
]


class RecursiveChunker(BaseChunker):
    """
    Cắt đệ quy theo danh sách dấu tách ưu tiên.

    Tham số
    -------
    chunk_size    : Kích thước chunk mong muốn, tính bằng ký tự.
    chunk_overlap : Số ký tự chồng lấn giữa hai chunk liên tiếp.
    """

    def split(self, docs: list[Document]) -> list[Document]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=_SEPARATORS,
            add_start_index=True,
            strip_whitespace=True,
        )
        return self._enrich(splitter.split_documents(docs))
