"""
chunking/hierarchical.py
========================
Chunking phân cấp cha–con.

Tạo hai mức chunk từ cùng một document:
  - Chunk con (nhỏ) → đem embed và index để truy hồi chính xác.
  - Chunk cha (lớn) → lấy ra lúc truy vấn để có đủ ngữ cảnh.

Liên kết qua metadata:
  cha : chunk_level="parent", parent_id=<pid>, children_ids=[...]
  con : chunk_level="child",  parent_id=<pid>, child_id=<cid>

Bước retrieval tìm trên chunk con nhưng trả về chunk cha, nên LLM có ngữ cảnh
dày mà embedding vẫn chính xác.

Dùng khi: tài liệu dài, có cấu trúc, cần cả độ chính xác tìm kiếm lẫn ngữ cảnh
đầy đủ khi sinh câu trả lời.
"""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from chunking.base import BaseChunker
from chunking.recursive import _SEPARATORS   # dùng chung danh sách dấu tách


class HierarchicalChunker(BaseChunker):
    """
    Dựng cây chunk hai mức cha–con.

    Trả về một list phẳng chứa CẢ chunk cha lẫn chunk con; phân biệt bằng
    ``metadata["chunk_level"]``.

    Cả hai mức dùng chung danh sách dấu tách của RecursiveChunker nên chunk luôn
    kết thúc ở ranh giới tự nhiên (heading → đoạn → câu → từ → ký tự).

    Tham số
    -------
    parent_chunk_size : Số ký tự mỗi chunk cha.
    child_chunk_size  : Số ký tự mỗi chunk con, phải nhỏ hơn chunk cha.
    parent_overlap    : Chồng lấn giữa hai chunk cha liên tiếp.
    child_overlap     : Chồng lấn giữa các chunk con trong cùng một cha. Thường
                        để 0 vì các con đã liên hệ với nhau qua chunk cha.
    """

    def __init__(
        self,
        parent_chunk_size: int = 1500,
        child_chunk_size:  int = 300,
        parent_overlap:    int = 100,
        child_overlap:     int = 0,
        # Tương thích BaseChunker — chunk_size ở đây ứng với mức con
        chunk_size:        int = 300,
        chunk_overlap:     int = 0,
    ):
        super().__init__(child_chunk_size, child_overlap)
        self.parent_chunk_size = parent_chunk_size
        self.parent_overlap    = parent_overlap

    def split(self, docs: list[Document]) -> list[Document]:
        parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.parent_chunk_size,
            chunk_overlap=self.parent_overlap,
            separators=_SEPARATORS,
            add_start_index=True,   # offset trong document gốc, tiện khi debug
            strip_whitespace=True,
        )
        child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=_SEPARATORS,
            # Không bật add_start_index ở mức con: offset sẽ tính từ text của
            # chunk cha chứ không phải document gốc, dễ gây hiểu nhầm. Quan hệ
            # con → cha đã ghi sẵn qua parent_id.
            strip_whitespace=True,
        )

        all_docs: list[Document] = []
        pid_counter = 0

        for doc in docs:
            parents = parent_splitter.split_documents([doc])
            for parent in parents:
                pid = f"parent_{pid_counter}"
                pid_counter += 1

                children      = child_splitter.split_documents([parent])
                children_ids  = [f"{pid}_child_{j}" for j in range(len(children))]

                parent.metadata.update({
                    "chunk_level":   "parent",
                    "parent_id":     pid,
                    "children_ids":  children_ids,
                })
                all_docs.append(parent)

                for j, child in enumerate(children):
                    child.metadata.update({
                        "chunk_level": "child",
                        "parent_id":   pid,
                        "child_id":    children_ids[j],
                    })
                    all_docs.append(child)

        return self._enrich(all_docs)
