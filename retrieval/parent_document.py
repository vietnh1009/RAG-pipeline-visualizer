"""
retrieval/parent_document.py
=============================
Tìm ở chunk con nhỏ, trả về chunk cha lớn.

Đi cùng HierarchicalChunker — bộ chunker lưu cả hai mức kèm metadata:
  con : chunk_level="child",  parent_id=<pid>
  cha : chunk_level="parent", parent_id=<pid>

Vì sao chất lượng sinh tốt hơn: chunk con nhỏ cho embedding chính xác nên
recall cao, còn chunk cha lớn cung cấp đủ ngữ cảnh để LLM viết câu trả lời trọn
vẹn. Không có bước này, LLM chỉ thấy mảnh vụn thiếu ngữ cảnh xung quanh.

Dùng khi: corpus được index bằng HierarchicalChunker.
"""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore

from retrieval.base import BaseRetriever
from utils.documents import deduplicate


class ParentDocumentRetriever(BaseRetriever):
    """
    Truy hồi chunk con rồi lấy chunk cha tương ứng trả về.

    Tham số
    -------
    vector_store : VectorStore chứa CẢ chunk cha lẫn chunk con.
    top_k        : Số chunk cha trả về.
    candidate_k  : Số chunk con lấy trước khi tra ngược lên cha.
    """

    def __init__(
        self,
        vector_store: VectorStore,
        top_k:        int = 5,
        candidate_k:  int = 20,
    ):
        super().__init__(vector_store, top_k)
        self.candidate_k = candidate_k

    def retrieve(self, result) -> list[Document]:
        query  = result.queries[0] if result.queries else result.original_query
        filter = result.metadata_filter or {}

        # Bước 1: chỉ truy hồi chunk con
        child_docs = self._search(
            query=query, k=self.candidate_k,
            filter={**filter, "chunk_level": "child"},
        )
        if not child_docs:
            child_docs = self._search(query=query, k=self.candidate_k, filter=filter)

        # Bước 2: gom parent_id không trùng, giữ thứ tự xếp hạng
        parent_ids: list[str] = []
        seen: set[str]        = set()
        for doc in child_docs:
            pid = doc.metadata.get("parent_id")
            if pid and pid not in seen:
                seen.add(pid)
                parent_ids.append(pid)
            if len(parent_ids) >= self.top_k:
                break

        # Bước 3: lấy chunk cha theo parent_id
        parent_docs: list[Document] = []
        for pid in parent_ids:
            parents = self._search(
                query=query, k=1,
                filter={"parent_id": pid, "chunk_level": "parent"},
            )
            if parents:
                parent_docs.append(parents[0])

        # Không tra được cha thì trả về chính chunk con
        if not parent_docs:
            return deduplicate(child_docs)[:self.top_k]

        return parent_docs[:self.top_k]
