"""
retrieval/sentence_window.py
=============================
Tìm theo câu rồi mở rộng ra các câu lân cận.

Hiệu quả nhất khi chunk rất nhỏ (mỗi chunk một câu). Sau khi tìm được câu liên
quan nhất, retriever nối thêm ±window_size câu hai bên để LLM có đủ ngữ cảnh.

Yêu cầu: chunk phải có ``chunk_index`` và ``source`` trong metadata — mọi lớp
con của BaseChunker đều tự thêm sẵn.
"""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore

from retrieval.base import BaseRetriever
from utils.documents import deduplicate


class SentenceWindowRetriever(BaseRetriever):
    """
    Tìm theo câu rồi mở rộng thành cửa sổ ±window_size câu.

    Tham số
    -------
    vector_store  : VectorStore đã nạp dữ liệu.
    all_documents : Toàn bộ document trong index, cần để mở rộng cửa sổ.
    top_k         : Số cửa sổ trả về.
    window_size   : Số câu lấy thêm ở mỗi bên câu khớp.
    """

    def __init__(
        self,
        vector_store:  VectorStore,
        all_documents: list[Document],
        top_k:         int = 5,
        window_size:   int = 2,
    ):
        super().__init__(vector_store, top_k)
        self.window_size = window_size
        # Bảng tra: (source, chunk_index) -> Document
        self._doc_map: dict[tuple[str, int], Document] = {
            (d.metadata.get("source", ""), d.metadata.get("chunk_index", -1)): d
            for d in all_documents
        }

    def retrieve(self, result) -> list[Document]:
        query  = result.queries[0] if result.queries else result.original_query
        filter = result.metadata_filter
        sentence_docs = self._search(query=query, k=self.top_k, filter=filter)

        expanded: list[Document] = []
        seen_keys: set[tuple[str, int]] = set()

        for doc in sentence_docs:
            source = doc.metadata.get("source", "")
            idx    = doc.metadata.get("chunk_index", -1)

            window: list[str] = []
            for offset in range(-self.window_size, self.window_size + 1):
                key = (source, idx + offset)
                if key in seen_keys:
                    continue
                neighbour = self._doc_map.get(key)
                if neighbour:
                    window.append(neighbour.page_content)
                    seen_keys.add(key)

            if window:
                expanded.append(Document(
                    page_content=" ".join(window),
                    metadata={
                        **doc.metadata,
                        "window_size":        self.window_size,
                        "retrieval_strategy": "sentence_window",
                    },
                ))
            else:
                expanded.append(doc)

        return deduplicate(expanded)[:self.top_k]
