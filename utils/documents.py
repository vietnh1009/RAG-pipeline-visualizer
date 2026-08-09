"""
utils/documents.py
==================
Tiện ích thao tác trên ``Document`` dùng chung giữa các bước.

Trước đây ``deduplicate`` được sao chép giống hệt ở ``retrieval/utils.py`` và
``post_retrieval/utils.py``. Gộp về một chỗ để hai bước không thể trôi khác
nhau — nếu định nghĩa "trùng lặp" thay đổi, nó phải thay đổi ở cả pipeline.

Public API
----------
    from utils.documents import deduplicate
"""

from __future__ import annotations

from langchain_core.documents import Document


def deduplicate(docs: list[Document]) -> list[Document]:
    """
    Bỏ các Document trùng nhau, giữ nguyên thứ tự xuất hiện đầu tiên.

    Khoá so sánh là ``page_content`` đã ``.strip()`` — trùng lặp CHÍNH XÁC,
    không phải trùng lặp ngữ nghĩa. Muốn khử trùng lặp theo ngữ nghĩa, dùng
    ``post_retrieval/redundancy_filter.py``.

    Tham số
    -------
    docs : Danh sách Document, có thể chứa phần tử trùng.

    Trả về
    ------
    Danh sách mới, không có phần tử trùng. Danh sách gốc không bị thay đổi.

    Ví dụ
    -----
    >>> deduplicate([Document(page_content="a"), Document(page_content="a ")])
    [Document(page_content='a')]
    """
    seen:   set[str]       = set()
    unique: list[Document] = []
    for doc in docs:
        key = doc.page_content.strip()
        if key not in seen:
            seen.add(key)
            unique.append(doc)
    return unique
