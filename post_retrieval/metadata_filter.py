"""
post_retrieval/metadata_filter.py
===================================
Lọc cứng theo metadata — chỉ giữ document khớp mọi điều kiện.

Khác với filter của vector DB (chạy trước/trong lúc tìm ANN), lớp này chạy SAU
truy hồi, làm lưới an toàn cho: điều kiện vector DB không hỗ trợ, điều kiện chỉ
biết được ở thời điểm sau truy hồi, và việc debug chất lượng truy hồi.

Toán tử hỗ trợ
--------------
  eq / ne            bằng / khác
  in                 thuộc danh sách
  gt / lt / gte / lte  lớn hơn / nhỏ hơn / lớn-bằng / nhỏ-bằng
  contains           chứa chuỗi con

Mọi điều kiện được nối bằng AND.
"""

from __future__ import annotations

from langchain_core.documents import Document

from post_retrieval.base import BasePostProcessor


class MetadataFilter(BasePostProcessor):
    """
    Chỉ giữ document khớp toàn bộ điều kiện metadata.

    Tham số
    -------
    conditions : List dict, mỗi dict gồm ``field`` (key metadata),
                 ``operator`` (eq | ne | in | gt | lt | gte | lte | contains)
                 và ``value`` (giá trị so sánh).

    Ví dụ
    -----
    >>> f = MetadataFilter(conditions=[
    ...     {"field": "language", "operator": "eq",       "value": "vi"},
    ...     {"field": "year",     "operator": "gte",      "value": 2022},
    ...     {"field": "source",   "operator": "contains", "value": "policy"},
    ... ])
    """

    def __init__(self, conditions: list[dict]):
        self.conditions = conditions

    def process(self, query: str, docs: list[Document]) -> list[Document]:
        return [doc for doc in docs if self._matches(doc)]

    def _matches(self, doc: Document) -> bool:
        for cond in self.conditions:
            field    = cond["field"]
            operator = cond["operator"]
            value    = cond["value"]
            fval     = doc.metadata.get(field)

            if operator == "eq"       and fval != value:                    return False
            if operator == "ne"       and fval == value:                    return False
            if operator == "in"       and fval not in value:                return False
            if operator == "gt"       and not (fval is not None and fval > value):  return False
            if operator == "lt"       and not (fval is not None and fval < value):  return False
            if operator == "gte"      and not (fval is not None and fval >= value): return False
            if operator == "lte"      and not (fval is not None and fval <= value): return False
            if operator == "contains" and value not in str(fval or ""):     return False
        return True
