"""
pre_retrieval/self_query.py
============================
Self-Query — trích metadata filter có cấu trúc từ câu hỏi ngôn ngữ tự nhiên.

Người dùng thường gài sẵn điều kiện lọc mà không nhận ra:
  "Cho tôi tài liệu chính sách tiếng Việt về tiểu đường năm 2024"
  → query ngữ nghĩa : "chính sách tiểu đường"
  → filter          : {"language": "vi", "doc_type": "policy", "year": 2024}

Filter được đẩy xuống vector DB để thu hẹp không gian tìm *trước* khi chạy ANN —
rẻ hơn nhiều so với lọc sau khi đã truy hồi.

Các trường lọc hợp lệ do tham số ``schema`` quy định.
"""

from __future__ import annotations

import json
import re

from pre_retrieval.base import BaseTransformer, TransformResult
from utils.llm import call_llm


class SelfQueryTransformer(BaseTransformer):
    """
    Trích metadata filter từ query của người dùng.

    Tham số
    -------
    schema   : Dict tên trường lọc → mô tả, ví dụ::
                 {"language": "ngôn ngữ tài liệu: 'vi' hoặc 'en'",
                  "year":     "năm xuất bản, kiểu số nguyên",
                  "source":   "tên file tài liệu nguồn"}
               Schema rỗng thì transformer trả query nguyên vẹn.
    language : "vi" | "en" | "both"
    """

    _PROMPT = (
        "You are a query parser. Extract a semantic search query and structured "
        "metadata filters from the user question below.\n\n"
        "Available metadata fields:\n{schema}\n\n"
        "User question: {query}\n\n"
        "Return ONLY valid JSON with two keys:\n"
        '  "query":  the semantic search string (remove filter conditions from it)\n'
        '  "filter": a dict of field→value pairs found in the question, or null\n\n'
        'Example: {{"query": "diabetes treatment", '
        '"filter": {{"language": "vi", "year": 2024}}}}'
    )

    def __init__(self, schema: dict | None = None, **kwargs):
        super().__init__(**kwargs)
        self.schema = schema or {}

    def transform(self, query: str) -> TransformResult:
        if not self.schema:
            return TransformResult(original_query=query, queries=[query])

        schema_str = "\n".join(f"  - {k}: {v}" for k, v in self.schema.items())
        raw        = call_llm(
            self._PROMPT.format(schema=schema_str, query=query),
            self.llm_provider, self.llm_model, max_tokens=256,
        )

        semantic_query  = query
        metadata_filter = None
        try:
            cleaned = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
            parsed  = json.loads(cleaned)
            semantic_query  = (parsed.get("query") or query).strip()
            metadata_filter = parsed.get("filter") or None
        except (json.JSONDecodeError, AttributeError):
            pass

        return TransformResult(
            original_query=query,
            queries=[semantic_query],
            metadata_filter=metadata_filter,
        )
