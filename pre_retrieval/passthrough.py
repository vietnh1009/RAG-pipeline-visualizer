"""
pre_retrieval/passthrough.py
============================
Passthrough — trả query nguyên vẹn, không biến đổi gì.

Dùng khi tắt pre-retrieval (``transformations: [none]`` trong config.yaml).
"""

from __future__ import annotations

from pre_retrieval.base import BaseTransformer, TransformResult


class PassthroughTransformer(BaseTransformer):
    """Transformer rỗng — chuyển tiếp query y nguyên."""

    def transform(self, query: str) -> TransformResult:
        return TransformResult(original_query=query, queries=[query])
