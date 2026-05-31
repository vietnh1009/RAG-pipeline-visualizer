"""
pre_retrieval/passthrough.py
============================
Passthrough — no transformation applied.

Returns the query unchanged. Use when pre-retrieval is disabled
(``transformations: [none]`` in config.yaml).
"""

from __future__ import annotations

from pre_retrieval.base import BaseTransformer, TransformResult


class PassthroughTransformer(BaseTransformer):
    """No-op transformer — passes the query through as-is."""

    def transform(self, query: str) -> TransformResult:
        return TransformResult(original_query=query, queries=[query])
