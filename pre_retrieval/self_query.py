"""
pre_retrieval/self_query.py
============================
Self-Query — extract structured metadata filters from natural language.

Users often embed filter conditions in their queries without realising it:
  "Show me Vietnamese policy documents about diabetes from 2024"

A self-query transformer parses this into:
  Semantic query   : "diabetes policy"
  Metadata filters : {"language": "vi", "doc_type": "policy", "year": 2024}

The filter is then passed to the vector DB to narrow the search space
*before* (or during) ANN retrieval, which is much more efficient than
post-retrieval filtering.

Available filter fields are defined by the ``schema`` parameter — a dict
mapping field names to their types and descriptions.

Use when: corpus has rich metadata; users frequently specify constraints
          in natural language (date, source, language, category, etc.).
"""

from __future__ import annotations

import json
import re

from pre_retrieval.base import BaseTransformer, TransformResult
from pre_retrieval.utils import call_llm


class SelfQueryTransformer(BaseTransformer):
    """
    Parse metadata filters from the user query.

    Parameters
    ----------
    schema   : Dict mapping filterable field names to descriptions.
               Example::
                 {
                   "language": "document language: 'vi' or 'en'",
                   "year":     "publication year as integer",
                   "source":   "filename of the source document",
                   "doc_type": "type: 'guideline', 'report', 'faq'"
                 }
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
