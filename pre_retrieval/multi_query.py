"""
pre_retrieval/multi_query.py
=============================
Multi-Query — decompose one query into N different retrieval perspectives.

Why it helps
------------
A single query vector may not capture all relevant aspects of a complex
question. Different phrasings retrieve different — and complementary —
chunks. By running N queries and merging results via RRF, recall improves
significantly at the cost of N extra vector searches (cheap) and one
LLM call (moderate).

Example (n=3)
-------------
Query: "Compare lifestyle vs medication for type 2 diabetes treatment"
Sub-queries:
  1. "lifestyle intervention for type 2 diabetes treatment"
  2. "medication options for managing type 2 diabetes"
  3. "diet exercise versus drug therapy diabetes comparison"

Use when: complex, multi-faceted queries; want higher recall at low cost.
"""

from __future__ import annotations

from pre_retrieval.base import BaseTransformer, TransformResult
from pre_retrieval.utils import call_llm, parse_json_list


class MultiQueryTransformer(BaseTransformer):
    """
    Decompose the query into N sub-queries covering different angles.

    Parameters
    ----------
    n_queries        : Number of sub-queries to generate.
    include_original : Also include the original query in the retrieval set.
    language         : "vi" | "en" | "both"
    """

    _PROMPT_EN = (
        "Generate {n} different search queries that together cover all aspects "
        "of the following question. Each query should approach the topic from "
        "a different angle to maximise document recall.\n\n"
        "Original question: {query}\n\n"
        "Return ONLY a JSON array of {n} query strings. "
        'Example: ["query 1", "query 2", "query 3"]'
    )

    _PROMPT_VI = (
        "Hãy tạo {n} câu truy vấn tìm kiếm khác nhau cùng bao phủ tất cả "
        "các khía cạnh của câu hỏi sau. Mỗi truy vấn tiếp cận chủ đề từ "
        "một góc độ khác nhau để tối đa hóa khả năng tìm kiếm.\n\n"
        "Câu hỏi gốc: {query}\n\n"
        "Chỉ trả về một JSON array gồm {n} câu truy vấn. "
        'Ví dụ: ["truy vấn 1", "truy vấn 2", "truy vấn 3"]'
    )

    def __init__(
        self,
        n_queries:        int  = 3,
        include_original: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.n_queries        = n_queries
        self.include_original = include_original

    def transform(self, query: str) -> TransformResult:
        tmpl = self._PROMPT_VI if self.language == "vi" else self._PROMPT_EN
        raw  = call_llm(
            tmpl.format(query=query, n=self.n_queries),
            self.llm_provider, self.llm_model, max_tokens=512,
        )
        sub_queries = parse_json_list(raw)[: self.n_queries]

        if not sub_queries:
            sub_queries = [query]

        queries = ([query] if self.include_original else []) + sub_queries
        return TransformResult(
            original_query=query,
            queries=queries,
            extra={"sub_queries": sub_queries},
        )
