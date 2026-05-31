"""
pre_retrieval/hyde.py
=====================
HyDE — Hypothetical Document Embeddings (Gao et al., 2022).

Instead of embedding the query directly, ask an LLM to write a short
hypothetical document that *would* answer the query. Embed that
hypothetical document instead of the original question.

Why it works
------------
Embedding models see queries and documents during training but often
treat short questions and long answer passages as very different objects
in the vector space. A hypothetical document uses the same vocabulary
and style as a real document, so it lands much closer to actual answers.

Mechanism
---------
Query: "What are the symptoms of metabolic syndrome?"
           ↓  LLM generates
Hypothetical doc: "Metabolic syndrome is characterised by central obesity,
hyperglycaemia, elevated triglycerides, reduced HDL-C, and hypertension..."
           ↓  embed hypothetical doc → retrieve with this vector

Use when: queries are short or abstract; asymmetric embedding problem.
Note    : adds one LLM call per query; use response caching to reduce cost.
"""

from __future__ import annotations

from pre_retrieval.base import BaseTransformer, TransformResult
from pre_retrieval.utils import call_llm


class HyDETransformer(BaseTransformer):
    """
    Generate a hypothetical answer document and use it as the retrieval query.

    Parameters
    ----------
    doc_length : Approximate word count of the generated document.
    language   : "vi" | "en" | "both" — controls generation language.
    """

    _PROMPT_EN = (
        "Write a short passage of about {length} words from a reference document "
        "that would directly and factually answer the following question. "
        "Write as if it is a real excerpt — dense and informative.\n\n"
        "Question: {query}\n\n"
        "Passage:"
    )

    _PROMPT_VI = (
        "Viết một đoạn văn khoảng {length} từ từ một tài liệu tham khảo "
        "sẽ trả lời trực tiếp và thực tế câu hỏi sau. "
        "Viết như thể đây là trích dẫn thực sự — đầy đủ thông tin và súc tích.\n\n"
        "Câu hỏi: {query}\n\n"
        "Đoạn văn:"
    )

    def __init__(self, doc_length: int = 100, **kwargs):
        super().__init__(**kwargs)
        self.doc_length = doc_length

    def transform(self, query: str) -> TransformResult:
        tmpl = self._PROMPT_VI if self.language == "vi" else self._PROMPT_EN
        hyp_doc = call_llm(
            tmpl.format(query=query, length=self.doc_length),
            self.llm_provider, self.llm_model,
            max_tokens=self.doc_length * 5,
        )
        if not hyp_doc:
            hyp_doc = query

        return TransformResult(
            original_query=query,
            queries=[hyp_doc],
            extra={"hypothetical_document": hyp_doc},
        )
