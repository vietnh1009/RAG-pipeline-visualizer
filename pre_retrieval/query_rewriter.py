"""
pre_retrieval/query_rewriter.py
================================
Query Rewriting — normalise the raw user query before retrieval.

Problems fixed
--------------
- Spelling / grammar errors      : "how dose rag work" → "how does RAG work"
- Colloquial / spoken language   : casual phrasing → formal written language
- Ambiguous pronouns             : "how does it work" → "how does RAG work"
- Overly verbose queries         : long ramblings → concise, keyword-rich form
- Vietnamese: normalises diacritics and common abbreviation patterns

The rewritten query replaces the original for downstream retrieval.
The original is always preserved in ``result.original_query``.

Use when: users submit conversational, misspelled, or verbose queries.
"""

from __future__ import annotations

from pre_retrieval.base import BaseTransformer, TransformResult
from pre_retrieval.utils import call_llm


class QueryRewriter(BaseTransformer):
    """
    Rewrite the raw user query into a cleaner, retrieval-optimised form.

    Parameters
    ----------
    language : "vi" | "en" | "both" — selects the prompt language.
    """

    _PROMPT_EN = (
        "Rewrite the following search query to make it more suitable for "
        "document retrieval. Fix spelling and grammar, resolve ambiguous "
        "pronouns if possible, and make it concise and keyword-rich.\n\n"
        "Original query: {query}\n\n"
        "Rewritten query (output ONLY the rewritten query, nothing else):"
    )

    _PROMPT_VI = (
        "Hãy viết lại câu truy vấn tìm kiếm sau để phù hợp hơn với việc "
        "tìm kiếm tài liệu. Sửa lỗi chính tả và ngữ pháp, giải quyết đại từ "
        "mơ hồ nếu có thể, và làm cho câu ngắn gọn, giàu từ khóa hơn.\n\n"
        "Câu truy vấn gốc: {query}\n\n"
        "Câu truy vấn đã viết lại (chỉ xuất câu viết lại, không có gì khác):"
    )

    def transform(self, query: str) -> TransformResult:
        tmpl      = self._PROMPT_VI if self.language == "vi" else self._PROMPT_EN
        rewritten = call_llm(
            tmpl.format(query=query),
            self.llm_provider, self.llm_model, max_tokens=128,
        ).strip().strip('"').strip("'")

        if not rewritten or len(rewritten) < 3:
            rewritten = query

        return TransformResult(original_query=query, queries=[rewritten])
