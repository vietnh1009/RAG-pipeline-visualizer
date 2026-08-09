"""
pre_retrieval/query_rewriter.py
================================
Query Rewriting — chuẩn hoá query thô của người dùng trước khi truy hồi.

Xử lý các lỗi thường gặp:
- Sai chính tả, sai ngữ pháp
- Văn nói, cách diễn đạt suồng sã → văn viết
- Đại từ mơ hồ: "nó hoạt động sao" → "RAG hoạt động sao"
- Query lê thê → ngắn gọn, giàu từ khoá
- Tiếng Việt: chuẩn hoá dấu và các dạng viết tắt phổ biến

Query đã viết lại thay thế query gốc ở các bước sau; bản gốc luôn còn trong
``result.original_query``.
"""

from __future__ import annotations

from pre_retrieval.base import BaseTransformer, TransformResult
from utils.llm import call_llm


class QueryRewriter(BaseTransformer):
    """
    Viết lại query thô thành dạng sạch, tối ưu cho truy hồi.

    Tham số
    -------
    language : "vi" | "en" | "both" — chọn ngôn ngữ prompt.
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
