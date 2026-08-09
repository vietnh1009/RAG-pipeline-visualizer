"""
pre_retrieval/query_decomposition.py
=====================================
Query Decomposition — chẻ câu hỏi phức thành các câu hỏi con đơn giản hơn.

Khác MultiQueryTransformer (sinh N cách diễn đạt của *cùng* một câu hỏi), lớp
này tách ra những câu hỏi con khác nhau về mặt logic, phải trả lời hết mới
giải quyết được câu hỏi gốc.

Hai chế độ
----------
sequential : Câu hỏi con phụ thuộc nhau — phải trả lời Q1 mới hỏi được Q2.
             "Ai sáng lập công ty đã mua DeepMind?"
             → Q1 "Công ty nào mua DeepMind?" → Q2 "Ai sáng lập [đáp án Q1]?"

parallel   : Câu hỏi con độc lập, truy hồi song song được.
             "So sánh BM25 và dense retrieval về precision và recall."
             → 4 câu hỏi con cho từng cặp phương pháp × chỉ số.

Dùng khi: câu hỏi ghép, suy luận nhiều bước, câu hỏi so sánh.
"""

from __future__ import annotations

from pre_retrieval.base import BaseTransformer, TransformResult
from utils.llm import call_llm, parse_json_list


class QueryDecompositionTransformer(BaseTransformer):
    """
    Chẻ câu hỏi phức thành các câu hỏi con đơn giản, cụ thể hơn.

    Tham số
    -------
    mode             : "parallel" | "sequential"
    max_sub_questions: Số câu hỏi con tối đa.
    include_original : Giữ luôn query gốc trong tập truy hồi.
    language         : "vi" | "en" | "both"
    """

    _PROMPT_EN = (
        "Decompose the following complex question into {n} simpler, specific "
        "sub-questions. Each sub-question should be independently answerable "
        "from a document and together they should fully answer the original.\n\n"
        "Question: {query}\n\n"
        "Return ONLY a JSON array of sub-question strings. "
        'Example: ["sub-question 1", "sub-question 2"]'
    )

    _PROMPT_VI = (
        "Hãy phân tách câu hỏi phức tạp sau thành {n} câu hỏi con đơn giản hơn. "
        "Mỗi câu hỏi con nên có thể trả lời độc lập từ tài liệu và "
        "cùng nhau giải quyết hoàn toàn câu hỏi gốc.\n\n"
        "Câu hỏi: {query}\n\n"
        "Chỉ trả về một JSON array các câu hỏi con. "
        'Ví dụ: ["câu hỏi con 1", "câu hỏi con 2"]'
    )

    def __init__(
        self,
        mode:              str  = "parallel",
        max_sub_questions: int  = 4,
        include_original:  bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.mode              = mode
        self.max_sub_questions = max_sub_questions
        self.include_original  = include_original

    def transform(self, query: str) -> TransformResult:
        tmpl = self._PROMPT_VI if self.language == "vi" else self._PROMPT_EN
        raw  = call_llm(
            tmpl.format(query=query, n=self.max_sub_questions),
            self.llm_provider, self.llm_model, max_tokens=512,
        )
        sub_questions = parse_json_list(raw)[: self.max_sub_questions]

        if not sub_questions:
            sub_questions = [query]

        queries = ([query] if self.include_original else []) + sub_questions
        return TransformResult(
            original_query=query,
            queries=queries,
            extra={
                "sub_questions":       sub_questions,
                "decomposition_mode":  self.mode,
            },
        )
