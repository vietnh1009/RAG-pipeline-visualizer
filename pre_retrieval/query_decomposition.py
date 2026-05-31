"""
pre_retrieval/query_decomposition.py
=====================================
Query Decomposition — break a complex question into simpler sub-questions.

Unlike MultiQueryTransformer (which generates N paraphrases of the same
question), QueryDecomposition identifies logically distinct sub-questions
that must each be answered to resolve the original compound question.

Types
-----
sequential : Sub-questions depend on each other (answer Q1 before Q2).
             E.g. "Who founded the company that acquired DeepMind?"
             → Q1: "Which company acquired DeepMind?"
             → Q2: "Who founded [answer to Q1]?"

parallel   : Sub-questions are independent and can be retrieved simultaneously.
             E.g. "Compare BM25 and dense retrieval on precision and recall."
             → Q1: "What is the precision of BM25 retrieval?"
             → Q2: "What is the recall of BM25 retrieval?"
             → Q3: "What is the precision of dense retrieval?"
             → Q4: "What is the recall of dense retrieval?"

Use when: compound questions, multi-hop reasoning, comparison queries.
"""

from __future__ import annotations

from pre_retrieval.base import BaseTransformer, TransformResult
from pre_retrieval.utils import call_llm, parse_json_list


class QueryDecompositionTransformer(BaseTransformer):
    """
    Decompose a complex question into simpler, targeted sub-questions.

    Parameters
    ----------
    mode             : "parallel" | "sequential"
    max_sub_questions: Maximum number of sub-questions to generate.
    include_original : Also include the original query in the retrieval set.
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
