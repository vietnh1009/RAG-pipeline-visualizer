"""
pre_retrieval/step_back.py
===========================
Step-Back Prompting (Zheng et al., 2023 — Google DeepMind).

Generates a broader, more abstract version of the user question to first
retrieve background context, then uses that context when answering the
specific question.

Example
-------
Specific  : "What is the recommended metformin dose for a 70 kg patient?"
Step-back : "What are the general principles of metformin dosing?"

The step-back query retrieves general background knowledge that helps
ground the answer to the specific question.

Retrieval behaviour
-------------------
Both the step-back query AND the original query are in ``result.queries``.
The retrieval stage retrieves for both and merges results, so the LLM
receives background context alongside specific evidence.

Use when: queries require background knowledge to answer correctly;
          users ask very specific questions that lack broader context.
"""

from __future__ import annotations

from pre_retrieval.base import BaseTransformer, TransformResult
from pre_retrieval.utils import call_llm


class StepBackTransformer(BaseTransformer):
    """
    Generate a broader background question alongside the original query.

    Parameters
    ----------
    include_original : Also retrieve with the original query (recommended).
    language         : "vi" | "en" | "both"
    """

    _PROMPT_EN = (
        "Given the following specific question, generate a more general, "
        "broader version that would help retrieve background knowledge useful "
        "for answering the specific question.\n\n"
        "Specific question: {query}\n\n"
        "Broader background question "
        "(output ONLY the broader question, nothing else):"
    )

    _PROMPT_VI = (
        "Với câu hỏi cụ thể sau, hãy tạo ra một phiên bản tổng quát hơn "
        "để giúp truy xuất kiến thức nền tảng hữu ích cho việc trả lời câu hỏi.\n\n"
        "Câu hỏi cụ thể: {query}\n\n"
        "Câu hỏi nền tảng rộng hơn "
        "(chỉ xuất câu hỏi, không có gì khác):"
    )

    def __init__(self, include_original: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.include_original = include_original

    def transform(self, query: str) -> TransformResult:
        tmpl       = self._PROMPT_VI if self.language == "vi" else self._PROMPT_EN
        step_back  = call_llm(
            tmpl.format(query=query),
            self.llm_provider, self.llm_model, max_tokens=128,
        ).strip().strip('"')

        if not step_back:
            step_back = query

        queries = ([query] if self.include_original else []) + [step_back]
        return TransformResult(
            original_query=query,
            queries=queries,
            extra={"step_back_query": step_back},
        )
