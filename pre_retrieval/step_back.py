"""
pre_retrieval/step_back.py
===========================
Step-Back Prompting (Zheng et al., 2023 — Google DeepMind).

Sinh thêm một câu hỏi rộng và trừu tượng hơn để lấy kiến thức nền, rồi dùng
kiến thức đó khi trả lời câu hỏi cụ thể.

    Cụ thể  : "Liều metformin khuyến cáo cho bệnh nhân 70 kg là bao nhiêu?"
    Lùi lại : "Nguyên tắc chung khi kê liều metformin là gì?"

Cả câu lùi lại lẫn query gốc đều nằm trong ``result.queries``, nên retrieval
chạy cả hai rồi gộp — LLM nhận được vừa bối cảnh nền vừa bằng chứng cụ thể.

Dùng khi: câu hỏi rất hẹp, muốn trả lời đúng phải có kiến thức nền.
"""

from __future__ import annotations

from pre_retrieval.base import BaseTransformer, TransformResult
from utils.llm import call_llm


class StepBackTransformer(BaseTransformer):
    """
    Sinh câu hỏi nền rộng hơn, đi kèm query gốc.

    Tham số
    -------
    include_original : Truy hồi thêm bằng query gốc (khuyến nghị bật).
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
