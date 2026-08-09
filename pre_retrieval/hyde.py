"""
pre_retrieval/hyde.py
=====================
HyDE — Hypothetical Document Embeddings (Gao et al., 2022).

Thay vì embed thẳng câu hỏi, nhờ LLM viết một đoạn tài liệu giả định *sẽ* trả
lời câu hỏi đó, rồi embed đoạn giả định này.

Vì sao hiệu quả: câu hỏi ngắn và đoạn văn trả lời nằm xa nhau trong không gian
vector. Tài liệu giả định dùng đúng từ vựng và văn phong tài liệu thật nên rơi
gần vùng chứa câu trả lời.

Dùng khi : query ngắn hoặc trừu tượng.
Lưu ý    : tốn thêm một lần gọi LLM cho mỗi query.
"""

from __future__ import annotations

from pre_retrieval.base import BaseTransformer, TransformResult
from utils.llm import call_llm


class HyDETransformer(BaseTransformer):
    """
    Sinh tài liệu giả định và dùng nó làm query truy hồi.

    Tham số
    -------
    doc_length : Số từ xấp xỉ của đoạn sinh ra.
    language   : "vi" | "en" | "both" — ngôn ngữ sinh.
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
