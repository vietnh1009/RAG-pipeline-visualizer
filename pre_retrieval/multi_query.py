"""
pre_retrieval/multi_query.py
=============================
Multi-Query — tách một query thành N góc nhìn truy hồi khác nhau.

Một vector query khó bao hết mọi khía cạnh của câu hỏi phức tạp. Chạy N query
rồi gộp bằng RRF giúp recall tăng rõ rệt, đổi lại N lần tìm vector (rẻ) và một
lần gọi LLM.

Ví dụ (n=3) với "So sánh lối sống và thuốc trong điều trị tiểu đường type 2":
  1. "can thiệp lối sống điều trị tiểu đường type 2"
  2. "các lựa chọn thuốc kiểm soát tiểu đường type 2"
  3. "so sánh ăn uống vận động với dùng thuốc"

Dùng khi: câu hỏi phức tạp, nhiều mặt; cần recall cao với chi phí thấp.
"""

from __future__ import annotations

from pre_retrieval.base import BaseTransformer, TransformResult
from utils.llm import call_llm, parse_json_list


class MultiQueryTransformer(BaseTransformer):
    """
    Tách query thành N truy vấn con theo các góc tiếp cận khác nhau.

    Tham số
    -------
    n_queries        : Số truy vấn con cần sinh.
    include_original : Giữ luôn query gốc trong tập truy hồi.
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
