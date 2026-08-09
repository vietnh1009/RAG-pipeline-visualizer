"""
pre_retrieval/query_expander.py
================================
Query Expansion — bổ sung từ đồng nghĩa và từ liên quan vào query.

Lệch từ vựng là một trong những nguyên nhân hỏng truy hồi phổ biến nhất: người
dùng gõ "đái tháo đường" còn tài liệu viết "tiểu đường" hoặc "glucose". Mở rộng
query để cả hai dạng cùng được tìm.

Hai chế độ
----------
``llm``     : LLM sinh các cách diễn đạt tương đương. Chất lượng tốt hơn, tốn token.
``wordnet`` : NLTK WordNet, chỉ tiếng Anh, miễn phí và chạy offline.

Query gốc luôn được giữ cùng với các từ mở rộng.

Dùng khi: thuật ngữ trong miền không nhất quán, người dùng hay dùng từ thông tục.
"""

from __future__ import annotations

from pre_retrieval.base import BaseTransformer, TransformResult
from utils.llm import call_llm, parse_json_list


class QueryExpander(BaseTransformer):
    """
    Mở rộng query bằng từ đồng nghĩa và cách diễn đạt khác.

    Tham số
    -------
    mode           : "llm" | "wordnet"
    num_expansions : Số từ / cụm bổ sung.
    language       : "vi" | "en" | "both"
    """

    _PROMPT_EN = (
        "Generate {n} alternative phrasings or closely related terms for "
        "the following search query. Include synonyms, abbreviations, and "
        "related concepts that might appear in relevant documents.\n\n"
        "Query: {query}\n\n"
        "Return ONLY a JSON array of {n} strings. "
        'Example: ["term 1", "term 2", "term 3"]'
    )

    _PROMPT_VI = (
        "Hãy tạo {n} cách diễn đạt khác nhau hoặc thuật ngữ liên quan cho "
        "câu truy vấn sau. Bao gồm từ đồng nghĩa, viết tắt và các khái niệm "
        "liên quan có thể xuất hiện trong tài liệu.\n\n"
        "Truy vấn: {query}\n\n"
        "Chỉ trả về một JSON array gồm {n} chuỗi. "
        'Ví dụ: ["thuật ngữ 1", "thuật ngữ 2", "thuật ngữ 3"]'
    )

    def __init__(
        self,
        mode:           str = "llm",
        num_expansions: int = 3,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.mode           = mode
        self.num_expansions = num_expansions

    def transform(self, query: str) -> TransformResult:
        expansions = (
            self._expand_wordnet(query)
            if self.mode == "wordnet"
            else self._expand_llm(query)
        )
        return TransformResult(
            original_query=query,
            queries=[query] + expansions,
            extra={"expansions": expansions},
        )

    def _expand_llm(self, query: str) -> list[str]:
        tmpl = self._PROMPT_VI if self.language == "vi" else self._PROMPT_EN
        raw  = call_llm(
            tmpl.format(query=query, n=self.num_expansions),
            self.llm_provider, self.llm_model, max_tokens=256,
        )
        return parse_json_list(raw)[: self.num_expansions]

    def _expand_wordnet(self, query: str) -> list[str]:
        import nltk
        from nltk.corpus import wordnet
        try:
            wordnet.synsets("test")
        except LookupError:
            nltk.download("wordnet", quiet=True)
            nltk.download("omw-1.4", quiet=True)

        expansions: set[str] = set()
        for word in query.split():
            for syn in wordnet.synsets(word):
                for lemma in syn.lemmas():
                    term = lemma.name().replace("_", " ")
                    if term.lower() != word.lower():
                        expansions.add(term)
            if len(expansions) >= self.num_expansions:
                break
        return list(expansions)[: self.num_expansions]
