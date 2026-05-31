"""
pre_retrieval/query_expander.py
================================
Query Expansion — enrich the query with synonyms and related terms.

Why it helps
------------
Vocabulary mismatch is one of the most common causes of retrieval failure:
the user writes "đái tháo đường" but the document uses "tiểu đường" or
"glucose". Expansion ensures both forms are searched simultaneously.

Two modes
---------
``llm``     : LLM generates semantically related alternative phrasings.
              Better quality; costs API tokens.
``wordnet`` : NLTK WordNet synonym expansion (English only, free, offline).

The original query is always included alongside the expanded terms.

Use when: domain terminology is inconsistent; users use informal synonyms.
"""

from __future__ import annotations

from pre_retrieval.base import BaseTransformer, TransformResult
from pre_retrieval.utils import call_llm, parse_json_list


class QueryExpander(BaseTransformer):
    """
    Expand the query with synonyms and alternative phrasings.

    Parameters
    ----------
    mode           : "llm" | "wordnet"
    num_expansions : Number of additional terms / phrases to add.
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
