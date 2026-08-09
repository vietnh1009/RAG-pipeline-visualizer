"""
post_retrieval/llm_reranker.py
================================
LLM Listwise Reranker (kiểu RankGPT, Sun et al. 2023).

Đưa query cùng toàn bộ document cho LLM trong một lần, yêu cầu trả về danh sách
chỉ số đã sắp theo độ liên quan ("3,1,5,2,4", đánh số từ 1). LLM nhìn được quan
hệ giữa các document cùng lúc — điều cross-encoder chấm từng cặp không làm được.

Dùng khi: không có GPU cho cross-encoder; top_n nhỏ (≤ 10) để prompt vừa context;
query phức tạp cần nhìn tổng thể.

Hạn chế: mỗi lần rerank tốn một lời gọi LLM, chậm khi N lớn.
"""

from __future__ import annotations

import re

from langchain_core.documents import Document

from post_retrieval.base import BasePostProcessor
from utils.llm import call_llm


class LLMReranker(BasePostProcessor):
    """
    Rerank listwise bằng LLM.

    Tham số
    -------
    llm_model     : LLM dùng để rerank.
    llm_provider  : "openai" | "anthropic" | "google"
    top_n         : Số document giữ lại.
    max_doc_chars : Số ký tự tối đa mỗi document đưa vào prompt.
    language      : "vi" | "en" | "both" — ngôn ngữ prompt.
    """

    _PROMPT_EN = (
        "You are a relevance ranking expert.\n\n"
        "Query: {query}\n\n"
        "Below are {n} retrieved passages numbered [1] to [{n}].\n"
        "Rank them from MOST to LEAST relevant to the query.\n\n"
        "{passages}\n\n"
        "Output ONLY a comma-separated list of indices in descending relevance order.\n"
        "Example for 5 passages: 3,1,5,2,4\n"
        "Ranking:"
    )

    _PROMPT_VI = (
        "Bạn là chuyên gia xếp hạng mức độ liên quan.\n\n"
        "Truy vấn: {query}\n\n"
        "Dưới đây là {n} đoạn văn được đánh số [1] đến [{n}].\n"
        "Hãy xếp hạng từ LIÊN QUAN NHẤT đến ÍT LIÊN QUAN NHẤT.\n\n"
        "{passages}\n\n"
        "Chỉ xuất danh sách các chỉ số cách nhau bằng dấu phẩy.\n"
        "Ví dụ: 3,1,5,2,4\n"
        "Xếp hạng:"
    )

    def __init__(
        self,
        llm_model:     str = "gpt-4.1-mini",
        llm_provider:  str = "openai",
        top_n:         int = 5,
        max_doc_chars: int = 400,
        language:      str = "both",
    ):
        self.llm_model     = llm_model
        self.llm_provider  = llm_provider
        self.top_n         = top_n
        self.max_doc_chars = max_doc_chars
        self.language      = language

    def process(self, query: str, docs: list[Document]) -> list[Document]:
        if not docs:
            return docs

        snippets = [
            f"[{i+1}] {doc.page_content[:self.max_doc_chars]}"
            for i, doc in enumerate(docs)
        ]
        tmpl   = self._PROMPT_VI if self.language == "vi" else self._PROMPT_EN
        prompt = tmpl.format(
            query=query, n=len(docs), passages="\n\n".join(snippets)
        )
        raw     = call_llm(prompt, self.llm_provider, self.llm_model, max_tokens=128)
        indices = self._parse(raw, len(docs))

        reranked: list[Document] = []
        for rank, idx in enumerate(indices):
            doc = docs[idx]
            doc.metadata["llm_rerank_position"] = rank + 1
            reranked.append(doc)

        # Nối thêm các document LLM không nhắc tới
        mentioned = set(indices)
        for i, doc in enumerate(docs):
            if i not in mentioned:
                reranked.append(doc)

        return reranked[:self.top_n]

    @staticmethod
    def _parse(raw: str, n: int) -> list[int]:
        """Đọc '3,1,5,2,4' → [2, 0, 4, 1, 3] (đánh số từ 0)."""
        indices: list[int] = []
        seen:    set[int]  = set()
        for num in re.findall(r"\d+", raw):
            idx = int(num) - 1
            if 0 <= idx < n and idx not in seen:
                indices.append(idx)
                seen.add(idx)
        return indices
