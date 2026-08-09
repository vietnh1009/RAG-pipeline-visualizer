"""
retrieval/multi_hop.py
=======================
Multi-Hop Retrieval — lặp truy hồi rồi suy luận, cho câu hỏi nhiều bước.

  Hỏi   : "Ai sáng lập công ty đã mua DeepMind?"
  Bước 1: truy hồi → "Google mua DeepMind"
  Bước 2: truy hồi → "Larry Page và Sergey Brin sáng lập Google"

Thuật toán
----------
1. Truy hồi k document cho query hiện tại.
2. Đưa toàn bộ ngữ cảnh đã gom cho LLM: đủ để trả lời chưa?
3. Chưa đủ → LLM sinh query tiếp theo, lặp lại.
4. Dừng khi LLM trả lời DONE hoặc chạm ``max_hops``.
5. Trả về toàn bộ document đã gom, khử trùng và cắt còn top_k.

Dùng khi: cần suy luận nhiều bước, nối quan hệ, hoặc so sánh giữa nhiều tài
liệu mà mỗi tài liệu chỉ chứa một phần thông tin.
"""

from __future__ import annotations

import logging

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore

from retrieval.base import BaseRetriever
from utils.documents import deduplicate

logger = logging.getLogger(__name__)

_HOP_PROMPT = (
    "You are answering a question by iteratively retrieving documents.\n\n"
    "Original question: {question}\n\n"
    "Retrieved context so far:\n{context}\n\n"
    "Do you have enough information to fully answer the original question?\n"
    "If YES → respond with: DONE\n"
    "If NO  → respond with a single follow-up search query to find the "
    "missing information (output ONLY the query, nothing else):"
)


class MultiHopRetriever(BaseRetriever):
    """
    Lặp truy hồi và suy luận cho tới khi đủ thông tin trả lời.

    Tham số
    -------
    vector_store : VectorStore của LangChain đã nạp dữ liệu.
    top_k        : Số document cuối, tính trên tất cả các bước.
    max_hops     : Số vòng truy hồi tối đa.
    candidate_k  : Số document lấy mỗi vòng.
    llm_model    : LLM quyết định dừng hay đi tiếp.
    llm_provider : "openai" | "anthropic" | "google"
    """

    def __init__(
        self,
        vector_store: VectorStore,
        top_k:        int = 5,
        max_hops:     int = 3,
        candidate_k:  int = 5,
        llm_model:    str = "gpt-4.1-mini",
        llm_provider: str = "openai",
    ):
        super().__init__(vector_store, top_k)
        self.max_hops     = max_hops
        self.candidate_k  = candidate_k
        self.llm_model    = llm_model
        self.llm_provider = llm_provider

    def retrieve(self, result) -> list[Document]:
        original_query = result.original_query
        current_query  = result.queries[0] if result.queries else original_query
        all_docs: list[Document] = []

        for hop in range(self.max_hops):
            hop_docs = self._search(
                query=current_query, k=self.candidate_k,
                filter=result.metadata_filter,
            )
            all_docs.extend(hop_docs)

            context  = "\n\n".join(d.page_content for d in all_docs[:10])
            followup = self._ask_followup(original_query, context)

            if not followup or followup.upper().startswith("DONE"):
                logger.debug("MultiHop: stopping at hop %d.", hop + 1)
                break

            logger.debug("MultiHop hop %d: follow-up='%s'.", hop + 1, followup[:60])
            current_query = followup

        return deduplicate(all_docs)[:self.top_k]

    def _ask_followup(self, question: str, context: str) -> str:
        prompt = _HOP_PROMPT.format(question=question, context=context[:3000])
        try:
            if self.llm_provider == "openai":
                from openai import OpenAI
                r = OpenAI().chat.completions.create(
                    model=self.llm_model, temperature=0, max_tokens=128,
                    messages=[{"role": "user", "content": prompt}],
                )
                return r.choices[0].message.content.strip()
            if self.llm_provider == "anthropic":
                import anthropic
                r = anthropic.Anthropic().messages.create(
                    model=self.llm_model, max_tokens=128,
                    messages=[{"role": "user", "content": prompt}],
                )
                return r.content[0].text.strip()
        except Exception as exc:
            logger.warning("MultiHop LLM call failed: %s", exc)
        return "DONE"
