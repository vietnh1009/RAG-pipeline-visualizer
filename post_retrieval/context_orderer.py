"""
post_retrieval/context_orderer.py
===================================
Sắp lại thứ tự chunk để giảm hiệu ứng "lost in the middle".

Liu et al. (2023) chỉ ra LLM chú ý mạnh nhất vào phần *đầu* và *cuối* context;
thông tin nằm giữa hay bị bỏ qua khi sinh câu trả lời.

Bốn chiến lược
--------------
relevance : Điểm liên quan giảm dần — liên quan nhất lên đầu.
reverse   : Điểm tăng dần — liên quan nhất xuống cuối, tận dụng recency bias.
sandwich  : Liên quan nhất ở đầu VÀ cuối, kém nhất nằm giữa. ← mặc định khuyến nghị.
original  : Giữ nguyên thứ tự retriever trả về.
"""

from __future__ import annotations

from langchain_core.documents import Document

from post_retrieval.base import BasePostProcessor


class ContextOrderer(BasePostProcessor):
    """
    Sắp lại chunk đã truy hồi để giảm suy giảm do "lost in the middle".

    Tham số
    -------
    ordering  : "relevance" | "reverse" | "sandwich" | "original"
    score_key : Key metadata chứa điểm liên quan. Nếu thiếu thì lần lượt thử
                relevance_score → rrf_score → hybrid_score.
    """

    def __init__(
        self,
        ordering:  str = "sandwich",
        score_key: str = "rerank_score",
    ):
        self.ordering  = ordering
        self.score_key = score_key

    def process(self, query: str, docs: list[Document]) -> list[Document]:
        if not docs or self.ordering == "original":
            return docs

        def get_score(doc: Document) -> float:
            for key in (self.score_key, "relevance_score", "rrf_score", "hybrid_score"):
                v = doc.metadata.get(key)
                if v is not None:
                    return float(v)
            return 0.0

        if self.ordering == "relevance":
            return sorted(docs, key=get_score, reverse=True)

        if self.ordering == "reverse":
            return sorted(docs, key=get_score)

        if self.ordering == "sandwich":
            sorted_docs = sorted(docs, key=get_score, reverse=True)
            n = len(sorted_docs)
            if n <= 2:
                return sorted_docs
            # Chunk tốt nhất lên đầu, kém nhất xuống cuối, còn lại kẹp ở giữa
            best  = sorted_docs[0]
            worst = sorted_docs[-1]
            mid   = sorted_docs[1:-1]
            half  = len(mid) // 2
            return [best] + mid[:half] + list(reversed(mid[half:])) + [worst]

        return docs
