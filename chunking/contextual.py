"""
chunking/contextual.py
======================
Contextual Chunking (Anthropic, 09/2024).

Vấn đề: chunk tách khỏi tài liệu là mất ngữ cảnh xung quanh. Một chunk mồ côi
kiểu "Tỷ lệ hiện mắc ước tính khoảng 25 %." không cho biết đang nói về bệnh gì.

Cách làm: với mỗi chunk do chunker nền sinh ra, LLM đọc toàn bộ tài liệu và viết
1–2 câu ngữ cảnh, gắn lên đầu chunk trước khi đem embed.

Hiệu quả công bố: kết hợp Contextual Retrieval + BM25 giảm tới 67 % tỉ lệ truy
hồi hỏng so với chunking thường (Anthropic, 2024).

Mẹo tiết kiệm: bật prompt caching cho phần tài liệu, chi phí giảm khoảng 80 %.

Dùng khi: chunk ngắn, thuật ngữ chuyên ngành, nhiều đại từ tham chiếu, và recall
quan trọng hơn chi phí lúc nạp dữ liệu.
"""

from __future__ import annotations

from langchain_core.documents import Document

from chunking.base import BaseChunker
from utils.llm import call_llm


class ContextualChunker(BaseChunker):
    """
    Gắn tiền tố ngữ cảnh do LLM sinh vào đầu mỗi chunk.

    Tham số
    -------
    base_strategy : Chiến lược chunking chạy trước; nhận mọi key trong registry
                    của factory, ví dụ "recursive", "sentence_aware".
    base_kwargs   : Tham số khởi tạo chuyển cho chunker nền.
    llm_model     : Nên chọn model nhanh và rẻ, vd claude-haiku-4-5-20251001.
    llm_provider  : "anthropic" | "openai" | "google"
    n_sentences   : Số câu ngữ cảnh cần gắn thêm.
    """

    _PROMPT = (
        "<document>\n{document}\n</document>\n\n"
        "Here is the chunk to situate within the document:\n"
        "<chunk>\n{chunk}\n</chunk>\n\n"
        "Write {n} concise sentence(s) placing this chunk in the context "
        "of the full document to improve retrieval. "
        "Output ONLY those sentences, nothing else."
    )

    def __init__(
        self,
        base_strategy: str  = "recursive",
        base_kwargs:   dict | None = None,
        llm_model:     str  = "claude-haiku-4-5-20251001",
        llm_provider:  str  = "anthropic",
        n_sentences:   int  = 2,
        chunk_size:    int  = 1000,
        chunk_overlap: int  = 150,
    ):
        super().__init__(chunk_size, chunk_overlap)
        self.base_strategy = base_strategy
        self.base_kwargs   = base_kwargs or {}
        self.llm_model     = llm_model
        self.llm_provider  = llm_provider
        self.n_sentences   = n_sentences

    def split(self, docs: list[Document]) -> list[Document]:
        from chunking.factory import get_chunker

        base_chunker = get_chunker(
            self.base_strategy,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            **self.base_kwargs,
        )
        base_chunks = base_chunker.split(docs)

        # Bảng tra source -> toàn văn, dùng khi sinh ngữ cảnh
        source_text: dict[str, str] = {}
        for doc in docs:
            src = doc.metadata.get("source", "")
            source_text.setdefault(src, "")
            source_text[src] += doc.page_content + "\n\n"

        enriched: list[Document] = []
        for chunk in base_chunks:
            src      = chunk.metadata.get("source", "")
            full_doc = source_text.get(src, chunk.page_content)
            context  = self._generate_context(full_doc, chunk.page_content)
            content  = f"{context}\n\n{chunk.page_content}" if context else chunk.page_content

            enriched.append(Document(
                page_content=content,
                metadata={
                    **chunk.metadata,
                    "chunking_strategy": "contextual",
                    "context_prefix":    context,
                },
            ))
        return self._enrich(enriched)

    def _generate_context(self, document: str, chunk: str) -> str:
        prompt = self._PROMPT.format(
            document=document[:8000],
            chunk=chunk,
            n=self.n_sentences,
        )
        return call_llm(prompt, self.llm_provider, self.llm_model, max_tokens=200)
