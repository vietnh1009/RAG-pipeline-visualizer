"""
chunking/sentence_aware.py
==========================
Sentence-aware chunking — chunk boundaries always fall at sentence ends.

Sentence detection priority:
    1. underthesea  (best for Vietnamese)
    2. NLTK punkt   (good multilingual fallback)
    3. Regex        (no-dependency fallback)

Use when : FAQ datasets, Q&A pairs, short narrative text,
           any content where sentence integrity is critical.
"""

from __future__ import annotations

import re

from langchain_core.documents import Document

from chunking.base import BaseChunker


def _split_sentences(text: str) -> list[str]:
    """Detect sentence boundaries with the best available library."""
    # 1. underthesea — best for Vietnamese
    try:
        from underthesea import sent_tokenize
        return [s.strip() for s in sent_tokenize(text) if s.strip()]
    except ImportError:
        pass

    # 2. NLTK punkt
    try:
        import nltk
        try:
            return [s.strip() for s in nltk.sent_tokenize(text) if s.strip()]
        except LookupError:
            nltk.download("punkt_tab", quiet=True)
            return [s.strip() for s in nltk.sent_tokenize(text) if s.strip()]
    except ImportError:
        pass

    # 3. Regex fallback
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


class SentenceChunker(BaseChunker):
    """
    Pack consecutive sentences into chunks that stay within ``chunk_size``
    characters, with an optional sentence-level overlap.

    Parameters
    ----------
    chunk_size          : Approximate target size in characters per chunk.
    chunk_overlap       : Characters of overlap (kept as whole sentences).
    sentences_per_chunk : If set, use a fixed sentence count instead of
                          the character budget.
    """

    def __init__(
        self,
        chunk_size:          int = 1000,
        chunk_overlap:       int = 0,
        sentences_per_chunk: int | None = None,
    ):
        super().__init__(chunk_size, chunk_overlap)
        self.sentences_per_chunk = sentences_per_chunk

    def split(self, docs: list[Document]) -> list[Document]:
        all_chunks: list[Document] = []
        for doc in docs:
            sentences = _split_sentences(doc.page_content)
            groups    = self._group(sentences)
            for group in groups:
                content = " ".join(group).strip()
                if content:
                    all_chunks.append(Document(
                        page_content=content,
                        metadata={**doc.metadata},
                    ))
        return self._enrich(all_chunks)

    def _group(self, sentences: list[str]) -> list[list[str]]:
        """Pack sentences into groups respecting chunk_size."""
        if self.sentences_per_chunk:
            step = max(1, self.sentences_per_chunk - self.chunk_overlap)
            return [
                sentences[i: i + self.sentences_per_chunk]
                for i in range(0, len(sentences), step)
            ]

        groups:  list[list[str]] = []
        current: list[str]       = []
        current_len = 0

        for sentence in sentences:
            s_len = len(sentence)
            if current_len + s_len > self.chunk_size and current:
                groups.append(current)
                # Build overlap from the tail of the current group
                overlap:     list[str] = []
                overlap_len: int       = 0
                for s in reversed(current):
                    if overlap_len + len(s) <= self.chunk_overlap:
                        overlap.insert(0, s)
                        overlap_len += len(s)
                    else:
                        break
                current     = overlap
                current_len = overlap_len
            current.append(sentence)
            current_len += s_len

        if current:
            groups.append(current)
        return groups
