"""
chunking/base.py
================
Abstract base class dùng chung cho tất cả chunking strategy.

Giao kèo của mọi chunker:
    chunker = SomeChunker(chunk_size=1000, chunk_overlap=150, **options)
    chunks  = chunker.split(docs)   -> list[Document]
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from langchain_core.documents import Document


class BaseChunker(ABC):
    """
    Abstract base for all chunking strategies.

    Parameters
    ----------
    chunk_size    : Target chunk size (characters, tokens, or sentences
                    depending on the strategy).
    chunk_overlap : Amount of overlap between consecutive chunks
                    (same unit as chunk_size).
    """

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 150):
        self.chunk_size    = chunk_size
        self.chunk_overlap = chunk_overlap

    @abstractmethod
    def split(self, docs: list[Document]) -> list[Document]:
        """
        Split a list of Documents into smaller chunks.

        Parameters
        ----------
        docs : Source documents (output of the loading stage).

        Returns
        -------
        List of chunk Documents.  Each chunk inherits the metadata
        of its source document, plus ``chunk_index`` and ``char_count``.
        """

    @staticmethod
    def _enrich(chunks: list[Document]) -> list[Document]:
        """Stamp sequential index and character count onto every chunk."""
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = i
            chunk.metadata["char_count"]  = len(chunk.page_content)
        return chunks
