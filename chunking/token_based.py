"""
chunking/token_based.py
=======================
Token-based chunking — counts tokens instead of characters.

Essential for multilingual corpora (especially Vietnamese) because
characters-per-token ratios vary significantly across languages.
A chunk of 500 Vietnamese characters may be 600-700 tokens with BPE
tokenisers, silently exceeding embedding model limits.

Use when : multilingual pipelines, strict token-limit compliance,
           production systems where silent truncation is unacceptable.
"""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_text_splitters import TokenTextSplitter

from chunking.base import BaseChunker


class TokenChunker(BaseChunker):
    """
    Split documents by token count using tiktoken.

    Parameters
    ----------
    chunk_size     : Number of tokens per chunk.
    chunk_overlap  : Overlap in tokens between consecutive chunks.
    encoding_name  : tiktoken encoding to use.
                     "cl100k_base" for OpenAI models (GPT-4, text-embedding-3-*).
                     "p50k_base"   for older GPT-3 models.
    """

    def __init__(
        self,
        chunk_size:    int = 512,
        chunk_overlap: int = 64,
        encoding_name: str = "cl100k_base",
    ):
        super().__init__(chunk_size, chunk_overlap)
        self.encoding_name = encoding_name

    def split(self, docs: list[Document]) -> list[Document]:
        splitter = TokenTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            encoding_name=self.encoding_name,
        )
        return self._enrich(splitter.split_documents(docs))
