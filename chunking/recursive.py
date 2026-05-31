"""
chunking/recursive.py
=====================
Recursive character splitting — the recommended default strategy.

Tries separators in priority order (paragraph → line → sentence → word →
character) so chunks always end at a natural boundary when possible.

Extra Markdown separators are included so heading lines (# / ##) and
code fences (```) act as preferred break points.

Use when : general-purpose default for plain text and Markdown documents.
"""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from chunking.base import BaseChunker

# Separator list ordered coarsest → finest
_SEPARATORS = [
    "\n#{1,6} ",   # Markdown headings
    "```\n",       # code fence
    "\n\n",        # paragraph break
    "\n",          # line break
    ". ",          # English sentence end
    "! ", "? ",
    "。", "，",    # CJK sentence / clause
    " ",           # word boundary
    "",            # character (last resort)
]


class RecursiveChunker(BaseChunker):
    """
    Recursively split text using a priority list of separators.

    Parameters
    ----------
    chunk_size    : Target chunk size in characters.
    chunk_overlap : Overlap in characters between consecutive chunks.
    """

    def split(self, docs: list[Document]) -> list[Document]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=_SEPARATORS,
            add_start_index=True,
            strip_whitespace=True,
        )
        return self._enrich(splitter.split_documents(docs))
