"""
chunking/format_aware.py
========================
Format-aware chunking -- uses document structure to find natural boundaries.

Three sub-strategies selected automatically (or explicitly):

  markdown : Splits on Markdown headings (# / ## / ###).
             Heading path stored in metadata for rich filtering.

  code     : AST-based splitting (LangChain Language enum).
             Boundaries always fall between top-level definitions.

  html     : Splits on semantic HTML tags (<h1> to <h6>).
             Boilerplate tags already stripped by HtmlLoader.

Secondary splitting (split_large_sections)
------------------------------------------
By default, format_aware preserves section boundaries exactly -- a section
of 5000 chars stays as one chunk regardless of chunk_size.

Set split_large_sections=True to add a second pass with
RecursiveCharacterTextSplitter for sections that exceed chunk_size.
Use this when your embedding model has a short context window
(e.g. VinAI/phobert-large: 256 tokens, mxbai-embed-large: 512 tokens).

Use when : source documents have clear structural markup (wikis, READMEs,
           API docs, source code repositories, HTML pages).
"""

from __future__ import annotations

from langchain_core.documents import Document

from chunking.base import BaseChunker
from chunking.recursive import RecursiveChunker

# Maps programming_language metadata value -> LangChain Language enum value
_LANG_ENUM_MAP = {
    "python":     "PYTHON",
    "javascript": "JS",
    "typescript": "TS",
    "java":       "JAVA",
    "cpp":        "CPP",
    "c":          "C",
    "go":         "GO",
    "rust":       "RUST",
    "ruby":       "RUBY",
    "sql":        "SQL",
}


class FormatAwareChunker(BaseChunker):
    """
    Split documents according to their structural format.

    Parameters
    ----------
    format_type          : "auto" | "markdown" | "code" | "html"
                           "auto" inspects metadata["file_type"] and
                           metadata["programming_language"] to decide.
    chunk_size           : Max characters per chunk for code sub-strategy,
                           and upper bound when split_large_sections=True.
    chunk_overlap        : Overlap in characters (code + secondary splitting).
    split_large_sections : If True, sections exceeding chunk_size are split
                           further with RecursiveCharacterTextSplitter.
                           Default False -- preserves structural boundaries.
    """

    def __init__(
        self,
        format_type:          str  = "auto",
        chunk_size:           int  = 1000,
        chunk_overlap:        int  = 100,
        split_large_sections: bool = False,
    ):
        super().__init__(chunk_size, chunk_overlap)
        self.format_type          = format_type
        self.split_large_sections = split_large_sections

    def split(self, docs: list[Document]) -> list[Document]:
        all_chunks: list[Document] = []
        for doc in docs:
            fmt = self._detect(doc)
            if fmt == "markdown":
                chunks = self._split_markdown(doc)
            elif fmt == "code":
                chunks = self._split_code(doc)
            elif fmt == "html":
                chunks = self._split_html(doc)
            else:
                chunks = RecursiveChunker(self.chunk_size, self.chunk_overlap).split([doc])
            all_chunks.extend(chunks)
        return self._enrich(all_chunks)

    # ------------------------------------------------------------------
    # Format detection
    # ------------------------------------------------------------------
    def _detect(self, doc: Document) -> str:
        if self.format_type != "auto":
            return self.format_type
        ft = doc.metadata.get("file_type", "")
        if ft in ("markdown", "md"):
            return "markdown"
        if ft == "html":
            return "html"
        if ft == "code" or doc.metadata.get("programming_language"):
            return "code"
        if doc.page_content.lstrip().startswith("#"):
            return "markdown"
        return "text"

    # ------------------------------------------------------------------
    # Secondary splitting helper
    # ------------------------------------------------------------------
    def _maybe_split_large(
        self,
        sections: list[Document],
        extra_meta: dict,
    ) -> list[Document]:
        """
        split_large_sections=False (default):
            Giu nguyen tung section, bat ke kich thuoc.
            Phu hop khi embedding model co ctx window lon (bge-m3: 8192,
            Qwen3: 32K) hoac khi muon chunk khop chinh xac voi cau truc heading.

        split_large_sections=True:
            Section vuot chunk_size bi chia nho them voi RecursiveCharacterTextSplitter.
            Dung khi model co ctx window ngan (phobert: 256, mxbai: 512 tokens).
        """
        if not self.split_large_sections:
            return [
                Document(
                    page_content=s.page_content.strip(),
                    metadata={**s.metadata, **extra_meta},
                )
                for s in sections
                if s.page_content.strip()
            ]

        from langchain_text_splitters import RecursiveCharacterTextSplitter
        char_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        chunks: list[Document] = []
        for s in sections:
            if not s.page_content.strip():
                continue
            base_meta = {**s.metadata, **extra_meta}
            if len(s.page_content) <= self.chunk_size:
                chunks.append(Document(
                    page_content=s.page_content.strip(),
                    metadata=base_meta,
                ))
            else:
                chunks.extend(char_splitter.split_documents(
                    [Document(page_content=s.page_content, metadata=base_meta)]
                ))
        return chunks

    # ------------------------------------------------------------------
    # Markdown
    # ------------------------------------------------------------------
    def _split_markdown(self, doc: Document) -> list[Document]:
        from langchain_text_splitters import MarkdownHeaderTextSplitter

        headers = [("#", "h1"), ("##", "h2"), ("###", "h3"), ("####", "h4")]
        splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers, strip_headers=False
        )
        sections = splitter.split_text(doc.page_content)
        for s in sections:
            s.metadata = {**doc.metadata, **s.metadata}
        return self._maybe_split_large(sections, {"format": "markdown"})

    # ------------------------------------------------------------------
    # Code (AST-based via LangChain Language enum)
    # ------------------------------------------------------------------
    def _split_code(self, doc: Document) -> list[Document]:
        from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

        prog_lang   = doc.metadata.get("programming_language", "python")
        enum_name   = _LANG_ENUM_MAP.get(prog_lang, "PYTHON")
        lc_language = Language[enum_name]

        splitter = RecursiveCharacterTextSplitter.from_language(
            language=lc_language,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        return splitter.split_documents([doc])

    # ------------------------------------------------------------------
    # HTML
    # ------------------------------------------------------------------
    def _split_html(self, doc: Document) -> list[Document]:
        from langchain_text_splitters import HTMLHeaderTextSplitter

        headers  = [("h1", "h1"), ("h2", "h2"), ("h3", "h3")]
        splitter = HTMLHeaderTextSplitter(headers_to_split_on=headers)
        try:
            sections = splitter.split_text(doc.page_content)
        except Exception:
            return RecursiveChunker(self.chunk_size, self.chunk_overlap).split([doc])

        for s in sections:
            s.metadata = {**doc.metadata, **s.metadata}
        return self._maybe_split_large(sections, {"format": "html"})
