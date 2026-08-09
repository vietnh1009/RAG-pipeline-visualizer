"""
chunking/format_aware.py
========================
Cắt theo cấu trúc tài liệu để tìm ranh giới tự nhiên.

Ba nhánh con, chọn tự động hoặc chỉ định thẳng:

  markdown : Cắt theo heading Markdown (# / ## / ###); đường dẫn heading được
             lưu vào metadata để lọc về sau.
  code     : Cắt theo AST qua enum Language của LangChain; ranh giới luôn rơi
             giữa các định nghĩa ở mức ngoài cùng.
  html     : Cắt theo thẻ ngữ nghĩa <h1>…<h6>.

Cắt tiếp lần hai (split_large_sections)
---------------------------------------
Mặc định giữ nguyên ranh giới section: một section 5000 ký tự vẫn là một chunk,
bất kể chunk_size.

Bật ``split_large_sections=True`` để chạy thêm một lượt
RecursiveCharacterTextSplitter cho những section vượt chunk_size. Cần khi model
embedding có context ngắn (VinAI/phobert-large 256 token, mxbai-embed-large 512).

Dùng khi: tài liệu nguồn có markup rõ ràng — wiki, README, tài liệu API, mã
nguồn, trang HTML.
"""

from __future__ import annotations

from langchain_core.documents import Document

from chunking.base import BaseChunker
from chunking.recursive import RecursiveChunker

# Ánh xạ metadata programming_language -> tên enum Language của LangChain
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
    Cắt document theo định dạng cấu trúc của nó.

    Tham số
    -------
    format_type          : "auto" | "markdown" | "code" | "html". Chế độ "auto"
                           dựa vào metadata ``file_type`` và
                           ``programming_language`` để quyết định.
    chunk_size           : Số ký tự tối đa mỗi chunk ở nhánh code, đồng thời là
                           trần khi bật split_large_sections.
    chunk_overlap        : Số ký tự chồng lấn (nhánh code và lượt cắt thứ hai).
    split_large_sections : True thì section vượt chunk_size được cắt tiếp bằng
                           RecursiveCharacterTextSplitter. Mặc định False để giữ
                           nguyên ranh giới cấu trúc.
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
    # Nhận diện định dạng
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
    # Helper cho lượt cắt thứ hai
    # ------------------------------------------------------------------
    def _maybe_split_large(
        self,
        sections: list[Document],
        extra_meta: dict,
    ) -> list[Document]:
        """
        Cắt tiếp các section quá lớn, tuỳ theo ``split_large_sections``.

        False (mặc định) : giữ nguyên từng section bất kể kích thước. Hợp khi
            model embedding có context lớn (bge-m3 8192, Qwen3 32K) hoặc khi
            muốn chunk khớp đúng cấu trúc heading.
        True : section vượt chunk_size được chia nhỏ tiếp bằng
            RecursiveCharacterTextSplitter. Dùng khi model có context ngắn
            (phobert 256, mxbai 512 token).
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
    # Code — cắt theo AST qua enum Language của LangChain
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
