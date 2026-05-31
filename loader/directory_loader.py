"""
loader/directory_loader.py
==========================
PDFDocumentLoader — điểm vào duy nhất cho bước loading (chỉ PDF).

Nhận đường dẫn đến 1 file PDF hoặc 1 thư mục, dispatch đến đúng PDF
loader class, và trả về list[Document] sẵn sàng cho bước chunking.

PDF strategies
--------------
  pypdf          Nhanh, text layer, không cần dep thêm          (mặc định)
  pymupdf        Nhanh hơn, layout tốt hơn
  pdfplumber     Trích bảng tốt nhất (text layer)
  unstructured   OCR + bảng + hình
  docling        IBM parser, Markdown output xuất sắc
  marker         Markdown chất lượng cao, bảng & công thức
  opendataloader #1 benchmark (0.90), bounding box, không cần GPU   # Java 11+
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal, Optional

from langchain_core.documents import Document

from loader.base import BaseLoader, Language
from loader.utils import content_hash

logger = logging.getLogger(__name__)

PDFStrategy = Literal[
    "pypdf", "pymupdf", "pdfplumber", "unstructured", "docling", "marker",
    "opendataloader",
]


def _build_pdf_loader(
    language:        Language,
    pdf_strategy:    PDFStrategy,
    extract_tables:  bool,
    extract_images:  bool,
    vision_model:    str,
    marker_device:   str        = "cpu",
    describe_images: bool       = False,
    vision_provider: str        = "openai",
    ollama_base_url: str        = "http://localhost:11434/v1",
    odl_hybrid:      str | None = None,
    odl_struct_tree: bool       = False,
) -> BaseLoader:
    """Khởi tạo đúng PDF loader theo pdf_strategy."""
    if pdf_strategy == "pypdf":
        from loader.pdf_loader import PyPDFLoader
        return PyPDFLoader(language)

    if pdf_strategy == "pymupdf":
        from loader.pdf_loader import PyMuPDFLoader
        return PyMuPDFLoader(language)

    if pdf_strategy == "pdfplumber":
        from loader.pdf_loader import PDFPlumberLoader
        return PDFPlumberLoader(language, extract_tables=extract_tables)

    if pdf_strategy == "docling":
        from loader.pdf_loader import DoclingPDFLoader
        return DoclingPDFLoader(language, extract_tables=extract_tables,
                                extract_images=extract_images)

    if pdf_strategy == "marker":
        from loader.pdf_loader import MarkerPDFLoader
        return MarkerPDFLoader(
            language,
            device=marker_device,
            describe_images=describe_images,
            vision_model=vision_model,
            vision_provider=vision_provider,
            ollama_base_url=ollama_base_url,
        )

    if pdf_strategy == "opendataloader":
        from loader.pdf_loader import OpenDataLoaderPDFLoader
        return OpenDataLoaderPDFLoader(
            language,
            hybrid=odl_hybrid,
            use_struct_tree=odl_struct_tree,
        )

    # unstructured (fallback)
    from loader.pdf_loader import UnstructuredPDFLoader
    return UnstructuredPDFLoader(
        language=language,
        extract_tables=extract_tables,
        extract_images=extract_images,
        vision_model=vision_model,
        vision_provider=vision_provider,
        ollama_base_url=ollama_base_url,
    )


class PDFDocumentLoader:
    """
    Điểm vào duy nhất để load file PDF.

    Nhận đường dẫn đến 1 file PDF hoặc 1 thư mục chứa PDF, dispatch đến
    PDF loader class tương ứng, và trả về list[Document] cho bước chunking.

    Tham số
    -------
    language        : Ngôn ngữ corpus (\"vi\" | \"en\" | \"both\").
    pdf_strategy    : Backend parse PDF.
    extract_tables  : Chuyển bảng thành Markdown (pdfplumber / docling).
    extract_images  : Dùng VLM mô tả hình ảnh trong PDF.
    vision_model    : Model VLM để mô tả hình.
    vision_provider : Provider của VLM (\"openai\" | \"ollama\").
    deduplicate     : Bỏ Document trùng nội dung (MD5 hash).
    marker_device   : \"cpu\" | \"cuda\" | \"mps\" — cho Marker loader.
    describe_images : Bật mô tả ảnh qua VLM cho Marker loader.
    ollama_base_url : URL Ollama server (dùng khi vision_provider=\"ollama\").
    odl_hybrid      : Hybrid mode cho OpenDataLoader.
    odl_struct_tree : Dùng struct tree cho OpenDataLoader.

    Ví dụ
    -----
    >>> loader = PDFDocumentLoader(pdf_strategy="marker")
    >>> docs = loader.load("report.pdf")
    >>> docs = loader.load("./data/")   # tất cả PDF trong thư mục
    """

    def __init__(
        self,
        language:        Language    = "both",
        pdf_strategy:    PDFStrategy = "pypdf",
        extract_tables:  bool        = True,
        extract_images:  bool        = False,
        vision_model:    str         = "gpt-4o-mini",
        deduplicate:     bool        = True,
        marker_device:   str         = "cpu",
        describe_images: bool        = False,
        vision_provider: str         = "openai",
        ollama_base_url: str         = "http://localhost:11434/v1",
        odl_hybrid:      str | None  = None,
        odl_struct_tree: bool        = False,
    ):
        self.deduplicate = deduplicate
        self._seen: set[str] = set()
        self._loader: BaseLoader = _build_pdf_loader(
            language        = language,
            pdf_strategy    = pdf_strategy,
            extract_tables  = extract_tables,
            extract_images  = extract_images,
            vision_model    = vision_model,
            marker_device   = marker_device,
            describe_images = describe_images,
            vision_provider = vision_provider,
            ollama_base_url = ollama_base_url,
            odl_hybrid      = odl_hybrid,
            odl_struct_tree = odl_struct_tree,
        )

    def load(self, path: str) -> list[Document]:
        """
        Load 1 file PDF hoặc toàn bộ PDF trong một thư mục.

        Tham số
        -------
        path : Đường dẫn đến file .pdf hoặc thư mục.

        Trả về
        ------
        list[Document] — đã deduplicate nếu self.deduplicate=True.
        """
        p = Path(path)
        if p.is_file():
            return self._load_file(p)
        if p.is_dir():
            return self._load_directory(p)
        raise ValueError(f"'{path}' không phải file .pdf hoặc thư mục hợp lệ.")

    # ── private ───────────────────────────────────────────────────────────────

    def _load_file(self, path: Path) -> list[Document]:
        if path.suffix.lower() != ".pdf":
            logger.warning("Bỏ qua '%s': không phải file PDF.", path.name)
            return []
        logger.info("Loading: %s", path.name)
        docs = self._loader.load(str(path))
        return self._dedup(docs) if self.deduplicate else docs

    def _load_directory(self, root: Path) -> list[Document]:
        pdf_files = sorted(root.rglob("*.pdf"))
        logger.info(
            "PDFDocumentLoader: tìm thấy %d file PDF trong '%s'.",
            len(pdf_files), root,
        )
        all_docs: list[Document] = []
        for pdf in pdf_files:
            all_docs.extend(self._load_file(pdf))
        logger.info("PDFDocumentLoader: tổng cộng %d documents.", len(all_docs))
        return all_docs

    def _dedup(self, docs: list[Document]) -> list[Document]:
        unique: list[Document] = []
        for doc in docs:
            h = content_hash(doc.page_content)
            if h not in self._seen:
                self._seen.add(h)
                unique.append(doc)
        return unique
