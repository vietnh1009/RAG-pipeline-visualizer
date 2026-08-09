"""
loader/
=======
Package load tài liệu PDF cho RAG indexing pipeline.

Public API
----------
    from loader import PDFDocumentLoader

    loader = PDFDocumentLoader(language="both", pdf_strategy="marker")
    docs   = loader.load(file_path)

Các loader riêng lẻ có thể import trực tiếp:

    from loader.pdf_loader import MarkerPDFLoader, DoclingPDFLoader, PyPDFLoader
"""

from loader.pdf_loader import MARKER_CACHE_DIR, DOCLING_CACHE_DIR
from loader.directory_loader import PDFDocumentLoader

__all__ = ["PDFDocumentLoader", "MARKER_CACHE_DIR", "DOCLING_CACHE_DIR"]
