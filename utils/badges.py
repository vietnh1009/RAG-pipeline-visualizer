import os
import sys
import re
from pathlib import Path

from core.constants import FILE_TYPE_COLORS
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

try:
    from dotenv import load_dotenv as _load_dotenv, dotenv_values as _dotenv_values
    _load_dotenv(override=True)
    _ENV = _dotenv_values()
except ImportError:
    _ENV = {}

import streamlit as st
def file_type_badge(file_type: str) -> str:
    """Tạo HTML badge màu cho loại file."""
    color = FILE_TYPE_COLORS.get(file_type, "#95a5a6")
    return (
        f'<span style="background:{color}; color:white; padding:2px 8px; '
        f'border-radius:10px; font-size:0.75rem; font-weight:bold;">'
        f'{file_type.upper()}</span>'
    )


def chunk_type_badge(idx: int, level: str = "") -> str:
    """Tạo badge màu cho chunk theo index."""
    colors = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12",
              "#9b59b6", "#1abc9c", "#e67e22", "#34495e"]
    color  = colors[idx % len(colors)]
    label  = f"Chunk {idx + 1}" + (f" [{level}]" if level else "")
    return (
        f'<span style="background:{color}; color:white; padding:2px 10px; '
        f'border-radius:10px; font-size:0.8rem; font-weight:bold;">{label}</span>'
    )


# Mapping lỗi import → hướng dẫn cài package
_IMPORT_ERROR_HINTS: dict[str, str] = {
    "unstructured_inference": "pip install unstructured-inference",
    "unstructured":           "pip install \'unstructured[pdf]\'",
    "fitz":                   "pip install pymupdf",
    "pdfplumber":             "pip install pdfplumber tabulate",
    "docling":                "pip install docling",
    "pytesseract":            "pip install pytesseract Pillow",
    "paddleocr":              "pip install paddlepaddle paddleocr",
    "ebooklib":               "pip install ebooklib beautifulsoup4",
}


def _friendly_import_error(exc: ModuleNotFoundError) -> str:
    """Chuyển ModuleNotFoundError thành thông báo hữu ích."""
    missing = exc.name or str(exc)
    for key, hint in _IMPORT_ERROR_HINTS.items():
        if key in missing:
            return (
                f"Thiếu package **`{missing}`**.\n\n"
                f"Cài bằng lệnh: `{hint}`"
            )
    return f"Thiếu module: `{missing}`. Chạy `pip install {missing}` để cài."



