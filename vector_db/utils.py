"""
vector_db/utils.py
==================
Vân tay corpus — giúp store cục bộ biết có phải index lại hay không.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def corpus_fingerprint(chunks) -> str:
    """
    Tính hash MD5 nhận diện duy nhất một tập chunk.

    Các store cục bộ (FAISS, Chroma, LanceDB) dùng để biết corpus có đổi so với
    lần index trước không. Hash trên nội dung text + đường dẫn nguồn của mỗi chunk.
    """
    hasher = hashlib.md5()
    for chunk in chunks:
        hasher.update(chunk.page_content.encode("utf-8", errors="replace"))
        src = str(chunk.metadata.get("source", ""))
        hasher.update(src.encode("utf-8", errors="replace"))
    return hasher.hexdigest()


def save_fingerprint(fingerprint: str, path: str) -> None:
    """Ghi vân tay corpus xuống đĩa."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"fingerprint": fingerprint}))


def load_fingerprint(path: str) -> str | None:
    """Đọc vân tay đã lưu; không có thì trả về None."""
    try:
        return json.loads(Path(path).read_text()).get("fingerprint")
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def corpus_changed(chunks, fingerprint_path: str) -> bool:
    """
    True nếu corpus đã đổi so với lần index trước.

    So vân tay MD5 hiện tại với bản lưu trên đĩa. Chưa có file vân tay thì trả
    về True, tức là cần index lại.
    """
    current  = corpus_fingerprint(chunks)
    previous = load_fingerprint(fingerprint_path)
    changed  = current != previous
    if not changed:
        import logging
        logging.getLogger(__name__).info(
            "Corpus unchanged — loading existing index."
        )
    return changed
