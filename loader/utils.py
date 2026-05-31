"""
loader/utils.py
===============
Hàm tiện ích dùng chung cho các loader module.
"""

from __future__ import annotations

import hashlib


def clean_text(text: str) -> str:
    """
    Chuẩn hoá văn bản sau khi trích xuất:
    xoá null byte, trim trailing whitespace từng dòng,
    thu gọn ≥3 dòng trắng xuống còn 2, strip toàn bộ.
    """
    text  = text.replace("\x00", "")
    lines = [line.rstrip() for line in text.splitlines()]

    cleaned: list[str] = []
    blank_run = 0
    for line in lines:
        if line == "":
            blank_run += 1
            if blank_run <= 2:
                cleaned.append(line)
        else:
            blank_run = 0
            cleaned.append(line)

    return "\n".join(cleaned).strip()


def content_hash(text: str) -> str:
    """MD5 fingerprint của chuỗi text — dùng cho deduplication chính xác."""
    return hashlib.md5(text.encode("utf-8", errors="replace")).hexdigest()


def table_html_to_markdown(html: str) -> str:
    """
    Chuyển HTML table sang Markdown qua pandas.
    Trả về raw HTML nếu chuyển đổi thất bại.
    """
    import pandas as pd
    from io import StringIO
    try:
        df = pd.read_html(StringIO(html))[0]
        return df.to_markdown(index=False)
    except Exception:
        return html
