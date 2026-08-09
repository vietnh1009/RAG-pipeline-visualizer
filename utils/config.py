"""
utils/config.py
===============
Đọc config.yaml.

Trước đây hàm ``load_config()`` được nhắc đến trong docstring của 5 factory
(``embedding``, ``retrieval``, ``post_retrieval``, ``generation``, ``vector_db``)
nhưng KHÔNG tồn tại trong repo — nghĩa là toàn bộ nhánh ``*_from_config()``
chưa bao giờ dùng được. Đây là phần bổ sung.

Lưu ý: app Streamlit KHÔNG đọc config.yaml — nó lấy cấu hình từ các panel
sidebar. config.yaml chỉ dùng cho đường chạy headless/CLI:

    python -m pipeline.indexing_pipeline --source ./data --config config.yaml
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def _expand_env(value: Any) -> Any:
    """Thay ${VAR} và ${VAR:-default} bằng giá trị biến môi trường."""
    if isinstance(value, str):
        def _sub(m: re.Match) -> str:
            return os.environ.get(m.group(1), m.group(2) or "")
        return _ENV_PATTERN.sub(_sub, value)
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    return value


def load_config(path: str | Path = "config.yaml") -> dict:
    """
    Đọc file YAML, mở rộng biến môi trường, trả về dict.

    Ném FileNotFoundError nếu không tìm thấy file — cố tình không im lặng
    trả về {} , vì chạy pipeline với config rỗng sẽ sinh lỗi khó hiểu hơn nhiều.
    """
    p = Path(path)
    if not p.is_absolute():
        p = Path(__file__).resolve().parent.parent / p
    if not p.exists():
        raise FileNotFoundError(
            f"Không tìm thấy config: {p}\n"
            f"Tạo file từ config.yaml mẫu ở thư mục gốc project."
        )

    try:
        import yaml
    except ImportError as exc:
        raise ImportError(
            "Thiếu PyYAML. Cài bằng: pip install PyYAML==6.0.2"
        ) from exc

    with p.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}

    if not isinstance(cfg, dict):
        raise ValueError(f"config phải là mapping ở cấp cao nhất, nhận: {type(cfg).__name__}")

    return _expand_env(cfg)
