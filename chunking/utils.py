"""
chunking/utils.py
=================
Hàm tiện ích riêng cho các chunking module.

``call_llm`` / ``parse_json_list`` đã chuyển sang ``utils/llm.py`` (dùng chung
cho mọi bước). Module này chỉ còn tiện ích đặc thù của chunking.
"""

from __future__ import annotations


def ensure_ollama_model(model: str, host: str = "http://localhost:11434") -> None:
    """
    Kiểm tra model Ollama đã có chưa, chưa có thì tự pull về.

    Dùng chung cho VLM (MarkerPDFLoader) và embedding (SemanticChunker).
    Thứ tự pull: SDK Python của ollama (hiện tiến độ đẹp hơn), không được thì
    lùi về subprocess ``ollama pull``.

    Tham số
    -------
    model : Tag model Ollama, vd "qwen3-embedding:8b".
    host  : URL Ollama dạng http://host:port, KHÔNG có hậu tố /v1.
    """
    # Chuẩn hoá: bỏ /v1 nếu có (app dùng OpenAI-compat URL, Ollama SDK dùng native URL)
    host = host.rstrip("/")
    if host.endswith("/v1"):
        host = host[:-3]

    # ── Bước 1: hỏi SDK xem model đã có chưa (nếu SDK đã cài) ────────────────
    already_present = False
    sdk_available   = False

    try:
        import ollama as _ollama
        sdk_available = True
        client  = _ollama.Client(host=host)
        models  = [m.model for m in client.list().models]
        # Chỉ khớp tuyệt đối. Khớp theo tên không kèm tag ("qwen3-embedding"
        # trùng cả :0.6b lẫn :8b) là quá lỏng: đã có qwen3-embedding:8b KHÔNG
        # được cản việc pull qwen3-embedding:0.6b khi cần đúng bản đó.
        if model in set(models):
            already_present = True
    except ImportError:
        pass   # SDK chưa cài — sẽ dùng subprocess
    except Exception as e:
        # Ollama server không phản hồi hoặc lỗi SDK — thử subprocess
        print(f"[Ollama] Không kiểm tra được model list qua SDK ({e}), thử subprocess...")

    if already_present:
        return

    # ── Bước 2: pull model ────────────────────────────────────────────────────
    print(f"[Ollama] Model '{model}' chưa có sẵn — đang pull về...")

    if sdk_available:
        try:
            import ollama as _ollama
            client = _ollama.Client(host=host)
            for progress in client.pull(model, stream=True):
                status    = getattr(progress, "status", "")
                completed = getattr(progress, "completed", None)
                total     = getattr(progress, "total", None)
                if total and completed:
                    pct = completed / total * 100
                    print(f"\r[Ollama] {status} {pct:.1f}%", end="", flush=True)
                elif status:
                    print(f"[Ollama] {status}", flush=True)
            print()
            print(f"[Ollama] Pull '{model}' hoàn tất.")
            return   # pull thành công qua SDK
        except Exception as e:
            # SDK pull thất bại — fallthrough sang subprocess
            print(f"\n[Ollama] SDK pull thất bại ({e}), thử lại bằng subprocess...")

    # ── Bước 3: phương án dự phòng bằng subprocess ────────────────────────────
    import subprocess
    print(f"[Ollama] Pulling '{model}' via subprocess (ollama pull)...")
    try:
        result = subprocess.run(
            ["ollama", "pull", model],
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"ollama pull '{model}' thất bại (exit {result.returncode}).\n"
                "Đảm bảo Ollama đang chạy: ollama serve"
            )
        print(f"[Ollama] Pull '{model}' hoàn tất (subprocess).")
    except FileNotFoundError:
        raise RuntimeError(
            f"Không tìm thấy lệnh 'ollama' trong PATH.\n\n"
            f"Model '{model}' chưa được pull và không thể tự động pull.\n\n"
            f"Giải pháp:\n"
            f"  1. Mở terminal và chạy: ollama pull {model}\n"
            f"  2. Hoặc cài ollama Python SDK: pip install ollama"
        )


# ── Cache pipeline HuggingFace chạy cục bộ ───────────────────────────────────
_HF_PIPELINE_CACHE: dict[str, object] = {}
