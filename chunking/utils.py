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
    Kiểm tra Ollama model đã có sẵn chưa, nếu chưa thì tự động pull về.
    Dùng chung cho VLM (MarkerPDFLoader) và embedding (SemanticChunker).

    Pull strategy (theo thứ tự ưu tiên):
      1. ollama Python SDK  — stream progress đẹp hơn
      2. subprocess         — fallback khi SDK chưa cài hoặc pull qua SDK thất bại

    Parameters
    ----------
    model : Ollama model tag, e.g. "qwen3-embedding:8b"
    host  : Ollama host URL — dạng http://host:port (KHÔNG có /v1 suffix)
    """
    # Chuẩn hoá: bỏ /v1 nếu có (app dùng OpenAI-compat URL, Ollama SDK dùng native URL)
    host = host.rstrip("/")
    if host.endswith("/v1"):
        host = host[:-3]

    # ── Bước 1: kiểm tra model đã có chưa qua SDK (nếu cài) ──────────────────
    already_present = False
    sdk_available   = False

    try:
        import ollama as _ollama
        sdk_available = True
        client  = _ollama.Client(host=host)
        models  = [m.model for m in client.list().models]
        # Exact match only — tag-based match (e.g. "qwen3-embedding" matching
        # both :0.6b and :8b) is too loose: having qwen3-embedding:8b must NOT
        # prevent pulling qwen3-embedding:0.6b when that specific version is needed.
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

    # ── Bước 3: subprocess fallback ───────────────────────────────────────────
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


# ── HuggingFace local pipeline cache ─────────────────────────────────────────
_HF_PIPELINE_CACHE: dict[str, object] = {}
