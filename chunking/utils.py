"""
chunking/utils.py
=================
Hàm tiện ích dùng chung cho các chunking module.
"""

from __future__ import annotations

import json
import re

from langchain_core.documents import Document


def call_llm(prompt: str, provider: str, model: str, max_tokens: int = 1024) -> str:
    """
    Call an LLM and return the raw text response.

    Supports OpenAI, Anthropic, and Google providers.
    Used by ContextualChunker, AgenticChunker, and PropositionChunker.
    """
    if provider == "openai":
        from openai import OpenAI
        _new_api     = any(model.startswith(p) for p in ("gpt-5", "o1", "o3", "o4"))
        _token_param = "max_completion_tokens" if _new_api else "max_tokens"
        _extra       = {} if _new_api else {"temperature": 0}
        r = OpenAI().chat.completions.create(
            model=model,
            **{_token_param: max_tokens},
            **_extra,
            messages=[{"role": "user", "content": prompt}],
        )
        content = r.choices[0].message.content
        if content is None:
            reasoning = getattr(r.choices[0].message, "reasoning_content", None)
            if reasoning:
                return reasoning.strip()
            finish_reason = r.choices[0].finish_reason
            raise ValueError(
                f"Model '{model}' trả về content=None (finish_reason={finish_reason}). "
                f"Model này có thể không hỗ trợ free-form text generation. "
                f"Thử dùng gpt-4o-mini hoặc gpt-5-mini thay thế."
            )
        return content.strip()

    if provider == "anthropic":
        import anthropic
        r = anthropic.Anthropic().messages.create(
            model=model, max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return r.content[0].text.strip()

    if provider == "google":
        import google.generativeai as genai
        return genai.GenerativeModel(model).generate_content(prompt).text.strip()

    if provider == "ollama":
        import os
        from openai import OpenAI
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
        r = OpenAI(base_url=base_url, api_key="ollama").chat.completions.create(
            model=model, temperature=0, max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return r.choices[0].message.content.strip()

    if provider == "huggingface":
        import os
        from chunking.utils import _get_hf_pipeline
        pipe   = _get_hf_pipeline(model, token=os.getenv("HF_TOKEN", ""))
        result = pipe(prompt, max_new_tokens=max_tokens, do_sample=False)
        full_text = result[0]["generated_text"]
        return full_text[len(prompt):].strip() if full_text.startswith(prompt) else full_text.strip()

    raise ValueError(f"Unsupported LLM provider: '{provider}'")


def parse_json_list(text: str) -> list[str]:
    """
    Parse a JSON array from LLM output that may be wrapped in markdown fences.
    Falls back to line-by-line extraction if JSON parsing fails.
    """
    cleaned = re.sub(r"```(?:json)?", "", text).strip().rstrip("`").strip()
    try:
        result = json.loads(cleaned)
        if isinstance(result, list):
            return [str(item).strip() for item in result if str(item).strip()]
    except json.JSONDecodeError:
        pass

    items = re.findall(r'"([^"]+)"', cleaned)
    if items:
        return items
    return [
        re.sub(r"^\d+[\.\)]\s*", "", line).strip()
        for line in cleaned.splitlines()
        if line.strip() and not line.strip().startswith(("[", "]"))
    ]


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


def _get_hf_pipeline(model_id: str, token: str = ""):
    """
    Trả về transformers text-generation pipeline cho model_id.
    Lần đầu: tự động download về HuggingFace cache (~/.cache/huggingface).
    Lần sau: dùng lại pipeline đã load (in-process cache).
    """
    if model_id in _HF_PIPELINE_CACHE:
        return _HF_PIPELINE_CACHE[model_id]

    import os
    if token:
        os.environ["HUGGING_FACE_HUB_TOKEN"] = token
        os.environ["HF_TOKEN"]               = token

    ensure_hf_model(model_id, token=token)

    try:
        import torch
        from transformers import pipeline as _pipeline

        device_map = "auto" if torch.cuda.is_available() else "cpu"
        pipe = _pipeline(
            "text-generation",
            model=model_id,
            token=token or None,
            device_map=device_map,
            torch_dtype="auto",
            trust_remote_code=True,
        )
    except ImportError:
        raise ImportError(
            "Cần cài transformers và torch để chạy HuggingFace model local:\n"
            "  pip install transformers torch accelerate"
        )

    _HF_PIPELINE_CACHE[model_id] = pipe
    return pipe


def ensure_hf_model(model_id: str, token: str = "") -> None:
    """
    Kiểm tra model đã có trong HuggingFace local cache chưa.
    Nếu chưa, tự động download về ~/.cache/huggingface/hub.
    """
    try:
        from huggingface_hub import try_to_load_from_cache, snapshot_download
        from huggingface_hub.utils import EntryNotFoundError

        cached = try_to_load_from_cache(model_id, filename="config.json")
        if cached and cached != "None":
            return

        print(f"[HuggingFace] Model '{model_id}' chưa có sẵn — đang download về local cache...")
        snapshot_download(
            repo_id=model_id,
            token=token or None,
            ignore_patterns=["*.msgpack", "*.h5", "flax_model*", "tf_model*"],
        )
        print(f"[HuggingFace] Download '{model_id}' hoàn tất.")

    except ImportError:
        raise ImportError(
            "Cần cài huggingface_hub:\n"
            "  pip install huggingface_hub"
        )
    except Exception as e:
        raise RuntimeError(
            f"[HuggingFace] Không thể download '{model_id}': {e}\n"
            f"Kiểm tra HF_TOKEN trong .env nếu đây là model gated."
        )
