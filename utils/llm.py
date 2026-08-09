"""
utils/llm.py
============
Điểm gọi LLM DUY NHẤT dùng chung cho mọi bước của pipeline.

Trước đây ``call_llm`` được sao chép ở ba nơi (``chunking/utils.py``,
``pre_retrieval/utils.py``, ``post_retrieval/utils.py``) và ba bản đã trôi khác
nhau: chỉ bản của ``chunking`` biết xử lý model dòng mới (gpt-5 / o1 / o3 / o4)
và hỗ trợ provider ``ollama`` / ``huggingface``; hai bản kia có tham số
``temperature`` nhưng chỉ chạy được openai / anthropic / google.

Module này là HỢP của cả ba — không bản nào bị mất tính năng.

Public API
----------
    from utils.llm import call_llm, parse_json_list

    raw   = call_llm(prompt, "openai", "gpt-4.1-mini", max_tokens=512)
    items = parse_json_list(raw)
"""

from __future__ import annotations

import json
import re

# Model dòng mới của OpenAI: dùng `max_completion_tokens` thay cho `max_tokens`,
# và TỪ CHỐI tham số `temperature`. Gửi sai tham số → API trả lỗi 400.
_OPENAI_NEW_API_PREFIXES = ("gpt-5", "o1", "o3", "o4")


def call_llm(
    prompt:      str,
    provider:    str,
    model:       str,
    max_tokens:  int   = 512,
    temperature: float = 0.0,
) -> str:
    """
    Gọi LLM và trả về text thô.

    Tham số
    -------
    prompt      : Nội dung prompt gửi cho model.
    provider    : "openai" | "anthropic" | "google" | "ollama" | "huggingface".
    model       : Tên model của provider tương ứng.
    max_tokens  : Giới hạn token sinh ra.
    temperature : Độ ngẫu nhiên. BỊ BỎ QUA với model dòng mới của OpenAI
                  (gpt-5/o1/o3/o4) vì các model đó không nhận tham số này.

    Trả về
    ------
    Chuỗi text đã ``.strip()``.

    Ngoại lệ
    --------
    ValueError : provider không hỗ trợ, hoặc OpenAI trả về content rỗng.

    Ví dụ
    -----
    >>> call_llm("Tóm tắt đoạn sau...", "openai", "gpt-4.1-mini", max_tokens=200)
    >>> call_llm("Xin chào", "ollama", "qwen2.5:7b", max_tokens=128)
    """
    if provider == "openai":
        from openai import OpenAI

        is_new_api   = any(model.startswith(p) for p in _OPENAI_NEW_API_PREFIXES)
        token_param  = "max_completion_tokens" if is_new_api else "max_tokens"
        extra        = {} if is_new_api else {"temperature": temperature}

        r = OpenAI().chat.completions.create(
            model=model,
            **{token_param: max_tokens},
            **extra,
            messages=[{"role": "user", "content": prompt}],
        )
        content = r.choices[0].message.content
        if content is None:
            # Model reasoning đôi khi trả content=None và đặt kết quả ở
            # reasoning_content. Nếu cũng không có thì báo lỗi rõ ràng thay vì
            # để AttributeError xảy ra ở chỗ khác.
            reasoning = getattr(r.choices[0].message, "reasoning_content", None)
            if reasoning:
                return reasoning.strip()
            raise ValueError(
                f"Model '{model}' trả về content=None "
                f"(finish_reason={r.choices[0].finish_reason}). "
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
            model=model, temperature=temperature, max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return r.choices[0].message.content.strip()

    if provider == "huggingface":
        import os
        pipe      = _get_hf_pipeline(model, token=os.getenv("HF_TOKEN", ""))
        result    = pipe(prompt, max_new_tokens=max_tokens, do_sample=False)
        full_text = result[0]["generated_text"]
        # Một số pipeline trả về cả prompt ở đầu output — cắt bỏ nếu có.
        return (full_text[len(prompt):].strip()
                if full_text.startswith(prompt) else full_text.strip())

    raise ValueError(f"Unsupported LLM provider: '{provider}'")


def parse_json_list(text: str) -> list[str]:
    """
    Tách một JSON array từ output của LLM, chấp nhận cả khi bị bọc trong
    markdown code fence.

    Chiến lược giảm dần: JSON hợp lệ → chuỗi trong dấu nháy kép → từng dòng
    (bỏ tiền tố đánh số kiểu "1." / "2)").

    Ví dụ
    -----
    >>> parse_json_list('```json\\n["a", "b"]\\n```')
    ['a', 'b']
    """
    cleaned = re.sub(r"```(?:json)?", "", text).strip().rstrip("`").strip()
    try:
        result = json.loads(cleaned)
        if isinstance(result, list):
            return [str(item).strip() for item in result if str(item).strip()]
    except json.JSONDecodeError:
        pass

    # Fallback: chuỗi trong nháy kép, hoặc danh sách đánh số theo dòng
    items = re.findall(r'"([^"]+)"', cleaned)
    if items:
        return items
    return [
        re.sub(r"^\d+[\.\)]\s*", "", line).strip()
        for line in cleaned.splitlines()
        if line.strip() and not line.strip().startswith(("[", "]"))
    ]


# ── HuggingFace local pipeline (chỉ phục vụ nhánh provider="huggingface") ────

_HF_PIPELINE_CACHE: dict[str, object] = {}


def _get_hf_pipeline(model_id: str, token: str = ""):
    """
    Trả về transformers text-generation pipeline cho ``model_id``.

    Lần đầu: tự động download về HuggingFace cache (~/.cache/huggingface).
    Lần sau: dùng lại pipeline đã load (cache trong tiến trình).
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
    Kiểm tra model đã có trong HuggingFace local cache chưa; nếu chưa thì
    tự động download về ``~/.cache/huggingface/hub``.
    """
    try:
        from huggingface_hub import try_to_load_from_cache, snapshot_download

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
