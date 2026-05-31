"""
generation/factory.py
=====================
Factory functions — điểm vào duy nhất cho generation stage.

    get_generator(provider, model_name, **kwargs) -> BaseGenerator
    build_generator_from_config(cfg)              -> BaseGenerator

Providers (rẻ / phổ biến / tự host):
  openai    gpt-4.1-mini (rẻ, nhanh) — gpt-4o (tốt nhất)
  anthropic claude-haiku-4-5 (rẻ nhất) — claude-sonnet-4-6 (balanced)
  google    gemini-2.0-flash (free tier) — gemini-1.5-pro (context dài)
  ollama    qwen2.5:7b, llama3.2:3b — fully local, không cần API
  cohere    command-r-plus (RAG-optimised)
"""

from __future__ import annotations

from generation.base import BaseGenerator

_REGISTRY: dict[str, tuple[str, str]] = {
    "openai":    ("generation.openai_generator",    "OpenAIGenerator"),
    "anthropic": ("generation.anthropic_generator", "AnthropicGenerator"),
    "google":    ("generation.google_generator",    "GoogleGenerator"),
    "ollama":    ("generation.ollama_generator",    "OllamaGenerator"),
    "cohere":    ("generation.cohere_generator",    "CohereGenerator"),
}


def get_generator(provider: str, model_name: str, **kwargs) -> BaseGenerator:
    """
    Khởi tạo generator theo provider.

    Tham số
    -------
    provider   : \"openai\" | \"anthropic\" | \"google\" | \"ollama\" | \"cohere\"
    model_name : Tên model theo provider.
    **kwargs   : temperature, max_tokens, streaming, base_url (ollama), auto_pull (ollama).

    Ví dụ
    -----
    >>> get_generator(\"openai\",    \"gpt-4.1-mini\", temperature=0.0)
    >>> get_generator(\"anthropic\", \"claude-haiku-4-5-20251001\", streaming=True)
    >>> get_generator(\"ollama\",    \"qwen2.5:7b\", base_url=\"http://localhost:11434\")
    >>> get_generator(\"google\",    \"gemini-2.0-flash\")
    """
    entry = _REGISTRY.get(provider)
    if entry is None:
        valid = ", ".join(sorted(_REGISTRY))
        raise ValueError(f"Unknown generation provider '{provider}'. Valid: {valid}")

    import importlib
    module_path, class_name = entry
    cls = getattr(importlib.import_module(module_path), class_name)
    return cls(model_name=model_name, **kwargs)


def build_generator_from_config(cfg: dict) -> BaseGenerator:
    """
    Khởi tạo generator từ section ``query_pipeline.generation`` của config.yaml.

    Config keys dùng
    ----------------
    query_pipeline.generation.provider    \"openai\" | \"anthropic\" | \"google\" | \"ollama\" | \"cohere\"
    query_pipeline.generation.model_name  tên model
    query_pipeline.generation.temperature float (mặc định 0.0)
    query_pipeline.generation.max_tokens  int   (mặc định 2048)
    query_pipeline.generation.streaming   bool  (mặc định True)
    query_pipeline.generation.base_url    str   (ollama only)
    """
    gen_cfg    = cfg["query_pipeline"]["generation"]
    provider   = gen_cfg.get("provider",    "openai")
    model_name = gen_cfg.get("model_name",  "gpt-4.1-mini")

    kwargs: dict = {
        "temperature": gen_cfg.get("temperature", 0.0),
        "max_tokens":  gen_cfg.get("max_tokens",  2048),
        "streaming":   gen_cfg.get("streaming",   True),
    }
    if provider == "ollama":
        kwargs["base_url"]   = gen_cfg.get("base_url",   "http://localhost:11434")
        kwargs["auto_pull"]  = gen_cfg.get("auto_pull",  True)

    return get_generator(provider=provider, model_name=model_name, **kwargs)
