"""
prompt/factory.py
=================
Factory function và config builder cho prompt template.

    get_prompt_builder(template, **kwargs)       -> BasePromptBuilder
    build_prompt_builder_from_config(cfg)        -> BasePromptBuilder
"""

from __future__ import annotations

from prompt.base import BasePromptBuilder

_REGISTRY: dict[str, tuple[str, str]] = {
    "basic":          ("prompt.basic",             "BasicPromptBuilder"),
    "citation":       ("prompt.citation",          "CitationPromptBuilder"),
    "conversational": ("prompt.conversational",    "ConversationalPromptBuilder"),
    "structured":     ("prompt.structured_output", "StructuredOutputPromptBuilder"),
}


def get_prompt_builder(template: str, **kwargs) -> BasePromptBuilder:
    """
    Khởi tạo prompt builder theo tên template.

    Tham số
    -------
    template : \"basic\" | \"citation\" | \"conversational\" | \"structured\"
    **kwargs : Truyền thẳng vào constructor của builder.

    Ví dụ
    -----
    >>> get_prompt_builder(\"citation\", language=\"vi\")
    >>> get_prompt_builder(\"conversational\", max_history_turns=3, language=\"both\")
    >>> get_prompt_builder(\"structured\", include_confidence=True)
    """
    entry = _REGISTRY.get(template)
    if entry is None:
        valid = ", ".join(sorted(_REGISTRY))
        raise ValueError(f"Unknown prompt template '{template}'. Valid: {valid}")

    import importlib
    module_path, class_name = entry
    cls = getattr(importlib.import_module(module_path), class_name)
    return cls(**kwargs)


def build_prompt_builder_from_config(cfg: dict) -> BasePromptBuilder:
    """
    Khởi tạo prompt builder từ section ``query_pipeline.prompt`` của config.yaml.

    Config keys dùng
    ----------------
    query_pipeline.prompt.template          \"basic\" | \"citation\" | \"conversational\" | \"structured\"
    query_pipeline.prompt.max_history_turns int (conversational only, mặc định 5)
    query_pipeline.prompt.validate_citations bool (citation only, mặc định True)
    query_pipeline.prompt.include_confidence bool (structured only, mặc định True)
    query_pipeline.prompt.system_instruction str bổ sung (tuỳ chọn)
    query_pipeline.prompt.max_context_chars  int (0 = không giới hạn)
    data.language                            corpus language
    """
    prompt_cfg = cfg["query_pipeline"]["prompt"]
    template   = prompt_cfg.get("template", "citation")
    language   = cfg["data"].get("language", "both")

    kwargs: dict = {
        "language":          language,
        "max_context_chars": prompt_cfg.get("max_context_chars", 0),
    }
    if prompt_cfg.get("system_instruction"):
        kwargs["system_instruction"] = prompt_cfg["system_instruction"]

    if template == "conversational":
        kwargs["max_history_turns"] = prompt_cfg.get("max_history_turns", 5)
    elif template == "citation":
        kwargs["validate_citations"] = prompt_cfg.get("validate_citations", True)
    elif template == "structured":
        kwargs["include_confidence"] = prompt_cfg.get("include_confidence", True)

    return get_prompt_builder(template, **kwargs)
