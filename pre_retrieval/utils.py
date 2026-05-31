"""
pre_retrieval/utils.py
======================
Shared helpers used across pre-retrieval transformer modules.
"""

from __future__ import annotations

import json
import re


def call_llm(
    prompt:       str,
    provider:     str,
    model:        str,
    max_tokens:   int = 512,
    temperature:  float = 0.0,
) -> str:
    """
    Call an LLM and return the raw text response.
    Supports OpenAI, Anthropic, and Google providers.
    """
    if provider == "openai":
        from openai import OpenAI
        r = OpenAI().chat.completions.create(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return r.choices[0].message.content.strip()

    if provider == "anthropic":
        import anthropic
        r = anthropic.Anthropic().messages.create(
            model=model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return r.content[0].text.strip()

    if provider == "google":
        import google.generativeai as genai
        return genai.GenerativeModel(model).generate_content(prompt).text.strip()

    raise ValueError(f"Unsupported LLM provider: '{provider}'")


def parse_json_list(text: str) -> list[str]:
    """
    Parse a JSON array from LLM output, handling markdown code fences.
    Falls back to line-by-line extraction if JSON parsing fails.
    """
    cleaned = re.sub(r"```(?:json)?", "", text).strip().rstrip("`").strip()
    try:
        result = json.loads(cleaned)
        if isinstance(result, list):
            return [str(item).strip() for item in result if str(item).strip()]
    except json.JSONDecodeError:
        pass

    # Fallback: quoted strings or numbered items
    items = re.findall(r'"([^"]+)"', cleaned)
    if items:
        return items
    return [
        re.sub(r"^\d+[\.\)]\s*", "", line).strip()
        for line in cleaned.splitlines()
        if line.strip() and not line.strip().startswith(("[", "]"))
    ]
