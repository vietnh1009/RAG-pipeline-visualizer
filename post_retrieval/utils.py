"""
post_retrieval/utils.py
=======================
Shared helpers used across post-retrieval modules.
"""

from __future__ import annotations

from langchain_core.documents import Document


def deduplicate(docs: list[Document]) -> list[Document]:
    """Remove exact-duplicate documents (same stripped page_content)."""
    seen:   set[str]       = set()
    unique: list[Document] = []
    for doc in docs:
        key = doc.page_content.strip()
        if key not in seen:
            seen.add(key)
            unique.append(doc)
    return unique


def call_llm(
    prompt:      str,
    provider:    str,
    model:       str,
    max_tokens:  int   = 512,
    temperature: float = 0.0,
) -> str:
    """Call an LLM and return the raw text response."""
    if provider == "openai":
        from openai import OpenAI
        r = OpenAI().chat.completions.create(
            model=model, temperature=temperature, max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return r.choices[0].message.content.strip()

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

    raise ValueError(f"Unsupported LLM provider: '{provider}'")
