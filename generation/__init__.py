"""
generation/
===========
Package sinh câu trả lời LLM cho RAG query pipeline.

Public API
----------
    from generation import get_generator, build_generator_from_config
    from generation import GenerationResult

    gen    = get_generator("openai", "gpt-4.1-mini")
    result = gen.generate(prompt_result)

    # result.answer        -> str    (câu trả lời đầy đủ)
    # result.cited_sources -> [1, 3] (nguồn được trích dẫn)
    # result.input_tokens  -> int
    # result.output_tokens -> int

    # Streaming (trong Streamlit):
    for chunk in gen.stream(prompt_result):
        st.write(chunk)

Providers
---------
  openai    gpt-4.1-mini ← rẻ, nhanh, mặc định
  anthropic claude-haiku-4-5 ← rẻ nhất; claude-sonnet-4-6 ← balanced
  google    gemini-2.0-flash ← free tier có sẵn
  ollama    qwen2.5:7b, llama3.2:3b ← fully local
  cohere    command-r-plus ← RAG-optimised
"""

from generation.base    import BaseGenerator, GenerationResult
from generation.factory import get_generator, build_generator_from_config

__all__ = [
    "BaseGenerator",
    "GenerationResult",
    "get_generator",
    "build_generator_from_config",
]
