"""
generation/
===========
Package sinh câu trả lời LLM cho RAG query pipeline.

API công khai
-------------
    from generation import get_generator, build_generator_from_config

    gen    = get_generator("openai", "gpt-4.1-mini")
    result = gen.generate(prompt_result)
    result.answer          # câu trả lời đầy đủ
    result.cited_sources   # các nguồn được trích dẫn, vd [1, 3]

    # sinh theo dòng, dùng trong Streamlit
    for chunk in gen.stream(prompt_result):
        st.write(chunk)

Provider
--------
  openai    gpt-4.1-mini — rẻ, nhanh, mặc định
  anthropic claude-haiku-4-5 rẻ nhất; claude-sonnet-4-6 cân bằng
  google    gemini-2.0-flash — có gói miễn phí
  ollama    qwen2.5:7b, llama3.2:3b — chạy hoàn toàn cục bộ
  cohere    command-r-plus — tối ưu cho RAG
"""

from generation.base    import BaseGenerator, GenerationResult
from generation.factory import get_generator, build_generator_from_config

__all__ = [
    "BaseGenerator",
    "GenerationResult",
    "get_generator",
    "build_generator_from_config",
]
