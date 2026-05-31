"""
prompt/
=======
Package xây dựng prompt cho RAG query pipeline.

Public API
----------
    from prompt import get_prompt_builder, build_prompt_builder_from_config
    from prompt import PromptResult

    builder = get_prompt_builder("citation", language="vi")
    result  = builder.build(query="...", docs=retrieved_docs)

    # result.messages    -> list[dict]  (chat API format)
    # result.full_prompt -> str         (completion API format)
    # result.n_sources   -> int         (số nguồn trong context)

Templates có sẵn
----------------
  basic         Tối giản, grounded chặt chẽ — prototype, Q&A đơn giản
  citation      Yêu cầu [NGUỒN N] inline — production, audit trail
  conversational Multi-turn với history — chatbot, follow-up questions
  structured    JSON output (claims + sources + confidence) — API integration
"""

from prompt.base    import BasePromptBuilder, PromptResult
from prompt.factory import get_prompt_builder, build_prompt_builder_from_config

__all__ = [
    "BasePromptBuilder",
    "PromptResult",
    "get_prompt_builder",
    "build_prompt_builder_from_config",
]
