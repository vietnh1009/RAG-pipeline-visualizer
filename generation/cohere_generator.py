"""
generation/cohere_generator.py
================================
Cohere Command generator.

Models được khuyến nghị:
  command-r-plus  ← RAG-optimised, 128K ctx, tốt nhất Cohere
  command-r       Rẻ hơn, đủ tốt cho Q&A đơn giản

Env var: COHERE_API_KEY
"""

from __future__ import annotations

from typing import Iterator

from prompt.base import PromptResult
from generation.base import BaseGenerator, GenerationResult


class CohereGenerator(BaseGenerator):
    """
    Tham số
    -------
    model_name  : Cohere model identifier.
    temperature : Sampling temperature.
    max_tokens  : Max tokens trong response.
    streaming   : Stream tokens.
    """

    def __init__(
        self,
        model_name:  str   = "command-r-plus",
        temperature: float = 0.0,
        max_tokens:  int   = 2048,
        streaming:   bool  = False,
    ):
        super().__init__(model_name, temperature, max_tokens, streaming)

    def generate(self, prompt_result: PromptResult) -> GenerationResult:
        import cohere
        client = cohere.ClientV2()

        # Cohere V2 API hỗ trợ messages format
        response = client.chat(
            model=self.model_name,
            messages=prompt_result.messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        answer = response.message.content[0].text.strip()

        usage = getattr(response, "usage", None)
        return self._post_process(
            answer        = answer,
            prompt_result = prompt_result,
            input_tokens  = getattr(usage.tokens, "input_tokens",  0) if usage and usage.tokens else 0,
            output_tokens = getattr(usage.tokens, "output_tokens", 0) if usage and usage.tokens else 0,
        )

    def stream(self, prompt_result: PromptResult) -> Iterator[str]:
        import cohere
        client = cohere.ClientV2()

        for event in client.chat_stream(
            model=self.model_name,
            messages=prompt_result.messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        ):
            if event.type == "content-delta":
                text = event.delta.message.content.text or ""
                if text:
                    yield text
