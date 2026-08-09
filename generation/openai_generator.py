"""
generation/openai_generator.py
================================
OpenAI ChatCompletion generator.

Models được khuyến nghị (giá rẻ / chất lượng tốt):
  gpt-4.1-mini   ← mặc định: nhanh, rẻ ($0.40/$1.60 per 1M), reasoning tốt
  gpt-4o-mini    Nhanh nhất, rẻ nhất ($0.15/$0.60 per 1M)
  gpt-4o         Chất lượng cao nhất, multimodal
  o3-mini        Reasoning sâu cho câu hỏi phức tạp

Env var: OPENAI_API_KEY
"""

from __future__ import annotations

from typing import Iterator

from prompt.base import PromptResult
from generation.base import BaseGenerator, GenerationResult


class OpenAIGenerator(BaseGenerator):
    """
    Tham số
    -------
    model_name  : OpenAI model identifier.
    temperature : 0 = deterministic.
    max_tokens  : Max tokens trong response.
    streaming   : Stream token ra stdout khi generate.
    """

    # Models của OpenAI dùng max_completion_tokens thay vì max_tokens
    _NEW_API_PREFIXES = ("o1", "o3", "o4")

    def __init__(
        self,
        model_name:  str   = "gpt-4.1-mini",
        temperature: float = 0.0,
        max_tokens:  int   = 2048,
        streaming:   bool  = False,
    ):
        super().__init__(model_name, temperature, max_tokens, streaming)

    def _is_reasoning_model(self) -> bool:
        return any(self.model_name.startswith(p) for p in self._NEW_API_PREFIXES)

    def _build_kwargs(self) -> dict:
        """Build kwargs phù hợp với từng loại model."""
        kw: dict = {"model": self.model_name, "messages": []}  # messages set later
        if self._is_reasoning_model():
            kw["max_completion_tokens"] = self.max_tokens
            # Reasoning model không nhận temperature
        else:
            kw["max_tokens"]  = self.max_tokens
            kw["temperature"] = self.temperature
        return kw

    def generate(self, prompt_result: PromptResult) -> GenerationResult:
        from openai import OpenAI
        client = OpenAI()

        kw = self._build_kwargs()
        kw["messages"] = prompt_result.messages

        response = client.chat.completions.create(**kw)
        choice   = response.choices[0]
        answer   = (choice.message.content or "").strip()

        usage = response.usage or None
        return self._post_process(
            answer        = answer,
            prompt_result = prompt_result,
            input_tokens  = getattr(usage, "prompt_tokens",     0) if usage else 0,
            output_tokens = getattr(usage, "completion_tokens", 0) if usage else 0,
            finish_reason = choice.finish_reason or "stop",
        )

    def stream(self, prompt_result: PromptResult) -> Iterator[str]:
        from openai import OpenAI
        client = OpenAI()

        kw = self._build_kwargs()
        kw["messages"] = prompt_result.messages
        kw["stream"]   = True

        for chunk in client.chat.completions.create(**kw):
            delta = chunk.choices[0].delta.content or ""
            if delta:
                yield delta
