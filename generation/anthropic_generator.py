"""
generation/anthropic_generator.py
====================================
Anthropic Claude generator.

Models được khuyến nghị:
  claude-haiku-4-5-20251001  ← rẻ nhất, nhanh nhất ($0.80/$4 per 1M)
  claude-sonnet-4-6          Cân bằng chất lượng/tốc độ — mặc định
  claude-opus-4-6            Mạnh nhất, đắt nhất

Env var: ANTHROPIC_API_KEY
"""

from __future__ import annotations

from typing import Iterator

from prompt.base import PromptResult
from generation.base import BaseGenerator, GenerationResult


class AnthropicGenerator(BaseGenerator):
    """
    Tham số
    -------
    model_name  : Anthropic model identifier.
    temperature : 0 = deterministic.
    max_tokens  : Max tokens trong response.
    streaming   : Stream token.
    """

    def __init__(
        self,
        model_name:  str   = "claude-sonnet-4-6",
        temperature: float = 0.0,
        max_tokens:  int   = 2048,
        streaming:   bool  = False,
    ):
        super().__init__(model_name, temperature, max_tokens, streaming)

    def _split_messages(self, messages: list[dict]) -> tuple[str, list[dict]]:
        """Tách system message khỏi chat messages (Anthropic API yêu cầu riêng)."""
        system = ""
        chat: list[dict] = []
        for m in messages:
            if m.get("role") == "system":
                system = m.get("content", "")
            else:
                chat.append(m)
        # Anthropic yêu cầu messages phải bắt đầu bằng user
        if chat and chat[0].get("role") != "user":
            chat = [{"role": "user", "content": ""}] + chat
        return system, chat

    def generate(self, prompt_result: PromptResult) -> GenerationResult:
        import anthropic
        client = anthropic.Anthropic()

        system, chat = self._split_messages(prompt_result.messages)

        kwargs: dict = {
            "model":      self.model_name,
            "max_tokens": self.max_tokens,
            "messages":   chat,
        }
        if system:
            kwargs["system"] = system
        if self.temperature > 0:
            kwargs["temperature"] = self.temperature

        response = client.messages.create(**kwargs)
        answer   = response.content[0].text.strip()

        return self._post_process(
            answer        = answer,
            prompt_result = prompt_result,
            input_tokens  = response.usage.input_tokens,
            output_tokens = response.usage.output_tokens,
            finish_reason = response.stop_reason or "stop",
        )

    def stream(self, prompt_result: PromptResult) -> Iterator[str]:
        import anthropic
        client = anthropic.Anthropic()

        system, chat = self._split_messages(prompt_result.messages)

        kwargs: dict = {
            "model":      self.model_name,
            "max_tokens": self.max_tokens,
            "messages":   chat,
        }
        if system:
            kwargs["system"] = system
        if self.temperature > 0:
            kwargs["temperature"] = self.temperature

        with client.messages.stream(**kwargs) as stream:
            for text in stream.text_stream:
                if text:
                    yield text
