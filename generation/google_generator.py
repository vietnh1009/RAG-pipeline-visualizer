"""
generation/google_generator.py
================================
Google Gemini generator.

Models được khuyến nghị:
  gemini-2.0-flash    ← mặc định: nhanh, rẻ (free tier có sẵn)
  gemini-2.0-flash-lite  Rẻ nhất
  gemini-1.5-pro      Context dài nhất (2M tokens)

Env var: GOOGLE_API_KEY
"""

from __future__ import annotations

from typing import Iterator

from prompt.base import PromptResult
from generation.base import BaseGenerator, GenerationResult


class GoogleGenerator(BaseGenerator):
    """
    Tham số
    -------
    model_name  : Google Gemini model identifier.
    temperature : Sampling temperature.
    max_tokens  : Max output tokens.
    streaming   : Stream tokens.
    """

    def __init__(
        self,
        model_name:  str   = "gemini-2.0-flash",
        temperature: float = 0.0,
        max_tokens:  int   = 2048,
        streaming:   bool  = False,
    ):
        super().__init__(model_name, temperature, max_tokens, streaming)

    def _messages_to_gemini(self, messages: list[dict]) -> tuple[str, list[dict]]:
        """
        Chuyển OpenAI-style messages sang Gemini format.
        Gemini: system_instruction riêng + contents list [{"role": "user"|"model", "parts": [...]}]
        """
        system = ""
        contents: list[dict] = []
        for m in messages:
            role    = m.get("role", "user")
            content = m.get("content", "")
            if role == "system":
                system = content
            elif role == "user":
                contents.append({"role": "user", "parts": [{"text": content}]})
            elif role == "assistant":
                contents.append({"role": "model", "parts": [{"text": content}]})
        return system, contents

    def generate(self, prompt_result: PromptResult) -> GenerationResult:
        import google.generativeai as genai

        system, contents = self._messages_to_gemini(prompt_result.messages)

        gen_config = genai.types.GenerationConfig(
            temperature=self.temperature,
            max_output_tokens=self.max_tokens,
        )
        model_kwargs: dict = {"generation_config": gen_config}
        if system:
            model_kwargs["system_instruction"] = system

        model    = genai.GenerativeModel(self.model_name, **model_kwargs)
        response = model.generate_content(contents)
        answer   = response.text.strip()

        usage = getattr(response, "usage_metadata", None)
        return self._post_process(
            answer        = answer,
            prompt_result = prompt_result,
            input_tokens  = getattr(usage, "prompt_token_count",     0) if usage else 0,
            output_tokens = getattr(usage, "candidates_token_count", 0) if usage else 0,
        )

    def stream(self, prompt_result: PromptResult) -> Iterator[str]:
        import google.generativeai as genai

        system, contents = self._messages_to_gemini(prompt_result.messages)

        gen_config = genai.types.GenerationConfig(
            temperature=self.temperature,
            max_output_tokens=self.max_tokens,
        )
        model_kwargs: dict = {"generation_config": gen_config}
        if system:
            model_kwargs["system_instruction"] = system

        model = genai.GenerativeModel(self.model_name, **model_kwargs)
        for chunk in model.generate_content(contents, stream=True):
            text = chunk.text or ""
            if text:
                yield text
