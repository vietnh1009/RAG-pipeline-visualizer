"""
generation/ollama_generator.py
================================
Ollama local LLM generator — hoàn toàn offline, không cần API key.

Models nhẹ được khuyến nghị (chạy tốt trên CPU):
  qwen2.5:7b      ← tốt nhất: tiếng Việt OK, 7B, lý luận tốt
  llama3.2:3b     Nhẹ nhất, nhanh nhất trên CPU
  mistral:7b      Chất lượng tốt, tiếng Anh
  gemma2:2b       Rất nhẹ (~1.6GB)
  phi4-mini       Nhỏ nhưng mạnh về reasoning

Pull model trước: ollama pull qwen2.5:7b
"""

from __future__ import annotations

from typing import Iterator

from prompt.base import PromptResult
from generation.base import BaseGenerator, GenerationResult


class OllamaGenerator(BaseGenerator):
    """
    Tham số
    -------
    model_name  : Ollama model name (đã pull sẵn).
    base_url    : Ollama server URL (mặc định localhost:11434).
    temperature : Sampling temperature.
    max_tokens  : Max tokens trong response.
    streaming   : Stream tokens.
    auto_pull   : Tự động pull model nếu chưa có (blocking).
    """

    def __init__(
        self,
        model_name:  str   = "qwen2.5:7b",
        base_url:    str   = "http://localhost:11434",
        temperature: float = 0.0,
        max_tokens:  int   = 2048,
        streaming:   bool  = False,
        auto_pull:   bool  = True,
    ):
        super().__init__(model_name, temperature, max_tokens, streaming)
        # Chuẩn hoá URL: bỏ /v1 suffix nếu có
        self.base_url  = base_url.rstrip("/").removesuffix("/v1")
        self.auto_pull = auto_pull

    def _ensure_model(self) -> None:
        """Pull model nếu chưa có (dùng chunking.utils để tái sử dụng logic)."""
        if not self.auto_pull:
            return
        try:
            from chunking.utils import ensure_ollama_model
            ensure_ollama_model(self.model_name, self.base_url)
        except Exception:
            pass  # Tiếp tục — Ollama sẽ tự báo lỗi nếu model chưa có

    def _get_client(self):
        from ollama import Client
        return Client(host=self.base_url)

    def generate(self, prompt_result: PromptResult) -> GenerationResult:
        self._ensure_model()
        client = self._get_client()

        # Ollama native chat API — hỗ trợ messages format chuẩn
        response = client.chat(
            model=self.model_name,
            messages=prompt_result.messages,
            options={
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
        )
        answer = response.message.content.strip()

        return self._post_process(
            answer        = answer,
            prompt_result = prompt_result,
            input_tokens  = getattr(response, "prompt_eval_count",  0),
            output_tokens = getattr(response, "eval_count",         0),
            finish_reason = "stop",
        )

    def stream(self, prompt_result: PromptResult) -> Iterator[str]:
        self._ensure_model()
        client = self._get_client()

        for chunk in client.chat(
            model=self.model_name,
            messages=prompt_result.messages,
            options={
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
            stream=True,
        ):
            text = chunk.message.content or ""
            if text:
                yield text
