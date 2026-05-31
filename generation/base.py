"""
generation/base.py
==================
Abstract base class cho tất cả LLM generator.

Giao kèo:
    generator = SomeGenerator(**options)
    result    = generator.generate(prompt_result) -> GenerationResult
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Iterator

from prompt.base import PromptResult


@dataclass
class GenerationResult:
    """
    Output của generation stage.

    Attributes
    ----------
    answer         : Câu trả lời đầy đủ (string).
    provider       : Tên provider đã dùng (\"openai\", \"ollama\"...).
    model_name     : Tên model đã dùng.
    input_tokens   : Số input tokens (nếu provider trả về).
    output_tokens  : Số output tokens (nếu provider trả về).
    finish_reason  : Lý do kết thúc (\"stop\", \"length\", \"content_filter\"...).
    structured     : Dict nếu template = \"structured\" và parse thành công; None nếu không.
    cited_sources  : List index nguồn được trích dẫn (1-indexed); [] nếu template không citation.
    """
    answer:        str              = ""
    provider:      str              = ""
    model_name:    str              = ""
    input_tokens:  int              = 0
    output_tokens: int              = 0
    finish_reason: str              = "stop"
    structured:    dict | None      = None
    cited_sources: list[int]        = field(default_factory=list)


class BaseGenerator(ABC):
    """
    Abstract base cho mọi LLM generator.

    Tham số
    -------
    model_name  : Tên model (provider-specific).
    temperature : Sampling temperature (0 = deterministic).
    max_tokens  : Số token tối đa trong response.
    streaming   : Nếu True, stream token khi generate.
    """

    def __init__(
        self,
        model_name:  str   = "",
        temperature: float = 0.0,
        max_tokens:  int   = 2048,
        streaming:   bool  = False,
    ):
        self.model_name  = model_name
        self.temperature = temperature
        self.max_tokens  = max_tokens
        self.streaming   = streaming

    @abstractmethod
    def generate(self, prompt_result: PromptResult) -> GenerationResult:
        """
        Sinh câu trả lời từ PromptResult.

        Tham số
        -------
        prompt_result : Output của prompt builder (chứa messages + metadata).

        Trả về
        ------
        GenerationResult với answer, token counts, cited sources, ...
        """

    @abstractmethod
    def stream(self, prompt_result: PromptResult) -> Iterator[str]:
        """
        Stream câu trả lời token-by-token.

        Tham số
        -------
        prompt_result : Output của prompt builder.

        Yields
        ------
        str — từng đoạn text nhỏ (token hoặc chunk).
        """

    def _post_process(
        self,
        answer:        str,
        prompt_result: PromptResult,
        input_tokens:  int = 0,
        output_tokens: int = 0,
        finish_reason: str = "stop",
    ) -> GenerationResult:
        """
        Post-process sau khi generate: parse JSON nếu structured,
        extract citation indices nếu citation template.
        """
        structured     = None
        cited_sources: list[int] = []

        # Parse structured output
        if prompt_result.template_name == "structured":
            from prompt.structured_output import StructuredOutputPromptBuilder
            structured = StructuredOutputPromptBuilder.parse_response(answer)

        # Extract citation indices
        if prompt_result.template_name == "citation":
            from prompt.citation import CitationPromptBuilder
            cited_sources = CitationPromptBuilder.extract_cited_indices(answer)
            # Validate: lọc bỏ index ngoài phạm vi
            cited_sources = [i for i in cited_sources if 1 <= i <= prompt_result.n_sources]

        return GenerationResult(
            answer        = answer.strip(),
            provider      = self.__class__.__name__.replace("Generator", "").lower(),
            model_name    = self.model_name,
            input_tokens  = input_tokens,
            output_tokens = output_tokens,
            finish_reason = finish_reason,
            structured    = structured,
            cited_sources = cited_sources,
        )
