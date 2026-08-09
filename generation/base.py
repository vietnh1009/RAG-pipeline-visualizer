"""
generation/base.py
==================
Lớp ABC cho mọi LLM generator.

Hợp đồng chung:
    result = SomeGenerator(**options).generate(prompt_result) -> GenerationResult
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Iterator

from prompt.base import PromptResult


@dataclass
class GenerationResult:
    """
    Kết quả của bước generation.

    Tham số
    -------
    answer        : Câu trả lời đầy đủ.
    provider      : Provider đã dùng (\"openai\", \"ollama\", …).
    model_name    : Tên model đã dùng.
    input_tokens  : Số token đầu vào, nếu provider có trả về.
    output_tokens : Số token đầu ra, nếu provider có trả về.
    finish_reason : Lý do dừng (\"stop\", \"length\", \"content_filter\", …).
    structured    : Dict khi template là \"structured\" và parse được; None nếu không.
    cited_sources : Chỉ số nguồn được trích dẫn, đánh số từ 1; [] nếu template
                    không phải citation.
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
    Lớp cơ sở cho mọi LLM generator.

    Tham số
    -------
    model_name  : Tên model, riêng theo từng provider.
    temperature : Nhiệt độ lấy mẫu; 0 là tất định.
    max_tokens  : Số token tối đa trong câu trả lời.
    streaming   : True thì trả token theo dòng khi sinh.
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
        prompt_result : Kết quả của prompt builder, gồm messages và metadata.

        Trả về
        ------
        GenerationResult chứa câu trả lời, số token, nguồn trích dẫn, …
        """

    @abstractmethod
    def stream(self, prompt_result: PromptResult) -> Iterator[str]:
        """
        Trả câu trả lời theo dòng, từng token một.

        Tham số
        -------
        prompt_result : Kết quả của prompt builder.

        Sinh ra
        -------
        str — từng mẩu text nhỏ (token hoặc chunk).
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
        Hậu xử lý sau khi sinh: parse JSON nếu template là structured, trích chỉ
        số nguồn nếu template là citation.
        """
        structured     = None
        cited_sources: list[int] = []

        # Parse kết quả JSON có cấu trúc
        if prompt_result.template_name == "structured":
            from prompt.structured_output import StructuredOutputPromptBuilder
            structured = StructuredOutputPromptBuilder.parse_response(answer)

        # Trích chỉ số nguồn được trích dẫn
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
