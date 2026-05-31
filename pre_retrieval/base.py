"""
pre_retrieval/base.py
=====================
Shared data container and abstract base class for all query transformers.

Every transformer follows the same contract:
    transformer = SomeTransformer(**options)
    result      = transformer.transform(query: str) -> TransformResult

TransformResult
---------------
The single data object that flows from pre-retrieval into retrieval.
It carries every piece of information the retrieval stage may need:
  - queries         : one or more query strings to retrieve for
  - metadata_filter : structured filter for the vector DB
  - intent          : classified query intent label
  - retrieval_path  : routing target (collection name or strategy name)
  - extra           : any additional transformer-specific metadata
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class TransformResult:
    """
    Output of any pre-retrieval transformer.

    Attributes
    ----------
    original_query  : The raw user query, unchanged.
    queries         : One or more transformed / generated queries.
                      The retrieval stage runs all of them and merges results.
    metadata_filter : Optional structured filter dict for the vector DB.
                      E.g. {"source": "policy_2024.pdf", "language": "vi"}
    intent          : Classified query intent label (set by IntentClassifier).
    retrieval_path  : Routing target set by QueryRouter.
    extra           : Additional transformer-specific metadata.
    """
    original_query:  str
    queries:         list[str]      = field(default_factory=list)
    metadata_filter: dict | None    = None
    intent:          str | None     = None
    retrieval_path:  str | None     = None
    extra:           dict           = field(default_factory=dict)

    def all_queries(self) -> list[str]:
        """Return original + transformed queries, deduplicated, order preserved."""
        seen:   set[str]  = set()
        result: list[str] = []
        for q in [self.original_query] + self.queries:
            key = q.strip().lower()
            if key and key not in seen:
                seen.add(key)
                result.append(q.strip())
        return result


class BaseTransformer(ABC):
    """
    Abstract base for all pre-retrieval transformers.

    Parameters
    ----------
    llm_model    : LLM model name used by LLM-based transformers.
    llm_provider : "openai" | "anthropic" | "google"
    language     : "vi" | "en" | "both" — controls prompt language.
    """

    def __init__(
        self,
        llm_model:    str = "gpt-4.1-mini",
        llm_provider: str = "openai",
        language:     str = "both",
    ):
        self.llm_model    = llm_model
        self.llm_provider = llm_provider
        self.language     = language

    @abstractmethod
    def transform(self, query: str) -> TransformResult:
        """Transform a raw query. Must be implemented by subclasses."""