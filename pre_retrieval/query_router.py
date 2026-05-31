"""
pre_retrieval/query_router.py
==============================
Query Routing — direct the query to the most appropriate retrieval path.

Different types of documents or domains may be stored in separate
collections or require different retrieval strategies. The router selects
the best path and stores it in ``TransformResult.retrieval_path``.

Routing modes
-------------
``llm``     : LLM classifies the query against route descriptions.
              Best quality; costs one API call per query.
``keyword`` : Fast rule-based routing using regex pattern matching.
              No LLM needed; deterministic; easy to maintain.
``semantic``: Cosine similarity between query embedding and route embeddings.
              Good balance between quality and cost.

Use when: knowledge base spans multiple domains or collections; different
          query types need different retrieval strategies or LLMs.
"""

from __future__ import annotations

import re

from pre_retrieval.base import BaseTransformer, TransformResult
from pre_retrieval.utils import call_llm


class QueryRouter(BaseTransformer):
    """
    Route the query to one of the defined retrieval paths.

    Parameters
    ----------
    routes        : Dict mapping route name → description.
                    Example::
                      {
                        "medical": "questions about diseases, treatments, drugs",
                        "legal":   "questions about laws, regulations, contracts",
                        "general": "all other questions"
                      }
    mode          : "llm" | "keyword" | "semantic"
    route_rules   : For keyword mode — list of (regex_pattern, route_name) tuples.
                    First match wins.
    default_route : Fallback when no match is found.
    language      : "vi" | "en" | "both"
    """

    _PROMPT_EN = (
        "Route the following query to the most appropriate knowledge base.\n\n"
        "Available routes:\n{routes}\n\n"
        "Query: {query}\n\n"
        "Return ONLY the route name. Valid values: {route_names}"
    )

    _PROMPT_VI = (
        "Hãy định tuyến câu truy vấn sau đến cơ sở tri thức phù hợp nhất.\n\n"
        "Các tuyến đường hiện có:\n{routes}\n\n"
        "Câu truy vấn: {query}\n\n"
        "Chỉ trả về tên tuyến đường. Giá trị hợp lệ: {route_names}"
    )

    def __init__(
        self,
        routes:        dict[str, str],
        mode:          str  = "llm",
        route_rules:   list[tuple[str, str]] | None = None,
        default_route: str  = "general",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.routes        = routes
        self.mode          = mode
        self.route_rules   = route_rules or []
        self.default_route = default_route

    def transform(self, query: str) -> TransformResult:
        if self.mode == "keyword":
            route = self._route_keyword(query)
        elif self.mode == "semantic":
            route = self._route_semantic(query)
        else:
            route = self._route_llm(query)

        return TransformResult(
            original_query=query,
            queries=[query],
            retrieval_path=route,
        )

    def _route_llm(self, query: str) -> str:
        routes_str  = "\n".join(f"  {name}: {desc}" for name, desc in self.routes.items())
        route_names = ", ".join(f'"{n}"' for n in self.routes)
        tmpl        = self._PROMPT_VI if self.language == "vi" else self._PROMPT_EN
        raw         = call_llm(
            tmpl.format(routes=routes_str, query=query, route_names=route_names),
            self.llm_provider, self.llm_model, max_tokens=30,
        )
        route = raw.strip().lower().strip('"\'')
        return route if route in self.routes else self.default_route

    def _route_keyword(self, query: str) -> str:
        q_lower = query.lower()
        for pattern, route in self.route_rules:
            if re.search(pattern, q_lower):
                return route
        return self.default_route

    def _route_semantic(self, query: str) -> str:
        from sentence_transformers import SentenceTransformer
        import numpy as np

        model      = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        query_emb  = model.encode(query, normalize_embeddings=True)
        route_embs = model.encode(list(self.routes.values()), normalize_embeddings=True)
        scores     = np.dot(route_embs, query_emb)
        best_idx   = int(scores.argmax())
        return list(self.routes.keys())[best_idx]
