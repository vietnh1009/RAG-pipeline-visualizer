"""
pre_retrieval/query_router.py
==============================
Query Routing — đưa query tới nhánh truy hồi phù hợp nhất.

Mỗi miền tài liệu có thể nằm ở collection riêng hoặc cần chiến lược truy hồi
riêng. Router chọn nhánh và ghi vào ``TransformResult.retrieval_path``.

Ba chế độ
---------
``llm``      : LLM phân loại query theo mô tả từng nhánh. Tốt nhất, tốn 1 lần gọi API.
``keyword``  : Khớp regex theo luật. Không cần LLM, tất định, dễ bảo trì.
``semantic`` : Cosine similarity giữa embedding query và embedding mô tả nhánh.

Dùng khi: kho tri thức trải nhiều miền, hoặc từng loại query cần chiến lược khác nhau.
"""

from __future__ import annotations

import re

from pre_retrieval.base import BaseTransformer, TransformResult
from utils.llm import call_llm


class QueryRouter(BaseTransformer):
    """
    Định tuyến query vào một trong các nhánh truy hồi đã khai báo.

    Tham số
    -------
    routes        : Dict tên nhánh → mô tả, ví dụ::
                      {"medical": "câu hỏi về bệnh, thuốc, điều trị",
                       "legal":   "câu hỏi về luật, quy định, hợp đồng",
                       "general": "các câu hỏi còn lại"}
    mode          : "llm" | "keyword" | "semantic"
    route_rules   : Chế độ keyword — list (regex, tên_nhánh), khớp đầu tiên thắng.
    default_route : Nhánh dự phòng khi không khớp gì.
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
