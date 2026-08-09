"""
embedding/pipeline.py
=====================
EmbeddingPipeline — ghép embedder dense với embedder sparse tuỳ chọn.

Đây là đối tượng mà ``vector_db`` và ``retrieval`` thực sự dùng: một giao diện
duy nhất sinh ra cả vector dense lẫn sparse.

Ví dụ
-----
    pipeline = EmbeddingPipeline(
        dense_provider="huggingface",
        dense_model="BAAI/bge-m3",
        enable_sparse=True,
        sparse_method="bm25",
    )
    pipeline.fit_sparse(corpus_texts)        # BM25 bắt buộc bước này
    pipeline.embed_documents(texts)          # {"dense": ..., "sparse": ...}
"""

from __future__ import annotations

from typing import Any

from langchain_core.embeddings import Embeddings

from embedding.sparse_embedder import SparseMethod, get_sparse_embedder


class EmbeddingPipeline:
    """
    Giao diện thống nhất cho embedding dense + sparse tuỳ chọn.

    Tham số
    -------
    dense_provider : Tên provider — xem ``factory.py`` để biết giá trị hợp lệ.
    dense_model    : Tên model cho embedder dense.
    dense_kwargs   : Tham số phụ chuyển thẳng cho constructor của embedder dense.
    enable_sparse  : Tính thêm vector sparse (BM25 / SPLADE).
    sparse_method  : "bm25" | "splade"
    sparse_model   : Tên model SPLADE; BM25 bỏ qua tham số này.
    """

    def __init__(
        self,
        dense_provider: str          = "openai",
        dense_model:    str          = "text-embedding-3-small",
        dense_kwargs:   dict | None  = None,
        enable_sparse:  bool         = False,
        sparse_method:  SparseMethod = "bm25",
        sparse_model:   str          = "naver/splade-cocondenser-ensembledistil",
    ):
        from embedding.factory import get_embedder

        self.enable_sparse  = enable_sparse
        self._dense         = get_embedder(dense_provider, dense_model, **(dense_kwargs or {}))
        self._sparse        = get_sparse_embedder(sparse_method, sparse_model) if enable_sparse else None

    # ------------------------------------------------------------------
    # Fit trên corpus — BM25 bắt buộc chạy trước khi embed
    # ------------------------------------------------------------------

    def fit_sparse(self, corpus_texts: list[str]) -> "EmbeddingPipeline":
        """Tính thống kê IDF của BM25 trên corpus. SPLADE không cần, bỏ qua."""
        if self._sparse:
            self._sparse.fit(corpus_texts)
        return self

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

    def embed_documents(self, texts: list[str]) -> dict[str, Any]:
        """
        Embed một danh sách text tài liệu.

        Trả về
        ------
        {"dense":  list[list[float]],             # luôn có
         "sparse": list[dict[str, float]] | None} # chỉ khi enable_sparse
        """
        return {
            "dense":  self._dense.embed_documents(texts),
            "sparse": self._sparse.embed_documents(texts) if self._sparse else None,
        }

    def embed_query(self, query: str) -> dict[str, Any]:
        """
        Embed một chuỗi query.

        Trả về ``{"dense": list[float], "sparse": dict[str, float] | None}``.
        """
        return {
            "dense":  self._dense.embed_query(query),
            "sparse": self._sparse.embed_query(query) if self._sparse else None,
        }

    @property
    def langchain_embedder(self) -> Embeddings:
        """
        Đối tượng Embeddings gốc của LangChain.

        Truyền cái này cho các vector store đòi đúng kiểu Embeddings chuẩn
        (Chroma, FAISS, pgvector…).
        """
        return self._dense.embedder
