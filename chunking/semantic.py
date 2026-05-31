"""
chunking/semantic.py
====================
Semantic chunking — detects topic boundaries via embedding similarity.

Each sentence is embedded; when cosine similarity between consecutive
sentences drops sharply (past a percentile threshold), a chunk boundary
is inserted.  Chunks are therefore grouped by topic, not by character count.

Use when : documents cover multiple unrelated topics; topic shifts need
           to be discovered automatically.

Supported embedding providers
------------------------------
API-based  : OpenAI, Google (Gemini), Cohere, Voyage AI, Jina AI
Self-hosted: Ollama (Qwen3-Embedding), HuggingFace sentence-transformers
             (BGE-M3, GTE-Multilingual, Multilingual-E5)

Note: the embedding model used for chunking and the one used for retrieval
should be the *same* model — cosine similarity boundaries only make sense
within a consistent vector space.
"""

from __future__ import annotations

import os
from langchain_core.documents import Document

from chunking.base import BaseChunker


# ── Embedding model registry ──────────────────────────────────────────────────
# Keys are the exact model identifiers passed to each provider's API/SDK.
EMBEDDING_MODELS: dict[str, dict] = {

    # ── API-based ─────────────────────────────────────────────────────────────

    "text-embedding-3-small": {
        "provider":     "openai",
        "display":      "OpenAI · text-embedding-3-small",
        "dim":          1536,
        "local":        False,
        "mteb":         "~52",
        "note":         "Rẻ nhất OpenAI ($0.02/1M tokens), đủ cho hầu hết RAG. Hỗ trợ Matryoshka.",
        "requires_env": "OPENAI_API_KEY",
        "install":      "pip install langchain-openai",
    },
    "text-embedding-3-large": {
        "provider":     "openai",
        "display":      "OpenAI · text-embedding-3-large",
        "dim":          3072,
        "local":        False,
        "mteb":         "64.6",
        "note":         "Tốt nhất trong dòng OpenAI, ecosystem tích hợp rộng. $0.13/1M tokens.",
        "requires_env": "OPENAI_API_KEY",
        "install":      "pip install langchain-openai",
    },
    "models/gemini-embedding-exp-03-07": {
        "provider":     "google",
        "display":      "Google · Gemini Embedding 2",
        "dim":          3072,
        "local":        False,
        "mteb":         "68.32 ⭐ SOTA",
        "note":         "SOTA MMTEB Multilingual, free tier có sẵn, hỗ trợ multimodal (text + ảnh + PDF).",
        "requires_env": "GOOGLE_API_KEY",
        "install":      "pip install langchain-google-genai",
    },
    "embed-multilingual-v3.0": {
        "provider":     "cohere",
        "display":      "Cohere · embed-multilingual-v3.0",
        "dim":          1024,
        "local":        False,
        "mteb":         "65.2",
        "note":         "Context window 128K token — tốt nhất cho tài liệu rất dài.",
        "requires_env": "COHERE_API_KEY",
        "install":      "pip install langchain-cohere",
    },
    "voyage-4-large": {
        "provider":     "voyage",
        "display":      "Voyage AI · voyage-4-large",
        "dim":          2048,
        "local":        False,
        "mteb":         "~65+",
        "note":         "Tối ưu cho document retrieval, độ chính xác cao nhất trong dòng Voyage.",
        "requires_env": "VOYAGE_API_KEY",
        "install":      "pip install langchain-voyageai",
    },
    "voyage-4": {
        "provider":     "voyage",
        "display":      "Voyage AI · voyage-4",
        "dim":          1024,
        "local":        False,
        "mteb":         "~64",
        "note":         "Cân bằng tốt giữa tốc độ và chất lượng trong dòng Voyage.",
        "requires_env": "VOYAGE_API_KEY",
        "install":      "pip install langchain-voyageai",
    },
    "jina-embeddings-v3": {
        "provider":     "jina",
        "display":      "Jina AI · jina-embeddings-v3",
        "dim":          1024,
        "local":        False,
        "mteb":         "65.5",
        "note":         "$0.02/1M token — rẻ ngang OpenAI small nhưng chất lượng cao hơn. Context 8192 token.",
        "requires_env": "JINA_API_KEY",
        "install":      "pip install langchain-community",
    },

    # ── Self-hosted · Ollama ──────────────────────────────────────────────────

    "qwen3-embedding:8b": {
        "provider": "ollama",
        "display":  "Qwen3-Embedding-8B · Ollama",
        "dim":      4096,
        "local":    True,
        "mteb":     "70.58 🏆",
        "note":     "#1 MTEB Multilingual (Jun 2025). Cần ~16GB VRAM hoặc ~32GB RAM (CPU).",
        "install":  "ollama pull qwen3-embedding:8b",
    },
    "qwen3-embedding:0.6b": {
        "provider": "ollama",
        "display":  "Qwen3-Embedding-0.6B · Ollama",
        "dim":      1024,
        "local":    True,
        "mteb":     "~65",
        "note":     "Nhỏ gọn (~2GB RAM), chạy tốt trên CPU. Vẫn vượt nhiều model lớn hơn trên benchmark.",
        "install":  "ollama pull qwen3-embedding:0.6b",
    },

    # ── Self-hosted · HuggingFace ─────────────────────────────────────────────

    "BAAI/bge-m3": {
        "provider": "huggingface",
        "display":  "BGE-M3 · BAAI",
        "dim":      1024,
        "local":    True,
        "mteb":     "63.0",
        "note":     "Dense + sparse + multi-vector trong 1 model. Context 8192 token. ~570MB.",
        "install":  "pip install langchain-huggingface sentence-transformers",
    },
    "Alibaba-NLP/gte-multilingual-base": {
        "provider": "huggingface",
        "display":  "GTE-Multilingual-Base · Alibaba",
        "dim":      768,
        "local":    True,
        "mteb":     "65.22 (VN-MTEB ⭐)",
        "note":     "Tốt nhất nhóm <500M params cho tiếng Việt theo VN-MTEB benchmark. ~300MB.",
        "install":  "pip install langchain-huggingface sentence-transformers",
    },
    "intfloat/multilingual-e5-large-instruct": {
        "provider": "huggingface",
        "display":  "Multilingual-E5-Large-Instruct · Microsoft",
        "dim":      1024,
        "local":    True,
        "mteb":     "~63",
        "note":     "Instruction-tuned, tốt cho cross-lingual retrieval. ~560MB.",
        "install":  "pip install langchain-huggingface sentence-transformers",
    },
}

# UI grouping — thứ tự quyết định thứ tự hiển thị trong selectbox
PROVIDER_GROUPS: dict[str, list[str]] = {
    "🔑 OpenAI API":                ["text-embedding-3-small", "text-embedding-3-large"],
    "🔑 Google API":                ["models/gemini-embedding-exp-03-07"],
    "🔑 Cohere API":                ["embed-multilingual-v3.0"],
    "🔑 Voyage AI API":             ["voyage-4-large", "voyage-4"],
    "🔑 Jina AI API":               ["jina-embeddings-v3"],
    "🏠 Self-hosted · Ollama":      ["qwen3-embedding:8b", "qwen3-embedding:0.6b"],
    "🏠 Self-hosted · HuggingFace": [
        "BAAI/bge-m3",
        "Alibaba-NLP/gte-multilingual-base",
        "intfloat/multilingual-e5-large-instruct",
    ],
}


class SemanticChunker(BaseChunker):
    """
    Split documents at topic-shift boundaries detected via sentence embeddings.

    Uses LangChain's SemanticChunker (Greg Kamradt implementation).

    Parameters
    ----------
    embedding_model_name : Model identifier — key in EMBEDDING_MODELS registry.
    breakpoint_type      : "percentile" | "standard_deviation" | "interquartile"
    breakpoint_threshold : Numeric threshold for the chosen breakpoint type.
    ollama_base_url      : Ollama URL (OpenAI-compat format, /v1 stripped internally).
    """

    def __init__(
        self,
        embedding_model_name: str   = "BAAI/bge-m3",
        breakpoint_type:      str   = "percentile",
        breakpoint_threshold: float = 95.0,
        chunk_size:           int   = 1000,   # inherited but unused by SemanticChunker
        chunk_overlap:        int   = 0,
        ollama_base_url:      str   = "http://localhost:11434/v1",
    ):
        super().__init__(chunk_size, chunk_overlap)
        self.embedding_model_name = embedding_model_name
        self.breakpoint_type      = breakpoint_type
        self.breakpoint_threshold = breakpoint_threshold
        self.ollama_base_url      = ollama_base_url

    # ------------------------------------------------------------------
    # Provider dispatch
    # ------------------------------------------------------------------

    def _build_embeddings(self):
        """Khởi tạo embeddings object phù hợp với provider của model đã chọn."""
        meta     = EMBEDDING_MODELS.get(self.embedding_model_name, {})
        provider = meta.get("provider", "huggingface")

        if provider == "openai":
            from langchain_openai import OpenAIEmbeddings
            return OpenAIEmbeddings(model=self.embedding_model_name)

        if provider == "google":
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
            return GoogleGenerativeAIEmbeddings(model=self.embedding_model_name)

        if provider == "cohere":
            from langchain_cohere import CohereEmbeddings
            return CohereEmbeddings(model=self.embedding_model_name)

        if provider == "voyage":
            from langchain_voyageai import VoyageAIEmbeddings
            return VoyageAIEmbeddings(model=self.embedding_model_name)

        if provider == "jina":
            from langchain_community.embeddings import JinaEmbeddings
            return JinaEmbeddings(
                jina_api_key=os.getenv("JINA_API_KEY", ""),
                model_name=self.embedding_model_name,
            )

        if provider == "ollama":
            from langchain_ollama import OllamaEmbeddings
            from chunking.utils import ensure_ollama_model

            # OllamaEmbeddings dùng native Ollama API, không phải OpenAI-compat → bỏ /v1
            host = self.ollama_base_url.rstrip("/")
            if host.endswith("/v1"):
                host = host[:-3]

            ensure_ollama_model(self.embedding_model_name, host)
            return OllamaEmbeddings(model=self.embedding_model_name, base_url=host)

        # Default: HuggingFace sentence-transformers — tự động tải từ Hub nếu chưa có
        from langchain_huggingface import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(model_name=self.embedding_model_name)

    # ------------------------------------------------------------------
    # Main split
    # ------------------------------------------------------------------

    def split(self, docs: list[Document]) -> list[Document]:
        from langchain_experimental.text_splitter import SemanticChunker as _SC

        embeddings = self._build_embeddings()
        splitter   = _SC(
            embeddings=embeddings,
            breakpoint_threshold_type=self.breakpoint_type,
            breakpoint_threshold_amount=self.breakpoint_threshold,
        )

        try:
            return self._enrich(splitter.split_documents(docs))
        except Exception as exc:
            # Safety net: nếu Ollama trả về 404 (model chưa pull) dù
            # ensure_ollama_model đã được gọi ở _build_embeddings(),
            # thử pull lại rồi retry một lần.
            meta     = EMBEDDING_MODELS.get(self.embedding_model_name, {})
            provider = meta.get("provider", "")
            if provider == "ollama" and _is_model_not_found_error(exc):
                from chunking.utils import ensure_ollama_model
                host = self.ollama_base_url.rstrip("/")
                if host.endswith("/v1"):
                    host = host[:-3]
                print(
                    f"[SemanticChunker] Model '{self.embedding_model_name}' chưa sẵn sàng "
                    f"— kích hoạt pull lần 2 và retry..."
                )
                ensure_ollama_model(self.embedding_model_name, host)
                # Khởi tạo lại embeddings sau khi pull hoàn tất
                embeddings = self._build_embeddings()
                splitter   = _SC(
                    embeddings=embeddings,
                    breakpoint_threshold_type=self.breakpoint_type,
                    breakpoint_threshold_amount=self.breakpoint_threshold,
                )
                return self._enrich(splitter.split_documents(docs))
            raise


# ── Helper ────────────────────────────────────────────────────────────────────

def _is_model_not_found_error(exc: Exception) -> bool:
    """
    Phát hiện lỗi 'model not found' từ Ollama API (status 404).
    Tránh dùng import cụ thể vì tên exception class thay đổi giữa các
    phiên bản ollama / langchain-ollama.
    """
    msg = str(exc).lower()
    return (
        "not found" in msg and ("pulling" in msg or "pull" in msg or "404" in msg)
    ) or "status code: 404" in msg
