import os
import sys
import re
from pathlib import Path

from core.constants import EMBED_PREVIEW_LIMIT
from utils.env import _friendly_import_error
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

try:
    from dotenv import load_dotenv as _load_dotenv, dotenv_values as _dotenv_values
    _load_dotenv(override=True)
    _ENV = _dotenv_values()
except ImportError:
    _ENV = {}

import streamlit as st

import inspect
import concurrent.futures
from langchain_core.documents import Document

def run_loader(
    source_path:     str,
    pdf_strategy:    str,
    extract_tables:  bool,
    language:        str,
    marker_device:   str  = "cpu",
    describe_images: bool = False,
    vision_model:    str  = "gpt-4o-mini",
    vision_provider: str  = "openai",
    ollama_base_url: str  = "http://localhost:11434/v1",
    odl_hybrid:      str | None = None,
    odl_struct_tree: bool = False,
) -> list:
    """Chạy PDFDocumentLoader và trả về list[Document]."""
    from loader.directory_loader import PDFDocumentLoader

    loader = PDFDocumentLoader(
        language=language,
        pdf_strategy=pdf_strategy,
        extract_tables=extract_tables,
        deduplicate=True,
        marker_device=marker_device,
        describe_images=describe_images,
        vision_model=vision_model,
        vision_provider=vision_provider,
        ollama_base_url=ollama_base_url,
        odl_hybrid=odl_hybrid,
        odl_struct_tree=odl_struct_tree,
    )

    path = Path(source_path)
    try:
        return loader.load(str(path))
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            f"{_friendly_import_error(exc)}\n\n"
            f"[Debug] Lỗi thật: {type(exc).__name__}: {exc}\n"
            f"[Debug] sys.executable: {__import__('sys').executable}"
        ) from exc


def run_chunker(
    docs: list,
    strategy: str,
    chunk_size: int,
    chunk_overlap: int,
    extra_kwargs: dict,
) -> list:
    """Chạy chunker đã chọn và trả về list[Document] chunks."""
    from chunking.factory import get_chunker

    kwargs = {"chunk_size": chunk_size, "chunk_overlap": chunk_overlap}
    kwargs.update(extra_kwargs)

    chunker = get_chunker(strategy, **kwargs)
    return chunker.split(docs)


def _ensure_ollama_model(model_name: str, base_url: str) -> None:
    """
    Kiểm tra và tự pull Ollama model nếu chưa có.
    Raise RuntimeError nếu server không kết nối được.
    """
    import ollama as _ollama

    host = base_url.rstrip("/")
    try:
        client = _ollama.Client(host=host)
        pulled = {m.model for m in client.list().models}
        if model_name not in pulled:
            client.pull(model_name)   # blocking — chạy trong thread riêng của run_embedder
    except Exception as exc:
        raise RuntimeError(
            f"Không kết nối được Ollama tại `{host}`. "
            f"Đảm bảo Ollama đang chạy (`ollama serve`). Chi tiết: {exc}"
        ) from exc


def _ensure_huggingface_model(model_name: str) -> None:
    """
    Đảm bảo HuggingFace model đã được cache local.
    Truyền HF_TOKEN để authenticate với gated/private models.
    Raise RuntimeError với thông báo rõ ràng nếu auth thất bại.
    """
    _TRUST_REMOTE_CODE_MODELS = {
        "nomic-ai/nomic-embed-text-v1.5",
        "Alibaba-NLP/gte-Qwen2-7B-instruct",
        "Qwen/Qwen3-Embedding-0.6B",
        "Qwen/Qwen3-Embedding-4B",
        "Qwen/Qwen3-Embedding-8B",
    }
    try:
        from huggingface_hub import snapshot_download

        hf_token = (
            os.environ.get("HF_TOKEN")
            or os.environ.get("HUGGINGFACE_API_TOKEN")
            or os.environ.get("HUGGINGFACEHUB_API_TOKEN")
        ) or None

        kwargs: dict = {
            "repo_id":   model_name,
            "repo_type": "model",
        }
        if hf_token:
            kwargs["token"] = hf_token
        if model_name in _TRUST_REMOTE_CODE_MODELS:
            kwargs["ignore_patterns"] = []

        snapshot_download(**kwargs)

    except ImportError:
        pass   # huggingface_hub chưa cài — sentence-transformers tự handle
    except Exception as exc:
        msg = str(exc)
        if "401" in msg or "403" in msg or "credential" in msg.lower():
            raise RuntimeError(
                f"**Xác thực thất bại** khi tải `{model_name}`.\n\n"
                "Kiểm tra:\n"
                "1. `HF_TOKEN` trong `.env` phải là token thật (bắt đầu bằng `hf_`).\n"
                "2. Token phải có quyền **read** trên HuggingFace.\n"
                "3. Với gated model (Qwen3, Llama…): vào trang model trên "
                "huggingface.co và chấp nhận điều khoản trước."
            ) from exc
        if "404" in msg or "not found" in msg.lower() or "repository" in msg.lower():
            raise RuntimeError(
                f"**Model không tìm thấy**: `{model_name}`\n\n"
                "Có thể do:\n"
                "1. Tên model sai — kiểm tra lại trên huggingface.co.\n"
                "2. Model là **gated repo** — cần `HF_TOKEN` hợp lệ và đã chấp nhận "
                "điều khoản trên trang model.\n"
                f"3. Thêm vào `.env`: `HF_TOKEN=hf_xxx...`"
            ) from exc
        # Lỗi khác (network, disk…) — bỏ qua, để HuggingFaceEmbeddings tự báo
        pass


@st.cache_resource(show_spinner="⏳ Đang load embedding model lên GPU...")
def _load_hf_embedder(
    model_name:           str,
    device:               str,
    trust_remote_code:    bool,
    hf_token:             str | None,
    query_instruction:    str | None,
    document_instruction: str | None,
    torch_dtype_str:      str = "auto",   # "auto" | "float32" | "float16" | "bfloat16"
    batch_size:           int = 32,
):
    """
    Load HuggingFaceEmbeddings một lần, cache theo key = tất cả tham số.
    @st.cache_resource giữ object sống suốt session → model không bị load lại
    mỗi lần ấn Process, tránh VRAM tích lũy gây OOM.

    torch_dtype_str: "auto" = fp16 trên CUDA, fp32 trên CPU.
    batch_size: số text xử lý song song — lớn hơn = nhanh hơn nhưng dùng nhiều VRAM hơn.
    """
    from langchain_huggingface import HuggingFaceEmbeddings as _HFEmb

    # ── Resolve dtype ────────────────────────────────────────────────────────
    _dtype_map = {
        "float32":  "float32",
        "float16":  "float16",
        "bfloat16": "bfloat16",
    }
    if torch_dtype_str == "auto":
        resolved_dtype = "float16" if device == "cuda" else "float32"
    else:
        resolved_dtype = _dtype_map.get(torch_dtype_str, "float32")

    try:
        import torch as _torch
        dtype_obj = getattr(_torch, resolved_dtype)
    except Exception:
        dtype_obj = None   # fallback: không set dtype, dùng default

    # ── Build model_kwargs ──────────────────────────────────────────────────
    # HuggingFaceEmbeddings truyền model_kwargs như **kwargs vào SentenceTransformer.
    # sentence-transformers 3.x nhận:
    #   - device, trust_remote_code, token  → top-level params
    #   - torch_dtype                       → phải lồng trong model_kwargs (nested)
    #     vì nó được forward xuống AutoModel.from_pretrained()
    model_kw: dict = {"device": device}
    if trust_remote_code:
        model_kw["trust_remote_code"] = True
    if hf_token:
        model_kw["token"] = hf_token

    # torch_dtype: lồng trong "model_kwargs" nested key
    if dtype_obj is not None and device == "cuda":
        model_kw["model_kwargs"] = {"torch_dtype": dtype_obj}

    # ── Build encode_kwargs ──────────────────────────────────────────────────
    encode_kw: dict = {
        "normalize_embeddings": True,
        "batch_size":           batch_size,
    }

    hf_init: dict = {
        "model_name":    model_name,
        "model_kwargs":  model_kw,
        "encode_kwargs": encode_kw,
    }
    if query_instruction:
        hf_init["query_instruction"]  = query_instruction
    if document_instruction:
        hf_init["embed_instruction"] = document_instruction

    return _HFEmb(**hf_init)


# Module-level constants — không rebuild mỗi lần gọi run_embedder
_CTX_CHAR_LIMITS: dict[str, int] = {
    "mxbai-embed-large":       512  * 3,
    "nomic-embed-text":        2048 * 3,
    "snowflake-arctic-embed":  512  * 3,
    "bge-m3":                  8192 * 3,
    "BAAI/bge-small-en-v1.5":  512  * 3,
    "BAAI/bge-base-en-v1.5":   512  * 3,
    "intfloat/multilingual-e5-small": 512 * 3,
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2": 512 * 3,
    "intfloat/multilingual-e5-large": 512  * 3,
    "VinAI/phobert-large":            256  * 4,
}

_HF_TRUST_RC: frozenset[str] = frozenset({
    "nomic-ai/nomic-embed-text-v1.5",
    "Alibaba-NLP/gte-Qwen2-7B-instruct",
    "Qwen/Qwen3-Embedding-0.6B",
    "Qwen/Qwen3-Embedding-4B",
    "Qwen/Qwen3-Embedding-8B",
})


def run_embedder(
    chunks: list,
    provider: str,
    model_name: str,
    enable_sparse: bool     = False,
    sparse_method: str      = "bm25",
    dimensions: int | None  = None,
    device: str             = "cpu",
    ollama_base_url: str    = "http://localhost:11434",
    input_type: str         = "search_document",
    query_instruction: str | None  = None,
    document_instruction: str | None = None,
    max_chunks: int         = EMBED_PREVIEW_LIMIT,
    torch_dtype_str: str    = "auto",
    batch_size: int         = 32,
) -> dict:
    """
    Chạy embedding trên tối đa `max_chunks` chunks, trả về:
    {
        "dense":   list[list[float]],
        "sparse":  list[dict[str, float]] | None,
        "dims":    int,
        "n_embedded": int,
        "truncated":  bool,
    }
    """
    from embedding.factory import get_embedder

    preview = chunks[:max_chunks]
    texts   = [c.page_content for c in preview]

    # ── Tự động pull / download model nếu chưa có local ─────────────────────
    if provider == "ollama":
        _ensure_ollama_model(model_name, ollama_base_url)
    elif provider == "huggingface":
        _ensure_huggingface_model(model_name)

    # ── Truncate theo context window TRƯỚC KHI embed ─────────────────────────
    _char_limit = _CTX_CHAR_LIMITS.get(model_name)
    if _char_limit:
        texts = [t[:_char_limit] for t in texts]

    # ── Embed ─────────────────────────────────────────────────────────────────
    if provider == "huggingface":
        hf_token = (
            os.environ.get("HF_TOKEN")
            or os.environ.get("HUGGINGFACE_API_TOKEN")
            or os.environ.get("HUGGINGFACEHUB_API_TOKEN")
        ) or None

        lc_embedder = _load_hf_embedder(
            model_name           = model_name,
            device               = device,
            trust_remote_code    = model_name in _HF_TRUST_RC,
            hf_token             = hf_token,
            query_instruction    = query_instruction,
            document_instruction = document_instruction,
            torch_dtype_str      = torch_dtype_str,
            batch_size           = batch_size,
        )
        dense = lc_embedder.embed_documents(texts)

        # Dọn CUDA fragment sau khi embed xong (không unload model)
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    else:
        extra: dict = {}
        if provider == "openai":
            if dimensions:
                extra["dimensions"] = dimensions
        elif provider == "cohere":
            extra["input_type"] = input_type
        elif provider == "ollama":
            extra["base_url"] = ollama_base_url
        # fastembed: không cần extra kwargs

        embedder = get_embedder(provider, model_name, **extra)
        dense    = embedder.embed_documents(texts)

    sparse = None
    if enable_sparse:
        from embedding.sparse_embedder import get_sparse_embedder
        sp = get_sparse_embedder(sparse_method)
        sp.fit(texts)
        sparse = sp.embed_documents(texts)

    # ── Sanitize: thay NaN/Inf bằng 0.0 (Ollama/local models đôi khi trả NaN) ─
    import math
    def _sanitize_vec(v: list[float]) -> list[float]:
        return [0.0 if (x != x or not math.isfinite(x)) else x for x in v]

    dense = [_sanitize_vec(v) for v in dense]
    if sparse:
        # sparse là list[dict[str, float]]
        def _sanitize_sparse(d: dict) -> dict:
            return {k: (0.0 if (v != v or not math.isfinite(v)) else v) for k, v in d.items()}
        sparse = [_sanitize_sparse(s) for s in sparse]

    dims = len(dense[0]) if dense else 0
    return {
        "dense":      dense,
        "sparse":     sparse,
        "dims":       dims,
        "n_embedded": len(dense),
        "truncated":  len(chunks) > max_chunks,
    }



