"""
app_visualizer.py
=================
RAG Pipeline Visualizer — trực quan hóa toàn bộ RAG indexing & query pipeline.

Khởi động: streamlit run app_visualizer.py
"""

import importlib
import importlib.util
import inspect
import os
import sys
import re
import tempfile
from pathlib import Path

# ── CUDA memory allocator — phải set TRƯỚC khi import torch ──────────────────
# expandable_segments: PyTorch trả memory về OS khi không dùng thay vì giữ lại
# → giảm fragmentation, tránh OOM khi chạy nhiều model liên tiếp
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# Dotenv setup: hai mục đích khác nhau
#   1. load_dotenv()    → inject vào os.environ để các thư viện (openai, cohere, …) đọc được
#   2. dotenv_values()  → đọc vào dict riêng để UI check placeholder, KHÔNG dùng os.environ
try:
    from dotenv import load_dotenv as _load_dotenv, dotenv_values as _dotenv_values
    _load_dotenv(override=True)   # inject vào os.environ; override=True → .env ghi đè system env vars
    _ENV = _dotenv_values()        # dict thuần để _get_env() detect placeholder
except ImportError:
    _ENV = {}   # python-dotenv chưa cài — dùng system env vars

def _get_env(key: str) -> str | None:
    """
    Lay gia tri key tu file .env.
    Tra ve None neu key khong ton tai, rong, hoac la placeholder chua dien.
    """
    val = _ENV.get(key)
    if not val:
        return None
    low = val.strip().lower()
    if (
        low.startswith("your_")
        or "_here" in low
        or low.startswith("<")
        or low.endswith(">")
        or low in ("", "none", "null", "placeholder", "changeme", "xxx")
    ):
        return None
    return val

import streamlit as st

from pipeline_cache import PipelineCache

# ── Global cache instance ─────────────────────────────────────────────────────
@st.cache_resource
def _get_pipeline_cache() -> PipelineCache:
    return PipelineCache("processed_data")

# Thêm thư mục gốc của project vào sys.path
# để import được các module loader/ và chunking/
sys.path.insert(0, str(Path(__file__).parent))

# ─── Hằng số & cấu hình UI ────────────────────────────────────────────────────

PAGE_TITLE = "RAG Pipeline Visualizer"

# Màu nhãn theo loại file
FILE_TYPE_COLORS = {
    "pdf":      "#e74c3c",
    "docx":     "#2980b9",
    "xlsx":     "#27ae60",
    "csv":      "#16a085",
    "markdown": "#8e44ad",
    "html":     "#e67e22",
    "code":     "#2c3e50",
    "txt":      "#7f8c8d",
    "email":    "#c0392b",
    "epub":     "#d35400",
    "image":    "#f39c12",
    "json":     "#1abc9c",
}

# Mô tả ngắn cho từng chiến lược chunking
CHUNK_DESCRIPTIONS = {
    "recursive":        "⭐ Cắt theo thứ tự ưu tiên: đoạn văn → dòng → câu → ký tự. Khuyến nghị mặc định.",
    "token_based":      "Đếm token thay vì ký tự — quan trọng với tiếng Việt & đa ngôn ngữ.",
    "sentence_aware":   "Ranh giới chunk luôn trùng với cuối câu — tốt cho FAQ, Q&A.",
    "semantic":         "Phát hiện ranh giới chủ đề qua cosine similarity — tốt cho văn bản đa chủ đề.",
    "contextual":       "LLM sinh context prefix cho mỗi chunk — cải thiện recall đáng kể (Anthropic 2024).",
    "hierarchical":     "Tạo cặp parent (lớn) + child (nhỏ) — search child, trả parent cho LLM.",
    "format_aware":     "Cắt theo cấu trúc tài liệu: Markdown heading, AST code, HTML tag.",
}

# Chiến lược cần LLM (sẽ hiện cảnh báo)
LLM_REQUIRED_STRATEGIES = {"contextual"}

# ── Embedding provider metadata ──────────────────────────────────────────────
# Mỗi entry: models, default, env_var, local, mrl_support, dim, vi_quality, note
EMBEDDING_PROVIDER_META: dict[str, dict] = {
    "openai": {
        "label":       "OpenAI",
        "icon":        "🤖",
        "models":      ["text-embedding-3-small", "text-embedding-3-large", "text-embedding-ada-002"],
        "default":     "text-embedding-3-small",
        "model_dims":  {"text-embedding-3-small": 1536, "text-embedding-3-large": 3072, "text-embedding-ada-002": 1536},
        "requires_env": "OPENAI_API_KEY",
        "local":       False,
        "mrl":         True,   # hỗ trợ truncation qua dimensions=
        "vi_quality":  "⭐⭐⭐",
        "note":        "$0.02/1M (3-small) · $0.13/1M (3-large). 3-small là mặc định tốt nhất cho tiếng Anh/đa ngôn ngữ cơ bản.",
        "install":     "pip install langchain-openai",
        "pkg_probe":   "langchain_openai",
    },
    "cohere": {
        "label":       "Cohere",
        "icon":        "🌀",
        "models":      ["embed-multilingual-v3.0", "embed-english-v3.0", "embed-v4.0"],
        "default":     "embed-multilingual-v3.0",
        "model_dims":  {"embed-multilingual-v3.0": 1024, "embed-english-v3.0": 1024, "embed-v4.0": 1536},
        "requires_env": "COHERE_API_KEY",
        "local":       False,
        "mrl":         False,
        "vi_quality":  "⭐⭐⭐⭐",
        "note":        "$0.10/1M · 108 ngôn ngữ · embed-multilingual-v3.0 là lựa chọn hàng đầu cho tiếng Việt qua API.",
        "install":     "pip install langchain-cohere",
        "pkg_probe":   "langchain_cohere",
        "asymmetric":  True,    # cần input_type khác cho doc vs query
        "doc_input_type": "search_document",
        "query_input_type": "search_query",
    },
    "huggingface": {
        "label":       "HuggingFace (local)",
        "icon":        "🤗",
        "models":      [
            "BAAI/bge-m3",
            "Qwen/Qwen3-Embedding-0.6B",
            "Qwen/Qwen3-Embedding-4B",
            "Qwen/Qwen3-Embedding-8B",
            "intfloat/multilingual-e5-large",
            "intfloat/e5-mistral-7b-instruct",
            "nomic-ai/nomic-embed-text-v1.5",
            "Alibaba-NLP/gte-Qwen2-7B-instruct",
            "VinAI/phobert-large",
        ],
        "default":     "BAAI/bge-m3",
        "model_dims":  {
            "BAAI/bge-m3": 1024,
            "Qwen/Qwen3-Embedding-0.6B": 1024,
            "Qwen/Qwen3-Embedding-4B": 2560,
            "Qwen/Qwen3-Embedding-8B": 4096,
            "intfloat/multilingual-e5-large": 1024,
            "intfloat/e5-mistral-7b-instruct": 4096,
            "nomic-ai/nomic-embed-text-v1.5": 768,
            "Alibaba-NLP/gte-Qwen2-7B-instruct": 3584,
            "VinAI/phobert-large": 768,
        },
        "model_notes": {
            "BAAI/bge-m3":                       "1024d · 8192 ctx · Dense+sparse+multivec · ⭐⭐⭐⭐ VI",
            "Qwen/Qwen3-Embedding-0.6B":         "1024d · 32K ctx · MTEB #1 nhẹ nhất · ⭐⭐⭐⭐⭐ VI",
            "Qwen/Qwen3-Embedding-4B":           "2560d · 32K ctx · MTEB #1 cân bằng · ⭐⭐⭐⭐⭐ VI",
            "Qwen/Qwen3-Embedding-8B":           "4096d · 32K ctx · MTEB #1 tốt nhất · ⭐⭐⭐⭐⭐ VI",
            "intfloat/multilingual-e5-large":    "1024d · 512 ctx · 100 ngôn ngữ · ⭐⭐⭐ VI",
            "intfloat/e5-mistral-7b-instruct":   "4096d · 32K ctx · Instruction-tuned · ⭐⭐⭐ VI",
            "nomic-ai/nomic-embed-text-v1.5":    "768d · 8192 ctx · MRL support · ⭐⭐⭐ VI",
            "Alibaba-NLP/gte-Qwen2-7B-instruct": "3584d · 131K ctx · Long context · ⭐⭐⭐⭐ VI",
            "VinAI/phobert-large":               "768d · 256 ctx · Vietnamese-specific · ⭐⭐⭐⭐ VI",
        },
        "requires_env": None,
        "local":       True,
        "mrl":         False,
        "vi_quality":  "⭐⭐⭐⭐⭐",
        "note":        "Chạy hoàn toàn local. Model tự động tải từ HuggingFace Hub lần đầu dùng.",
        "install":     "pip install langchain-huggingface sentence-transformers",
        "pkg_probe":   "sentence_transformers",
    },
    "ollama": {
        "label":       "Ollama (local)",
        "icon":        "🦙",
        "models":      ["nomic-embed-text", "mxbai-embed-large", "bge-m3", "snowflake-arctic-embed"],
        "default":     "nomic-embed-text",
        "model_dims":  {"nomic-embed-text": 768, "mxbai-embed-large": 1024, "bge-m3": 1024, "snowflake-arctic-embed": 1024},
        "model_notes": {
            "nomic-embed-text":      "768d · ctx 2048 token · Fast · English",
            "mxbai-embed-large":     "1024d · ctx 512 token · Strong general purpose",
            "bge-m3":                "1024d · ctx 8192 token · Multilingual · tốt nhất cho tiếng Việt",
            "snowflake-arctic-embed":"1024d · ctx 512 token · Strong English retrieval",
        },
        "requires_env": None,
        "local":       True,
        "mrl":         False,
        "vi_quality":  "⭐⭐⭐",
        "note":        "Cần Ollama server đang chạy. Pull model trước: `ollama pull <model>`.",
        "install":     "pip install langchain-ollama",
        "pkg_probe":   "langchain_ollama",
    },
    "fastembed": {
        "label":       "FastEmbed (local · CPU)",
        "icon":        "⚡",
        "models":      [
            "BAAI/bge-small-en-v1.5",
            "BAAI/bge-base-en-v1.5",
            "intfloat/multilingual-e5-small",
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        ],
        "default":     "BAAI/bge-small-en-v1.5",
        "model_dims":  {
            "BAAI/bge-small-en-v1.5": 384,
            "BAAI/bge-base-en-v1.5": 768,
            "intfloat/multilingual-e5-small": 384,
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2": 384,
        },
        "model_notes": {
            "BAAI/bge-small-en-v1.5": "384d · Nhanh nhất · English",
            "BAAI/bge-base-en-v1.5":  "768d · Chất lượng tốt hơn · English",
            "intfloat/multilingual-e5-small": "384d · Đa ngôn ngữ (VI ok) · ONNX",
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2": "384d · Đa ngôn ngữ · Nhẹ",
        },
        "requires_env": None,
        "local":       True,
        "mrl":         False,
        "vi_quality":  "⭐⭐⭐",
        "note":        "CPU-optimised ONNX. Không cần GPU. Nhanh hơn sentence-transformers trên CPU.",
        "install":     "pip install fastembed langchain-community",
        "pkg_probe":   "fastembed",
    },
}

# Số lượng chunk tối đa embed trong preview (tránh tốn API cost / RAM)
EMBED_PREVIEW_LIMIT = 20

# Package cần cài thêm cho từng PDF strategy: (module_to_probe, install_cmd)
PDF_STRATEGY_DEPS = {
    "pypdf":        None,
    "pymupdf":      ("fitz",          "pip install pymupdf"),
    "pdfplumber":   ("pdfplumber",    "pip install pdfplumber tabulate"),
    "unstructured": ("unstructured",  "pip install 'unstructured[pdf]' unstructured-inference"),
    "docling":      ("docling",       "pip install docling"),
    "marker":       ("marker",        "pip install marker-pdf"),
    "opendataloader": ("opendataloader_pdf", "pip install opendataloader-pdf  # Java 11+ required"),
}


# ── Vector DB provider metadata ───────────────────────────────────────────────
VECTOR_DB_PROVIDER_META: dict[str, dict] = {
    "faiss": {
        "icon": "💾", "label": "FAISS",
        "tier": "free", "tier_icon": "🆓",
        "mode": "local", "local": True,
        "scale": "< 10M vectors",
        "hybrid": False, "filtering": "post-filter",
        "requires_env": None,
        "pkg_probe": "langchain_openai",
        "note": (
            "**Meta AI.** Chạy hoàn toàn in-process — không server, không API key. "
            "Lý tưởng cho prototype, offline pipeline, CI/CD. Index được lưu vào disk.\n\n"
            "⚠️ Không có metadata filtering native — filter sau khi retrieve."
        ),
        "install": "pip install faiss-cpu langchain-community",
        "pkg_probe": "faiss",
        "params": ["persist_dir"],
    },
    "chroma": {
        "icon": "🎨", "label": "Chroma",
        "tier": "free", "tier_icon": "🆓",
        "mode": "local", "local": True,
        "scale": "< 10M vectors",
        "hybrid": False, "filtering": "native",
        "requires_env": None,
        "pkg_probe": "langchain_cohere",
        "note": (
            "**Chroma.** SQLite backend, Python-native, hỗ trợ metadata filtering. "
            "Dễ dùng nhất cho dev và demo. Collection persist giữa các lần chạy.\n\n"
            "✅ Metadata filtering · 🆓 Zero config · 📦 Tích hợp LangChain tốt nhất"
        ),
        "install": "pip install langchain-chroma",
        "pkg_probe": "chromadb",
        "params": ["persist_dir"],
    },
    "lancedb": {
        "icon": "🏹", "label": "LanceDB",
        "tier": "free", "tier_icon": "🆓",
        "mode": "local / cloud", "local": True,
        "scale": "~ 1B vectors",
        "hybrid": False, "filtering": "native",
        "requires_env": None,
        "pkg_probe": "sentence_transformers",
        "note": (
            "**LanceDB.** Columnar format (Lance ≈ Parquet), embedded mode không cần server. "
            "Tích hợp tốt với pandas/Arrow/DuckDB. Hỗ trợ LanceDB Cloud (serverless).\n\n"
            "✅ Không server · ✅ Columnar = đọc nhanh · ✅ DuckDB SQL trên vectors"
        ),
        "install": "pip install lancedb langchain-community",
        "pkg_probe": "lancedb",
        "params": ["persist_dir", "distance"],
    },
    "qdrant": {
        "icon": "🎯", "label": "Qdrant",
        "tier": "free / paid", "tier_icon": "🔓",
        "mode": "self-host / cloud", "local": False,
        "scale": "~ 1B+ vectors",
        "hybrid": True, "filtering": "native (ACORN)",
        "requires_env": "QDRANT_URL",
        "note": (
            "**Qdrant.** Vector DB tốt nhất cho metadata filtering phức tạp nhờ thuật toán "
            "**ACORN** (filter trong graph traversal, không phải sau). Hỗ trợ hybrid dense+sparse. "
            "Có thể self-host bằng Docker hoặc dùng Qdrant Cloud.\n\n"
            "✅ ACORN filtering · ✅ Hybrid search · ✅ Self-host hoặc Cloud"
        ),
        "install": "pip install qdrant-client langchain-qdrant",
        "pkg_probe": "qdrant_client",
        "params": ["url", "api_key", "distance", "on_disk"],
    },
    "weaviate": {
        "icon": "🕸️", "label": "Weaviate",
        "tier": "free / paid", "tier_icon": "🔓",
        "mode": "self-host / cloud", "local": False,
        "scale": "~ 1B vectors",
        "hybrid": True, "filtering": "native",
        "requires_env": "WEAVIATE_URL",
        "note": (
            "**Weaviate.** Lưu cả dense vector lẫn BM25 term freq natively → hybrid search "
            "out-of-the-box không cần config thêm. GraphQL API cho filtering phức tạp.\n\n"
            "✅ Hybrid search mặc định · ✅ GraphQL API · ✅ Schema linh hoạt"
        ),
        "install": "pip install weaviate-client langchain-weaviate",
        "pkg_probe": "weaviate",
        "params": ["url", "api_key"],
    },
    "pgvector": {
        "icon": "🐘", "label": "pgvector (PostgreSQL)",
        "tier": "free", "tier_icon": "🆓",
        "mode": "self-host", "local": False,
        "scale": "< 50M vectors",
        "hybrid": False, "filtering": "native (SQL)",
        "requires_env": "DATABASE_URL",
        "note": (
            "**pgvector.** Extension PostgreSQL — ACID, JOINs với các bảng khác, SQL filtering "
            "đầy đủ. Best choice nếu đã có PostgreSQL trong stack.\n\n"
            "✅ ACID transactions · ✅ SQL filtering · ✅ Join với bảng khác · "
            "⚠️ Không scale > 50M vectors tốt"
        ),
        "install": "pip install langchain-postgres psycopg",
        "pkg_probe": "psycopg",
        "params": ["connection_string", "distance_strategy"],
    },
    "pinecone": {
        "icon": "🌲", "label": "Pinecone",
        "tier": "paid", "tier_icon": "💳",
        "mode": "managed cloud", "local": False,
        "scale": "~ 1B vectors",
        "hybrid": True, "filtering": "native",
        "requires_env": "PINECONE_API_KEY",
        "note": (
            "**Pinecone Serverless.** Fully managed, auto-scaling, zero ops. "
            "API key là bắt buộc. Tốt cho startup muốn time-to-market nhanh.\n\n"
            "✅ Zero ops · ✅ Auto-scale · 💳 Paid (free tier giới hạn) · ⚠️ Vendor lock-in"
        ),
        "install": "pip install pinecone langchain-pinecone",
        "pkg_probe": "pinecone",
        "params": ["cloud", "region"],
    },
}


# ─── Helper functions ──────────────────────────────────────────────────────────

@st.cache_data
def get_chunking_strategies() -> dict[str, str]:
    """
    Đọc danh sách chiến lược chunking từ chunking/factory.py.
    Tự động cập nhật nếu thêm/xóa strategy trong thư mục chunking/.
    """
    from chunking.factory import _REGISTRY
    return {name: CHUNK_DESCRIPTIONS.get(name, "") for name in _REGISTRY}


@st.cache_data
def get_pdf_strategies() -> list[str]:
    """Đọc danh sách PDF strategy — cache sau lần đầu."""
    from loader.directory_loader import PDFStrategy
    return list(PDFStrategy.__args__)


@st.cache_data
def _check_java_version() -> tuple[bool, str]:
    """Kiểm tra Java — cache kết quả, không chạy subprocess mỗi rerun."""
    import subprocess as _sp
    try:
        r = _sp.run(["java", "-version"], capture_output=True, text=True, timeout=5)
        if r.returncode == 0:
            ver = (r.stderr or r.stdout).splitlines()[0]
            return True, ver
        return False, "Java không hoạt động"
    except FileNotFoundError:
        return False, "not_found"
    except Exception:
        return False, "unknown"


@st.cache_data
def _is_pdf_strategy_installed(strategy: str) -> bool:
    """Cache kết quả find_spec — tránh scan module mỗi render."""
    entry = PDF_STRATEGY_DEPS.get(strategy)
    if entry is None:
        return True
    module_name, _ = entry
    return importlib.util.find_spec(module_name) is not None


def file_type_badge(file_type: str) -> str:
    """Tạo HTML badge màu cho loại file."""
    color = FILE_TYPE_COLORS.get(file_type, "#95a5a6")
    return (
        f'<span style="background:{color}; color:white; padding:2px 8px; '
        f'border-radius:10px; font-size:0.75rem; font-weight:bold;">'
        f'{file_type.upper()}</span>'
    )


def chunk_type_badge(idx: int, level: str = "") -> str:
    """Tạo badge màu cho chunk theo index."""
    colors = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12",
              "#9b59b6", "#1abc9c", "#e67e22", "#34495e"]
    color  = colors[idx % len(colors)]
    label  = f"Chunk {idx + 1}" + (f" [{level}]" if level else "")
    return (
        f'<span style="background:{color}; color:white; padding:2px 10px; '
        f'border-radius:10px; font-size:0.8rem; font-weight:bold;">{label}</span>'
    )


# Mapping lỗi import → hướng dẫn cài package
_IMPORT_ERROR_HINTS: dict[str, str] = {
    "unstructured_inference": "pip install unstructured-inference",
    "unstructured":           "pip install \'unstructured[pdf]\'",
    "fitz":                   "pip install pymupdf",
    "pdfplumber":             "pip install pdfplumber tabulate",
    "docling":                "pip install docling",
    "pytesseract":            "pip install pytesseract Pillow",
    "paddleocr":              "pip install paddlepaddle paddleocr",
    "ebooklib":               "pip install ebooklib beautifulsoup4",
}


def _friendly_import_error(exc: ModuleNotFoundError) -> str:
    """Chuyển ModuleNotFoundError thành thông báo hữu ích."""
    missing = exc.name or str(exc)
    for key, hint in _IMPORT_ERROR_HINTS.items():
        if key in missing:
            return (
                f"Thiếu package **`{missing}`**.\n\n"
                f"Cài bằng lệnh: `{hint}`"
            )
    return f"Thiếu module: `{missing}`. Chạy `pip install {missing}` để cài."


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

    dims = len(dense[0]) if dense else 0
    return {
        "dense":      dense,
        "sparse":     sparse,
        "dims":       dims,
        "n_embedded": len(dense),
        "truncated":  len(chunks) > max_chunks,
    }


@st.cache_resource
def _init_uploads_dir() -> Path:
    """
    Tạo thư mục uploads trong thư mục tạm của OS (không nằm trong project).
    Chạy đúng 1 lần khi app khởi động; xóa sạch session cũ nếu còn tồn tại.
    Dùng tempfile.gettempdir() → tự dọn dẹp khi reboot, không làm bẩn project.
    """
    import shutil, tempfile
    uploads = Path(tempfile.gettempdir()) / "rag_visualizer_uploads"
    if uploads.exists():
        shutil.rmtree(uploads)
    uploads.mkdir(parents=True)
    return uploads


def save_uploaded_files(uploaded_files) -> str:
    """
    Lưu file upload vào thư mục tạm của OS.
    Dùng tên file gốc → path ổn định giữa các lần upload → pipeline cache hoạt động đúng.
    """
    uploads_dir = _init_uploads_dir()
    for uf in uploaded_files:
        dest = uploads_dir / uf.name
        dest.write_bytes(uf.read())
    return str(uploads_dir)


@st.cache_resource
def _detect_gpu() -> tuple[str, str]:
    """
    Trả về (device, label) mô tả GPU khả dụng. Cache suốt session.
    device: "cuda" | "mps" | "cpu"
    """
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory // (1024 ** 2)
            return "cuda", f"NVIDIA {name} ({vram:,} MB VRAM)"
        if torch.backends.mps.is_available():
            return "mps", "Apple Silicon MPS"
    except ImportError:
        pass
    return "cpu", "CPU (không phát hiện GPU)"


# ─── Smart pipeline suggestions ──────────────────────────────────────────────

@st.cache_data
def _get_file_profile(source_path: str) -> dict:
    """Phân tích loại file trong input, trả về profile dict. Cache theo path."""
    path = Path(source_path)
    files = [f for f in path.rglob("*") if f.is_file()] if path.is_dir() else [path]

    CODE_EXTS = {"py", "js", "ts", "java", "cpp", "c", "go", "rs", "rb", "sql"}
    ext_counts: dict[str, int] = {}
    for f in files:
        ext = f.suffix.lower().lstrip(".")
        ext_counts[ext] = ext_counts.get(ext, 0) + 1

    total  = max(len(files), 1)
    pdf_r  = ext_counts.get("pdf", 0) / total
    md_r   = (ext_counts.get("md", 0) + ext_counts.get("markdown", 0)) / total
    html_r = (ext_counts.get("html", 0) + ext_counts.get("htm", 0)) / total
    code_r = sum(ext_counts.get(e, 0) for e in CODE_EXTS) / total

    return {
        "total": total, "ext_counts": ext_counts,
        "pdf_r": pdf_r, "md_r": md_r, "html_r": html_r, "code_r": code_r,
    }


@st.cache_data
def get_pipeline_suggestions(source_path: str, local_only: bool = False) -> list[dict]:
    """
    Trả về tối đa 3 gợi ý pipeline (loader + chunking + embedding) dựa trên loại file.
    Mỗi gợi ý: {rank, title, pdf_strategy, chunking_strategy, chunking_extra,
                 emb_provider, emb_model, emb_sparse, emb_reason, reason}
    Khi local_only=True: chỉ gợi ý các công cụ chạy hoàn toàn local.
    """
    p = _get_file_profile(source_path)
    pdf_r, md_r, html_r, code_r = p["pdf_r"], p["md_r"], p["html_r"], p["code_r"]
    suggestions: list[dict] = []

    # ── PDF dominant ──────────────────────────────────────────────────────────
    if pdf_r > 0.4:
        suggestions.append({
            "rank": 1,
            "title": "PDF phức tạp — bảng, layout nhiều cột, cần độ chính xác cao",
            "pdf_strategy": "opendataloader",
            "chunking_strategy": "format_aware",
            "chunking_extra": {"format_type": "markdown"},
            "loader_reason": (
                "**opendataloader** — #1 benchmark (0.90 overall, 0.93 table). "
                "Không cần GPU, Java backend local. Fast mode: 0.05s/trang. "
                "Hybrid mode: 0.43s/trang với AI backend cho bảng phức tạp & scan."
            ),
            "chunking_reason": (
                "**Format-aware (markdown)** cắt theo heading → chunk khớp section gốc. "
                "Lý tưởng với output Markdown từ opendataloader."
            ),
            # Embedding: PDF tiếng Việt thường có nhiều bảng, danh mục, số liệu
            # → cần model mạnh về VI + sparse để match chính xác mã/số/tên riêng
            "emb_provider": "huggingface" if local_only else "cohere",
            "emb_model":    "BAAI/bge-m3" if local_only else "embed-multilingual-v3.0",
            "emb_sparse":   True,
            "emb_reason": (
                "**Cohere embed-multilingual-v3.0** ⭐⭐⭐⭐ VI — top API choice cho tài liệu "
                "PDF tiếng Việt, 1024d, 108 ngôn ngữ. "
                "**BM25** bắt buộc: PDF thường chứa tên riêng, mã số, số liệu cần khớp chính xác."
            ) if not local_only else (
                "**BAAI/bge-m3** ⭐⭐⭐⭐ VI — local, 1024d, 8192 ctx, hỗ trợ "
                "dense+sparse+multivec. Tốt nhất cho PDF đa ngôn ngữ chạy offline. "
                "**BM25** để match chính xác số liệu, mã bảng, tên riêng trong PDF."
            ),
        "vdb_provider": "qdrant" if local_only else "chroma",
        "vdb_reason": (
            "🎯 Qdrant** phù hợp nhất: PDF thường cần filter theo trang, section, metadata tài liệu. ACORN filtering giữ recall cao khi kết hợp dense+BM25."
        ) if local_only else (
            "🎨 **Chroma** — persistent local, filtering native theo metadata (source, page). Zero config, phù hợp prototype offline."
        ),
            "ret_strategy":  "hybrid",
            "pre_transforms": "rewrite",
            "post_reranker":  "cross_encoder",
            "ret_reason":     "**hybrid** (dense+BM25) — bắt buộc khi dùng BM25 embedding. RRF fusion cân bằng semantic recall và exact match. **rewrite** pre-retrieval: chuẩn hoá query trước khi tìm kiếm.",
            "pre_reason":     "**rewrite** — chuẩn hoá lỗi chính tả, đại từ mơ hồ. Tùy chọn: **multi_query** để tăng recall.",
            "post_reason":    "**cross_encoder** reranker (BAAI/bge-reranker-v2-m3) — re-score sau hybrid merge, cải thiện precision đáng kể. Không cần GPU.",
        })
        suggestions.append({
            "rank": 2,
            "title": "PDF phức tạp — text, bảng, ảnh & công thức",
            "pdf_strategy": "marker",
            "chunking_strategy": "format_aware",
            "chunking_extra": {"format_type": "auto"},
            "loader_reason": (
                "**Marker** chuyển PDF sang Markdown chuẩn, giữ nguyên bảng, công thức LaTeX "
                "và caption ảnh. Tốt nhất cho PDF kỹ thuật, báo cáo khoa học."
            ),
            "chunking_reason": (
                "**Format-aware (auto)** cắt theo Markdown heading → chunk khớp đúng "
                "với section gốc. Recall cao nhất khi dùng với loader ra Markdown."
            ),
            # Marker output là Markdown → chunk thường dài (section-level)
            # → cần model có ctx window lớn + hiểu cả công thức/ký hiệu
            "emb_provider": "huggingface",
            "emb_model":    "BAAI/bge-m3",
            "emb_sparse":   True,
            "emb_reason": (
                "**BAAI/bge-m3** — 8192 ctx lý tưởng cho chunk section-level dài từ Marker. "
                "Hỗ trợ cả dense + sparse natively. "
                "**BM25** giúp match công thức, ký hiệu, tên riêng trong tài liệu kỹ thuật."
            ),
        "vdb_provider": "chroma" if local_only else "qdrant",
        "vdb_reason": (
            "🎨 **Chroma** — chunk section-level từ Marker thường lớn. Chroma lưu metadata heading để filter theo cấu trúc tài liệu."
        ) if local_only else (
            "🎯 **Qdrant** — BM25+dense hybrid. ACORN filtering đặc biệt hiệu quả với chunk Markdown section có metadata phong phú từ Marker."
        ),
            "ret_strategy":  "hybrid",
            "pre_transforms": "rewrite",
            "post_reranker":  "cross_encoder",
            "ret_reason":     "**hybrid** (dense+BM25) — bắt buộc khi dùng BM25 embedding. RRF fusion cân bằng semantic recall và exact match. **rewrite** pre-retrieval: chuẩn hoá query trước khi tìm kiếm.",
            "pre_reason":     "**rewrite** — chuẩn hoá lỗi chính tả, đại từ mơ hồ. Tùy chọn: **multi_query** để tăng recall.",
            "post_reason":    "**cross_encoder** reranker (BAAI/bge-reranker-v2-m3) — re-score sau hybrid merge, cải thiện precision đáng kể. Không cần GPU.",
        })
        suggestions.append({
            "rank": 3,
            "title": "PDF có bảng — không cần xử lý ảnh/công thức",
            "pdf_strategy": "pdfplumber",
            "chunking_strategy": "recursive",
            "chunking_extra": {},
            "loader_reason": (
                "**pdfplumber** trích bảng tốt nhất trong nhóm rule-based, "
                "nhanh hơn Marker đáng kể. Không cần Java. Output text + Markdown table sạch."
            ),
            "chunking_reason": (
                "**Recursive** — lựa chọn mặc định an toàn, cân bằng tốc độ và chất lượng. "
                "Cắt ưu tiên: đoạn văn → dòng → câu → ký tự."
            ),
            # Chunk ngắn hơn (recursive) → model nhỏ hơn đủ dùng
            "emb_provider": "fastembed" if local_only else "openai",
            "emb_model":    "intfloat/multilingual-e5-small" if local_only else "text-embedding-3-small",
            "emb_sparse":   False,
            "emb_reason": (
                "**text-embedding-3-small** — cân bằng tốt giữa tốc độ, chi phí ($0.02/1M) "
                "và chất lượng. Chunk ngắn (recursive) không cần ctx window lớn. "
                "Sparse không cần thiết cho PDF text thuần."
            ) if not local_only else (
                "**FastEmbed · multilingual-e5-small** — ONNX, CPU-only, không cần GPU. "
                "Đủ nhanh cho prototype với chunk ngắn từ recursive splitting."
            ),
        "vdb_provider": "faiss" if local_only else "chroma",
        "vdb_reason": (
            "💾 **FAISS** — chunk ngắn (recursive), không cần filtering phức tạp. Nhanh nhất cho prototype local, zero config."
        ) if local_only else (
            "🎨 **Chroma** — chunk ngắn, use-case đơn giản. Persistent, dễ reset khi thay đổi embedding model."
        ),
            "ret_strategy":  "dense",
            "pre_transforms": "none",
            "post_reranker":  "none",
            "ret_reason":     "**dense** — đủ cho corpus text thuần, query ngữ nghĩa. Nếu corpus có nhiều tên riêng hoặc mã số, hãy chuyển sang **hybrid**.",
            "pre_reason":     "Pre-retrieval không bắt buộc cho use-case đơn giản. Tuỳ chọn: **rewrite** để chuẩn hoá query.",
            "post_reason":    "Reranker không bắt buộc cho use-case đơn giản. Tuỳ chọn: **cross_encoder** nếu cần precision cao hơn.",
        })

    # ── Markdown dominant ─────────────────────────────────────────────────────
    elif md_r > 0.4:
        suggestions.append({
            "rank": 1,
            "title": "Markdown có cấu trúc heading (#, ##, ###)",
            "pdf_strategy": "pypdf",
            "chunking_strategy": "format_aware",
            "chunking_extra": {"format_type": "markdown"},
            "loader_reason": (
                "**pymupdf** / **pdfplumber** đọc PDF text sạch và nhanh. "
                "Output plain text đủ cho chunking không cần Markdown."
            ),
            "chunking_reason": (
                "**Format-aware (markdown)** cắt theo heading, mỗi chunk là một section hoàn "
                "chỉnh. Metadata heading giữ lại → filtering theo chủ đề chính xác."
            ),
            # Section-level chunk → cần ctx window lớn + hiểu ngữ nghĩa sâu
            "emb_provider": "huggingface",
            "emb_model":    "BAAI/bge-m3",
            "emb_sparse":   False,
            "emb_reason": (
                "**BAAI/bge-m3** — 8192 ctx phù hợp với section chunk lớn từ format_aware. "
                "Dense capture ngữ nghĩa section tốt hơn sparse. "
                "Không cần BM25 khi chunk đã khớp cấu trúc heading."
            ),
        "vdb_provider": "chroma" if local_only else "chroma",
        "vdb_reason": (
            "🎨 **Chroma** — section chunk từ format_aware có metadata heading đầy đủ. Filtering theo heading level để retrieve đúng section."
        ) if local_only else (
            "🎨 **Chroma** — đủ cho tài liệu Markdown structured. Filtering theo heading metadata; dense-only, không cần hybrid."
        ),
            "ret_strategy":  "dense",
            "pre_transforms": "none",
            "post_reranker":  "none",
            "ret_reason":     "**dense** — đủ cho corpus text thuần, query ngữ nghĩa. Nếu corpus có nhiều tên riêng hoặc mã số, hãy chuyển sang **hybrid**.",
            "pre_reason":     "Pre-retrieval không bắt buộc cho use-case đơn giản. Tuỳ chọn: **rewrite** để chuẩn hoá query.",
            "post_reason":    "Reranker không bắt buộc cho use-case đơn giản. Tuỳ chọn: **cross_encoder** nếu cần precision cao hơn.",
        })
        suggestions.append({
            "rank": 2,
            "title": "Markdown hỗn hợp, ít heading nhất quán",
            "pdf_strategy": "pypdf",
            "chunking_strategy": "recursive",
            "chunking_extra": {},
            "loader_reason": (
                "**pypdf** — nhanh nhất, không cần dep extra. "
                "Phù hợp PDF text thuần, không bảng phức tạp."
            ),
            "chunking_reason": (
                "**Recursive** — an toàn khi Markdown không có heading nhất quán. "
                "Cắt theo đoạn văn → dòng → câu → ký tự."
            ),
            "emb_provider": "fastembed" if local_only else "openai",
            "emb_model":    "intfloat/multilingual-e5-small" if local_only else "text-embedding-3-small",
            "emb_sparse":   False,
            "emb_reason": (
                "**text-embedding-3-small** — default tốt cho chunk paragraph-level. "
                "MRL support: có thể cắt xuống 256d để tiết kiệm ~83% RAM nếu cần scale."
            ) if not local_only else (
                "**FastEmbed · multilingual-e5-small** — đủ tốt cho Markdown đa ngôn ngữ, "
                "không cần GPU, tự download ONNX model."
            ),
        "vdb_provider": "faiss" if local_only else "chroma",
        "vdb_reason": (
            "💾 **FAISS** — chunk paragraph đơn giản, không cần filtering. Nhanh nhất local."
        ) if local_only else (
            "🎨 **Chroma** — chunk ngắn (recursive), use-case đơn giản. Dễ dùng nhất."
        ),
            "ret_strategy":  "dense",
            "pre_transforms": "none",
            "post_reranker":  "none",
            "ret_reason":     "**dense** — đủ cho corpus text thuần, query ngữ nghĩa. Nếu corpus có nhiều tên riêng hoặc mã số, hãy chuyển sang **hybrid**.",
            "pre_reason":     "Pre-retrieval không bắt buộc cho use-case đơn giản. Tuỳ chọn: **rewrite** để chuẩn hoá query.",
            "post_reason":    "Reranker không bắt buộc cho use-case đơn giản. Tuỳ chọn: **cross_encoder** nếu cần precision cao hơn.",
        })
        suggestions.append({
            "rank": 3,
            "title": "Tài liệu Markdown dài — nhiều section, cần context rộng",
            "pdf_strategy": "pypdf",
            "chunking_strategy": "hierarchical",
            "chunking_extra": {"parent_chunk_size": 2000},
            "loader_reason": (
                "**marker** / **docling** — output Markdown có heading rõ ràng, "
                "lý tưởng để hierarchical chunking phát hiện ranh giới section."
            ),
            "chunking_reason": (
                "**Hierarchical** lưu cả parent (section lớn) và child (đoạn nhỏ). "
                "Retrieve child → trả parent cho LLM → đủ context, giảm hallucination."
            ),
            # Hierarchical: embed child chunk nhỏ để search, parent chunk lớn cho LLM
            # → model cần hiểu cả 2 granularity → BM25 giúp match term chính xác trong child
            "emb_provider": "huggingface",
            "emb_model":    "BAAI/bge-m3",
            "emb_sparse":   True,
            "emb_reason": (
                "**BAAI/bge-m3** — ctx 8192 xử lý được cả parent (2000 chars) và child chunk. "
                "**BM25** đặc biệt hiệu quả với child chunk ngắn: exact term matching bù đắp "
                "cho dense khi chunk thiếu ngữ cảnh đầy đủ."
            ),
        "vdb_provider": "chroma" if local_only else "qdrant",
        "vdb_reason": (
            "🎨 **Chroma** — hỗ trợ filtering theo metadata để phân biệt parent/child chunk khi retrieve."
        ) if local_only else (
            "🎯 **Qdrant** — ACORN filtering cần thiết để phân biệt parent chunk vs child chunk. Retrieve child → filter trả về parent đúng cặp."
        ),
            "ret_strategy":  "hybrid",
            "pre_transforms": "rewrite",
            "post_reranker":  "cross_encoder",
            "ret_reason":     "**hybrid** (dense+BM25) — bắt buộc khi dùng BM25 embedding. RRF fusion cân bằng semantic recall và exact match. **rewrite** pre-retrieval: chuẩn hoá query trước khi tìm kiếm.",
            "pre_reason":     "**rewrite** — chuẩn hoá lỗi chính tả, đại từ mơ hồ. Tùy chọn: **multi_query** để tăng recall.",
            "post_reason":    "**cross_encoder** reranker (BAAI/bge-reranker-v2-m3) — re-score sau hybrid merge, cải thiện precision đáng kể. Không cần GPU.",
        })

    # ── Code dominant ─────────────────────────────────────────────────────────
    elif code_r > 0.4:
        suggestions.append({
            "rank": 1,
            "title": "Source code — Python, JS, TS, Java, Go…",
            "pdf_strategy": "pypdf",
            "chunking_strategy": "format_aware",
            "chunking_extra": {"format_type": "code"},
            "loader_reason": (
                "**pypdf** / **pdfplumber** đủ cho tài liệu kỹ thuật text. "
                "Chọn pdfplumber nếu có bảng API reference, pypdf nếu cần tốc độ."
            ),
            "chunking_reason": (
                "**Format-aware (code)** dùng AST-based splitting — ranh giới chunk luôn nằm "
                "giữa top-level definition (class, function). Không bao giờ cắt giữa thân hàm."
            ),
            # Code search: identifier name matching cực kỳ quan trọng → sparse bắt buộc
            "emb_provider": "huggingface" if local_only else "cohere",
            "emb_model":    "BAAI/bge-m3" if local_only else "text-embedding-3-small",
            "emb_sparse":   True,
            "emb_reason": (
                "**text-embedding-3-small** — balances cost and quality for "
                "code search. OpenAI ecosystem, tốt cho documentation + code hỗn hợp."
            ) if not local_only else (
                "**BAAI/bge-m3** local — dense+sparse natively, 8192 ctx. "
                "**BM25** bắt buộc: tên hàm, class cần exact match."
            ),
        "vdb_provider": "chroma" if local_only else "qdrant",
        "vdb_reason": (
            "🎨 **Chroma** — filter theo metadata: ngôn ngữ lập trình, tên file, loại definition (class/function)."
        ) if local_only else (
            "🎯 **Qdrant** — ACORN filtering để filter theo ngôn ngữ, file path, definition type. Hybrid dense+BM25 đặc biệt mạnh cho code identifier search."
        ),
            "ret_strategy":  "hybrid",
            "pre_transforms": "rewrite",
            "post_reranker":  "cross_encoder",
            "ret_reason":     "**hybrid** (dense+BM25) — bắt buộc khi dùng BM25 embedding. RRF fusion cân bằng semantic recall và exact match. **rewrite** pre-retrieval: chuẩn hoá query trước khi tìm kiếm.",
            "pre_reason":     "**rewrite** — chuẩn hoá lỗi chính tả, đại từ mơ hồ. Tùy chọn: **multi_query** để tăng recall.",
            "post_reason":    "**cross_encoder** reranker (BAAI/bge-reranker-v2-m3) — re-score sau hybrid merge, cải thiện precision đáng kể. Không cần GPU.",
        })
        suggestions.append({
            "rank": 2,
            "title": "Code hỗn hợp ngôn ngữ hoặc script ngắn",
            "pdf_strategy": "pypdf",
            "chunking_strategy": "recursive",
            "chunking_extra": {},
            "loader_reason": (
                "**pypdf** / **pymupdf** đủ cho PDF văn bản. "
                "Không cần parser nặng nếu không có bảng phức tạp."
            ),
            "chunking_reason": (
                "**Recursive** dùng separator đặc trưng của code (def, class, …) "
                "trước khi fallback về ký tự. Đơn giản hơn AST."
            ),
            "emb_provider": "huggingface" if local_only else "openai",
            "emb_model":    "BAAI/bge-m3" if local_only else "text-embedding-3-small",
            "emb_sparse":   True,
            "emb_reason": (
                "**text-embedding-3-small** — đủ tốt cho code hỗn hợp. "
                "**BM25** vẫn nên bật: identifier, API name, error code cần exact match."
            ) if not local_only else (
                "**BAAI/bge-m3** local + **BM25**: combo đủ mạnh cho code search offline. "
                "Identifier matching qua BM25 là không thể thiếu với code."
            ),
        "vdb_provider": "chroma" if local_only else "qdrant",
        "vdb_reason": (
            "🎨 **Chroma** — filter theo file extension, ngôn ngữ. Đủ cho code hỗn hợp ít file."
        ) if local_only else (
            "🎯 **Qdrant** — hybrid BM25+dense với ACORN filtering theo ngôn ngữ và file path. Phù hợp khi codebase lớn hơn."
        ),
            "ret_strategy":  "hybrid",
            "pre_transforms": "rewrite",
            "post_reranker":  "cross_encoder",
            "ret_reason":     "**hybrid** (dense+BM25) — bắt buộc khi dùng BM25 embedding. RRF fusion cân bằng semantic recall và exact match. **rewrite** pre-retrieval: chuẩn hoá query trước khi tìm kiếm.",
            "pre_reason":     "**rewrite** — chuẩn hoá lỗi chính tả, đại từ mơ hồ. Tùy chọn: **multi_query** để tăng recall.",
            "post_reason":    "**cross_encoder** reranker (BAAI/bge-reranker-v2-m3) — re-score sau hybrid merge, cải thiện precision đáng kể. Không cần GPU.",
        })

    # ── HTML dominant ─────────────────────────────────────────────────────────
    elif html_r > 0.4:
        suggestions.append({
            "rank": 1,
            "title": "HTML có cấu trúc heading rõ ràng",
            "pdf_strategy": "pypdf",
            "chunking_strategy": "format_aware",
            "chunking_extra": {"format_type": "html"},
            "loader_reason": (
                "**pymupdf** / **pdfplumber** đủ cho PDF. "
                "Strip boilerplate (header, nav, footer) trước khi load nếu có thể."
            ),
            "chunking_reason": (
                "**Format-aware (html)** cắt theo thẻ semantic <h1>…<h6>, phân tách "
                "đúng theo cấu trúc trang. Tốt cho web scraping hoặc documentation."
            ),
            # HTML thường là web content → đa ngôn ngữ, VI là phổ biến
            "emb_provider": "huggingface" if local_only else "cohere",
            "emb_model":    "BAAI/bge-m3" if local_only else "embed-multilingual-v3.0",
            "emb_sparse":   False,
            "emb_reason": (
                "**Cohere embed-multilingual-v3.0** ⭐⭐⭐⭐ VI — tốt nhất cho web content "
                "tiếng Việt qua API, 108 ngôn ngữ. Dense đủ mạnh khi chunk đã khớp heading."
            ) if not local_only else (
                "**BAAI/bge-m3** local — đa ngôn ngữ, hiểu web content tốt. "
                "Dense đủ khi chunk đã được cắt sạch theo heading HTML."
            ),
        "vdb_provider": "chroma" if local_only else "weaviate",
        "vdb_reason": (
            "🎨 **Chroma** — filter theo URL/domain metadata từ web scraping. Dense-only đủ khi chunk đã clean."
        ) if local_only else (
            "🕸️ **Weaviate** — native hybrid search (dense + BM25) không cần config. Phù hợp web content đa ngôn ngữ cần search cả keyword lẫn ngữ nghĩa."
        ),
            "ret_strategy":  "dense",
            "pre_transforms": "none",
            "post_reranker":  "none",
            "ret_reason":     "**dense** — đủ cho corpus text thuần, query ngữ nghĩa. Nếu corpus có nhiều tên riêng hoặc mã số, hãy chuyển sang **hybrid**.",
            "pre_reason":     "Pre-retrieval không bắt buộc cho use-case đơn giản. Tuỳ chọn: **rewrite** để chuẩn hoá query.",
            "post_reason":    "Reranker không bắt buộc cho use-case đơn giản. Tuỳ chọn: **cross_encoder** nếu cần precision cao hơn.",
        })
        suggestions.append({
            "rank": 2,
            "title": "HTML có nhiều boilerplate / nav / footer",
            "pdf_strategy": "pypdf",
            "chunking_strategy": "recursive",
            "chunking_extra": {},
            "loader_reason": (
                "**pypdf** / **pymupdf** đủ cho PDF text. "
                "Không cần parser nặng nếu HTML/PDF đã được strip sạch."
            ),
            "chunking_reason": (
                "**Recursive** — ổn ngay cả khi HTML không có heading nhất quán. "
                "An toàn hơn khi content đã được strip boilerplate."
            ),
            "emb_provider": "fastembed" if local_only else "openai",
            "emb_model":    "intfloat/multilingual-e5-small" if local_only else "text-embedding-3-small",
            "emb_sparse":   False,
            "emb_reason": (
                "**text-embedding-3-small** — lựa chọn an toàn cho web content hỗn hợp, "
                "không cần sparse khi content đã được strip boilerplate."
            ) if not local_only else (
                "**FastEmbed · multilingual-e5-small** — ONNX nhẹ, không cần GPU, "
                "đủ cho HTML content đã strip."
            ),
        "vdb_provider": "faiss" if local_only else "chroma",
        "vdb_reason": (
            "💾 **FAISS** — HTML đã clean, chunk đơn giản. FAISS đủ nhanh và đơn giản nhất."
        ) if local_only else (
            "🎨 **Chroma** — dễ dùng, persistent, đủ cho HTML content nhỏ-vừa."
        ),
            "ret_strategy":  "dense",
            "pre_transforms": "none",
            "post_reranker":  "none",
            "ret_reason":     "**dense** — đủ cho corpus text thuần, query ngữ nghĩa. Nếu corpus có nhiều tên riêng hoặc mã số, hãy chuyển sang **hybrid**.",
            "pre_reason":     "Pre-retrieval không bắt buộc cho use-case đơn giản. Tuỳ chọn: **rewrite** để chuẩn hoá query.",
            "post_reason":    "Reranker không bắt buộc cho use-case đơn giản. Tuỳ chọn: **cross_encoder** nếu cần precision cao hơn.",
        })

    # ── Mixed / generic ───────────────────────────────────────────────────────
    else:
        suggestions.append({
            "rank": 1,
            "title": "Văn bản hỗn hợp — lựa chọn mặc định tốt nhất",
            "pdf_strategy": "pymupdf",
            "chunking_strategy": "recursive",
            "chunking_extra": {},
            "loader_reason": (
                "**pymupdf** — nhanh, ổn định, layout tốt cho hầu hết PDF. "
                "Lựa chọn mặc định an toàn cho corpus hỗn hợp."
            ),
            "chunking_reason": (
                "**Recursive** — mặc định được khuyến nghị, cân bằng tốt giữa "
                "tốc độ và chất lượng ngữ nghĩa. Không phụ thuộc cấu trúc."
            ),
            "emb_provider": "huggingface" if local_only else "openai",
            "emb_model":    "BAAI/bge-m3" if local_only else "text-embedding-3-small",
            "emb_sparse":   False,
            "emb_reason": (
                "**text-embedding-3-small** — default phổ biến nhất, đủ tốt cho văn bản "
                "hỗn hợp. MRL support: trim xuống 512d để giảm RAM nếu cần."
            ) if not local_only else (
                "**BAAI/bge-m3** local — đa ngôn ngữ, dense chất lượng cao, "
                "không cần API key."
            ),
        "vdb_provider": "chroma" if local_only else "chroma",
        "vdb_reason": (
            "🎨 **Chroma** — lựa chọn mặc định an toàn. Persistent, filtering cơ bản, zero config. Phù hợp cho hầu hết use case."
        ) if local_only else (
            "🎨 **Chroma** — lựa chọn mặc định tốt nhất. Dễ migrate sang provider khác sau khi prototype xong."
        ),
            "ret_strategy":  "dense",
            "pre_transforms": "none",
            "post_reranker":  "none",
            "ret_reason":     "**dense** — đủ cho corpus text thuần, query ngữ nghĩa. Nếu corpus có nhiều tên riêng hoặc mã số, hãy chuyển sang **hybrid**.",
            "pre_reason":     "Pre-retrieval không bắt buộc cho use-case đơn giản. Tuỳ chọn: **rewrite** để chuẩn hoá query.",
            "post_reason":    "Reranker không bắt buộc cho use-case đơn giản. Tuỳ chọn: **cross_encoder** nếu cần precision cao hơn.",
        })
        suggestions.append({
            "rank": 2,
            "title": "Tài liệu đa ngôn ngữ (Việt + Anh)",
            "pdf_strategy": "pymupdf",
            "chunking_strategy": "token_based",
            "chunking_extra": {"encoding_name": "cl100k_base"},
            "loader_reason": (
                "**pymupdf** / **pdfplumber** đọc PDF text sạch. "
                "Tốt cho corpus tiếng Việt, không bị ảnh hưởng bởi encoding đặc thù."
            ),
            "chunking_reason": (
                "**Token-based** đếm token thay vì ký tự — quan trọng với tiếng Việt. "
                "Tránh chunk bị cắt lệch so với ngưỡng token của embedding model."
            ),
            # Đa ngôn ngữ VI+EN → cần model tốt nhất về VI
            "emb_provider": "huggingface" if local_only else "cohere",
            "emb_model":    "BAAI/bge-m3" if local_only else "embed-multilingual-v3.0",
            "emb_sparse":   True,
            "emb_reason": (
                "**Cohere embed-multilingual-v3.0** ⭐⭐⭐⭐ VI — top API choice cho corpus "
                "song ngữ Việt–Anh. **BM25** giúp match từ khoá tiếng Việt không dấu hoặc "
                "tên riêng mà dense có thể bỏ sót khi cross-lingual."
            ) if not local_only else (
                "**BAAI/bge-m3** ⭐⭐⭐⭐ VI local — tốt nhất cho song ngữ Việt–Anh offline. "
                "**BM25** match chính xác từ khoá tiếng Việt đặc thù."
            ),
        "vdb_provider": "chroma" if local_only else "qdrant",
        "vdb_reason": (
            "🎨 **Chroma** — filtering theo ngôn ngữ (VI/EN) trong metadata. Phù hợp corpus song ngữ local."
        ) if local_only else (
            "🎯 **Qdrant** — ACORN filtering theo ngôn ngữ trong metadata. Hybrid dense+BM25 đặc biệt hiệu quả cho corpus song ngữ Việt–Anh."
        ),
            "ret_strategy":  "hybrid",
            "pre_transforms": "rewrite",
            "post_reranker":  "cross_encoder",
            "ret_reason":     "**hybrid** (dense+BM25) — bắt buộc khi dùng BM25 embedding. RRF fusion cân bằng semantic recall và exact match. **rewrite** pre-retrieval: chuẩn hoá query trước khi tìm kiếm.",
            "pre_reason":     "**rewrite** — chuẩn hoá lỗi chính tả, đại từ mơ hồ. Tùy chọn: **multi_query** để tăng recall.",
            "post_reason":    "**cross_encoder** reranker (BAAI/bge-reranker-v2-m3) — re-score sau hybrid merge, cải thiện precision đáng kể. Không cần GPU.",
        })

    # Lọc strategy cần external LLM API nếu local_only
    if local_only:
        NON_LOCAL = {"contextual"}
        suggestions = [s for s in suggestions if s["chunking_strategy"] not in NON_LOCAL]

    return suggestions[:3]


# ─── UI: Loader settings panel ────────────────────────────────────────────────

def _render_vlm_panel(key_prefix: str) -> None:
    """
    Panel chọn VLM để mô tả ảnh — dùng chung cho marker và unstructured.
    Kết quả được ghi vào session_state với key có prefix:
      _vlm_describe_{key_prefix}  : bool
      _vlm_provider_{key_prefix}  : "openai" | "ollama"
      _vlm_model_{key_prefix}     : str
      _vlm_url_{key_prefix}       : str
    """
    describe = st.checkbox(
        "Dùng VLM mô tả ảnh trong PDF",
        value=False,
        key=f"_vlm_describe_{key_prefix}",
        help="VLM đọc ảnh và sinh mô tả text để embedding. Cần API key hoặc Ollama local.",
    )

    if not describe:
        return

    vision_provider = st.radio(
        "VLM Provider",
        ["openai", "ollama"],
        horizontal=True,
        key=f"_vlm_provider_{key_prefix}",
        help="openai: cần OPENAI_API_KEY · ollama: chạy local, miễn phí",
    )

    if vision_provider == "openai":
        st.selectbox(
            "VLM Model",
            ["gpt-4o-mini", "gpt-4o"],
            key=f"_vlm_model_{key_prefix}",
            help="gpt-4o-mini: nhanh & rẻ · gpt-4o: chất lượng cao hơn",
        )
        st.caption("💡 `OPENAI_API_KEY` được load từ `.env` hoặc biến môi trường.")
    else:
        st.selectbox(
            "VLM Model (Ollama)",
            ["llava:7b", "llava:13b", "llava-llama3", "moondream",
             "minicpm-v", "qwen2-vl:7b", "qwen2.5vl:7b", "qwen2.5vl:3b", "glm4v"],
            key=f"_vlm_model_{key_prefix}",
            help="Cần pull model trước: `ollama pull <model>`. App tự pull nếu chưa có.",
        )
        st.text_input(
            "Ollama base URL",
            value="http://localhost:11434/v1",
            key=f"_vlm_url_{key_prefix}",
            help="Thay đổi nếu Ollama chạy ở host/port khác",
        )
        if st.button("🔌 Test kết nối Ollama", key=f"test_ollama_{key_prefix}"):
            _url = st.session_state.get(f"_vlm_url_{key_prefix}", "http://localhost:11434/v1")
            _mdl = st.session_state.get(f"_vlm_model_{key_prefix}", "llava:7b")
            with st.spinner("Đang kiểm tra..."):
                try:
                    from openai import OpenAI as _OAI
                    c = _OAI(base_url=_url, api_key="ollama")
                    models = [m.id for m in c.models.list().data]
                    if _mdl in set(models):
                        st.success(f"✅ Kết nối OK · `{_mdl}` sẵn sàng")
                    else:
                        st.warning(
                            f"⚠️ Kết nối OK nhưng `{_mdl}` chưa được pull\n\n"
                            f"Models có sẵn: `{'`, `'.join(models[:6])}`"
                        )
                        st.info("💡 Ấn **Process** — app sẽ tự động pull model trước khi mô tả ảnh.")
                except Exception as e:
                    st.error(f"❌ Không kết nối được Ollama: `{e}`")
                    st.caption("Đảm bảo Ollama đang chạy: `ollama serve`")


def render_loader_settings() -> dict:
    """Hiển thị panel cài đặt loader, trả về dict các tham số."""
    st.subheader("⚙️ Cài đặt Loader")

    col1, col2 = st.columns(2)
    with col1:
        strategies = get_pdf_strategies()
        # Khởi tạo default một lần — tránh conflict khi Apply button ghi vào session state
        if "sel_pdf_strategy" not in st.session_state:
            st.session_state["sel_pdf_strategy"] = "pypdf"
        pdf_strategy = st.selectbox(
            "PDF Strategy",
            options=strategies,
            key="sel_pdf_strategy",
            help=(
                "**pypdf** — nhanh, chỉ text layer (không cần cài thêm)\n\n"
                "**pymupdf** — nhanh hơn, layout tốt hơn · `pip install pymupdf`\n\n"
                "**pdfplumber** — trích bảng tốt nhất · `pip install pdfplumber`\n\n"
                "**unstructured** — tốt nhất: OCR + bảng + hình · `pip install 'unstructured[pdf]' unstructured-inference`\n\n"
                "**docling** — IBM parser, Markdown output xuất sắc · `pip install docling`\n\n"
                "**marker** — Markdown chất lượng cao, bảng & LaTeX · `pip install marker-pdf`\n\n"
                "**opendataloader** — #1 benchmark (0.90), bounding box, no GPU · `pip install opendataloader-pdf` · **Java 11+ required**"
            ),
        )
        # Chỉ hiện cảnh báo khi thư viện chưa được cài
        if not _is_pdf_strategy_installed(pdf_strategy):
            entry = PDF_STRATEGY_DEPS.get(pdf_strategy)
            if entry:
                _, install_cmd = entry
                st.caption(f"⚠️ Cần cài thêm: `{install_cmd}`")
        extract_tables = st.checkbox("Trích xuất bảng → Markdown", value=True)

    with col2:
        # Các strategy có OCR tích hợp sẵn → disable OCR engine selector
        _BUILTIN_OCR: dict[str, str] = {
            "marker":         "Marker tích hợp **Surya OCR** — nhận dạng 90+ ngôn ngữ, không cần chọn thêm.",
            "docling":        "Docling tích hợp **RapidOCR** — không cần chọn thêm OCR engine.",
            "opendataloader": "OpenDataLoader tích hợp OCR trong hybrid mode — không cần chọn thêm OCR engine.",
            "pypdf":          "pypdf chỉ đọc text layer — OCR không áp dụng cho strategy này.",
            "pymupdf":        "PyMuPDF chỉ đọc text layer — OCR không áp dụng cho strategy này.",
            "pdfplumber":     "pdfplumber chỉ đọc text layer — OCR không áp dụng cho strategy này.",
        }
        ocr_disabled = pdf_strategy in _BUILTIN_OCR



        language = st.selectbox(
            "Ngôn ngữ corpus",
            options=["both", "vi", "en"],
            index=0,
            help="Dùng để chọn ngôn ngữ cho OCR và NLP tools"
        )

    # ── Device + VLM (full width, chỉ hiện khi marker) ───────────────────────
    marker_device   = "cpu"
    describe_images = False
    vision_provider = "openai"
    vision_model    = "gpt-4o-mini"
    ollama_base_url = "http://localhost:11434/v1"
    # ── OpenDataLoader options (chỉ hiện khi opendataloader) ─────────────────
    odl_hybrid      = None   # None = fast mode; "docling-fast" = hybrid mode
    odl_struct_tree = False

    if pdf_strategy == "marker":
        st.markdown("---")
        st.subheader("🖥️ Device cho Marker")
        auto_device, gpu_label = _detect_gpu()
        if auto_device != "cpu":
            st.success(f"✅ Phát hiện GPU: {gpu_label}")
        else:
            st.warning("⚠️ Không phát hiện GPU — Marker sẽ chạy trên CPU (~30-60s/trang)")

        device_options = [auto_device, "cpu"] if auto_device != "cpu" else ["cpu"]
        marker_device = st.radio(
            "Chọn device",
            options=device_options,
            index=0,
            horizontal=True,
            help="cuda: NVIDIA GPU · mps: Apple Silicon · cpu: chậm nhất",
        )

        st.markdown("---")
        _render_vlm_panel(key_prefix="marker")
        describe_images = st.session_state.get("_vlm_describe_marker", False)
        vision_provider  = st.session_state.get("_vlm_provider_marker", "openai")
        vision_model     = st.session_state.get("_vlm_model_marker", "gpt-4o-mini")
        ollama_base_url  = st.session_state.get("_vlm_url_marker", "http://localhost:11434/v1")

    elif pdf_strategy == "unstructured":
        st.markdown("---")
        st.subheader("🖼️ Mô tả ảnh bằng VLM")
        st.caption(
            "Unstructured detect được hình ảnh trong PDF (khi dùng `hi_res` strategy với OCR). "
            "Bật option này để dùng VLM sinh mô tả text cho từng ảnh, giúp embedding capture nội dung hình."
        )
        _render_vlm_panel(key_prefix="unstructured")
        describe_images = st.session_state.get("_vlm_describe_unstructured", False)
        vision_provider  = st.session_state.get("_vlm_provider_unstructured", "openai")
        vision_model     = st.session_state.get("_vlm_model_unstructured", "gpt-4o-mini")
        ollama_base_url  = st.session_state.get("_vlm_url_unstructured", "http://localhost:11434/v1")



    elif pdf_strategy == "opendataloader":
        st.markdown("---")
        st.subheader("⚙️ Cài đặt OpenDataLoader PDF")

        # ── Java availability check (cached) ─────────────────────────────────
        _java_ok, _java_msg = _check_java_version()
        if _java_ok:
            st.success(f"✅ Java đã sẵn sàng: `{_java_msg}`")
        elif _java_msg == "not_found":
            st.error(
                "❌ **Java chưa được cài đặt** — opendataloader-pdf yêu cầu Java 11+.\n\n"
                "Tải tại: https://adoptium.net/"
            )
        else:
            st.error("❌ Java không hoạt động. Cần Java 11+.")

        # Only show install hint if package is actually missing
        if not _is_pdf_strategy_installed("opendataloader"):
            st.caption("`pip install opendataloader-pdf`")

        # ── Mode: fast vs hybrid ──────────────────────────────────────────────
        st.markdown("")
        odl_mode = st.radio(
            "Mode",
            options=["fast", "hybrid"],
            index=0,
            horizontal=True,
            help=(
                "**fast** — Java local, deterministic, 0.05s/trang, accuracy 0.72. "
                "Không cần cài thêm.\n\n"
                "**hybrid** — AI backend routing, 0.43s/trang, accuracy **0.90 (#1 benchmark)**. "
                "Tốt cho bảng phức tạp, PDF scan, công thức. "
                "Cần: `pip install \"opendataloader-pdf[hybrid]\"` và server đang chạy."
            ),
            key="sel_odl_mode",
        )

        if odl_mode == "hybrid":
            odl_hybrid = "docling-fast"   # backend mặc định theo README
            st.info(
                "**Hybrid mode** yêu cầu server đang chạy trong terminal riêng:\n\n"
                "```\n"
                "opendataloader-pdf-hybrid --port 5002\n"
                "```\n\n"
                "Với PDF scan / tiếng Việt:\n"
                "```\n"
                "opendataloader-pdf-hybrid --port 5002 --force-ocr --ocr-lang \"vi,en\"\n"
                "```"
            )
            _odl_port = st.number_input(
                "Hybrid server port",
                min_value=1024, max_value=65535,
                value=5002, step=1,
                help="Port của opendataloader-pdf-hybrid server. Mặc định: 5002.",
                key="odl_hybrid_port",
            )
            st.session_state["_odl_hybrid_port"] = int(_odl_port)
        else:
            odl_hybrid = None   # fast mode

        # ── use_struct_tree ───────────────────────────────────────────────────
        odl_struct_tree = st.checkbox(
            "Dùng native PDF structure tags (`use_struct_tree`)",
            value=False,
            help=(
                "Nếu PDF đã được tagged (Tagged PDF), bật để đọc layout trực tiếp từ "
                "structure tree của PDF — reading order chính xác nhất, không cần heuristic.\n\n"
                "Tắt (mặc định): dùng XY-Cut++ layout analysis — tốt cho hầu hết PDF thông thường."
            ),
            key="odl_struct_tree",
        )

    if pdf_strategy == "docling":
        st.markdown("---")
        st.info(
            "🖼️ **Docling tự động xử lý ảnh** — không cần VLM.\n\n"
            "Docling extract ảnh, encode thành base64 và nhúng trực tiếp vào Markdown output "
            "(`![Figure N](data:image/png;base64,...)`). App visualizer render được các ảnh này "
            "trong tab Loader.\n\n"
            "Để **mô tả nội dung ảnh bằng VLM** (sinh text caption để embedding), "
            "hãy dùng **Marker** hoặc **Unstructured** thay thế."
        )

    return {
        "pdf_strategy":    pdf_strategy,
        "extract_tables":  extract_tables,
        "language":        language,
        "marker_device":   marker_device,
        "describe_images": describe_images,
        "vision_model":    vision_model,
        "vision_provider": vision_provider,
        "ollama_base_url": ollama_base_url,
        "odl_hybrid":      odl_hybrid,
        "odl_struct_tree": odl_struct_tree,
    }


# ─── UI: Chunking settings panel ──────────────────────────────────────────────

def render_chunking_settings(local_only: bool = False) -> tuple[str, int, int, dict]:
    """Hiển thị panel cài đặt chunking, trả về (strategy, chunk_size, overlap, extra)."""
    st.subheader("✂️ Cài đặt Chunking")

    strategies = get_chunking_strategies()
    strategy_names = list(strategies.keys())
    # Khởi tạo default một lần — tránh conflict khi Apply button ghi vào session state
    if "sel_chunking_strategy" not in st.session_state:
        st.session_state["sel_chunking_strategy"] = "recursive"
    strategy = st.selectbox(
        "Chunking Strategy",
        options=strategy_names,
        key="sel_chunking_strategy",
        format_func=lambda s: f"{s}  —  {strategies[s][:55]}…" if len(strategies[s]) > 55 else f"{s}  —  {strategies[s]}",
    )

    # Mô tả đầy đủ của strategy đã chọn
    if strategies.get(strategy):
        st.info(f"ℹ️ {strategies[strategy]}")

    if strategy in LLM_REQUIRED_STRATEGIES:
        st.warning(
            "⚠️ Strategy này cần LLM API (chậm và tốn phí). "
            "Demo sẽ dùng 'recursive' để tạo chunk cơ sở trước khi gọi LLM."
        )

    # Strategies không dùng chunk_size/overlap — boundary do model/LLM tự quyết định:
    #   semantic   → cosine similarity giữa các câu
    #   late       → token embedding của model quyết định
    _NO_SIZE_STRATEGIES: dict[str, str] = {
        "semantic":    "ranh giới được xác định tự động qua cosine similarity giữa các câu",
    }
    _no_size = strategy in _NO_SIZE_STRATEGIES

    # format_aware: chunk_size/overlap chỉ dùng khi split_large_sections=True
    # → render sau khi người dùng tick checkbox (xem block format_aware bên dưới)
    _defer_size = (strategy == "format_aware")

    if not _defer_size:
        col1, col2 = st.columns(2)
        with col1:
            chunk_size = st.number_input(
                "Chunk size (chars)",
                min_value=50, max_value=8000, value=500, step=50,
                disabled=_no_size,
                help="Kích thước tối đa mỗi chunk (tính bằng ký tự)"
            )
        with col2:
            chunk_overlap = st.number_input(
                "Chunk overlap (chars)",
                min_value=0, max_value=2000, value=100, step=25,
                disabled=_no_size,
                help="Số ký tự chồng lấp giữa các chunk liên tiếp"
            )
        if _no_size:
            st.caption(f"ℹ️ Chunk size và overlap không áp dụng cho **{strategy}** — {_NO_SIZE_STRATEGIES[strategy]}.")
    else:
        # Placeholder — sẽ được render bên trong block format_aware
        chunk_size    = 1000
        chunk_overlap = 100

    # Tham số bổ sung tuỳ strategy
    extra: dict = {}

    if strategy == "token_based":
        extra["encoding_name"] = st.selectbox(
            "Tokenizer encoding",
            ["cl100k_base", "p50k_base", "r50k_base"],
            help="cl100k_base: GPT-4 / text-embedding-3"
        )

    elif strategy == "semantic":
        from chunking.semantic import EMBEDDING_MODELS, PROVIDER_GROUPS

        # Lọc group theo local_only
        available_groups = {
            g: models for g, models in PROVIDER_GROUPS.items()
            if (not local_only) or g.startswith("🏠")
        }
        if not available_groups:
            available_groups = {g: m for g, m in PROVIDER_GROUPS.items() if g.startswith("🏠")}

        # Bước 1: chọn provider group
        default_group = next(
            (g for g in available_groups if g.startswith("🏠 Self-hosted · HuggingFace")),
            list(available_groups.keys())[0],
        )
        provider_group = st.selectbox(
            "Embedding Provider",
            options=list(available_groups.keys()),
            index=list(available_groups.keys()).index(default_group),
            help="API: cần key, nhanh, không cần GPU.\nSelf-hosted: hoàn toàn local, tự động tải về khi chạy.",
        )

        # Bước 2: chọn model trong group
        models_in_group = available_groups[provider_group]

        def _model_label(m: str) -> str:
            meta = EMBEDDING_MODELS.get(m, {})
            return f"{meta.get('display', m)}  ·  MTEB {meta.get('mteb','?')}  ·  {meta.get('dim','?')}d"

        embedding_model = st.selectbox(
            "Embedding Model",
            options=models_in_group,
            format_func=_model_label,
        )
        extra["embedding_model_name"] = embedding_model

        meta     = EMBEDDING_MODELS.get(embedding_model, {})
        provider = meta.get("provider", "")

        # Thông tin model
        st.info(f"ℹ️ {meta.get('note', '')}")

        # ── API key check ──────────────────────────────────────────────
        if not meta.get("local", True) and meta.get("requires_env"):
            env_key = meta["requires_env"]
            if _get_env(env_key):
                st.success(f"✅ `{env_key}` đã được cấu hình.")
            else:
                st.warning(f"⚠️ Cần `{env_key}` trong file `.env` hoặc biến môi trường.")
                pkg_probe = meta.get("pkg_probe", "")
                if meta.get("install") and pkg_probe and importlib.util.find_spec(pkg_probe) is None:
                    st.caption(f"📦 Cần cài: `{meta['install']}`")

        # ── Ollama: test connection + check pull status ────────────────
        if provider == "ollama":
            ollama_url = st.text_input(
                "Ollama base URL",
                value=st.session_state.get("ollama_embed_url", "http://localhost:11434/v1"),
                key="ollama_embed_url",
                help="URL Ollama server. Model sẽ tự động pull nếu chưa có khi ấn Process.",
            )
            extra["ollama_base_url"] = ollama_url

            if st.button("🔌 Test Ollama & kiểm tra model", key="test_ollama_embed"):
                with st.spinner("Đang kiểm tra..."):
                    try:
                        import ollama as _ollama
                        host = ollama_url.rstrip("/")
                        if host.endswith("/v1"):
                            host = host[:-3]
                        client     = _ollama.Client(host=host)
                        pulled     = [m.model for m in client.list().models]
                        is_present = embedding_model in set(pulled)
                        if is_present:
                            st.success(f"✅ Kết nối OK · `{embedding_model}` đã sẵn sàng.")
                        else:
                            st.warning(
                                f"⚠️ Kết nối OK nhưng `{embedding_model}` chưa được pull.\n\n"
                                f"Ấn **▶️ Process** — app sẽ tự pull trước khi chạy.\n"
                                f"Hoặc chủ động: `ollama pull {embedding_model}`"
                            )
                    except Exception as exc:
                        st.error(f"❌ Không kết nối được Ollama: `{exc}`")
                        st.caption("Đảm bảo Ollama đang chạy: `ollama serve`")

        # ── HuggingFace: thông báo auto-download ──────────────────────
        elif provider == "huggingface":
            st.caption(
                f"💡 Model sẽ tự động tải từ HuggingFace Hub khi chạy lần đầu "
                f"(~{meta.get('note','').split('~')[-1].split('.')[0].strip()} nếu chưa cache).\n\n"
                f"`{meta.get('install', '')}`"
            )

        # ── Breakpoint settings (chung cho mọi provider) ──────────────
        extra["breakpoint_type"] = st.selectbox(
            "Breakpoint detection type",
            ["percentile", "standard_deviation", "interquartile"],
            help=(
                "**percentile**: cắt tại các điểm similarity thấp nhất theo phần trăm — khuyến nghị.\n\n"
                "**standard_deviation**: cắt khi similarity thấp hơn mean - N×std.\n\n"
                "**interquartile**: phát hiện outlier bằng IQR — ít nhạy cảm với tham số nhất."
            ),
        )
        extra["breakpoint_threshold"] = st.slider(
            "Breakpoint threshold",
            min_value=50.0, max_value=99.0, value=95.0, step=1.0,
            help="Với 'percentile': càng cao → ít chunk hơn (chỉ cắt ở những chỗ drop rất lớn).",
        )


    elif strategy == "format_aware":
        if "sel_format_type" not in st.session_state:
            st.session_state["sel_format_type"] = "auto"
        extra["format_type"] = st.selectbox(
            "Format type",
            ["auto", "markdown", "code", "html"],
            key="sel_format_type",
            help="auto: tự phát hiện từ metadata file_type"
        )
        # ── split_large_sections: gợi ý dựa trên embedding model đang chọn ──
        # Đọc ctx window của model embedding hiện tại (nếu đã chọn)
        _emb_provider  = st.session_state.get("sel_emb_provider", "")
        _emb_model     = st.session_state.get("sel_emb_model", "")
        _emb_skip      = st.session_state.get("emb_skip", False)

        # Ctx window ngắn (token) của các model đã biết
        _SHORT_CTX_MODELS = {
            # Ollama
            "mxbai-embed-large":       512,
            "snowflake-arctic-embed":  512,
            "nomic-embed-text":       2048,
            # HuggingFace
            "VinAI/phobert-large":     256,
            "intfloat/multilingual-e5-large": 512,
            "intfloat/e5-mistral-7b-instruct": None,  # 32K — đủ lớn
            # FastEmbed
            "BAAI/bge-small-en-v1.5":  512,
            "BAAI/bge-base-en-v1.5":   512,
            "intfloat/multilingual-e5-small": 512,
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2": 512,
        }
        _LARGE_CTX_MODELS = {
            "BAAI/bge-m3",
            "Qwen/Qwen3-Embedding-0.6B",
            "Qwen/Qwen3-Embedding-4B",
            "Qwen/Qwen3-Embedding-8B",
            "Alibaba-NLP/gte-Qwen2-7B-instruct",
            "nomic-ai/nomic-embed-text-v1.5",
            "bge-m3",
            # API models đều có ctx lớn
            "text-embedding-3-small", "text-embedding-3-large", "text-embedding-ada-002",
            "embed-multilingual-v3.0", "embed-english-v3.0", "embed-v4.0",
        }

        _ctx_tokens = _SHORT_CTX_MODELS.get(_emb_model)
        _is_large_ctx = _emb_model in _LARGE_CTX_MODELS or _emb_provider in ("openai", "cohere")
        _is_short_ctx = _ctx_tokens is not None and _ctx_tokens <= 512

        # Suggest default value based on model
        _suggested_split = _is_short_ctx

        extra["split_large_sections"] = st.checkbox(
            "Chia nhỏ section quá dài",
            value=_suggested_split,
            key="sel_split_large_sections",
        )
        _split_on = extra["split_large_sections"]

        # chunk_size và chunk_overlap chỉ có nghĩa khi split_large_sections=True
        st.markdown("")
        col1, col2 = st.columns(2)
        with col1:
            chunk_size = st.number_input(
                "Chunk size (chars)",
                min_value=50, max_value=8000, value=1000, step=50,
                disabled=not _split_on,
                key="fa_chunk_size",
                help="Kích thước tối đa mỗi sub-chunk sau khi chia nhỏ section"
            )
        with col2:
            chunk_overlap = st.number_input(
                "Chunk overlap (chars)",
                min_value=0, max_value=2000, value=100, step=25,
                disabled=not _split_on,
                key="fa_chunk_overlap",
                help="Số ký tự chồng lấp giữa các sub-chunk"
            )
        if not _split_on:
            st.caption(
                "ℹ️ Chunk size và overlap không áp dụng — "
                "mỗi section heading được giữ nguyên, không bị cắt thêm. "
                "Tick checkbox trên để kích hoạt."
            )

        # Guidance box — nội dung thay đổi theo context
        if _emb_skip:
            st.info(
                "ℹ️ **Bạn đang bỏ qua bước Embedding.**\n\n"
                "Nếu sau này dùng embedding với model có **ctx window ngắn** "
                "(phobert-large: 256 token, mxbai-embed-large: 512 token, "
                "các FastEmbed model: 512 token) → **nên tick**.\n\n"
                "Nếu dùng model ctx lớn (bge-m3: 8192, Qwen3: 32K, các API như "
                "OpenAI / Cohere) → **không cần tick**, "
                "giữ nguyên section để chunk khớp đúng cấu trúc tài liệu."
            )
        elif _is_short_ctx:
            st.warning(
                f"⚠️ **`{_emb_model}`** có context window chỉ **{_ctx_tokens} token** "
                f"(~{_ctx_tokens * 3:,} ký tự).\n\n"
                "Section dài hơn giới hạn này sẽ bị **truncate silently** khi embed "
                "→ mất nội dung cuối section mà không có cảnh báo.\n\n"
                "**Khuyến nghị: bật checkbox này** để chia nhỏ section trước khi embed."
            )
        elif _is_large_ctx:
            _ctx_label = {
                "BAAI/bge-m3": "8192 token", "bge-m3": "8192 token",
                "nomic-ai/nomic-embed-text-v1.5": "8192 token",
                "Qwen/Qwen3-Embedding-0.6B": "32K token",
                "Qwen/Qwen3-Embedding-4B": "32K token",
                "Qwen/Qwen3-Embedding-8B": "32K token",
                "Alibaba-NLP/gte-Qwen2-7B-instruct": "131K token",
            }.get(_emb_model, "ctx window lớn")
            _provider_label = {
                "openai": "OpenAI (8191 token)",
                "cohere": "Cohere (512 token input, tự truncate)",
            }.get(_emb_provider, "")
            _display = _provider_label or f"{_emb_model} ({_ctx_label})"
            st.success(
                f"✅ **{_display}** có ctx window đủ lớn.\n\n"
                "**Không cần tick** — giữ nguyên section để mỗi chunk khớp "
                "chính xác với một heading, tối ưu cho retrieval theo chủ đề."
            )
        elif _emb_model:
            # Model được chọn nhưng không có trong danh sách đã biết
            st.info(
                f"ℹ️ Không có thông tin ctx window cho `{_emb_model}`.\n\n"
                "**Tắt (mặc định):** giữ nguyên section — phù hợp với hầu hết model ctx lớn.\n\n"
                "**Bật:** nếu model có ctx window ngắn (≤512 token) để tránh truncation."
            )
        else:
            # Chưa chọn embedding model
            st.info(
                "ℹ️ **Chưa chọn Embedding model.**\n\n"
                "Quyết định tùy thuộc vào ctx window của model bạn sẽ dùng:\n\n"
                "• **Ctx ngắn** (phobert: 256, mxbai / FastEmbed: 512 token) → **nên tick**\n\n"
                "• **Ctx lớn** (bge-m3: 8192, Qwen3: 32K, OpenAI / Cohere) → **không cần tick**"
            )

    elif strategy == "hierarchical":
        extra["parent_chunk_size"] = st.number_input(
            "Parent chunk size (chars)", min_value=200, max_value=10000,
            value=2000, step=100,
        )
        # child_chunk_size = chunk_size


    return strategy, int(chunk_size), int(chunk_overlap), extra


# ─── UI: Embedding settings panel ────────────────────────────────────────────

def render_embedding_settings(local_only: bool = False, force_skip: bool = False) -> dict:
    """
    Hiển thị panel cài đặt Embedding trong sidebar.
    Trả về dict cấu hình để truyền vào run_embedder().
    """
    st.subheader("🧮 Cài đặt Embedding")

    # ── Bỏ qua embedding ────────────────────────────────────────────────────
    skip = st.checkbox(
        "Bỏ qua bước Embedding",
        value=force_skip or st.session_state.get("emb_skip", False),
        key="emb_skip",
        disabled=force_skip,
        help="Tắt nếu chỉ muốn kiểm tra Loading & Chunking mà chưa cần embed."
             + (" (tự động bỏ qua vì bước trước đã bị tắt)" if force_skip else ""),
    )
    if skip or force_skip:
        if force_skip:
            st.caption("⏭️ Tự động bỏ qua vì bước Chunking đã bị tắt.")
        return {"skip": True}

    # ── Provider selection ──────────────────────────────────────────────────
    all_providers = list(EMBEDDING_PROVIDER_META.keys())
    if local_only:
        providers = [p for p in all_providers if EMBEDDING_PROVIDER_META[p]["local"]]
    else:
        providers = all_providers

    def _provider_label(p: str) -> str:
        m = EMBEDDING_PROVIDER_META[p]
        return f"{m['icon']} {m['label']}"

    if "sel_emb_provider" not in st.session_state:
        st.session_state["sel_emb_provider"] = (
            "huggingface" if local_only else "openai"
        )
    # Ensure saved provider is still in filtered list
    if st.session_state["sel_emb_provider"] not in providers:
        st.session_state["sel_emb_provider"] = providers[0]

    provider = st.selectbox(
        "Embedding Provider",
        options=providers,
        key="sel_emb_provider",
        format_func=_provider_label,
    )

    meta = EMBEDDING_PROVIDER_META[provider]

    # ── Model selection ─────────────────────────────────────────────────────
    models = meta["models"]
    if "sel_emb_model" not in st.session_state:
        st.session_state["sel_emb_model"] = meta["default"]
    # Reset model if provider changed
    if st.session_state["sel_emb_model"] not in models:
        st.session_state["sel_emb_model"] = meta["default"]

    model_notes = meta.get("model_notes", {})

    def _model_label(m: str) -> str:
        note = model_notes.get(m, "")
        dims = meta["model_dims"].get(m, "?")
        base = f"{m}  ·  {dims}d"
        return f"{base}  —  {note}" if note else base

    model_name = st.selectbox(
        "Model",
        options=models,
        key="sel_emb_model",
        format_func=_model_label,
    )

    dims_this_model = meta["model_dims"].get(model_name, 0)

    # ── Note / description ──────────────────────────────────────────────────
    st.info(
        f"**{meta['icon']} {meta['label']}** · VI {meta['vi_quality']}\n\n"
        f"{meta['note']}"
    )

    # ── API key check (API providers) ────────────────────────────────────────
    env_key = meta.get("requires_env")
    if env_key:
        if _get_env(env_key):
            st.success(f"✅ `{env_key}` đã được cấu hình.")
        else:
            st.warning(f"⚠️ Cần `{env_key}` trong file `.env` hoặc biến môi trường.")
            pkg_probe = meta.get("pkg_probe", "")
            if meta.get("install") and pkg_probe and importlib.util.find_spec(pkg_probe) is None:
                st.caption(f"📦 Cần cài: `{meta['install']}`")

    # ── Provider-specific options ────────────────────────────────────────────
    dimensions:          int | None = None
    device:              str        = "cpu"
    ollama_base_url:     str        = "http://localhost:11434"
    input_type:          str        = meta.get("doc_input_type", "search_document")
    query_instruction:   str | None = None
    document_instruction: str | None = None
    torch_dtype_str:     str        = "auto"
    batch_size:          int        = 32

    # MRL — chỉ OpenAI text-embedding-3-*
    if meta.get("mrl") and "3" in model_name:
        st.markdown("")
        use_mrl = st.checkbox(
            "Dùng MRL (Matryoshka) — cắt giảm số chiều",
            value=False,
            help="text-embedding-3-* hỗ trợ cắt giảm chiều với chỉ ~5% quality loss. "
                 "Giảm RAM & tăng tốc ANN search.",
        )
        if use_mrl:
            dimensions = st.select_slider(
                "Số chiều mục tiêu",
                options=[64, 128, 256, 512, 768, 1024, 1536],
                value=512,
                help=f"Mặc định: {dims_this_model}d → cắt xuống còn n chiều đầu.",
            )
            st.caption(f"💾 Tiết kiệm RAM: {dims_this_model}d → {dimensions}d "
                       f"({dimensions/dims_this_model*100:.0f}% kích thước gốc)")

    # Asymmetric input_type (Cohere)
    if meta.get("asymmetric"):
        doc_type   = meta["doc_input_type"]
        query_type = meta["query_input_type"]
        st.caption(
            f"ℹ️ Dùng **`{doc_type}`** khi embed tài liệu (index time). "
            f"Dùng **`{query_type}`** khi embed câu hỏi (query time). "
            f"Phải nhất quán giữa 2 bước."
        )
        input_type = doc_type   # visualizer luôn ở chế độ document

    # HuggingFace local options
    if provider == "huggingface":
        st.markdown("")
        auto_device, gpu_label = _detect_gpu()
        if auto_device != "cpu":
            st.success(f"✅ Phát hiện GPU: {gpu_label}")
        device_opts = [auto_device, "cpu"] if auto_device != "cpu" else ["cpu"]
        device = st.radio(
            "Device",
            options=device_opts,
            horizontal=True,
            help="cuda: NVIDIA GPU · mps: Apple Silicon · cpu: chậm nhất",
            key="sel_emb_device",
        )
        st.caption(meta.get("install", ""))

        # ── Precision (dtype) ────────────────────────────────────────────────
        st.markdown("")
        _dtype_opts = ["auto", "float16", "bfloat16", "float32"]
        torch_dtype_str = st.select_slider(
            "Precision (dtype)",
            options=_dtype_opts,
            value="auto",
            key="sel_emb_dtype",
            help=(
                "**auto** = fp16 trên CUDA, fp32 trên CPU — khuyến nghị.\n\n"
                "**float16** = giảm 50% VRAM, nhanh hơn ~1.5–2×, chất lượng gần như tương đương.\n\n"
                "**bfloat16** = ổn định hơn fp16 (ít overflow), cần GPU Ampere+ (RTX 30xx+) "
                "hoặc có hỗ trợ bf16 (RTX 20xx hỗ trợ hạn chế).\n\n"
                "**float32** = chất lượng tốt nhất, dùng VRAM gấp đôi fp16."
            ),
        )

        # VRAM estimate
        _model_dims = meta["model_dims"].get(model_name, 0)
        _dtype_multiplier = {"auto": 0.5 if device == "cuda" else 1.0,
                             "float16": 0.5, "bfloat16": 0.5, "float32": 1.0}
        _base_vram_gb = _model_dims * 4 / 1e9 * 1000   # very rough heuristic
        _est_vram = _base_vram_gb * _dtype_multiplier.get(torch_dtype_str, 1.0)
        st.caption(f"💾 VRAM ước tính: ~{_est_vram:.1f}× baseline  "
                   f"({'giảm 50% vs fp32' if torch_dtype_str in ('float16','bfloat16') else 'full precision' if torch_dtype_str == 'float32' else 'tự động chọn tốt nhất'})")

        # ── Batch size ───────────────────────────────────────────────────────
        _batch_default = 8 if device == "cuda" and "7B" in model_name else 32
        batch_size = st.select_slider(
            "Batch size",
            options=[1, 2, 4, 8, 16, 32, 64, 128],
            value=_batch_default,
            key="sel_emb_batch",
            help=(
                "Số văn bản xử lý song song mỗi forward pass.\n\n"
                "**Lớn hơn** = nhanh hơn (throughput cao hơn) nhưng dùng nhiều VRAM hơn.\n\n"
                "**Nhỏ hơn** = an toàn hơn với VRAM ít hoặc model lớn.\n\n"
                "Gợi ý: 32–64 cho model nhỏ (<1B) · 8–16 cho model lớn (4B–8B)."
            ),
        )

        # ── Unload model button ──────────────────────────────────────────────
        st.markdown("")
        col_unload, col_info = st.columns([2, 3])
        with col_unload:
            if st.button("🗑️ Unload model khỏi GPU", key="btn_unload_hf",
                         help="Giải phóng VRAM ngay lập tức. Lần Process tiếp theo sẽ load lại."):
                _load_hf_embedder.clear()
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.ipc_collect()
                except Exception:
                    pass
                st.success("✅ Đã unload model — VRAM được giải phóng.")
        with col_info:
            try:
                import torch
                if torch.cuda.is_available():
                    free_mb  = torch.cuda.mem_get_info()[0] // (1024**2)
                    total_mb = torch.cuda.mem_get_info()[1] // (1024**2)
                    used_mb  = total_mb - free_mb
                    st.caption(f"🖥️ VRAM: **{used_mb:,} MB** used / {total_mb:,} MB total")
            except Exception:
                pass

        # Models cần HF_TOKEN (gated hoặc yêu cầu xác thực)
        _GATED_MODELS = {
            "Qwen/Qwen3-Embedding-0.6B",
            "Qwen/Qwen3-Embedding-4B",
            "Qwen/Qwen3-Embedding-8B",
            "Alibaba-NLP/gte-Qwen2-7B-instruct",
            "intfloat/e5-mistral-7b-instruct",
        }
        if model_name in _GATED_MODELS:
            hf_token_present = bool(
                os.environ.get("HF_TOKEN")
                or os.environ.get("HUGGINGFACE_API_TOKEN")
                or os.environ.get("HUGGINGFACEHUB_API_TOKEN")
            )
            if hf_token_present:
                st.success("✅ `HF_TOKEN` đã được cấu hình — model sẽ tự download.")
            else:
                st.warning(
                    f"⚠️ **`{model_name}`** là model gated — cần `HF_TOKEN` để download.\n\n"
                    "1. Vào https://huggingface.co/settings/tokens → tạo token\n"
                    "2. Accept license tại trang model trên HuggingFace\n"
                    f"3. Thêm vào `.env`: `HF_TOKEN=hf_xxx...`"
                )

        # Instruction-following models
        _INSTRUCTION_MODELS = {
            "Qwen/Qwen3-Embedding-0.6B",
            "Qwen/Qwen3-Embedding-4B",
            "Qwen/Qwen3-Embedding-8B",
            "intfloat/e5-mistral-7b-instruct",
            "Alibaba-NLP/gte-Qwen2-7B-instruct",
        }
        if model_name in _INSTRUCTION_MODELS:
            with st.expander("⚙️ Instruction prefix (tuỳ chọn)"):
                query_instruction = st.text_input(
                    "Query instruction",
                    value="",
                    placeholder="Retrieve relevant passages for the question:",
                    help="Prefix cho câu hỏi (query time). Để trống = dùng mặc định của model.",
                )
                document_instruction = st.text_input(
                    "Document instruction",
                    value="",
                    placeholder="Represent this Vietnamese document:",
                    help="Prefix cho tài liệu (index time). Để trống = dùng mặc định của model.",
                )
                if query_instruction or document_instruction:
                    st.warning(
                        "⚠️ Instruction prefix PHẢI dùng nhất quán giữa index time và query time."
                    )
        query_instruction    = query_instruction    or None
        document_instruction = document_instruction or None

    # Ollama options
    elif provider == "ollama":
        st.markdown("")

        # Cảnh báo ctx window ngắn
        _OLLAMA_CTX: dict[str, int] = {
            "mxbai-embed-large":      512,
            "snowflake-arctic-embed": 512,
            "nomic-embed-text":      2048,
            "bge-m3":                8192,
        }
        _ctx = _OLLAMA_CTX.get(model_name)
        if _ctx and _ctx <= 512:
            st.warning(
                f"⚠️ **`{model_name}`** có context window chỉ **{_ctx} token** (~{_ctx*3} ký tự). "
                f"Chunk dài hơn sẽ bị **tự động truncate** trước khi embed để tránh lỗi 400. "
                f"Nếu chunk của bạn dài, hãy cân nhắc dùng **`bge-m3`** (8192 token) thay thế."
            )
        elif _ctx:
            st.caption(f"ℹ️ Context window: {_ctx:,} token (~{_ctx*3:,} ký tự). "
                       f"Chunk dài hơn sẽ tự động bị truncate.")

        ollama_base_url = st.text_input(
            "Ollama base URL",
            value=st.session_state.get("ollama_emb_url", "http://localhost:11434"),
            key="ollama_emb_url",
            help="URL Ollama server (không có /v1). Model tự pull nếu chưa có.",
        )
        if st.button("🔌 Test Ollama & kiểm tra model", key="test_ollama_emb"):
            with st.spinner("Đang kiểm tra..."):
                try:
                    import ollama as _ollama
                    client = _ollama.Client(host=ollama_base_url)
                    pulled = [m.model for m in client.list().models]
                    if model_name in set(pulled):
                        st.success(f"✅ Kết nối OK · `{model_name}` đã sẵn sàng.")
                    else:
                        st.warning(
                            f"⚠️ Kết nối OK nhưng `{model_name}` chưa pull.\n\n"
                            f"`ollama pull {model_name}`"
                        )
                except Exception as exc:
                    st.error(f"❌ Không kết nối được: `{exc}`")

    elif provider == "fastembed":
        st.caption(meta.get("install", ""))

    # ── Sparse / Hybrid ──────────────────────────────────────────────────────
    st.markdown("")
    st.markdown("**🔀 Hybrid Retrieval (Dense + Sparse)**")
    enable_sparse = st.checkbox(
        "Bật sparse embedding (hybrid retrieval)",
        value=False,
        key="emb_enable_sparse",
        help="Kết hợp dense vector với sparse vector (BM25/SPLADE). "
             "Hybrid luôn tốt hơn chỉ dense, đặc biệt với từ khoá kỹ thuật & tiếng Việt.",
    )
    sparse_method = "bm25"
    if enable_sparse:
        sparse_method = st.radio(
            "Sparse method",
            ["bm25", "splade"],
            horizontal=True,
            help=(
                "**BM25**: không cần GPU, không cần training, phải gọi .fit() trước. "
                "**SPLADE**: transformer-based, mở rộng từ vựng ngữ nghĩa, cần GPU."
            ),
            key="emb_sparse_method",
        )
        if sparse_method == "splade":
            st.caption("💡 `pip install transformers torch` và cần GPU cho tốc độ hợp lý.")
        st.info(
            "ℹ️ BM25 bắt buộc phải **fit trên toàn bộ corpus** trước khi embed — "
            "bước này được thực hiện tự động khi ấn **Process**."
        )

    # ── Preview limit ────────────────────────────────────────────────────────
    st.markdown("")
    max_preview = st.number_input(
        "Số chunk tối đa để embed (preview)",
        min_value=1,
        max_value=500,
        value=EMBED_PREVIEW_LIMIT,
        step=5,
        help=(
            f"Giới hạn số chunk gửi lên embedder để tránh tốn API cost hoặc quá tải RAM. "
            f"Mặc định: {EMBED_PREVIEW_LIMIT}. "
            "Tăng lên để xem nhiều vector hơn."
        ),
        key="emb_max_preview",
    )

    return {
        "skip":                False,
        "provider":            provider,
        "model_name":          model_name,
        "dims":                dims_this_model,
        "enable_sparse":       enable_sparse,
        "sparse_method":       sparse_method,
        "dimensions":          dimensions,
        "device":              device,
        "ollama_base_url":     ollama_base_url,
        "input_type":          input_type,
        "query_instruction":   query_instruction,
        "document_instruction": document_instruction,
        "max_preview":         int(max_preview),
        "torch_dtype_str":     torch_dtype_str if provider == "huggingface" else "auto",
        "batch_size":          batch_size      if provider == "huggingface" else 32,
    }


# ─── UI: Vector DB settings panel ────────────────────────────────────────────


def render_vector_db_settings(local_only: bool = False, force_skip: bool = False) -> dict:
    """
    Hiển thị panel cài đặt Vector Database trong sidebar.
    Trả về dict cấu hình để truyền vào get_vector_store().
    """
    st.subheader("🗃️ Cài đặt Vector Database")

    skip = st.checkbox(
        "Bỏ qua bước Vector DB",
        value=force_skip or st.session_state.get("vdb_skip", False),
        key="vdb_skip",
        disabled=force_skip,
        help="Tắt nếu chỉ muốn kiểm tra Loading/Chunking/Embedding mà chưa cần lưu vào vector DB."
             + (" (tự động bỏ qua vì bước trước đã bị tắt)" if force_skip else ""),
    )
    if skip or force_skip:
        if force_skip:
            st.caption("⏭️ Tự động bỏ qua vì bước Embedding đã bị tắt.")
        return {"skip": True}

    # ── Provider selection ──────────────────────────────────────────────────
    all_providers = list(VECTOR_DB_PROVIDER_META.keys())
    if local_only:
        providers = [p for p in all_providers if VECTOR_DB_PROVIDER_META[p]["local"]]
    else:
        providers = all_providers

    def _vdb_label(p: str) -> str:
        m = VECTOR_DB_PROVIDER_META[p]
        return f"{m['icon']} {m['label']}  {m['tier_icon']}  ·  {m['scale']}"

    if "sel_vdb_provider" not in st.session_state:
        st.session_state["sel_vdb_provider"] = "chroma"
    if st.session_state["sel_vdb_provider"] not in providers:
        st.session_state["sel_vdb_provider"] = providers[0]

    provider = st.selectbox(
        "Vector DB Provider",
        options=providers,
        key="sel_vdb_provider",
        format_func=_vdb_label,
    )

    meta = VECTOR_DB_PROVIDER_META[provider]

    # ── Note / description ──────────────────────────────────────────────────
    st.info(
        f"**{meta['icon']} {meta['label']}** · {meta['tier_icon']} {meta['tier']} "
        f"· {meta['scale']}\n\n{meta['note']}"
    )

    # ── API key / env check ─────────────────────────────────────────────────
    env_key = meta.get("requires_env")
    if env_key:
        if _get_env(env_key):
            st.success(f"✅ `{env_key}` đã được cấu hình.")
        else:
            st.warning(f"⚠️ Cần `{env_key}` trong file `.env` hoặc biến môi trường.")
            pkg_probe = meta.get("pkg_probe", "")
            if meta.get("install") and pkg_probe and importlib.util.find_spec(pkg_probe) is None:
                st.caption(f"📦 Cần cài: `{meta['install']}`")

    # ── Common params ───────────────────────────────────────────────────────
    collection_name = st.text_input(
        "Collection / Index name",
        value="rag",
        key="vdb_collection_name",
        help="Tên collection/index trong vector DB. Mặc định: 'rag'.",
    )

    force_reindex = st.checkbox(
        "Force reindex",
        value=False,
        key="vdb_force_reindex",
        help="Xoá collection cũ và build lại từ đầu. Cần thiết khi đổi embedding model.",
    )

    # ── Provider-specific params ────────────────────────────────────────────
    cfg: dict = {
        "provider":        provider,
        "collection_name": collection_name,
        "force_reindex":   force_reindex,
        "skip":            False,
    }

    params = meta.get("params", [])

    # Local providers — persist_dir
    if "persist_dir" in params:
        default_dir = f"./storage/{provider}"
        persist_dir = st.text_input(
            "Persist directory",
            value=default_dir,
            key=f"vdb_persist_dir_{provider}",
            help="Thư mục lưu index trên disk.",
        )
        cfg["persist_dir"] = persist_dir

    # Distance metric (LanceDB + Qdrant)
    if "distance" in params:
        dist_opts = {
            "lancedb": ["cosine", "l2", "dot"],
            "qdrant":  ["Cosine", "Dot", "Euclid"],
        }.get(provider, ["cosine", "l2", "dot"])
        distance = st.selectbox(
            "Distance metric",
            options=dist_opts,
            key="vdb_distance",
        )
        cfg["distance"] = distance

    # Qdrant extras
    if provider == "qdrant":
        url = st.text_input(
            "Qdrant URL",
            value=_get_env("QDRANT_URL") or "http://localhost:6333",
            key="vdb_qdrant_url",
            help=":memory: cho in-process, http://localhost:6333 cho Docker, hoặc Qdrant Cloud URL.",
        )
        on_disk = st.checkbox(
            "Store vectors on disk (giảm RAM)",
            value=False,
            key="vdb_qdrant_on_disk",
        )
        cfg["url"]     = url
        cfg["on_disk"] = on_disk

    # Weaviate extras
    elif provider == "weaviate":
        url = st.text_input(
            "Weaviate URL",
            value=_get_env("WEAVIATE_URL") or "http://localhost:8080",
            key="vdb_weaviate_url",
        )
        cfg["url"] = url

    # pgvector extras
    elif provider == "pgvector":
        conn_str = st.text_input(
            "DATABASE_URL",
            value=_get_env("DATABASE_URL") or "",
            key="vdb_pg_conn",
            type="password",
            help="postgresql+psycopg://user:pass@host:5432/dbname",
        )
        if "distance_strategy" in params:
            dist_strat = st.selectbox(
                "Distance strategy",
                ["cosine", "euclidean", "inner_product"],
                key="vdb_pg_dist",
            )
            cfg["distance_strategy"] = dist_strat
        cfg["connection_string"] = conn_str

    # Pinecone extras
    elif provider == "pinecone":
        if "cloud" in params:
            col1, col2 = st.columns(2)
            with col1:
                cloud = st.selectbox("Cloud", ["aws", "gcp", "azure"], key="vdb_pc_cloud")
            with col2:
                region = st.text_input("Region", value="us-east-1", key="vdb_pc_region")
            cfg["cloud"]  = cloud
            cfg["region"] = region

    return cfg


# ─── UI: Vector DB results panel ──────────────────────────────────────────────


def render_vector_db_results(vdb_result: dict, vdb_cfg: dict):
    """
    Hiển thị kết quả sau khi đã index vào vector DB.
    """
    provider = vdb_cfg.get("provider", "unknown")
    meta     = VECTOR_DB_PROVIDER_META.get(provider, {})

    # ── Summary metrics ─────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Provider",    f"{meta.get('icon','')} {meta.get('label', provider)}")
    col2.metric("Vectors",     f"{vdb_result.get('n_vectors', 0):,}")
    col3.metric("Collection",  vdb_result.get("collection_name", "—"))
    col4.metric("Status",      "✅ Loaded" if vdb_result.get("loaded_from_existing") else "✅ Indexed")

    st.markdown("---")

    # ── Provider info ───────────────────────────────────────────────────────
    info_col, detail_col = st.columns([1, 1])
    with info_col:
        st.markdown("**📋 Thông tin Provider**")
        st.markdown(f"- **Tier:** {meta.get('tier_icon','')} {meta.get('tier','')}")
        st.markdown(f"- **Mode:** {meta.get('mode','')}")
        st.markdown(f"- **Scale:** {meta.get('scale','')}")
        st.markdown(f"- **Hybrid search:** {'✅' if meta.get('hybrid') else '❌'}")
        st.markdown(f"- **Filtering:** {meta.get('filtering','')}")

    with detail_col:
        st.markdown("**⚙️ Cấu hình đã dùng**")
        ignore = {"skip", "provider"}
        for k, v in vdb_cfg.items():
            if k not in ignore and v is not None and v != "":
                # Mask sensitive values
                display_v = "••••••" if any(s in k for s in ("key", "password", "token", "conn")) and v else v
                st.markdown(f"- **{k}:** `{display_v}`")

    # ── Quick test search ───────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("**🔎 Test search nhanh**")
    test_query = st.text_input(
        "Nhập câu hỏi thử:",
        placeholder="Ví dụ: What is the main topic of this document?",
        key="vdb_test_query",
    )
    top_k = st.slider("Top-K", min_value=1, max_value=10, value=3, key="vdb_test_k")

    if st.button("🔍 Search", key="vdb_test_btn") and test_query:
        vector_store = vdb_result.get("store")
        if vector_store is not None:
            try:
                with st.spinner("Đang search..."):
                    results = vector_store.similarity_search(test_query, k=top_k)
                st.success(f"Tìm thấy {len(results)} kết quả:")
                for i, doc in enumerate(results, 1):
                    src = doc.metadata.get("source", "")
                    src_label = f"  ·  `{src}`" if src else ""
                    with st.expander(f"**Kết quả #{i}**{src_label}", expanded=(i == 1)):
                        st.text(doc.page_content[:800] + ("…" if len(doc.page_content) > 800 else ""))
                        if doc.metadata:
                            st.json({k: v for k, v in doc.metadata.items() if k != "source"})
            except Exception as e:
                st.error(f"❌ Search thất bại: {e}")
        else:
            st.warning("Vector store chưa khả dụng.")


# ─── UI: Pipeline suggestion cards ───────────────────────────────────────────

def render_pipeline_suggestions(suggestions: list[dict]):
    """Hiển thị tối đa 3 gợi ý pipeline. Click 'Áp dụng' để tự điền sidebar."""
    if not suggestions:
        return

    RANK_META = {
        1: ("#27ae60", "#27ae60",  "🏆 Tốt nhất"),
        2: ("#2980b9", "#2980b9",  "⚡ Thay thế tốt"),
        3: ("#e67e22", "#e67e22",  "💡 Phương án khác"),
    }

    st.subheader("💡 Gợi ý cấu hình cho input của bạn")
    st.caption(
        "Dựa trên loại file đã phát hiện. Click **✅ Áp dụng** để tự động điền vào sidebar, "
        "rồi ấn **▶️ Process**."
    )

    cols = st.columns(len(suggestions))
    for col, sug in zip(cols, suggestions):
        rank              = sug["rank"]
        color, _, rlbl    = RANK_META.get(rank, ("#95a5a6", "#95a5a6", f"#{rank}"))
        pdf_strat         = sug["pdf_strategy"]
        chunk_strat       = sug["chunking_strategy"]
        fmt_type          = sug.get("chunking_extra", {}).get("format_type", "")
        fmt_badge         = f" ({fmt_type})" if fmt_type else ""

        emb_provider      = sug.get("emb_provider", "")
        emb_model         = sug.get("emb_model", "")
        emb_sparse        = sug.get("emb_sparse", False)
        emb_reason        = sug.get("emb_reason", "")
        vdb_provider      = sug.get("vdb_provider", "")
        vdb_reason        = sug.get("vdb_reason", "")

        ret_strategy   = sug.get("ret_strategy", "")
        pre_transforms = sug.get("pre_transforms", "none")
        post_reranker  = sug.get("post_reranker", "none")
        ret_reason     = sug.get("ret_reason", "")
        pre_reason     = sug.get("pre_reason", "")
        post_reason    = sug.get("post_reason", "")

        emb_meta          = EMBEDDING_PROVIDER_META.get(emb_provider, {})
        emb_icon          = emb_meta.get("icon", "🧮")
        emb_short         = emb_model.split("/")[-1]
        sparse_badge      = " + BM25" if emb_sparse else ""

        vdb_meta          = VECTOR_DB_PROVIDER_META.get(vdb_provider, {})
        vdb_icon          = vdb_meta.get("icon", "🗃️")
        vdb_label         = vdb_meta.get("label", vdb_provider)

        # ── Inline markdown → HTML ────────────────────────────────────────────
        def _md(text: str) -> str:
            text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)
            text = re.sub(r'`([^`]+)`',
                          r'<code style="background:var(--color-background-tertiary);'
                          r'padding:1px 5px;border-radius:4px;font-size:0.82em;'
                          r'font-family:var(--font-mono);">\1</code>', text)
            return text

        # ── Shared style constants ────────────────────────────────────────────
        _B = (
            "display:inline-block;padding:4px 10px;border-radius:6px;"
            "font-size:0.78rem;font-family:var(--font-mono);font-weight:500;"
            "margin:0 4px 4px 0;white-space:nowrap;"
        )
        _NOTE = (
            "margin-top:10px;padding:9px 11px;"
            "border-left:3px solid {c};border-radius:0 6px 6px 0;"
            "background:{bg};font-size:0.82rem;"
            "color:var(--color-text-primary);line-height:1.6;"
        )

        # ── Tech badges ───────────────────────────────────────────────────────
        s_ret  = (f'<span style="{_B}background:rgba(41,128,185,0.12);color:#2471a3;'
                  f'border:1px solid rgba(41,128,185,0.35);">🔎 {ret_strategy}</span>'
                  ) if ret_strategy else ""
        s_pre  = (f'<span style="{_B}background:rgba(142,68,173,0.10);color:#7d3c98;'
                  f'border:1px solid rgba(142,68,173,0.30);">🔄 {pre_transforms}</span>'
                  ) if pre_transforms and pre_transforms != "none" else ""
        s_post = (f'<span style="{_B}background:rgba(192,57,43,0.10);color:#c0392b;'
                  f'border:1px solid rgba(192,57,43,0.30);">⚙️ {post_reranker}</span>'
                  ) if post_reranker and post_reranker != "none" else ""
        s_loader  = (f'<span style="{_B}background:rgba(26,107,58,0.12);color:#1a6b3a;'
                     f'border:1px solid rgba(26,107,58,0.35);">📄 {pdf_strat}</span>')
        s_chunker = (f'<span style="{_B}background:rgba(180,83,9,0.12);color:#b45309;'
                     f'border:1px solid rgba(180,83,9,0.35);">✂️ {chunk_strat}{fmt_badge}</span>')
        s_emb = (f'<span style="{_B}background:rgba(124,58,237,0.12);color:#7c3aed;'
                 f'border:1px solid rgba(124,58,237,0.35);">{emb_icon} {emb_short}{sparse_badge}</span>'
                 ) if emb_provider else ""
        s_vdb = (f'<span style="{_B}background:rgba(11,110,79,0.12);color:#0b6e4f;'
                 f'border:1px solid rgba(11,110,79,0.35);">{vdb_icon} {vdb_label}</span>'
                 ) if vdb_provider else ""

        # ── Note boxes ────────────────────────────────────────────────────────
        loader_reason     = sug.get("loader_reason", sug.get("reason", ""))
        chunking_reason   = sug.get("chunking_reason", "")

        # ── Note boxes ────────────────────────────────────────────────────────
        _loader_note = (
            f'<div style="{_NOTE.format(c="#1a6b3a", bg="rgba(26,107,58,0.06)")}">'
            f'<span style="font-weight:600;color:#1a6b3a;">📄 Loader</span>'
            f'<div style="margin-top:3px;color:var(--color-text-secondary);">{_md(loader_reason)}</div>'
            f'</div>'
        ) if loader_reason else ""

        _chunking_note = (
            f'<div style="{_NOTE.format(c="#b45309", bg="rgba(180,83,9,0.06)")}">'
            f'<span style="font-weight:600;color:#b45309;">✂️ Chunking</span>'
            f'<div style="margin-top:3px;color:var(--color-text-secondary);">{_md(chunking_reason)}</div>'
            f'</div>'
        ) if chunking_reason else ""

        _emb_note = (
            f'<div style="{_NOTE.format(c="#7c3aed", bg="rgba(124,58,237,0.06)")}">'
            f'<span style="font-weight:600;color:#7c3aed;">🧮 Embedding</span>'
            f'<div style="margin-top:3px;color:var(--color-text-secondary);">{_md(emb_reason)}</div>'
            f'</div>'
        ) if emb_reason else ""

        _vdb_note = (
            f'<div style="{_NOTE.format(c="#0b6e4f", bg="rgba(11,110,79,0.06)")}">'
            f'<span style="font-weight:600;color:#0b6e4f;">🗃️ Vector DB</span>'
            f'<div style="margin-top:3px;color:var(--color-text-secondary);">{_md(vdb_reason)}</div>'
            f'</div>'
        ) if vdb_reason else ""



        def _make_note(color, bg, label, body):
            if not body:
                return ""
            style = ("margin-top:8px;padding:8px 10px;"
                     f"border-left:3px solid {color};"
                     "border-radius:0 6px 6px 0;"
                     f"font-size:0.82rem;background:{bg};")
            return (
                f'<div style="{style}">'
                f'<span style="font-weight:600;color:{color};">{label}</span>'
                f'<div style="margin-top:3px;color:var(--color-text-secondary);">{_md(body)}</div>'
                f'</div>'
            )

        _ret_note  = _make_note("#2471a3", "rgba(41,128,185,0.06)",  "Retrieval",                ret_reason)
        _pre_note  = _make_note("#7d3c98", "rgba(142,68,173,0.05)", "Pre-retrieval (optional)",  pre_reason)
        _post_note = _make_note("#c0392b", "rgba(192,57,43,0.05)",  "Post-retrieval (optional)", post_reason)


        # ── Card HTML ─────────────────────────────────────────────────────────
        card_html = (
            f'<div style="border:1.5px solid {color}40;border-top:3px solid {color};'
            f'border-radius:10px;padding:16px 18px;'
            f'background:var(--color-background-secondary);">'

            # rank pill
            f'<span style="display:inline-block;background:{color}18;color:{color};'
            f'border:1px solid {color}40;padding:2px 10px;border-radius:20px;'
            f'font-size:0.72rem;font-weight:700;letter-spacing:0.3px;margin-bottom:10px;">'
            f'{rlbl}</span>'

            # title
            f'<div style="font-weight:600;font-size:0.92rem;line-height:1.45;'
            f'color:var(--color-text-primary);margin-bottom:10px;">{sug["title"]}</div>'

            # tech badges — flex-wrap prevents misalignment
            f'<div style="display:flex;flex-wrap:wrap;gap:4px;margin-bottom:12px;">'
            f'{s_loader}{s_chunker}{s_emb}{s_vdb}{s_pre}{s_ret}{s_post}</div>'

            # 4 note boxes
            f'{_loader_note}{_chunking_note}{_emb_note}{_vdb_note}{_pre_note}{_ret_note}{_post_note}'

            f'</div>'
        )

        with col:
            st.markdown(card_html, unsafe_allow_html=True)
            st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
            if st.button(
                "✅ Áp dụng cấu hình này",
                key=f"apply_sug_{rank}",
                width="stretch",
                type="secondary",
            ):
                st.session_state["_want_pdf_strategy"]      = pdf_strat
                st.session_state["_want_chunking_strategy"] = chunk_strat
                if fmt_type:
                    st.session_state["_want_format_type"]   = fmt_type
                if emb_provider:
                    st.session_state["_want_emb_provider"]  = emb_provider
                    st.session_state["_want_emb_model"]     = emb_model
                if emb_sparse:
                    st.session_state["_want_emb_sparse"]        = True
                    st.session_state["_want_emb_sparse_method"] = "bm25"
                else:
                    st.session_state["_want_emb_sparse"]        = False
                if vdb_provider:
                    st.session_state["_want_vdb_provider"]  = vdb_provider
                if ret_strategy:
                    st.session_state["_want_ret_strategy"]  = ret_strategy
                if pre_transforms and pre_transforms != "none":
                    st.session_state["_want_pre_transforms"] = [pre_transforms]
                if post_reranker and post_reranker != "none":
                    st.session_state["_want_post_reranker"] = post_reranker
                st.rerun()

    st.markdown("")

# ─── UI: Query pipeline results ───────────────────────────────────────────────


def render_query_pipeline_results(
    query:      str,
    pre_cfg:    dict,
    ret_cfg:    dict,
    post_cfg:   dict,
    vdb_result: dict,
    emb_cfg:    dict,
    prompt_cfg: dict | None = None,
    gen_cfg:    dict | None = None,
    history:    list[dict] | None = None,
):
    """
    Chạy toàn bộ query pipeline (pre → retrieval → post → prompt → generation)
    và hiển thị kết quả từng bước để người dùng debug.
    """
    prompt_cfg = prompt_cfg or {"template": "citation", "language": "both"}
    gen_cfg    = gen_cfg    or {"provider": "openai", "model_name": "gpt-4.1-mini",
                                "temperature": 0.0, "max_tokens": 2048, "streaming": True}
    vector_store = vdb_result.get("store")
    if vector_store is None:
        st.warning("Vector store chưa khả dụng — chạy Process trước.")
        return

    # ── Bước 1: Pre-retrieval ──────────────────────────────────────────────
    st.markdown("#### 🔄 Bước 7 — Pre-retrieval")
    pre_transforms = pre_cfg.get("transformations", ["none"])

    with st.spinner("Đang chạy pre-retrieval..."):
        try:
            from pre_retrieval import build_pipeline
            pre_pipeline = build_pipeline(
                transformations=pre_transforms,
                llm_model=pre_cfg.get("llm_model", "gpt-4.1-mini"),
                llm_provider=pre_cfg.get("llm_provider", "openai"),
            )
            transform_result = pre_pipeline.transform(query)
        except Exception as e:
            st.error(f"Pre-retrieval lỗi: {e}")
            return

    if pre_transforms == ["none"] or pre_transforms == ["none"]:
        st.caption("_Passthrough — query không được biến đổi._")
    else:
        for i, q in enumerate(transform_result.queries, 1):
            label = "Query gốc" if q == query else f"Query #{i}"
            st.markdown(
                f'<div style="padding:6px 12px;margin:4px 0;background:var(--color-background-secondary);'
                f'border-left:3px solid #7d3c98;border-radius:0 6px 6px 0;font-size:0.87rem;">'
                f'<b style="color:#7d3c98;">{label}:</b> {q}</div>',
                unsafe_allow_html=True,
            )
        if transform_result.metadata_filter:
            st.json({"metadata_filter": transform_result.metadata_filter})
        if transform_result.intent:
            st.caption(f"Intent: `{transform_result.intent}`")
        if transform_result.retrieval_path:
            st.caption(f"Route: `{transform_result.retrieval_path}`")

    st.markdown("---")

    # ── Bước 2: Retrieval ──────────────────────────────────────────────────
    st.markdown("#### 🔎 Bước 8 — Retrieval")
    ret_strategy = ret_cfg.get("strategy", "dense")
    top_k        = ret_cfg.get("top_k", 10)

    with st.spinner(f"Đang retrieval ({ret_strategy}, top-{top_k})..."):
        try:
            from retrieval import get_retriever
            chunks = st.session_state.get("chunks", [])
            retriever = get_retriever(
                strategy=ret_strategy,
                vector_store=vector_store,
                documents=chunks,
                top_k=top_k,
                fusion_method=ret_cfg.get("fusion_method", "rrf"),
                alpha=ret_cfg.get("hybrid_alpha", 0.5),
            )
            retrieved_docs = retriever.retrieve(transform_result)
        except Exception as e:
            st.error(f"Retrieval lỗi: {e}")
            return

    st.caption(f"Tìm được **{len(retrieved_docs)}** documents với strategy `{ret_strategy}`")
    for i, doc in enumerate(retrieved_docs, 1):
        src   = doc.metadata.get("source", "")
        page  = doc.metadata.get("page", "")
        score = (doc.metadata.get("rrf_score")
                 or doc.metadata.get("relevance_score")
                 or doc.metadata.get("bm25_score", 0))
        score_str = f" · score={score:.3f}" if score else ""
        label = f"#{i}  {src}" + (f" p.{page}" if page else "") + score_str
        with st.expander(label, expanded=(i <= 3)):
            st.text(doc.page_content[:600] + ("…" if len(doc.page_content) > 600 else ""))

    st.markdown("---")

    # ── Bước 3: Post-retrieval ─────────────────────────────────────────────
    st.markdown("#### ⚙️ Bước 9 — Post-retrieval")
    reranker = post_cfg.get("reranker", "none")

    if reranker == "none" and not post_cfg.get("apply_redundancy") and not post_cfg.get("apply_mmr"):
        st.caption("_Post-retrieval bị tắt — kết quả giữ nguyên từ Retrieval._")
        final_docs = retrieved_docs
    else:
        with st.spinner("Đang xử lý post-retrieval..."):
            try:
                from post_retrieval import build_pipeline as build_post
                post_pipeline = build_post(
                    reranker=reranker,
                    top_n=post_cfg.get("top_n", 5),
                    cross_encoder_model=post_cfg.get("cross_encoder_model", "BAAI/bge-reranker-v2-m3"),
                    apply_redundancy=post_cfg.get("apply_redundancy", True),
                    apply_mmr=post_cfg.get("apply_mmr", False),
                    apply_compression=post_cfg.get("apply_compression", False),
                    apply_llm_filter=post_cfg.get("apply_llm_filter", False),
                    context_ordering=post_cfg.get("context_ordering", "sandwich"),
                )
                final_docs = post_pipeline.process(query, retrieved_docs)
            except Exception as e:
                st.error(f"Post-retrieval lỗi: {e}")
                return

        st.caption(
            f"Sau post-retrieval: **{len(retrieved_docs)}** → **{len(final_docs)}** documents"
            + (f" (reranker: `{reranker}`)" if reranker != "none" else "")
        )

    st.markdown("**📋 Kết quả cuối (đưa vào LLM):**")
    for i, doc in enumerate(final_docs, 1):
        src  = doc.metadata.get("source", "")
        page = doc.metadata.get("page", "")
        rs   = (doc.metadata.get("rerank_score")
                or doc.metadata.get("rrf_score")
                or doc.metadata.get("relevance_score", 0))
        rs_str = f" · rerank={rs:.3f}" if rs else ""
        label  = f"#{i}  {src}" + (f" p.{page}" if page else "") + rs_str
        with st.expander(label, expanded=(i <= 2)):
            st.text(doc.page_content[:600] + ("…" if len(doc.page_content) > 600 else ""))

    st.markdown("---")

    # ── Bước 10: Prompt ───────────────────────────────────────────────────────
    st.markdown("#### 📝 Bước 10 — Prompt")

    try:
        from prompt import get_prompt_builder
        prompt_builder = get_prompt_builder(
            template          = prompt_cfg.get("template", "citation"),
            language          = prompt_cfg.get("language", "both"),
            max_context_chars = prompt_cfg.get("max_context_chars", 0),
            **({"max_history_turns": prompt_cfg["max_history_turns"]}
               if prompt_cfg.get("template") == "conversational" else {}),
            **({"validate_citations": True}
               if prompt_cfg.get("template") == "citation" else {}),
        )
        prompt_result = prompt_builder.build(
            query   = query,
            docs    = final_docs,
            history = history or [],
        )
    except Exception as e:
        st.error(f"Prompt builder lỗi: {e}")
        return

    template_name = prompt_cfg.get("template", "citation")
    st.caption(
        f"Template: `{template_name}` · Ngôn ngữ: `{prompt_cfg.get('language','both')}` · "
        f"{prompt_result.n_sources} nguồn trong context"
    )

    with st.expander("👁️ Xem prompt đầy đủ gửi lên LLM", expanded=False):
        # Hiển thị từng message riêng biệt
        for msg in prompt_result.messages:
            role_label = {"system": "⚙️ System", "user": "👤 User", "assistant": "🤖 Assistant"}.get(
                msg["role"], msg["role"].capitalize()
            )
            st.markdown(f"**{role_label}**")
            st.text_area(
                label=f"msg_{msg['role']}",
                value=msg["content"],
                height=min(300, max(80, msg["content"].count("\n") * 20 + 80)),
                disabled=True,
                label_visibility="collapsed",
                key=f"prompt_msg_{msg['role']}_{id(msg)}",
            )

    st.markdown("---")

    # ── Bước 11: Generation ───────────────────────────────────────────────────
    st.markdown("#### 🤖 Bước 11 — Generation")

    provider   = gen_cfg.get("provider",   "openai")
    model_name = gen_cfg.get("model_name", "gpt-4.1-mini")
    streaming  = gen_cfg.get("streaming",  True)

    st.caption(f"Provider: `{provider}` · Model: `{model_name}` · Streaming: `{streaming}`")

    try:
        from generation import get_generator
        generator = get_generator(
            provider   = provider,
            model_name = model_name,
            temperature = gen_cfg.get("temperature", 0.0),
            max_tokens  = gen_cfg.get("max_tokens",  2048),
            streaming   = streaming,
            **({"base_url":   gen_cfg.get("base_url", "http://localhost:11434"),
                "auto_pull":  gen_cfg.get("auto_pull", True)}
               if provider == "ollama" else {}),
        )
    except Exception as e:
        st.error(f"Khởi tạo generator lỗi: {e}")
        return

    st.markdown("**💬 Câu trả lời:**")
    answer_placeholder = st.empty()

    try:
        if streaming:
            # ── Streaming mode ──────────────────────────────────────────────
            full_answer = ""
            with st.spinner(""):
                for chunk in generator.stream(prompt_result):
                    full_answer += chunk
                    answer_placeholder.markdown(full_answer + "▌")
            answer_placeholder.markdown(full_answer)

            # Post-process để lấy citations
            from generation.base import GenerationResult
            from prompt.citation import CitationPromptBuilder
            from prompt.structured_output import StructuredOutputPromptBuilder

            cited: list[int] = []
            structured = None
            if template_name == "citation":
                cited = CitationPromptBuilder.extract_cited_indices(full_answer)
                cited = [i for i in cited if 1 <= i <= prompt_result.n_sources]
            elif template_name == "structured":
                structured = StructuredOutputPromptBuilder.parse_response(full_answer)

            gen_result = GenerationResult(
                answer        = full_answer,
                provider      = provider,
                model_name    = model_name,
                cited_sources = cited,
                structured    = structured,
            )
        else:
            # ── Non-streaming mode ──────────────────────────────────────────
            with st.spinner(f"Đang sinh câu trả lời ({model_name})..."):
                gen_result = generator.generate(prompt_result)
            answer_placeholder.markdown(gen_result.answer)

    except Exception as e:
        st.error(f"Generation lỗi: {e}")
        st.exception(e)
        return

    # ── Metadata: token usage + citations ────────────────────────────────────
    meta_cols = st.columns(4)
    meta_cols[0].metric("Provider",  f"{provider}")
    meta_cols[1].metric("Model",     model_name.split("/")[-1].split(":")[0])
    meta_cols[2].metric("Input tok", f"{gen_result.input_tokens:,}"  if gen_result.input_tokens  else "—")
    meta_cols[3].metric("Output tok", f"{gen_result.output_tokens:,}" if gen_result.output_tokens else "—")

    # ── Citation analysis ─────────────────────────────────────────────────────
    if template_name == "citation" and prompt_result.n_sources > 0:
        st.markdown("")
        if gen_result.cited_sources:
            cited_docs = [
                final_docs[i - 1]
                for i in gen_result.cited_sources
                if i <= len(final_docs)
            ]
            st.markdown(f"**📚 Nguồn được trích dẫn: {gen_result.cited_sources}**")
            for i, idx in enumerate(gen_result.cited_sources):
                if idx <= len(final_docs):
                    doc = final_docs[idx - 1]
                    src = doc.metadata.get("source", "")
                    pg  = doc.metadata.get("page", "")
                    label = f"[NGUỒN {idx}] {src}" + (f" p.{pg}" if pg else "")
                    with st.expander(label, expanded=False):
                        st.text(doc.page_content[:400] + ("…" if len(doc.page_content) > 400 else ""))
        else:
            st.caption("ℹ️ Câu trả lời không trích dẫn nguồn cụ thể nào.")

    # ── Structured output display ─────────────────────────────────────────────
    if template_name == "structured" and gen_result.structured:
        st.markdown("")
        st.markdown("**📊 Structured Output:**")
        parsed = gen_result.structured
        if parsed.get("claims"):
            st.markdown("**Claims:**")
            for claim in parsed["claims"]:
                st.markdown(f"- {claim}")
        if parsed.get("sources"):
            st.markdown(f"**Sources:** {parsed['sources']}")
        if parsed.get("confidence"):
            conf_color = {"high": "🟢", "medium": "🟡", "low": "🔴"}.get(parsed["confidence"], "⚪")
            st.markdown(f"**Confidence:** {conf_color} {parsed['confidence']}")
        if parsed.get("unanswered"):
            st.info(f"**Chưa trả lời được:** {parsed['unanswered']}")

    # Lưu vào session_state để có thể reuse trong conversational mode
    _hist = st.session_state.get("_query_history", [])
    _hist.append({"role": "user",      "content": query})
    _hist.append({"role": "assistant", "content": gen_result.answer})
    st.session_state["_query_history"] = _hist[-20:]  # giữ tối đa 10 lượt


# ─── Helper: render text+images mixed content ─────────────────────────────────

def _render_images_in_text(content: str, area_key: str, area_height: int):
    """
    Render plain-text mode: text_area cho text, st.image cho ảnh xen kẽ.
    Path ảnh trong markdown đã được MarkerPDFLoader convert sang tuyệt đối.
    """
    IMAGE_RE = re.compile(
        r'!\[([^\]]*)\]\((data:image/[^)]+|[^)]+\.(jpe?g|png|webp|gif|bmp))\)',
        re.IGNORECASE
    )

    if not IMAGE_RE.search(content):
        st.text_area(
            "Nội dung",
            value=content,
            height=area_height,
            key=area_key,
            disabled=True,
            label_visibility="collapsed",
        )
        return

    parts   = IMAGE_RE.split(content)
    matches = IMAGE_RE.findall(content)

    text_before = parts[0]
    if text_before.strip():
        st.text_area(
            "Nội dung",
            value=text_before,
            height=min(area_height, max(60, len(text_before) // 3)),
            key=f"{area_key}_t0",
            disabled=True,
            label_visibility="collapsed",
        )

    for m_idx, (alt, img_path, _ext) in enumerate(matches):
        caption = alt if alt else "Figure"

        if img_path.startswith("data:"):
            # Docling embedded mode: data URI → decode và render
            import base64 as _b64
            try:
                header, b64data = img_path.split(",", 1)
                img_bytes = _b64.b64decode(b64data)
                st.image(img_bytes, caption=caption, width="stretch")
            except Exception:
                st.caption(f"🖼️ *(không render được data URI)*")
        else:
            img_file = Path(img_path)
            if img_file.exists():
                st.image(str(img_file), caption=caption, width="stretch")
            else:
                st.caption(f"🖼️ `{img_path}` *(ảnh không tìm thấy trên disk)*")

        text_after_idx = 1 + m_idx * 4 + 3
        if text_after_idx < len(parts):
            text_after = parts[text_after_idx]
            if text_after.strip():
                st.text_area(
                    "Nội dung",
                    value=text_after,
                    height=min(area_height, max(60, len(text_after) // 3)),
                    key=f"{area_key}_t{m_idx+1}",
                    disabled=True,
                    label_visibility="collapsed",
                )


def _local_images_to_base64(content: str) -> str:
    """
    Scan Markdown content tìm ![](<local_path>), đọc file và
    chuyển thành data URI base64 để st.markdown() render được.
    URL (http/https) và path không tồn tại được giữ nguyên.
    """
    import base64, mimetypes

    IMAGE_RE = re.compile(r'!\[([^\]]*)\]\(([^)]+)\)')

    def _to_data_uri(m: re.Match) -> str:
        alt     = m.group(1)
        src     = m.group(2)
        # Bỏ qua URL và data URI (Docling embedded mode đã là base64 rồi)
        if src.startswith("http://") or src.startswith("https://") or src.startswith("data:"):
            return m.group(0)
        img_path = Path(src)
        if not img_path.exists():
            return m.group(0)   # giữ nguyên, sẽ hiện broken image
        mime, _ = mimetypes.guess_type(str(img_path))
        mime = mime or "image/jpeg"
        b64  = base64.b64encode(img_path.read_bytes()).decode()
        return f"![{alt}](data:{mime};base64,{b64})"

    return IMAGE_RE.sub(_to_data_uri, content)


def render_content_with_images(content: str, area_key: str, area_height: int,
                               display_mode: str = "Text"):
    """
    Hiển thị nội dung theo display_mode được truyền vào từ global toggle.

    - Text    : text_area thuần, ảnh local render inline bằng st.image
    - Markdown: st.markdown với ảnh được convert sang base64 data URI
    """
    st.caption("Nội dung")
    if display_mode == "Text":
        _render_images_in_text(content, area_key, area_height)
    else:
        # Nếu content có literal \n (2 ký tự, thường do JSON fallback), chuyển thành newline thật
        if r"\n" in content and "\n" not in content:
            content = content.replace(r"\n", "\n")
        md_with_embedded = _local_images_to_base64(content)
        st.markdown(md_with_embedded, unsafe_allow_html=True)


# ─── UI: Loader results ───────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def _compute_loader_stats(doc_contents: tuple[str, ...], doc_metas: tuple[str, ...]) -> dict:
    """Cache thống kê loader — chỉ tính lại khi docs thay đổi."""
    import json
    total_chars = sum(len(c) for c in doc_contents)
    file_types: dict[str, int] = {}
    for m_str in doc_metas:
        ft = json.loads(m_str).get("file_type", "unknown")
        file_types[ft] = file_types.get(ft, 0) + 1
    return {"total_chars": total_chars, "file_types": file_types}


@st.cache_data(show_spinner=False)
def _compute_chunking_stats(chunk_contents: tuple[str, ...]) -> dict:
    """Cache thống kê chunking — chỉ tính lại khi chunks thay đổi."""
    if not chunk_contents:
        return {"total_chars": 0, "avg": 0, "min": 0, "max": 0, "sizes": ()}
    sizes = tuple(len(c) for c in chunk_contents)
    return {
        "total_chars": sum(sizes),
        "avg":  sum(sizes) // len(sizes),
        "min":  min(sizes),
        "max":  max(sizes),
        "sizes": sizes,
    }


_LOADER_PAGE_SIZE  = 20
_CHUNKING_PAGE_SIZE = 30


def render_loader_results(docs: list, display_mode: str = "Text"):
    """Hiển thị kết quả sau bước loading."""
    import json

    st.markdown("---")
    st.header("📂 Kết quả bước Loading")

    # --- Thống kê (cached) ---
    stats = _compute_loader_stats(
        tuple(d.page_content for d in docs),
        tuple(json.dumps(d.metadata, default=str, sort_keys=True) for d in docs),
    )
    total_chars = stats["total_chars"]
    file_types  = stats["file_types"]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("📄 Tổng documents", len(docs))
    col2.metric("🔤 Tổng ký tự", f"{total_chars:,}")
    col3.metric("📊 Ký tự TB / doc", f"{total_chars // max(len(docs), 1):,}")
    col4.metric("📁 Loại file", len(file_types))

    if file_types:
        st.markdown("**Phân bổ loại file:**")
        cols = st.columns(min(len(file_types), 6))
        for i, (ft, count) in enumerate(sorted(file_types.items())):
            cols[i % len(cols)].markdown(
                f"{file_type_badge(ft)} **×{count}**", unsafe_allow_html=True
            )
        st.markdown("")

    # --- Search + Pagination ---
    st.markdown(f"**Chi tiết {len(docs)} document(s):**")
    search_query = st.text_input("🔍 Tìm kiếm trong nội dung", placeholder="Nhập từ khoá…",
                                 key="loader_search")

    filtered = [
        (i, doc) for i, doc in enumerate(docs)
        if not search_query or search_query.lower() in doc.page_content.lower()
    ]

    total_pages = max(1, (len(filtered) + _LOADER_PAGE_SIZE - 1) // _LOADER_PAGE_SIZE)
    if "loader_page" not in st.session_state or search_query != st.session_state.get("_loader_search_prev"):
        st.session_state["loader_page"] = 0
        st.session_state["_loader_search_prev"] = search_query
    page = st.session_state["loader_page"]

    if total_pages > 1:
        col_prev, col_info, col_next = st.columns([1, 3, 1])
        with col_prev:
            if st.button("◀ Trước", key="loader_prev", disabled=(page == 0)):
                st.session_state["loader_page"] = max(0, page - 1)
        with col_info:
            st.caption(f"Trang {page+1}/{total_pages}  ·  {len(filtered)} kết quả")
        with col_next:
            if st.button("Sau ▶", key="loader_next", disabled=(page >= total_pages - 1)):
                st.session_state["loader_page"] = min(total_pages - 1, page + 1)

    start = page * _LOADER_PAGE_SIZE
    page_items = filtered[start: start + _LOADER_PAGE_SIZE]

    for i, doc in page_items:
        content = doc.page_content
        meta    = doc.metadata
        ft      = meta.get("file_type", "unknown")
        source  = Path(meta.get("source", "")).name or "unknown"
        page_n  = meta.get("page", "")
        chars   = len(content)
        page_info = f" · Trang {page_n}" if page_n else ""
        label = f"Doc {i+1}  |  {source}{page_info}  |  {chars:,} ký tự"

        with st.expander(label, expanded=(i == 0 and page == 0)):
            st.markdown(file_type_badge(ft), unsafe_allow_html=True)
            st.markdown("")
            render_content_with_images(
                content,
                area_key=f"doc_content_{i}",
                area_height=min(300, max(100, len(content) // 3)),
                display_mode=display_mode,
            )
            st.markdown("**Metadata:**")
            meta_clean = {k: v for k, v in meta.items() if v is not None and v != ""}
            st.json(meta_clean, expanded=False)


# ─── UI: Chunking results ─────────────────────────────────────────────────────

def render_chunking_results(chunks: list, strategy: str, display_mode: str = "Text"):
    """Hiển thị kết quả sau bước chunking."""
    st.markdown("---")
    st.header("✂️ Kết quả bước Chunking")

    if not chunks:
        st.warning("Không có chunk nào được tạo ra.")
        return

    # --- Thống kê (cached) ---
    stats = _compute_chunking_stats(tuple(c.page_content for c in chunks))
    total_chars = stats["total_chars"]
    sizes       = stats["sizes"]

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("🧩 Tổng chunks",  len(chunks))
    col2.metric("🔤 Tổng ký tự",   f"{total_chars:,}")
    col3.metric("📊 TB / chunk",    f"{stats['avg']:,}")
    col4.metric("⬇️ Nhỏ nhất",     f"{stats['min']:,}")
    col5.metric("⬆️ Lớn nhất",     f"{stats['max']:,}")

    st.markdown("**Phân bổ kích thước chunk:**")
    size_buckets = {"<200": 0, "200-500": 0, "500-1000": 0, "1000-2000": 0, ">2000": 0}
    for s in sizes:
        if s < 200:   size_buckets["<200"]      += 1
        elif s < 500: size_buckets["200-500"]    += 1
        elif s < 1000:size_buckets["500-1000"]   += 1
        elif s < 2000:size_buckets["1000-2000"]  += 1
        else:         size_buckets[">2000"]      += 1

    bucket_cols = st.columns(5)
    for i, (label, count) in enumerate(size_buckets.items()):
        pct = count / len(chunks) * 100
        bucket_cols[i].metric(label, f"{count} ({pct:.0f}%)")

    levels = set(c.metadata.get("chunk_level", "") for c in chunks)
    levels.discard("")
    if levels:
        st.info(f"🏗️ Hierarchical chunks: {', '.join(sorted(levels))}")

    st.markdown(f"**Chi tiết {len(chunks)} chunk(s):**")

    # Bộ lọc
    col_f1, col_f2 = st.columns([3, 1])
    with col_f1:
        chunk_search = st.text_input(
            "🔍 Tìm trong chunk", placeholder="Nhập từ khoá…", key="chunk_search"
        )
    with col_f2:
        if levels:
            level_filter = st.selectbox("Lọc level", ["Tất cả"] + sorted(levels))
        else:
            level_filter = "Tất cả"

    # Lọc trước khi paginate
    filtered_chunks = [
        (i, c) for i, c in enumerate(chunks)
        if (not chunk_search or chunk_search.lower() in c.page_content.lower())
        and (level_filter == "Tất cả" or c.metadata.get("chunk_level", "") == level_filter)
    ]

    if not filtered_chunks:
        st.info("Không có chunk nào khớp với bộ lọc.")
        return

    total_pages = max(1, (len(filtered_chunks) + _CHUNKING_PAGE_SIZE - 1) // _CHUNKING_PAGE_SIZE)
    _prev_search = st.session_state.get("_chunk_search_prev")
    _prev_level  = st.session_state.get("_chunk_level_prev")
    if chunk_search != _prev_search or level_filter != _prev_level:
        st.session_state["chunk_page"] = 0
        st.session_state["_chunk_search_prev"] = chunk_search
        st.session_state["_chunk_level_prev"]  = level_filter
    if "chunk_page" not in st.session_state:
        st.session_state["chunk_page"] = 0
    page = st.session_state["chunk_page"]

    if total_pages > 1:
        col_prev, col_info, col_next = st.columns([1, 3, 1])
        with col_prev:
            if st.button("◀ Trước", key="chunk_prev", disabled=(page == 0)):
                st.session_state["chunk_page"] = max(0, page - 1)
        with col_info:
            st.caption(f"Trang {page+1}/{total_pages}  ·  {len(filtered_chunks)} kết quả")
        with col_next:
            if st.button("Sau ▶", key="chunk_next", disabled=(page >= total_pages - 1)):
                st.session_state["chunk_page"] = min(total_pages - 1, page + 1)

    start = page * _CHUNKING_PAGE_SIZE
    page_items = filtered_chunks[start: start + _CHUNKING_PAGE_SIZE]

    for i, chunk in page_items:
        content     = chunk.page_content
        meta        = chunk.metadata
        char_count  = len(content)
        source      = Path(meta.get("source", "")).name or ""
        chunk_level = meta.get("chunk_level", "")

        source_info = f" · {source}" if source else ""
        level_info  = f" · {chunk_level}" if chunk_level else ""
        label = f"Chunk {i+1}{level_info}{source_info}  |  {char_count:,} ký tự"

        with st.expander(label, expanded=(i < 3 and page == 0)):
            badge_html = chunk_type_badge(i, chunk_level)
            st.markdown(badge_html, unsafe_allow_html=True)
            st.markdown("")
            render_content_with_images(
                content,
                area_key=f"chunk_{i}",
                area_height=min(250, max(80, char_count // 2)),
                display_mode=display_mode,
            )
            with st.container():
                meta_display = {
                    k: v for k, v in meta.items()
                    if v is not None and v != "" and k not in ("late_embedding",)
                }
                st.markdown("**Metadata:**")
                st.json(meta_display, expanded=False)

    st.caption(f"Hiển thị {len(page_items)} / {len(filtered_chunks)} chunks (trang {page+1})")


@st.cache_data(show_spinner=False)
def _compute_cosine_heatmap_html(dense_preview: tuple, n_heat: int) -> str:
    """Tính cosine similarity matrix và sinh HTML heatmap. Cache theo vectors."""
    import numpy as np

    vecs  = np.array(dense_preview[:n_heat], dtype=np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1e-9
    vecs_norm  = vecs / norms
    sim_matrix = vecs_norm @ vecs_norm.T

    def _cell_color(val: float) -> str:
        t = (val + 1) / 2
        if t < 0.5:
            r = int(255 * 2 * t); g = r; b = 255
        else:
            t2 = (t - 0.5) * 2
            r = 255; g = int(255 * (1 - t2)); b = g
        return f"rgb({r},{g},{b})"

    def _text_color(val: float) -> str:
        t = (val + 1) / 2
        return "#111" if 0.25 < t < 0.75 else ("#fff" if t <= 0.25 else "#111")

    labels = [f"C{i+1}" for i in range(n_heat)]
    header = (
        "<tr><th style='background:#1a252f;color:#fff;padding:6px 10px;'></th>"
        + "".join(f"<th style='background:#1a252f;color:#fff;padding:6px 10px;font-size:0.8rem'>{l}</th>" for l in labels)
        + "</tr>"
    )
    rows = []
    for i in range(n_heat):
        cells = [f"<td style='background:#1a252f;color:#fff;padding:6px 10px;font-size:0.8rem;font-weight:600'>{labels[i]}</td>"]
        for j in range(n_heat):
            v  = float(sim_matrix[i, j])
            bg = _cell_color(v); fg = _text_color(v)
            cells.append(
                f"<td style='background:{bg};color:{fg};padding:6px 10px;"
                f"text-align:center;font-size:0.82rem;font-weight:{'700' if i==j else '400'}'>"
                f"{v:.2f}</td>"
            )
        rows.append("<tr>" + "".join(cells) + "</tr>")

    return (
        "<div style='overflow-x:auto'>"
        "<table style='border-collapse:collapse;border-radius:8px;overflow:hidden;font-family:monospace'>"
        f"<thead>{header}</thead><tbody>{''.join(rows)}</tbody>"
        "</table></div>"
    )



# ─── UI: Embedding results ────────────────────────────────────────────────────

def render_embedding_results(chunks: list, embed_result: dict, emb_cfg: dict):
    """Hiển thị kết quả sau bước embedding."""
    import math

    st.markdown("---")
    st.header("🧮 Kết quả bước Embedding")

    if not embed_result or not embed_result.get("dense"):
        st.warning("Không có vector nào được tạo ra.")
        return

    dense        = embed_result["dense"]
    sparse       = embed_result.get("sparse")
    dims         = embed_result["dims"]
    n_embedded   = embed_result["n_embedded"]
    truncated    = embed_result["truncated"]
    provider     = emb_cfg["provider"]
    model_name   = emb_cfg["model_name"]
    meta         = EMBEDDING_PROVIDER_META.get(provider, {})

    # ── Summary metrics ──────────────────────────────────────────────────────
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("🧮 Vectors tạo ra",   n_embedded)
    col2.metric("📐 Số chiều (dim)",   f"{dims:,}")
    col3.metric("💾 RAM / vector",      f"{dims * 4 / 1024:.1f} KB")
    col4.metric("📦 Tổng RAM (dense)", f"{n_embedded * dims * 4 / (1024**2):.2f} MB")
    col5.metric("🔀 Sparse",           "✅ BẬT" if sparse else "❌ TẮT")

    if truncated:
        st.warning(
            f"⚠️ Chỉ embed **{n_embedded}/{len(chunks)}** chunks đầu tiên (giới hạn preview). "
            "Tăng **Số chunk tối đa** trong sidebar để embed thêm."
        )

    # ── Provider + model info ────────────────────────────────────────────────
    st.markdown(
        f"**Provider:** {meta.get('icon', '')} `{provider}` &nbsp;&nbsp; "
        f"**Model:** `{model_name}` &nbsp;&nbsp; "
        f"**VI Quality:** {meta.get('vi_quality', '—')}",
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # ── Cosine similarity heatmap ─────────────────────────────────────────────
    n_heat = min(10, len(dense))
    if n_heat >= 2:
        st.markdown(f"**🌡️ Cosine Similarity giữa {n_heat} chunks đầu**")
        st.caption(
            "Giá trị gần 1.0 → hai chunk rất giống nhau về ngữ nghĩa. "
            "Diagonal luôn = 1.0 (self-similarity). Màu đậm → similarity cao."
        )
        table_html = _compute_cosine_heatmap_html(
            tuple(tuple(v) for v in dense[:n_heat]), n_heat
        )
        st.markdown(table_html, unsafe_allow_html=True)
        st.markdown("")

    # ── Per-chunk vector preview ──────────────────────────────────────────────
    st.markdown(f"**📋 Chi tiết {n_embedded} vector(s)**")
    search_emb = st.text_input(
        "🔍 Tìm trong nội dung chunk", placeholder="Nhập từ khoá…", key="emb_search"
    )

    PREVIEW_DIMS = 8

    for i in range(n_embedded):
        if i >= len(chunks) or i >= len(dense):
            break
        chunk   = chunks[i]
        content = chunk.page_content
        vec     = dense[i]
        norm    = math.sqrt(sum(v * v for v in vec))

        if search_emb and search_emb.lower() not in content.lower():
            continue

        preview_vals = [f"{v:+.4f}" for v in vec[:PREVIEW_DIMS]]
        vec_preview  = "  ".join(preview_vals)
        if dims > PREVIEW_DIMS:
            vec_preview += f"  … (+{dims - PREVIEW_DIMS} chiều)"

        _typical_norm = math.sqrt(dims) * 0.3
        bar_pct = min(int(norm / _typical_norm * 100), 100) if _typical_norm > 0 else 50

        label = (
            f"Chunk {i+1}  |  {len(content):,} ký tự  |  "
            f"{dims}d  |  ‖v‖ = {norm:.4f}"
        )
        with st.expander(label, expanded=(i < 2)):
            st.markdown(
                f"<div style='margin-bottom:6px'>"
                f"<div style='background:#eee;border-radius:4px;height:8px;width:100%'>"
                f"<div style='background:#2980b9;border-radius:4px;height:8px;width:{bar_pct}%'></div>"
                f"</div>"
                f"<span style='font-size:0.75rem;color:#666'>Norm: {norm:.4f}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )
            st.code(f"[{vec_preview}]", language=None)
            st.caption("Nội dung chunk:")
            st.text_area(
                "content",
                value=content[:600] + ("…" if len(content) > 600 else ""),
                height=100,
                key=f"emb_content_{i}",
                disabled=True,
                label_visibility="collapsed",
            )
            if sparse and i < len(sparse):
                sv = sparse[i]
                if sv:
                    top_tokens = sorted(sv.items(), key=lambda x: -x[1])[:10]
                    st.markdown("**Sparse (top-10 tokens):**")
                    token_badges = " &nbsp; ".join(
                        f'<span style="background:#16a085;color:#fff;padding:2px 8px;'
                        f'border-radius:10px;font-size:0.78rem;font-family:monospace">'
                        f'{tok} <b>{score:.2f}</b></span>'
                        for tok, score in top_tokens
                    )
                    st.markdown(token_badges, unsafe_allow_html=True)
                    st.caption(f"Tổng {len(sv):,} token non-zero")

    # ── Tips ─────────────────────────────────────────────────────────────────
    with st.expander("💡 Mẹo chọn embedding model"):
        st.markdown("""
| Tình huống | Gợi ý |
|---|---|
| Tiếng Việt, có GPU | `BAAI/bge-m3` (HuggingFace) hoặc `Cohere embed-multilingual-v3.0` |
| Tiếng Việt, chất lượng cao nhất | `Qwen/Qwen3-Embedding-0.6B` (MTEB #1) |
| Hoàn toàn local, không GPU | `FastEmbed · intfloat/multilingual-e5-small` |
| Local + privacy, có GPU | `Ollama · bge-m3` |
| Budget API thấp | `FastEmbed multilingual-e5-small` (free, CPU) |
| OpenAI ecosystem | `text-embedding-3-small` (default) |
| Hybrid retrieval | Bật **BM25** (nhanh, không GPU) hoặc **SPLADE** (chất lượng cao hơn) |
""")

def main():
    st.set_page_config(
        page_title=PAGE_TITLE,
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # ── Transfer pending suggestion values → widget keys ─────────────────────
    # Phải làm TRƯỚC khi sidebar render để tránh StreamlitAPIException.
    # Apply button ghi vào _want_* keys, đây là bước chuyển sang sel_* keys
    # mà các selectbox trong sidebar dùng.
    _PENDING = [
        ("_want_pdf_strategy",       "sel_pdf_strategy"),
        ("_want_chunking_strategy",  "sel_chunking_strategy"),
        ("_want_format_type",        "sel_format_type"),
        ("_want_emb_provider",       "sel_emb_provider"),
        ("_want_emb_model",          "sel_emb_model"),
        ("_want_emb_sparse",         "emb_enable_sparse"),
        ("_want_emb_sparse_method",  "emb_sparse_method"),
        ("_want_vdb_provider",       "sel_vdb_provider"),
        ("_want_ret_strategy",       "ret_strategy"),
        ("_want_pre_transforms",     "pre_ret_strategies"),
        ("_want_post_reranker",      "post_ret_reranker"),
    ]
    for want_key, widget_key in _PENDING:
        if want_key in st.session_state:
            st.session_state[widget_key] = st.session_state.pop(want_key)

    # Khởi tạo trạng thái pipeline nếu chưa có
    if "_pipeline_running" not in st.session_state:
        st.session_state["_pipeline_running"] = False

    # ── Global CSS overrides ─────────────────────────────────────────────────
    st.markdown(
        """
        <style>
        /* ── Tăng font size toàn app ── */
        html, body, [class*="css"] {
            font-size: 16px !important;
        }
        /* Sidebar text */
        section[data-testid="stSidebar"] * {
            font-size: 15px !important;
        }
        section[data-testid="stSidebar"] label,
        section[data-testid="stSidebar"] .stSelectbox label,
        section[data-testid="stSidebar"] .stRadio label,
        section[data-testid="stSidebar"] .stCheckbox label {
            font-size: 15px !important;
            font-weight: 500 !important;
        }
        /* Selectbox, radio, checkbox labels in main area */
        .stSelectbox label, .stRadio label, .stCheckbox label,
        .stTextInput label, .stNumberInput label {
            font-size: 15px !important;
            font-weight: 500 !important;
        }
        /* Selectbox dropdown value */
        .stSelectbox div[data-baseweb="select"] {
            font-size: 15px !important;
        }
        /* Buttons */
        .stButton button {
            font-size: 15px !important;
        }
        /* Expander header */
        .streamlit-expanderHeader {
            font-size: 15px !important;
            font-weight: 600 !important;
        }
        /* Caption */
        .stCaption, .stCaption p {
            font-size: 13px !important;
        }
        /* Disabled text-area: màu chữ rõ, size lớn hơn */
        textarea[disabled],
        .stTextArea textarea:disabled {
            color: #1a1a1a !important;
            font-size: 15px !important;
            line-height: 1.7 !important;
            opacity: 1 !important;
            -webkit-text-fill-color: #1a1a1a !important;
        }
        /* Markdown content */
        .stMarkdown p, .stMarkdown li, .stMarkdown td, .stMarkdown th {
            font-size: 15px !important;
            line-height: 1.75 !important;
        }
        .stMarkdown h1 { font-size: 1.8rem !important; }
        .stMarkdown h2 { font-size: 1.5rem !important; }
        .stMarkdown h3 { font-size: 1.25rem !important; }
        /* Ảnh trong Markdown mode: full width */
        .stMarkdown img {
            max-width: 100% !important;
            height: auto !important;
            display: block !important;
            margin: 0.5rem auto !important;
        }
        /* Formula placeholder */
        .stMarkdown code {
            background: #fff3cd !important;
            color: #856404 !important;
            padding: 2px 6px !important;
            border-radius: 3px !important;
            font-size: 14px !important;
        }
        /* st.success / st.error / st.info messages */
        .stAlert p { font-size: 15px !important; }
        /* JSON viewer */
        .stJson { font-size: 13px !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("🔬 RAG Pipeline Visualizer")
    st.markdown(
        "Trực quan hóa kết quả từng bước trong RAG pipeline. "
        "Chọn file/thư mục, cài đặt loader & chunker, rồi ấn **Process**."
    )

    # ── Sidebar: chọn nguồn dữ liệu ─────────────────────────────────────────
    with st.sidebar:
        st.header("📁 Nguồn dữ liệu")

        input_method = st.radio(
            "Cách chọn file",
            ["Upload file(s)", "Nhập đường dẫn"],
            help="Upload: kéo thả file trực tiếp\nĐường dẫn: nhập path tuyệt đối"
        )

        source_path: str | None = None

        if input_method == "Upload file(s)":
            uploaded = st.file_uploader(
                "Chọn 1 hoặc nhiều file",
                accept_multiple_files=True,
                type=["pdf", "txt", "md", "docx", "pptx", "xlsx", "csv",
                      "html", "htm", "json", "jsonl", "eml", "epub",
                      "py", "js", "ts", "java", "cpp", "go", "sql"],
            )
            if uploaded:
                source_path = save_uploaded_files(uploaded)
                st.success(f"✅ {len(uploaded)} file(s) đã upload")
        else:
            path_input = st.text_input(
                "Đường dẫn file hoặc thư mục",
                placeholder="/path/to/your/data",
            )
            if path_input:
                p = Path(path_input)
                if p.exists():
                    source_path = str(p)
                    kind = "thư mục" if p.is_dir() else "file"
                    st.success(f"✅ Đã chọn {kind}: `{p.name}`")
                else:
                    st.error("❌ Đường dẫn không tồn tại")

        st.markdown("---")
        local_only = st.checkbox(
            "🏠 Chỉ dùng model / công cụ local & self-hosted",
            value=False,
            help=(
                "Khi bật: các gợi ý chỉ bao gồm công cụ chạy hoàn toàn local "
                "(Marker, pymupdf, pdfplumber, Ollama, …) — không dùng OpenAI "
                "hay bất kỳ API bên ngoài nào."
            ),
        )

        st.markdown("---")

        # ── Loader settings ─────────────────────────────────────────────────
        loader_cfg = render_loader_settings()

        st.markdown("---")

        # ── Chunking settings ────────────────────────────────────────────────
        strategy, chunk_size, chunk_overlap, extra_kwargs = render_chunking_settings(local_only=local_only)

        st.markdown("---")

        # ── Embedding settings ───────────────────────────────────────────────
        _chunk_skipped = st.session_state.get("chunk_skip", False)
        emb_cfg = render_embedding_settings(
            local_only=local_only,
            force_skip=_chunk_skipped,
        )

        st.markdown("---")

        # ── Vector DB settings ───────────────────────────────────────────────
        _emb_skipped = emb_cfg.get("skip", False)
        vdb_cfg = render_vector_db_settings(
            local_only=local_only,
            force_skip=_emb_skipped,
        )

        st.markdown("---")

        # ── Query pipeline settings ───────────────────────────────────────────
        st.subheader("🔎 Query Pipeline")
        st.caption(
            "Cấu hình cho các bước 7–11 (chạy lúc truy vấn, không phải lúc indexing).  \n"
            "**Bước 7, 9 tuỳ chọn** — chỉ bắt buộc cấu hình Bước 8, 10, 11."
        )

        with st.expander("7️⃣ Pre-retrieval — biến đổi query  *(tuỳ chọn)*", expanded=False):
            st.info(
                "**Tuỳ chọn.** Mặc định `none` — query được truyền thẳng vào Retrieval.  \n"
                "Bật khi: query có lỗi chính tả (`rewrite`), muốn tăng recall (`multi_query`), "
                "query quá ngắn/trừu tượng (`hyde`), hoặc corpus đa domain (`route`).",
                icon="ℹ️",
            )
            pre_strategies = ["none", "rewrite", "expand",
                              "step_back", "multi_query", "decompose", "self_query", "route"]
            pre_sel = st.multiselect(
                "Transformations (theo thứ tự áp dụng)",
                options=pre_strategies,
                default=["none"],
                key="pre_ret_strategies",
                help="none = passthrough. Các bước áp dụng tuần tự.",
            )
            pre_llm_provider = st.selectbox(
                "LLM Provider", ["openai", "anthropic"],
                key="pre_ret_llm_provider",
            )
            pre_llm_model = st.text_input(
                "LLM Model",
                value="gpt-4.1-mini" if pre_llm_provider == "openai" else "claude-haiku-4-5-20251001",
                key="pre_ret_llm_model",
            )
            if "multi_query" in pre_sel:
                pre_n_queries = st.slider("Số sub-queries", 2, 6, 3, key="pre_ret_n_queries")
            else:
                pre_n_queries = 3
            st.session_state["query_pre_cfg"] = {
                "transformations":   pre_sel or ["none"],
                "llm_provider":      pre_llm_provider,
                "llm_model":         pre_llm_model,
                "multi_query_count": pre_n_queries,
            }

        with st.expander("8️⃣ Retrieval — chiến lược tìm kiếm", expanded=False):
            ret_strategies = ["hybrid", "dense", "sparse", "multi_query",
                              "parent_document", "sentence_window", "multi_hop", "contextual"]
            ret_strategy = st.selectbox(
                "Strategy",
                ret_strategies,
                key="ret_strategy",
                help="hybrid = dense + BM25 (khuyến nghị mặc định).",
            )
            ret_top_k = st.slider("Top-K kết quả", 3, 30, 10, key="ret_top_k")
            if ret_strategy == "hybrid":
                ret_fusion = st.selectbox(
                    "Fusion method", ["rrf", "weighted", "dbsf"],
                    key="ret_fusion",
                    help="rrf = Reciprocal Rank Fusion (mặc định, robust nhất).",
                )
                ret_alpha = st.slider(
                    "Alpha (dense weight)", 0.0, 1.0, 0.5, 0.05,
                    key="ret_alpha",
                    help="1.0 = pure dense · 0.0 = pure sparse. Chỉ dùng với weighted.",
                )
            else:
                ret_fusion, ret_alpha = "rrf", 0.5
            st.session_state["query_ret_cfg"] = {
                "strategy":      ret_strategy,
                "top_k":         ret_top_k,
                "fusion_method": ret_fusion,
                "hybrid_alpha":  ret_alpha,
            }

        with st.expander("9️⃣ Post-retrieval — reranking & filtering  *(tuỳ chọn)*", expanded=False):
            st.info(
                "**Tuỳ chọn.** Mặc định `none` — kết quả retrieval được dùng trực tiếp.  \n"
                "Bật `cross_encoder` reranker khi cần precision cao hơn (PDF kỹ thuật, bảng phức tạp).  \n"
                "Bật `context ordering` (sandwich) luôn có ích — giảm lost-in-the-middle khi có nhiều chunks.",
                icon="ℹ️",
            )
            reranker_opts = ["none", "cross_encoder", "cohere", "llm"]
            reranker = st.selectbox(
                "Reranker",
                reranker_opts,
                key="post_ret_reranker",
                help="cross_encoder: BAAI/bge-reranker-v2-m3 — tốt nhất cho VI, không cần GPU.\ncohere: API, tốt nhất về chất lượng.\nllm: không cần GPU, chậm hơn.",
            )
            post_top_n = st.slider("Top-N sau reranking", 1, 15, 5, key="post_ret_top_n")
            if reranker == "cross_encoder":
                ce_model = st.text_input(
                    "Cross-encoder model",
                    value="BAAI/bge-reranker-v2-m3",
                    key="post_ret_ce_model",
                )
            else:
                ce_model = "BAAI/bge-reranker-v2-m3"

            col_a, col_b = st.columns(2)
            with col_a:
                apply_redundancy = st.checkbox("Semantic dedup", value=True, key="post_apply_redundancy")
                apply_mmr        = st.checkbox("MMR diversity",  value=False, key="post_apply_mmr")
            with col_b:
                apply_compress   = st.checkbox("Compress context", value=False, key="post_apply_compress")
                apply_llm_filter = st.checkbox("LLM filter",     value=False, key="post_apply_llm_filter")

            ordering = st.selectbox(
                "Context ordering",
                ["sandwich", "relevance", "reverse", "original"],
                key="post_ret_ordering",
                help="sandwich = most relevant first + last (tốt nhất cho lost-in-middle).",
            )
            st.session_state["query_post_cfg"] = {
                "reranker":           reranker,
                "top_n":              post_top_n,
                "cross_encoder_model": ce_model,
                "apply_redundancy":   apply_redundancy,
                "apply_mmr":          apply_mmr,
                "apply_compression":  apply_compress,
                "apply_llm_filter":   apply_llm_filter,
                "context_ordering":   ordering,
            }

        with st.expander("🔟 Prompt — xây dựng prompt  *(chạy lúc query)*", expanded=False):
            st.info(
                "Chọn template prompt để đưa context vào LLM.\n\n"
                "- **citation** ← khuyến nghị: yêu cầu trích dẫn [NGUỒN N]\n"
                "- **basic**: tối giản, grounded chặt\n"
                "- **conversational**: hội thoại nhiều lượt\n"
                "- **structured**: output JSON (claims + sources + confidence)",
                icon="ℹ️",
            )
            prompt_template = st.selectbox(
                "Template",
                ["citation", "basic", "conversational", "structured"],
                key="prompt_template",
                help="citation: yêu cầu [NGUỒN N] inline — tốt nhất cho production.\nbasic: tối giản.\nconversational: multi-turn với lịch sử.\nstructured: JSON output.",
            )
            prompt_language = st.selectbox(
                "Ngôn ngữ prompt",
                ["both", "vi", "en"],
                key="prompt_language",
                help="vi: prompt tiếng Việt · en: prompt tiếng Anh · both: dùng vi",
            )
            prompt_max_ctx = st.number_input(
                "Giới hạn context (ký tự, 0 = không giới hạn)",
                min_value=0, max_value=100_000, value=0, step=1000,
                key="prompt_max_ctx",
                help="Truncate context nếu tổng vượt giới hạn — tránh vượt context window LLM.",
            )
            if prompt_template == "conversational":
                prompt_max_hist = st.slider(
                    "Số lượt lịch sử tối đa", 1, 10, 5, key="prompt_max_hist"
                )
            else:
                prompt_max_hist = 5
            st.session_state["query_prompt_cfg"] = {
                "template":        prompt_template,
                "language":        prompt_language,
                "max_context_chars": int(prompt_max_ctx),
                "max_history_turns": prompt_max_hist,
            }

        with st.expander("1️⃣1️⃣ Generation — LLM sinh câu trả lời  *(chạy lúc query)*", expanded=False):
            st.info(
                "Chọn LLM để sinh câu trả lời cuối cùng từ prompt đã xây dựng.",
                icon="ℹ️",
            )
            # Provider selection
            gen_provider = st.selectbox(
                "Provider",
                ["openai", "anthropic", "google", "ollama", "cohere"],
                key="gen_provider",
                help="openai: GPT-4.1-mini (rẻ, nhanh) · anthropic: Claude · google: Gemini (free tier) · ollama: local · cohere: RAG-optimised",
            )

            # Model presets per provider
            _GEN_MODEL_PRESETS: dict[str, list[str]] = {
                "openai":    ["gpt-4.1-mini", "gpt-4o-mini", "gpt-4o", "o3-mini"],
                "anthropic": ["claude-haiku-4-5-20251001", "claude-sonnet-4-6", "claude-opus-4-6"],
                "google":    ["gemini-2.0-flash", "gemini-2.0-flash-lite", "gemini-1.5-pro"],
                "ollama":    ["qwen2.5:7b", "llama3.2:3b", "mistral:7b", "gemma2:2b", "phi4-mini"],
                "cohere":    ["command-r-plus", "command-r"],
            }
            _GEN_MODEL_NOTES: dict[str, str] = {
                "gpt-4.1-mini":              "⭐ $0.40/$1.60 per 1M — mặc định tốt nhất",
                "gpt-4o-mini":               "$0.15/$0.60 per 1M — nhanh nhất, rẻ nhất",
                "gpt-4o":                    "$2.50/$10 per 1M — chất lượng cao nhất",
                "o3-mini":                   "Reasoning model, câu hỏi phức tạp",
                "claude-haiku-4-5-20251001": "⭐ $0.80/$4 per 1M — rẻ nhất Anthropic",
                "claude-sonnet-4-6":         "$3/$15 per 1M — balanced",
                "claude-opus-4-6":           "$15/$75 per 1M — mạnh nhất",
                "gemini-2.0-flash":          "⭐ Free tier · nhanh · đa ngôn ngữ",
                "gemini-2.0-flash-lite":     "Free tier · rẻ nhất Gemini",
                "gemini-1.5-pro":            "2M context window · mạnh nhất Gemini",
                "qwen2.5:7b":                "⭐ Local · tốt nhất tiếng Việt · ~4.7GB",
                "llama3.2:3b":               "Local · nhẹ nhất · ~2GB · CPU OK",
                "mistral:7b":                "Local · tiếng Anh · ~4.1GB",
                "gemma2:2b":                 "Local · siêu nhẹ · ~1.6GB",
                "phi4-mini":                 "Local · reasoning tốt · ~2.5GB",
                "command-r-plus":            "⭐ RAG-optimised · 128K ctx · $2.5/$10/1M",
                "command-r":                 "$0.15/$0.60/1M — đủ tốt cho Q&A",
            }
            presets = _GEN_MODEL_PRESETS.get(gen_provider, [])
            # Reset model selection khi đổi provider
            if st.session_state.get("_gen_provider_prev") != gen_provider:
                st.session_state["gen_model"] = presets[0] if presets else ""
                st.session_state["_gen_provider_prev"] = gen_provider

            def _gen_model_label(m: str) -> str:
                note = _GEN_MODEL_NOTES.get(m, "")
                return f"{m}  —  {note}" if note else m

            gen_model = st.selectbox(
                "Model",
                presets,
                key="gen_model",
                format_func=_gen_model_label,
            )

            if gen_provider == "ollama":
                gen_ollama_url = st.text_input(
                    "Ollama URL", value="http://localhost:11434",
                    key="gen_ollama_url",
                    help="URL Ollama server. Model tự pull nếu chưa có.",
                )
                gen_auto_pull = st.checkbox("Tự động pull model nếu chưa có", value=True, key="gen_auto_pull")
            else:
                gen_ollama_url = "http://localhost:11434"
                gen_auto_pull  = True

            col_gen1, col_gen2 = st.columns(2)
            with col_gen1:
                gen_temperature = st.slider(
                    "Temperature", 0.0, 1.0, 0.0, 0.05,
                    key="gen_temperature",
                    help="0 = deterministic. Tăng để câu trả lời đa dạng hơn.",
                )
            with col_gen2:
                gen_max_tokens = st.number_input(
                    "Max tokens", min_value=256, max_value=8192, value=2048, step=256,
                    key="gen_max_tokens",
                    help="Số token tối đa trong câu trả lời.",
                )
            gen_streaming = st.checkbox(
                "Streaming (hiển thị từng token)",
                value=True,
                key="gen_streaming",
                help="Bật để xem câu trả lời xuất hiện dần — trải nghiệm tốt hơn với câu trả lời dài.",
            )

            # API key check
            _GEN_ENV_KEYS = {
                "openai":    "OPENAI_API_KEY",
                "anthropic": "ANTHROPIC_API_KEY",
                "google":    "GOOGLE_API_KEY",
                "cohere":    "COHERE_API_KEY",
            }
            _env_key = _GEN_ENV_KEYS.get(gen_provider)
            if _env_key:
                if _get_env(_env_key):
                    st.success(f"✅ `{_env_key}` đã được cấu hình.")
                else:
                    st.warning(f"⚠️ Cần `{_env_key}` trong file `.env` hoặc biến môi trường.")

            st.session_state["query_gen_cfg"] = {
                "provider":    gen_provider,
                "model_name":  gen_model,
                "temperature": gen_temperature,
                "max_tokens":  int(gen_max_tokens),
                "streaming":   gen_streaming,
                "base_url":    gen_ollama_url,
                "auto_pull":   gen_auto_pull,
            }

        st.markdown("---")
        _is_running = st.session_state.get("_pipeline_running", False)
        process_btn = st.button(
            "⏳ Đang xử lý..." if _is_running else "▶️ Process",
            key="btn_process",
            type="primary",
            width="stretch",
            disabled=(source_path is None or _is_running),
        )


        # ── Pipeline Cache management ─────────────────────────────────────────
        st.markdown("---")
        with st.expander("🗄️ Pipeline Cache", expanded=False):
            _pc = _get_pipeline_cache()
            _total_mb = _pc.total_size_mb()
            _entries  = _pc.list_entries()

            col_a, col_b = st.columns(2)
            col_a.metric("Dung lượng", f"{_total_mb:.1f} MB")
            col_b.metric("Inputs cached", len(_entries))

            for _entry in _entries:
                _src   = _entry["source_path"] or _entry["input_short"]
                _label = Path(_src).name if _src else _entry["input_short"]
                with st.expander(f"📁 {_label}  ·  {_entry['total_size_mb']} MB"):
                    st.caption(
                        f"Hash: `{_entry['input_short']}`  ·  {_entry['created_at']}\n\n"
                        f"`{_src}`"
                    )
                    for _step, _step_entries in _entry["steps"].items():
                        for _se in _step_entries:
                            _stats = _se.get("stats", {})
                            _stat_str = "  ·  ".join(
                                f"{k}: {v}" for k, v in _stats.items()
                                if k not in ("source_path",) and v is not None
                            )
                            st.markdown(
                                f"**{_step}** &nbsp;`{_se['key_short']}`&nbsp; "
                                f"{_se['size_mb']} MB · {_se['saved_at']}<br>"
                                f"<small>{_stat_str}</small>",
                                unsafe_allow_html=True,
                            )
                    if st.button("🗑️ Xoá input này",
                                 key=f"del_input_{_entry['input_short']}"):
                        _pc.clear_input(_entry["input_short"])
                        st.success("Đã xoá."); st.rerun()

            st.markdown("")
            _col_c, _col_d = st.columns(2)
            with _col_c:
                if st.button("🗑️ Xoá tất cả", type="secondary", key="clear_pipeline_cache"):
                    _pc.clear_all(); st.success("Đã xoá."); st.rerun()
            with _col_d:
                if st.button("🧹 Prune >30 ngày", type="secondary", key="prune_cache"):
                    _n = _pc.prune_old(30)
                    st.success(f"Đã xoá {_n} input cũ."); st.rerun()

        # ── Image cache management ───────────────────────────────────────────
        st.markdown("---")
        st.markdown("**🗂️ Image Cache**")
        try:
            from loader.pdf_loader import MARKER_CACHE_DIR, DOCLING_CACHE_DIR
            import shutil as _shutil

            # Chỉ hiện cache của loader đang được chọn
            _active_strategy = st.session_state.get("sel_pdf_strategy", "")
            _LOADER_CACHE_MAP = {
                "marker":  ("Marker",  MARKER_CACHE_DIR),
                "docling": ("Docling", DOCLING_CACHE_DIR),
            }
            _relevant = _LOADER_CACHE_MAP.get(_active_strategy)

            if _relevant is None:
                # Loader hiện tại không sinh image cache (pypdf, pdfplumber, v.v.)
                st.caption("_(loader này không sinh image cache)_")
            else:
                _loader_name, _cache_dir = _relevant
                if not _cache_dir.exists():
                    st.caption(f"_(chưa có cache cho {_loader_name})_")
                else:
                    _img_files  = list(_cache_dir.rglob("*.*"))
                    _n_files    = len(_img_files)
                    _total_mb   = sum(f.stat().st_size for f in _img_files if f.is_file()) / (1024**2)
                    _n_pdfs     = len([d for d in _cache_dir.iterdir() if d.is_dir()])
                    try:
                        _rel = _cache_dir.relative_to(Path.cwd())
                    except ValueError:
                        _rel = _cache_dir
                    st.caption(
                        f"📁 **{_loader_name}** `{_rel}`  \n"
                        f"{_n_pdfs} PDF · {_n_files} ảnh · {_total_mb:.1f} MB"
                    )
                    if st.button(
                        f"🗑️ Clear {_loader_name} cache",
                        key=f"btn_clear_{_loader_name.lower()}_img_cache",
                        width="stretch",
                    ):
                        _shutil.rmtree(str(_cache_dir), ignore_errors=True)
                        st.success(f"✅ Đã xóa {_loader_name} image cache.")
                        st.rerun()
        except ImportError:
            pass

    # ── Main area: hướng dẫn khi chưa chọn file ─────────────────────────────
    if source_path is None:
        st.info(
            "👈 **Bắt đầu bằng cách chọn file hoặc nhập đường dẫn ở thanh bên trái**, "
            "sau đó cài đặt Loader và Chunking, rồi ấn **Process**."
        )

        with st.expander("📖 Hướng dẫn sử dụng"):
            st.markdown("""
**Bước 1 — Chọn file / thư mục**
Upload trực tiếp hoặc nhập đường dẫn tuyệt đối.
Hỗ trợ: PDF, DOCX, PPTX, XLSX, CSV, TXT, MD, HTML, JSON, code files, EML, EPUB.

**Bước 2 — Cài đặt Loader** · *chọn cách đọc file*

**Bước 3 — Cài đặt Chunking** · *chọn cách cắt nhỏ văn bản*

**Bước 4 — Cài đặt Embedding** · *chọn model để vector hoá*

**Bước 5 — Cài đặt Vector DB** · *chọn nơi lưu trữ vector*

**Bước 6 — Ấn ▶️ Process** · chạy 5 bước indexing, kết quả hiện ở 4 tab bên dưới.

---
*Query pipeline — cấu hình cho bước truy vấn:*

**Bước 7 — Pre-retrieval** · *biến đổi query trước khi tìm kiếm (tuỳ chọn)*

**Bước 8 — Retrieval** · *chiến lược tìm kiếm trong vector DB*

**Bước 9 — Post-retrieval** · *reranking, filtering, context ordering (tuỳ chọn)*

**Bước 10 — Prompt** · *chọn template xây dựng prompt (citation, basic, conversational, structured)*

**Bước 11 — Generation** · *chọn LLM sinh câu trả lời cuối cùng*
""")

        with st.expander("⚙️ Loader — chọn PDF strategy nào?"):
            st.markdown("""\
**So sánh nhanh 7 strategy:**

| Strategy | Tốc độ | Bảng | Scan/OCR | Output | Best for |
|---|---|---|---|---|---|
| `pypdf` | ⚡⚡⚡ | ❌ | ❌ | Plain text | Prototype nhanh, PDF text thuần |
| `pymupdf` | ⚡⚡⚡ | ⚠️ | ❌ | Plain text | Nhanh nhất, layout tốt hơn pypdf |
| `pdfplumber` | ⚡⚡ | ✅ | ❌ | Text + Markdown table | Bảng rule-based, không cần Java |
| `unstructured` | ⚡ | ✅ | ✅ | Text / HTML | OCR + hình ảnh, cần hi_res mode |
| `docling` | ⚡ | ✅ | ✅ | **Markdown** | IBM parser, cấu trúc heading tốt |
| `marker` | 🐢 | ✅ | ✅ | **Markdown** | LaTeX, caption ảnh, tốt nhất về chất lượng |
| `opendataloader` | ⚡⚡ | ✅✅ | ✅ | **Markdown** | **#1 benchmark** (0.90), cần Java 11+ |

**Chọn theo mục tiêu:**

📄 **PDF text thuần, cần nhanh** → `pypdf` hoặc `pymupdf` — không cần dep extra, chạy ngay

📊 **PDF có nhiều bảng** → `pdfplumber` (rule-based, nhanh) hoặc `opendataloader` (AI-powered, chính xác hơn)

📐 **PDF kỹ thuật: công thức, LaTeX, ảnh** → `marker` — output Markdown chuẩn nhất, hỗ trợ describe ảnh qua VLM

🌐 **Cần Markdown cấu trúc tốt, đa ngôn ngữ** → `docling` (IBM) hoặc `opendataloader` (hybrid mode)

🖨️ **PDF scan (ảnh chụp giấy)** → `unstructured` — cần bật OCR nội bộ (hi_res strategy)

🏆 **Muốn accuracy cao nhất, có Java 11+** → `opendataloader [hybrid]` — overall 0.90, table 0.93

**Output dạng Markdown quan trọng không?**
Nếu dùng chunking `format_aware` → **bắt buộc** chọn loader cho output Markdown (`docling`, `marker`, `opendataloader`).
Nếu dùng `recursive` hoặc `token_based` → text thuần từ `pypdf`/`pymupdf` là đủ.
""")

        with st.expander("✂️ Chunking — chiến lược nào phù hợp với tôi?"):
            st.markdown("""\
**So sánh nhanh các strategy:**

| Strategy | Ranh giới cắt | Cần LLM? | Best for |
|---|---|---|---|
| `recursive` ⭐ | Đoạn → dòng → câu → ký tự | ❌ | Mặc định tốt nhất cho mọi loại PDF |
| `token_based` | Token (BPE/tiktoken) | ❌ | Tiếng Việt, đa ngôn ngữ, kiểm soát token budget |
| `format_aware` | Heading / code block / HTML tag | ❌ | PDF → Markdown (marker/docling/opendataloader) |
| `sentence_aware` | Cuối câu (dấu chấm) | ❌ | FAQ, Q&A, văn bản câu ngắn |
| `semantic` | Cosine similarity giữa câu | ❌ | Văn bản nhiều chủ đề, chủ đề chuyển tiếp tự nhiên |
| `late` | Token embedding tự học ranh giới | ❌ | Nhiều coreference, đại từ, ngữ cảnh phức tạp |
| `hierarchical` | Parent (section) + child (chunk nhỏ) | ❌ | Tài liệu dài có cấu trúc rõ ràng |
| `hybrid` | Kết hợp 2 strategy bất kỳ | ❌ | Cần flexibility cao |
| `contextual` | Recursive + LLM thêm context prefix | ✅ | Khi chunk thiếu ngữ cảnh, có API budget |

**Chọn theo loại tài liệu:**

📄 **PDF text thông thường** → `recursive` — cắt thứ tự ưu tiên đoạn → dòng → câu, ổn định nhất

📐 **PDF → Markdown (marker / docling / opendataloader)** → `format_aware` — tôn trọng heading H1/H2/H3, chunk khớp cấu trúc gốc

🌐 **Tiếng Việt hoặc đa ngôn ngữ** → `token_based` — đếm token thực tế, tránh cắt giữa từ ghép

📊 **Tài liệu dài, nhiều section** → `hierarchical` — parent chunk giữ ngữ cảnh rộng, child chunk để retrieve chính xác

🧠 **Cần chất lượng cao nhất, có LLM budget** → `contextual` (thêm prefix ngữ cảnh cho mỗi chunk)

**Chunk size gợi ý:**

| Use case | Chunk size | Overlap |
|---|---|---|
| Q&A ngắn, FAQ | 200–400 chars | 50 |
| Tài liệu phổ thông | 400–600 chars ⭐ | 100 |
| Tài liệu kỹ thuật dài | 800–1500 chars | 150–200 |
| `semantic` | tự động | — |
""")

        with st.expander("🧮 Embedding — model và provider nào phù hợp với tôi?"):
            st.markdown("""\
**Chọn theo use case:**

| Tình huống | Provider | Model |
|---|---|---|
| Tiếng Việt, API, chất lượng cao | Cohere | `embed-multilingual-v3.0` ⭐ |
| Tiếng Việt, local, có GPU | HuggingFace | `BAAI/bge-m3` hoặc `Qwen3-Embedding` |
| Tiếng Việt, local, không GPU | FastEmbed | `intfloat/multilingual-e5-small` |
| Tiếng Việt, local + private | Ollama | `bge-m3` |
| Source code search | OpenAI | `text-embedding-3-small` |
| Ecosystem OpenAI, đơn giản | OpenAI | `text-embedding-3-small` |
| Budget rất thấp | FastEmbed | `intfloat/multilingual-e5-small` (free, CPU) |

**Dense hay Hybrid (Dense + Sparse)?**

| Corpus | Khuyến nghị |
|---|---|
| Văn bản thông thường | Dense đủ dùng |
| Có tên riêng, mã số, số phiên bản | Bật **BM25** |
| Source code (tên hàm, class, API) | Bật **BM25** bắt buộc |
| Cần recall cao nhất, có GPU | Bật **SPLADE** |

**BM25 vs SPLADE:** BM25 = từ khoá chính xác, không cần GPU, nhanh.
SPLADE = mở rộng từ vựng ngữ nghĩa (tự thêm từ liên quan), cần GPU.
""")

        with st.expander("🗃️ Vector DB — chọn loại nào?"):
            st.markdown("""\
**So sánh nhanh theo tiêu chí quan trọng:**

| Provider | Tier | Mode | Scale | Hybrid | Filtering | Best for |
|---|---|---|---|---|---|---|
| 💾 FAISS | 🆓 Free | Local | < 10M | ❌ | Post | Prototype, offline |
| 🎨 Chroma | 🆓 Free | Local | < 10M | ❌ | ✅ Native | Dev, demo, Python |
| 🏹 LanceDB | 🆓 Free | Local/Cloud | ~ 1B | ❌ | ✅ Native | Python workflow, serverless |
| 🐘 pgvector | 🆓 Free | Self-host | < 50M | ❌ | ✅ SQL | Đã có PostgreSQL |
| 🎯 Qdrant | 🆓/💳 | Self/Cloud | 1B+ | ✅ | ✅ ACORN | Filtering phức tạp |
| 🕸️ Weaviate | 🆓/💳 | Self/Cloud | ~ 1B | ✅ | ✅ | Hybrid out-of-the-box |
| 🌲 Pinecone | 💳 Paid | Managed | ~ 1B | ✅ | ✅ | Zero ops, startup |


**Chọn theo ngân sách & tình huống:**

🆓 **Miễn phí, không server** → **FAISS** (đơn giản nhất) hoặc **Chroma** (có filtering)

🆓 **Đã có PostgreSQL** → **pgvector** — ACID + SQL filtering, zero infra mới

🎯 **Cần metadata filtering phức tạp** → **Qdrant** (ACORN algorithm tốt nhất)

🕸️ **Cần hybrid search out-of-the-box** → **Weaviate** (BM25 + dense không cần config thêm)

🚀 **Zero ops, startup, nhanh chạy** → **Pinecone** (managed) hoặc **Chroma** (local dev)

🏹 **Python workflow, serverless, columnar** → **LanceDB** (tích hợp pandas/DuckDB tốt)

**Hybrid search quan trọng không?**
Nếu corpus có tên riêng, mã sản phẩm, số phiên bản → nên chọn Qdrant hoặc Weaviate.
Nếu corpus thuần text ngữ nghĩa → Dense-only (FAISS, Chroma, LanceDB) thường đủ.
""")

        with st.expander("🔄 Pre-retrieval — khi nào nên dùng?"):
            st.markdown("""\
**Các transformation và khi nào dùng:**

| Transformation | Khi nào dùng | LLM? |
|---|---|---|
| `none` | Query rõ ràng, không cần biến đổi | ❌ |
| `rewrite` | Query có lỗi chính tả, đại từ mơ hồ, văn nói | ✅ |
| `expand` | Vocabulary mismatch (đái tháo đường ↔ tiểu đường) | ✅/❌ |
| `step_back` | Câu hỏi cụ thể cần background context | ✅ |
| `multi_query` | Câu hỏi phức tạp, muốn tăng recall | ✅ |
| `decompose` | Câu hỏi ghép nhiều vế độc lập | ✅ |
| `self_query` | Query chứa filter ẩn ("tài liệu 2024 về...") | ✅ |
| `route` | KB nhiều domain, cần định tuyến | ✅/❌ |

**Combo phổ biến nhất:**
- `rewrite` → sạch query trước
- `rewrite` + `multi_query` → sạch rồi tạo biến thể (nhờ chain fix, multi_query nhận query đã rewrite)
- `self_query` + `multi_query` → filter metadata + tăng recall
- `hyde` → khi embedding model không nhận dạng được query ngắn
""")

        with st.expander("🔎 Retrieval — chiến lược nào?"):
            st.markdown("""\
**So sánh nhanh:**

| Strategy | Dùng khi | Cần corpus? |
|---|---|---|
| `hybrid` ⭐ | Default tốt nhất — dense + BM25 | ✅ |
| `dense` | Query ngữ nghĩa thuần, không cần exact match | ❌ |
| `sparse` | Tên riêng, mã số, jargon kỹ thuật | ✅ |
| `multi_query` | Pre-retrieval sinh nhiều queries | ❌ |
| `parent_document` | Dùng HierarchicalChunker | ✅ |
| `sentence_window` | Dùng SentenceAwareChunker | ✅ |
| `multi_hop` | Câu hỏi multi-step, cần chain reasoning | ❌ |
| `contextual` | Dense + score filter + MMR diversity | ❌ |

**Fusion method (hybrid):**
- `rrf` — Reciprocal Rank Fusion. Mặc định, robust nhất, không phụ thuộc score scale.
- `weighted` — Alpha-blend normalized scores. Cần tuning alpha.
- `dbsf` — Z-score normalized. Robust hơn weighted với outlier scores.
""")

        with st.expander("⚙️ Post-retrieval — reranking & filtering"):
            st.markdown("""\
**Reranker:**

| Reranker | Khi nào dùng | GPU? |
|---|---|---|
| `none` | Retrieval đã đủ tốt, muốn latency thấp | ❌ |
| `cross_encoder` ⭐ | Default tốt nhất. BAAI/bge-reranker-v2-m3 = best VI | ❌ |
| `cohere` | Best API quality, 100+ ngôn ngữ | ❌ |
| `llm` | Không có GPU, top_n ≤ 10 | ❌ |

**Pipeline order (cố định):**
1. MetadataFilter → 2. RedundancyFilter → 3. Reranker → 4. LLMFilter → 5. MMRFilter → 6. Compressor → 7. Orderer

**Context ordering:**
- `sandwich` ⭐ — Most relevant đầu và cuối, ít relevant giữa. Giảm lost-in-middle tốt nhất.
- `relevance` — Theo thứ tự relevance giảm dần.
- `reverse` — Theo relevance tăng dần (most relevant cuối).

**Khi nào bật thêm:**
- **Semantic dedup** → luôn bật khi dùng multi_query hoặc hybrid
- **MMR diversity** → khi nhiều chunk giống nhau về nội dung
- **Compress context** → khi chunk lớn (>1000 chars) và chỉ một phần liên quan
- **LLM filter** → khi retrieval trả về nhiều kết quả off-topic
""")

        with st.expander("📝 Prompt — template nào phù hợp với tôi?"):
            st.markdown("""\
**So sánh nhanh 4 template:**

| Template | Dùng khi | Output |
|---|---|---|
| `citation` ⭐ | Mặc định tốt nhất — cần truy xuất nguồn gốc | Text + [NGUỒN N] |
| `basic` | Prototype nhanh, grounding chặt, ít overhead | Plain text |
| `conversational` | Chatbot, follow-up question nhiều lượt | Text + lịch sử |
| `structured` | Code cần parse câu trả lời, audit trail | JSON |

**Chọn theo mục tiêu:**

📌 **Production / cần verify fact** → `citation` — LLM bắt buộc trích dẫn `[NGUỒN N]` cho mỗi claim, app hiển thị đúng đoạn nguồn tương ứng.

🚀 **Prototype nhanh** → `basic` — Prompt tối giản, overhead thấp nhất, grounded chặt chẽ.

💬 **Chatbot / hỏi đáp nhiều lượt** → `conversational` — Inject lịch sử hội thoại để LLM resolve follow-up ("cái đó" → refers to what?). Nên bật với Bước 11 streaming.

🔧 **API / downstream parsing** → `structured` — LLM trả JSON: `answer`, `claims[]`, `sources[]`, `confidence`. App tự parse và hiển thị từng claim riêng.

**Ngôn ngữ prompt:**
- `vi` — Toàn bộ instruction bằng tiếng Việt → LLM thiên về trả lời tiếng Việt.
- `en` — Instruction tiếng Anh → tốt hơn với corpus tiếng Anh thuần.
- `both` — Dùng instruction tiếng Việt (mặc định an toàn cho corpus song ngữ).

**Giới hạn context (max_context_chars):**
- `0` = không giới hạn (mặc định).
- Đặt giới hạn khi dùng model có context window nhỏ (ví dụ Ollama CPU model) để tránh lỗi vượt token limit.
- Gợi ý: `llama3.2:3b` → 4000, `qwen2.5:7b` → 12000, các API model → thường không cần giới hạn.
""")

        with st.expander("🤖 Generation — LLM nào phù hợp với tôi?"):
            st.markdown("""\
**So sánh theo tiêu chí quan trọng:**

| Provider | Model | Giá (input/output /1M) | Tiếng Việt | Best for |
|---|---|---|---|---|
| OpenAI | `gpt-4.1-mini` ⭐ | $0.40 / $1.60 | ⭐⭐⭐ | Mặc định tốt nhất: rẻ + nhanh + chất lượng cao |
| OpenAI | `gpt-4o-mini` | $0.15 / $0.60 | ⭐⭐⭐ | Rẻ nhất API, đủ tốt cho Q&A đơn giản |
| OpenAI | `gpt-4o` | $2.50 / $10 | ⭐⭐⭐ | Chất lượng cao nhất OpenAI |
| Anthropic | `claude-haiku-4-5` ⭐ | $0.80 / $4 | ⭐⭐⭐⭐ | Rẻ nhất Anthropic, nhanh, tiếng Việt tốt |
| Anthropic | `claude-sonnet-4-6` | $3 / $15 | ⭐⭐⭐⭐ | Cân bằng chất lượng / chi phí |
| Google | `gemini-2.0-flash` ⭐ | Free tier | ⭐⭐⭐⭐ | Miễn phí (có giới hạn), đa ngôn ngữ tốt |
| Ollama | `qwen2.5:7b` ⭐ | Miễn phí | ⭐⭐⭐⭐ | Tốt nhất local: tiếng Việt OK, ~4.7 GB |
| Ollama | `llama3.2:3b` | Miễn phí | ⭐⭐ | Nhẹ nhất (~2 GB), chạy tốt trên CPU |
| Cohere | `command-r-plus` | $2.5 / $10 | ⭐⭐⭐ | RAG-optimised, context 128K |

**Chọn theo tình huống:**

💰 **Tiết kiệm nhất (API)** → `gpt-4o-mini` hoặc `claude-haiku-4-5` — dưới $1/1M tokens.

🆓 **Miễn phí hoàn toàn** → `gemini-2.0-flash` (Google free tier) hoặc Ollama local.

🇻🇳 **Tiếng Việt chất lượng cao** → `claude-haiku-4-5` hoặc `gemini-2.0-flash` (API); `qwen2.5:7b` (local).

🔒 **Privacy / offline** → Ollama với `qwen2.5:7b` (tốt nhất) hoặc `llama3.2:3b` (nhẹ nhất).

**Temperature:**
- `0.0` ⭐ — Deterministic, câu trả lời nhất quán. Khuyến nghị cho RAG.
- `0.1–0.3` — Đa dạng hơn một chút, vẫn grounded.
- `> 0.5` — Sáng tạo nhưng dễ hallucinate — không nên dùng với RAG.

**Streaming:**
- Bật ⭐ — Hiển thị câu trả lời từng token, trải nghiệm tốt hơn với câu trả lời dài.
- Tắt — Chờ response hoàn chỉnh rồi mới hiển thị (dùng khi cần đo thời gian chính xác).
""")

        return

    # ── Gợi ý cấu hình khi chưa có kết quả ─────────────────────────────────
    if "loader_docs" not in st.session_state:
        suggestions = get_pipeline_suggestions(source_path, local_only)
        render_pipeline_suggestions(suggestions)

    # ── Chạy pipeline khi ấn Process ─────────────────────────────────────────
    if process_btn:
        st.session_state["_pipeline_running"] = True
        import threading, concurrent.futures
        import time as _time

        cache = _get_pipeline_cache()

        # ── Tính toán keys cho toàn bộ chain ──────────────────────────────────
        with st.spinner("🔍 Đang kiểm tra cache..."):
            input_hash  = cache.compute_input_hash(source_path)
            loader_cfg_for_cache = {k: v for k, v in loader_cfg.items()
                                    if k not in ("ollama_base_url",)}  # bỏ URL khỏi key
            # Chunking: loại ollama_base_url khỏi extra_kwargs trước khi hash
            # (URL không ảnh hưởng đến kết quả chunking, chỉ là địa chỉ server)
            _CHUNK_URL_KEYS = {"ollama_base_url"}
            chunk_cfg_for_cache = {
                "strategy":      strategy,
                "chunk_size":    chunk_size,
                "chunk_overlap": chunk_overlap,
                **{k: v for k, v in extra_kwargs.items() if k not in _CHUNK_URL_KEYS},
            }
            # Các key ảnh hưởng đến GIÁ TRỊ vector → phải nằm trong cache key
            # Các key chỉ ảnh hưởng tốc độ/memory → KHÔNG nằm trong cache key
            _EMBED_SPEED_PARAMS = {
                "skip",            # meta flag
                "dims",            # raw model dim, không phải target dim
                "max_preview",     # số chunk embed trong preview, không ảnh hưởng vector
                "ollama_base_url", # URL server, không ảnh hưởng model output
                "device",          # cuda vs cpu → cùng vector, chỉ khác tốc độ
                "torch_dtype_str", # fp16 vs fp32 → semantically same vectors
                "batch_size",      # throughput param, zero effect on output
            }
            embed_cfg_for_cache = {
                k: v for k, v in emb_cfg.items()
                if k not in _EMBED_SPEED_PARAMS
            }

            # Vector DB: chỉ các param ảnh hưởng đến *cấu trúc dữ liệu* trong DB
            # (loại bỏ URL/connection details vì chúng là "where" chứ không phải "what")
            _VDB_INFRA_PARAMS = {
                "skip",             # meta flag
                "force_reindex",    # runtime flag
                # connection details — thay đổi URL không đổi data
                "url", "uri", "redis_url", "connection_string",
                "endpoint", "api_key", "user", "password", "token",
            }
            vdb_cfg_for_cache = {
                k: v for k, v in vdb_cfg.items()
                if k not in _VDB_INFRA_PARAMS
            }

            loader_key = cache.make_step_key(input_hash,  loader_cfg_for_cache)
            chunk_key  = cache.make_step_key(loader_key,  chunk_cfg_for_cache)
            embed_key  = cache.make_step_key(chunk_key,   embed_cfg_for_cache)
            # vdb_key chains từ embed_key → nếu embedding thay đổi, vdb cache tự động miss
            vdb_key    = cache.make_step_key(embed_key,   vdb_cfg_for_cache)

        stop_event = threading.Event()
        st.session_state["_stop_event"] = stop_event

        _skip_embed = emb_cfg.get("skip")
        _skip_vdb   = vdb_cfg.get("skip")
        _n_steps = 2 if _skip_embed else (3 if _skip_vdb else 4)

        st.markdown("#### ⏳ Đang xử lý...")

        # ── Bước 1: Loading ───────────────────────────────────────────────────
        cached_docs = cache.load_loader(input_hash, loader_key)
        if cached_docs is not None:
            docs = cached_docs
            st.session_state["loader_docs"] = docs
            st.success(
                f"⚡ Bước 1/{_n_steps} — Loading từ cache "
                f"({len(docs)} docs · {cache._step_dir(input_hash, 'loader', loader_key).name})"
            )
        else:
            prog_load = st.progress(0, text=f"Bước 1/{_n_steps} — Đang đọc tài liệu...")
            stop_load_placeholder = st.empty()

            def _do_load():
                return run_loader(
                    source_path=source_path,
                    pdf_strategy=loader_cfg["pdf_strategy"],
                    extract_tables=loader_cfg["extract_tables"],
                    language=loader_cfg["language"],
                    marker_device=loader_cfg["marker_device"],
                    describe_images=loader_cfg["describe_images"],
                    vision_model=loader_cfg["vision_model"],
                    vision_provider=loader_cfg["vision_provider"],
                    ollama_base_url=loader_cfg["ollama_base_url"],
                    odl_hybrid=loader_cfg.get("odl_hybrid"),
                    odl_struct_tree=loader_cfg.get("odl_struct_tree", False),
                )

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(_do_load)
                with stop_load_placeholder:
                    if st.button("🛑 Stop", key="stop_loading", type="secondary"):
                        stop_event.set(); future.cancel()
                        prog_load.empty()
                        st.session_state["_pipeline_running"] = False
                        st.warning("⏹️ Đã dừng.")
                        st.stop()
                tick = 0
                while not future.done():
                    prog_load.progress(
                        min(0.05 + tick * 0.007, 0.88),
                        text=f"Bước 1/{_n_steps} — Đang đọc tài liệu... ({tick}s)"
                    )
                    _time.sleep(1); tick += 1
                try:
                    docs = future.result()
                except Exception as e:
                    prog_load.empty()
                    if stop_event.is_set():
                        st.warning("⏹️ Đã dừng."); st.stop()
                    st.error(f"❌ Lỗi khi loading: {e}")
                    st.exception(e); st.stop()

            stop_load_placeholder.empty()
            prog_load.progress(1.0, text=f"✅ Bước 1/{_n_steps} — Loading hoàn tất!")
            cache.save_loader(input_hash, loader_key, docs, loader_cfg_for_cache, source_path)
            st.session_state["loader_docs"] = docs
            st.success(f"✅ Loading thành công: {len(docs)} document(s)")

        # Hiển thị lỗi VLM nếu có
        if loader_cfg.get("describe_images"):
            vlm_errors: list[str] = []
            for doc in docs:
                vlm_errors.extend(doc.metadata.get("_vlm_errors", []))
            if st.session_state.get("_vlm_errors"):
                vlm_errors.extend(st.session_state.pop("_vlm_errors", []))
            if vlm_errors:
                with st.expander(f"⚠️ VLM gặp lỗi với {len(vlm_errors)} ảnh", expanded=True):
                    for err in vlm_errors[:10]:
                        st.error(f"• {err}")
                    if len(vlm_errors) > 10:
                        st.caption(f"... và {len(vlm_errors)-10} lỗi khác")

        # ── Bước 2: Chunking ──────────────────────────────────────────────────
        if docs and not stop_event.is_set():
            cached_chunks = cache.load_chunking(input_hash, chunk_key)
            if cached_chunks is not None:
                chunks = cached_chunks
                st.session_state["chunks"] = chunks
                st.session_state["active_tab"] = 1
                st.success(
                    f"⚡ Bước 2/{_n_steps} — Chunking từ cache "
                    f"({len(chunks)} chunks · {cache._step_dir(input_hash, 'chunking', chunk_key).name})"
                )
            else:
                prog_chunk = st.progress(0, text=f"Bước 2/{_n_steps} — Đang chunking...")
                stop_chunk_placeholder = st.empty()

                def _do_chunk():
                    return run_chunker(
                        docs=docs,
                        strategy=strategy,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                        extra_kwargs=extra_kwargs,
                    )

                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(_do_chunk)
                    with stop_chunk_placeholder:
                        if st.button("🛑 Stop", key="stop_chunking", type="secondary"):
                            stop_event.set(); future.cancel()
                            prog_chunk.empty()
                            st.session_state["_pipeline_running"] = False
                            st.warning("⏹️ Đã dừng sau bước Loading."); st.stop()
                    tick = 0
                    while not future.done():
                        prog_chunk.progress(
                            min(0.05 + tick * 0.03, 0.90),
                            text=f"Bước 2/{_n_steps} — Đang chunking... ({tick}s)"
                        )
                        _time.sleep(0.5); tick += 1
                    try:
                        chunks = future.result()
                    except Exception as e:
                        prog_chunk.empty()
                        st.error(f"❌ Lỗi khi chunking: {e}")
                        st.exception(e); st.stop()

                stop_chunk_placeholder.empty()
                prog_chunk.progress(1.0, text=f"✅ Bước 2/{_n_steps} — Chunking hoàn tất!")
                cache.save_chunking(input_hash, chunk_key, chunks, chunk_cfg_for_cache)
                st.session_state["chunks"] = chunks
                st.session_state["active_tab"] = 1
                st.success(f"✅ Chunking thành công: {len(chunks)} chunk(s)")

        # ── Bước 3: Embedding ─────────────────────────────────────────────────
        if (
            "chunks" in st.session_state
            and not stop_event.is_set()
            and not _skip_embed
        ):
            chunks_for_embed = st.session_state["chunks"]

            cached_embed = cache.load_embedding(input_hash, embed_key)
            if cached_embed is not None:
                st.session_state["embed_result"] = cached_embed
                st.session_state["emb_cfg_used"] = emb_cfg
                st.session_state["active_tab"] = 2
                st.success(
                    f"⚡ Bước 3/3 — Embedding từ cache "
                    f"({cached_embed['n_embedded']} vectors · {cached_embed['dims']}d · "
                    f"{cache._step_dir(input_hash, 'embedding', embed_key).name})"
                )
            else:
                prog_embed = st.progress(0, text="Bước 3/3 — Đang embedding...")
                stop_embed_placeholder = st.empty()

                def _do_embed():
                    return run_embedder(
                        chunks=chunks_for_embed,
                        provider=emb_cfg["provider"],
                        model_name=emb_cfg["model_name"],
                        enable_sparse=emb_cfg["enable_sparse"],
                        sparse_method=emb_cfg["sparse_method"],
                        dimensions=emb_cfg.get("dimensions"),
                        device=emb_cfg.get("device", "cpu"),
                        ollama_base_url=emb_cfg.get("ollama_base_url", "http://localhost:11434"),
                        input_type=emb_cfg.get("input_type", "search_document"),
                        query_instruction=emb_cfg.get("query_instruction"),
                        document_instruction=emb_cfg.get("document_instruction"),
                        max_chunks=emb_cfg.get("max_preview", EMBED_PREVIEW_LIMIT),
                        torch_dtype_str=emb_cfg.get("torch_dtype_str", "auto"),
                        batch_size=emb_cfg.get("batch_size", 32),
                    )

                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(_do_embed)
                    with stop_embed_placeholder:
                        if st.button("🛑 Stop", key="stop_embedding", type="secondary"):
                            stop_event.set(); future.cancel()
                            prog_embed.empty()
                            st.session_state["_pipeline_running"] = False
                            st.warning("⏹️ Đã dừng sau bước Chunking."); st.stop()
                    tick = 0
                    while not future.done():
                        prog_embed.progress(
                            min(0.05 + tick * 0.015, 0.90),
                            text=f"Bước 3/3 — Đang embedding... ({tick}s)"
                        )
                        _time.sleep(0.5); tick += 1
                    try:
                        embed_result = future.result()
                    except Exception as e:
                        prog_embed.empty()
                        st.error(f"❌ Lỗi khi embedding: {e}")
                        st.exception(e); st.stop()

                stop_embed_placeholder.empty()
                prog_embed.progress(1.0, text=f"✅ Bước 3/{_n_steps} — Embedding hoàn tất!")
                cache.save_embedding(input_hash, embed_key, embed_result, embed_cfg_for_cache)
                st.session_state["embed_result"] = embed_result
                st.session_state["emb_cfg_used"] = emb_cfg
                st.session_state["active_tab"] = 2
                st.success(
                    f"✅ Embedding thành công: {embed_result['n_embedded']} vectors · "
                    f"{embed_result['dims']}d"
                )

        # ── Bước 4: Vector DB ─────────────────────────────────────────────────
        if (
            "embed_result" in st.session_state
            and not stop_event.is_set()
            and not _skip_vdb
            and not _skip_embed
        ):
            chunks_for_vdb = st.session_state["chunks"]

            # ── Helper: build embedder kwargs an toàn cho mọi provider ────────
            # Mỗi provider của get_embedder chỉ nhận đúng các kwargs riêng của nó.
            # Truyền ollama_base_url/device/input_type cho OpenAI → LangChain dump
            # vào model_kwargs → API reject "unexpected keyword argument".
            # Logic này mirror chính xác phần extra-building trong run_embedder().
            def _embedder_kwargs() -> dict:
                _p  = emb_cfg["provider"]
                _mn = emb_cfg["model_name"]
                _ex: dict = {}
                if _p == "openai":
                    _dims = emb_cfg.get("dimensions")
                    if _dims:
                        _ex["dimensions"] = _dims
                elif _p == "cohere":
                    _ex["input_type"] = emb_cfg.get("input_type", "search_document")
                elif _p == "ollama":
                    _ex["base_url"] = emb_cfg.get("ollama_base_url", "http://localhost:11434")
                elif _p == "huggingface":
                    _ex["device"] = emb_cfg.get("device", "cpu")
                # fastembed: không cần extra kwargs
                return {"provider": _p, "model_name": _mn, **_ex}

            # ── Kiểm tra pipeline cache ────────────────────────────────────
            cached_vdb = cache.load_vector_db(input_hash, vdb_key)
            if cached_vdb is not None and not vdb_cfg.get("force_reindex"):
                # Cache hit: reconnect to existing DB (không embed lại)
                with st.spinner("🔍 VDB cache hit — đang kết nối lại..."):
                    try:
                        from vector_db import get_vector_store
                        from embedding.factory import get_embedder
                        _embedder = get_embedder(**_embedder_kwargs())
                        _provider = vdb_cfg["provider"]
                        _conn_kwargs = {k: v for k, v in vdb_cfg.items()
                                        if k not in ("provider", "skip", "force_reindex")}
                        # force_reindex=False → provider tự load collection đã có
                        _store = get_vector_store(
                            provider=_provider,
                            chunks=chunks_for_vdb,      # cần để FAISS/Chroma kiểm tra fingerprint
                            embedder=_embedder,
                            force_reindex=False,
                            **_conn_kwargs,
                        )
                        vdb_result = {
                            "store":                _store,
                            "n_vectors":            cached_vdb.get("n_vectors", len(chunks_for_vdb)),
                            "collection_name":      cached_vdb.get("collection_name", vdb_cfg.get("collection_name", "rag")),
                            "loaded_from_existing": True,
                        }
                    except Exception as _e:
                        st.warning(f"⚠️ Không thể kết nối lại VDB cache: {_e}. Sẽ index lại.")
                        cached_vdb = None   # fall through to re-index below

                if cached_vdb is not None:
                    st.session_state["vdb_result"]   = vdb_result
                    st.session_state["vdb_cfg_used"] = vdb_cfg
                    st.session_state["active_tab"]   = 3
                    st.success(
                        f"⚡ Bước 4/4 — VDB từ cache "
                        f"({vdb_result['n_vectors']:,} vectors · "
                        f"{vdb_cfg['provider']}:{vdb_cfg.get('collection_name','rag')} · "
                        f"{cache._step_dir(input_hash, 'vector_db', vdb_key).name})"
                    )

            # ── Cache miss hoặc force_reindex: index từ đầu ───────────────
            if cached_vdb is None or vdb_cfg.get("force_reindex"):
                prog_vdb = st.progress(0, text="Bước 4/4 — Đang index vào Vector DB...")
                stop_vdb_placeholder = st.empty()

                def _do_vdb():
                    from vector_db import get_vector_store
                    from embedding.factory import get_embedder
                    _embedder = get_embedder(**_embedder_kwargs())
                    _provider     = vdb_cfg["provider"]
                    _force        = vdb_cfg.get("force_reindex", False)
                    _conn_kwargs  = {k: v for k, v in vdb_cfg.items()
                                     if k not in ("provider", "skip", "force_reindex")}
                    _store = get_vector_store(
                        provider=_provider,
                        chunks=chunks_for_vdb,
                        embedder=_embedder,
                        force_reindex=_force,
                        **_conn_kwargs,
                    )
                    return {
                        "store":                _store,
                        "n_vectors":            len(chunks_for_vdb),
                        "collection_name":      vdb_cfg.get("collection_name", "rag"),
                        "provider":             _provider,
                        "loaded_from_existing": False,
                    }

                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(_do_vdb)
                    with stop_vdb_placeholder:
                        if st.button("🛑 Stop", key="stop_vdb", type="secondary"):
                            stop_event.set(); future.cancel()
                            prog_vdb.empty()
                            st.session_state["_pipeline_running"] = False
                            st.warning("⏹️ Đã dừng sau bước Embedding."); st.stop()
                    tick = 0
                    while not future.done():
                        prog_vdb.progress(
                            min(0.05 + tick * 0.02, 0.90),
                            text=f"Bước 4/4 — Đang index vào Vector DB... ({tick}s)"
                        )
                        _time.sleep(0.5); tick += 1
                    try:
                        vdb_result = future.result()
                    except Exception as e:
                        prog_vdb.empty()
                        st.error(f"❌ Lỗi khi index Vector DB: {e}")
                        st.exception(e); st.stop()

                stop_vdb_placeholder.empty()
                prog_vdb.progress(1.0, text="✅ Bước 4/4 — Vector DB index hoàn tất!")

                # Lưu metadata vào pipeline cache
                cache.save_vector_db(
                    input_hash,
                    vdb_key,
                    {
                        "n_vectors":       vdb_result["n_vectors"],
                        "collection_name": vdb_result["collection_name"],
                        "provider":        vdb_result["provider"],
                    },
                    vdb_cfg_for_cache,
                )

                st.session_state["vdb_result"]   = vdb_result
                st.session_state["vdb_cfg_used"] = vdb_cfg
                st.session_state["active_tab"]   = 3
                st.success(
                    f"✅ Vector DB: {vdb_result['n_vectors']:,} vectors → "
                    f"{vdb_cfg['provider']}:{vdb_cfg.get('collection_name','rag')}"
                )

        # Rerun sạch để render results — tránh conflict giữa process widgets
        # và results widgets trong cùng 1 script run
        st.session_state["_pipeline_running"] = False
        st.rerun()

    # ── Hiển thị kết quả (giữ nguyên khi thay đổi settings) ──────────────────
    if "loader_docs" in st.session_state:
        if "active_tab" not in st.session_state:
            st.session_state["active_tab"] = 0
        if "display_mode" not in st.session_state:
            st.session_state["display_mode"] = "Text"

        n_docs    = len(st.session_state["loader_docs"])
        n_chunks  = len(st.session_state.get("chunks", []))
        n_vectors = st.session_state.get("embed_result", {}).get("n_embedded", 0)
        n_vdb     = st.session_state.get("vdb_result", {}).get("n_vectors", 0)

        # on_click callbacks — chạy TRƯỚC khi script rerun
        # → active_tab đã đúng khi buttons render, tránh highlight lệch 1 step
        def _tab(idx: int):
            st.session_state["active_tab"] = idx

        col_tab1, col_tab2, col_tab3, col_tab4, col_tab5, col_spacer, col_mode = st.columns([3, 3, 3, 3, 3, 1, 2])
        with col_tab1:
            st.button(
                f"📂 Loader  ({n_docs} docs)",
                key="tab_btn_loader",
                width="stretch",
                type="primary" if st.session_state["active_tab"] == 0 else "secondary",
                on_click=_tab, args=(0,),
            )
        with col_tab2:
            st.button(
                f"✂️ Chunking  ({n_chunks} chunks)",
                key="tab_btn_chunking",
                width="stretch",
                type="primary" if st.session_state["active_tab"] == 1 else "secondary",
                on_click=_tab, args=(1,),
            )
        with col_tab3:
            emb_label = f"🧮 Embedding  ({n_vectors} vecs)" if n_vectors else "🧮 Embedding"
            st.button(
                emb_label,
                key="tab_btn_embedding",
                width="stretch",
                type="primary" if st.session_state["active_tab"] == 2 else "secondary",
                disabled=("embed_result" not in st.session_state),
                on_click=_tab, args=(2,),
            )
        with col_tab4:
            vdb_meta  = VECTOR_DB_PROVIDER_META.get(st.session_state.get("vdb_cfg_used", {}).get("provider", ""), {})
            vdb_icon  = vdb_meta.get("icon", "🗃️")
            vdb_label = f"{vdb_icon} Vector DB  ({n_vdb:,} vecs)" if n_vdb else f"{vdb_icon} Vector DB"
            st.button(
                vdb_label,
                key="tab_btn_vdb",
                width="stretch",
                type="primary" if st.session_state["active_tab"] == 3 else "secondary",
                disabled=("vdb_result" not in st.session_state),
                on_click=_tab, args=(3,),
            )
        with col_tab5:
            st.button(
                "🔎 Query",
                key="tab_btn_query",
                width="stretch",
                type="primary" if st.session_state["active_tab"] == 4 else "secondary",
                disabled=("vdb_result" not in st.session_state),
                on_click=_tab, args=(4,),
            )
        with col_mode:
            mode = st.radio(
                "Hiển thị",
                options=["Text", "Markdown"],
                index=0 if st.session_state["display_mode"] == "Text" else 1,
                horizontal=True,
                key="global_display_mode",
            )
            st.session_state["display_mode"] = mode

        st.markdown("---")

        display_mode = st.session_state["display_mode"]
        if st.session_state["active_tab"] == 0:
            render_loader_results(st.session_state["loader_docs"], display_mode)
        elif st.session_state["active_tab"] == 1:
            if "chunks" in st.session_state:
                render_chunking_results(st.session_state["chunks"], strategy, display_mode)
            else:
                st.info("Ấn **Process** để thực hiện chunking.")
        elif st.session_state["active_tab"] == 2:
            if "embed_result" in st.session_state:
                render_embedding_results(
                    st.session_state["chunks"],
                    st.session_state["embed_result"],
                    st.session_state.get("emb_cfg_used", emb_cfg),
                )
            else:
                if emb_cfg.get("skip"):
                    st.info("Embedding bị tắt. Bỏ tick **Bỏ qua bước Embedding** trong sidebar rồi ấn **Process** lại.")
                else:
                    st.info("Ấn **Process** để thực hiện embedding.")
        elif st.session_state["active_tab"] == 3:  # tab 3 — Vector DB
            if "vdb_result" in st.session_state:
                render_vector_db_results(
                    st.session_state["vdb_result"],
                    st.session_state.get("vdb_cfg_used", vdb_cfg),
                )
            else:
                if vdb_cfg.get("skip"):
                    st.info("Vector DB bị tắt. Bỏ tick **Bỏ qua bước Vector DB** trong sidebar rồi ấn **Process** lại.")
                elif emb_cfg.get("skip"):
                    st.info("Cần bật Embedding trước khi index Vector DB.")
                else:
                    st.info("Ấn **Process** để index vào Vector DB.")

        elif st.session_state["active_tab"] == 4:  # tab 4 — Query Pipeline
            st.markdown("### 🔎 Query Pipeline — Trực quan hóa từng bước truy vấn")
            st.caption(
                "Nhập câu hỏi bên dưới để chạy toàn bộ pipeline: "
                "**Pre-retrieval → Retrieval → Post-retrieval → Prompt → Generation**. "
                "Cấu hình từng bước ở sidebar (Bước 7–11)."
            )

            # Nút xóa lịch sử hội thoại (conversational mode)
            if st.session_state.get("_query_history"):
                if st.button("🗑️ Xóa lịch sử hội thoại", key="btn_clear_hist", type="secondary"):
                    st.session_state["_query_history"] = []
                    st.rerun()

            _query_input = st.text_input(
                "Câu hỏi",
                placeholder="Ví dụ: What are the main contributions of this paper?",
                key="query_pipeline_input",
            )
            _run_query = st.button("▶️ Run Query Pipeline", key="btn_run_query", type="primary")

            if _run_query and _query_input.strip():
                render_query_pipeline_results(
                    query=_query_input.strip(),
                    pre_cfg=st.session_state.get("query_pre_cfg", {"transformations": ["none"]}),
                    ret_cfg=st.session_state.get("query_ret_cfg", {"strategy": "dense", "top_k": 10}),
                    post_cfg=st.session_state.get("query_post_cfg", {"reranker": "none", "top_n": 5}),
                    vdb_result=st.session_state.get("vdb_result", {}),
                    emb_cfg=st.session_state.get("emb_cfg_used", emb_cfg),
                    prompt_cfg=st.session_state.get("query_prompt_cfg", {"template": "citation", "language": "both"}),
                    gen_cfg=st.session_state.get("query_gen_cfg", {"provider": "openai", "model_name": "gpt-4.1-mini",
                                                                    "temperature": 0.0, "max_tokens": 2048, "streaming": True}),
                    history=st.session_state.get("_query_history") if st.session_state.get(
                        "query_prompt_cfg", {}).get("template") == "conversational" else None,
                )
            elif not _query_input.strip():
                st.info("Nhập câu hỏi ở trên rồi nhấn **▶️ Run Query Pipeline**.")


if __name__ == "__main__":
    main()
