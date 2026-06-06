"""utils/env.py — Environment helpers, package detection, uploads, GPU, file profiling."""
import os, sys, importlib, importlib.util, tempfile
from pathlib import Path
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
try:
    from dotenv import load_dotenv as _ld, dotenv_values as _dv
    _ld(override=True); _ENV = _dv()
except ImportError:
    _ENV = {}
import streamlit as st
from pipeline_cache import PipelineCache
from core.constants import (
    EMBED_PREVIEW_LIMIT, PAGE_TITLE, FILE_TYPE_COLORS,
    CHUNK_DESCRIPTIONS, LLM_REQUIRED_STRATEGIES, PDF_STRATEGY_DEPS,
)

def _is_installed(pkg: str) -> bool:
    """Kiểm tra nhanh package đã được cài chưa bằng importlib.util.find_spec."""
    import importlib.util
    # Một số package có tên import khác tên pip (e.g. Pillow → PIL)
    _IMPORT_MAP = {
        "torch":              "torch",
        "transformers":       "transformers",
        "sentence_transformers": "sentence_transformers",
        "fastembed":          "fastembed",
        "chromadb":           "chromadb",
        "faiss":              "faiss",
        "faiss-cpu":          "faiss",
        "qdrant-client":      "qdrant_client",
        "lancedb":            "lancedb",
        "weaviate-client":    "weaviate",
        "pinecone-client":    "pinecone",
        "cohere":             "cohere",
        "anthropic":          "anthropic",
        "google-generativeai":"google.generativeai",
        "langchain-ollama":   "langchain_ollama",
        "ollama":             "ollama",
        "marker-pdf":         "marker",
        "docling":            "docling",
        "unstructured":       "unstructured",
        "opendataloader-pdf": "opendataloader_pdf",
    }
    import_name = _IMPORT_MAP.get(pkg, pkg.replace("-", "_"))
    return importlib.util.find_spec(import_name) is not None


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

# ── Global cache instance ─────────────────────────────────────────────────────

@st.cache_resource
def _init_uploads_dir() -> Path:
    """
    Trả về Path thư mục uploads (tạo nếu chưa có, KHÔNG xóa file cũ).
    Việc xóa file cũ được thực hiện trong save_uploaded_files() để đảm bảo
    chỉ xóa đúng lúc user upload batch mới.
    """
    import tempfile
    uploads = Path(tempfile.gettempdir()) / "rag_visualizer_uploads"
    uploads.mkdir(parents=True, exist_ok=True)
    return uploads


def save_uploaded_files(uploaded_files) -> str:
    """
    Lưu file upload vào thư mục tạm của OS.
    - Xóa TOÀN BỘ file cũ trước khi lưu để tránh lẫn file từ lần upload trước.
    - Dùng tên file gốc → path ổn định giữa các lần upload → pipeline cache hoạt động đúng.
    - Lưu tên file gốc vào session_state["_upload_display_names"] để hiển thị.
    """
    import shutil, streamlit as _st
    uploads_dir = Path(_init_uploads_dir_path())
    if uploads_dir.exists():
        shutil.rmtree(uploads_dir)
    uploads_dir.mkdir(parents=True, exist_ok=True)
    names = []
    for uf in uploaded_files:
        dest = uploads_dir / uf.name
        dest.write_bytes(uf.read())
        names.append(uf.name)
    # Lưu tên gốc để hiển thị thay vì temp path
    _st.session_state["_upload_display_names"] = names
    return str(uploads_dir)


def _init_uploads_dir_path() -> str:
    """Trả về path thư mục uploads (không xóa)."""
    import tempfile
    return str(Path(tempfile.gettempdir()) / "rag_visualizer_uploads")


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




@st.cache_resource
def _get_pipeline_cache() -> PipelineCache:
    return PipelineCache("processed_data")



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




