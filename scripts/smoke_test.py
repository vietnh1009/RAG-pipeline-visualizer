#!/usr/bin/env python3
"""
scripts/smoke_test.py
=====================
Bộ test tự động cho INDEXING STAGE của RAG-pipeline-visualizer.

Mục đích: phát hiện lỗi phát sinh trước khi người dùng gặp phải, bằng cách
chạy thật từng option (và tổ hợp option) của 4 bước:
    loader  →  chunking  →  embedding  →  vector DB

Ba chế độ
---------
  --mode single    Test RIÊNG LẺ từng option của từng bước.
                   Chi phí: O(7 + 7 + 5 + 7) ≈ 26 test. Chạy cái này TRƯỚC.

  --mode pairs     Test cặp liền kề (loader×chunking, chunking×embedding,
                   embedding×vdb). Bắt được ~95% lỗi tích hợp với chi phí
                   O(n²) thay vì O(n⁴). Đây là "pairwise testing".

  --mode matrix    Toàn bộ tích Descartes của các option KHẢ DỤNG.
                   Có thể lên tới hàng trăm test — dùng --max-tests để giới hạn.

Cách chạy
---------
    # Bước 1 — luôn chạy cái này trước, nhanh nhất
    python scripts/smoke_test.py --pdf data/sample.pdf --mode single

    # Bước 2 — kiểm tra tích hợp giữa các bước
    python scripts/smoke_test.py --pdf data/sample.pdf --mode pairs

    # Bước 3 — quét toàn bộ (chỉ chạy khi 2 bước trên đã xanh)
    python scripts/smoke_test.py --pdf data/sample.pdf --mode matrix --max-tests 400

    # Chỉ dùng option chạy offline (không gọi API, không tốn tiền)
    python scripts/smoke_test.py --pdf data/sample.pdf --mode pairs --offline

Kết quả
-------
  - Bảng màu in ra terminal
  - smoke_report.json  (máy đọc — dùng cho CI)
  - smoke_report.md    (người đọc — dán vào GitHub Issue)

Quy ước trạng thái
------------------
  PASS  Chạy xong, output hợp lệ.
  WARN  Chạy xong nhưng output đáng ngờ (0 document / 0 chunk / vector toàn 0).
  SKIP  Không test được vì thiếu thư viện, thiếu API key, hoặc thiếu Java.
        SKIP KHÔNG phải lỗi — nó cho biết môi trường chưa đủ.
  FAIL  Ném exception. Đây là thứ cần sửa.
"""

from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass, field, asdict
from pathlib import Path

# Console Windows mặc định là cp1252 → bảng kết quả tiếng Việt + emoji sẽ ném
# UnicodeEncodeError giữa chừng, mất sạch kết quả của cả lượt chạy.
# Ép stdout/stderr về UTF-8 trước khi in bất cứ thứ gì.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

# ── Đưa project root vào sys.path ────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except ImportError:
    pass


# ═════════════════════════════════════════════════════════════════════════════
# Khai báo option — ĐỌC TRỰC TIẾP TỪ REGISTRY CỦA REPO
# Không hard-code danh sách, để khi bạn thêm option mới thì test tự bắt được.
# ═════════════════════════════════════════════════════════════════════════════

def _discover(fn, fallback: list[str], what: str) -> list[str]:
    """Đọc registry của repo; nếu import lỗi (thiếu dep core) thì dùng danh sách dự phòng."""
    try:
        return fn()
    except Exception as exc:
        print(f"  {C_YELLOW}Không đọc được registry {what} ({type(exc).__name__}: {exc}) "
              f"— dùng danh sách dự phòng.{C_RESET}")
        return list(fallback)


def discover_loaders() -> list[str]:
    def _f():
        from loader.directory_loader import PDFStrategy
        return list(PDFStrategy.__args__)
    return _discover(_f, list(LOADER_REQS), "loader")


def discover_chunkers() -> list[str]:
    def _f():
        from chunking.factory import _REGISTRY
        return list(_REGISTRY)
    return _discover(_f, list(CHUNKER_REQS), "chunking")


def discover_embedders() -> list[str]:
    def _f():
        from embedding.factory import _REGISTRY
        return list(_REGISTRY)
    return _discover(_f, list(EMBEDDER_REQS), "embedding")


def discover_vector_dbs() -> list[str]:
    def _f():
        from vector_db.factory import _REGISTRY
        return list(_REGISTRY)
    return _discover(_f, list(VDB_REQS), "vector_db")


# ── Điều kiện tiên quyết của từng option ─────────────────────────────────────
# (tên module cần import được, biến môi trường cần có, có cần Java không)

_CORE = ["langchain_core"]

# Binary hệ thống bắt buộc, tra theo TÊN OPTION.
#
# Vì sao cần: `pip install unstructured` thành công không có nghĩa là chạy được.
# Nó còn cần tesseract (OCR) và poppler (pdftoppm) — hai thứ pip không cài được.
# Thiếu chúng, unstructured KHÔNG ném ImportError mà TREO vô hạn rồi segfault,
# giết luôn cả lượt smoke test (toàn bộ tiến trình, vì test chạy chung process).
# Kiểm tra binary trước để SKIP sạch sẽ thay vì chết giữa chừng.
BINARY_REQS: dict[str, list[str]] = {
    "unstructured": ["tesseract", "pdftoppm"],
}

LOADER_REQS = {
    "pypdf":          (_CORE + ["pypdf", "langchain_community"], [], False),
    "pymupdf":        (_CORE + ["fitz"], [], False),
    "pdfplumber":     (_CORE + ["pdfplumber"], [], False),
    "unstructured":   (_CORE + ["unstructured"], [], False),
    "docling":        (_CORE + ["docling"], [], False),
    "marker":         (_CORE + ["marker"], [], False),
    "opendataloader": (_CORE + ["opendataloader_pdf"], [], True),
}

CHUNKER_REQS = {
    "recursive":      (_CORE + ["langchain_text_splitters"], [], False),
    "token_based":    (_CORE + ["tiktoken"], [], False),
    "format_aware":   (_CORE + ["langchain_text_splitters"], [], False),
    "sentence_aware": (_CORE, [], False),        # có regex fallback, luôn chạy được
    "semantic":       (_CORE + ["langchain_experimental"], [], False),
    "hierarchical":   (_CORE + ["langchain_text_splitters"], [], False),
    "contextual":     (_CORE, ["ANTHROPIC_API_KEY|OPENAI_API_KEY|GOOGLE_API_KEY"], False),
}

EMBEDDER_REQS = {
    "openai":      (_CORE + ["langchain_openai"],      ["OPENAI_API_KEY"], False),
    "cohere":      (_CORE + ["langchain_cohere"],      ["COHERE_API_KEY"], False),
    "huggingface": (_CORE + ["sentence_transformers"], [], False),
    "fastembed":   (_CORE + ["fastembed"],             [], False),
    "ollama":      (_CORE + ["langchain_ollama"],      [], False),
}

VDB_REQS = {
    "faiss":    (_CORE + ["faiss", "langchain_community"],   [], False),
    "chroma":   (_CORE + ["chromadb", "langchain_chroma"],   [], False),
    "lancedb":  (_CORE + ["lancedb"],                        [], False),
    "qdrant":   (_CORE + ["qdrant_client", "langchain_qdrant"], ["QDRANT_URL"], False),
    "weaviate": (_CORE + ["weaviate", "langchain_weaviate"],    ["WEAVIATE_URL"], False),
    "pgvector": (_CORE + ["psycopg", "langchain_postgres"],     ["DATABASE_URL"], False),
    "pinecone": (_CORE + ["pinecone", "langchain_pinecone"],    ["PINECONE_API_KEY"], False),
}

# Option nào gọi API trả tiền / cần mạng — bị loại khi chạy --offline
NETWORK_OPTIONS = {
    "contextual", "openai", "cohere",
    "qdrant", "weaviate", "pgvector", "pinecone",
}

# Model nhẹ nhất cho mỗi embedding provider, để test nhanh và rẻ
EMBED_TEST_MODEL = {
    "openai":      "text-embedding-3-small",
    "cohere":      "embed-multilingual-v3.0",
    "huggingface": "sentence-transformers/all-MiniLM-L6-v2",
    "fastembed":   "BAAI/bge-small-en-v1.5",
    "ollama":      "nomic-embed-text",
}


# ═════════════════════════════════════════════════════════════════════════════
# Hạ tầng test
# ═════════════════════════════════════════════════════════════════════════════

C_RESET, C_RED, C_GREEN, C_YELLOW, C_BLUE, C_DIM = (
    "\033[0m", "\033[31m", "\033[32m", "\033[33m", "\033[36m", "\033[2m"
)
if os.environ.get("NO_COLOR") or not sys.stdout.isatty():
    C_RESET = C_RED = C_GREEN = C_YELLOW = C_BLUE = C_DIM = ""

_STATUS_COLOR = {"PASS": C_GREEN, "WARN": C_YELLOW, "SKIP": C_DIM, "FAIL": C_RED}


@dataclass
class Result:
    stage:   str
    name:    str
    status:  str            # PASS | WARN | SKIP | FAIL
    seconds: float = 0.0
    detail:  str = ""
    error:   str = ""
    trace:   str = ""
    metrics: dict = field(default_factory=dict)


def _module_ok(mod: str) -> bool:
    try:
        return importlib.util.find_spec(mod) is not None
    except (ImportError, ValueError, ModuleNotFoundError):
        return False


def _java_ok() -> bool:
    exe = shutil.which("java")
    if not exe:
        return False
    try:
        r = subprocess.run([exe, "-version"], capture_output=True, timeout=10)
        return r.returncode == 0
    except Exception:
        return False


def _env_ok(spec: str) -> bool:
    """spec có thể là 'A' hoặc 'A|B|C' (thoả một trong các biến là đủ)."""
    return any(os.environ.get(v.strip()) for v in spec.split("|"))


def check_prereqs(reqs: tuple, offline: bool, name: str) -> str | None:
    """Trả về lý do SKIP, hoặc None nếu đủ điều kiện chạy."""
    modules, envs, needs_java = reqs
    if offline and name in NETWORK_OPTIONS:
        return "bỏ qua ở chế độ --offline"
    missing = [m for m in modules if not _module_ok(m)]
    if missing:
        return f"thiếu package: {', '.join(missing)}"
    missing_bin = [b for b in BINARY_REQS.get(name, []) if shutil.which(b) is None]
    if missing_bin:
        return f"thiếu binary hệ thống: {', '.join(missing_bin)}"
    missing_env = [e for e in envs if not _env_ok(e)]
    if missing_env:
        return f"thiếu env var: {', '.join(missing_env)}"
    if needs_java and not _java_ok():
        return "thiếu Java 11+"
    return None


def timed(fn):
    t0 = time.perf_counter()
    try:
        out = fn()
        return out, time.perf_counter() - t0, None
    except BaseException as exc:            # BaseException: bắt cả SystemExit từ lib
        return None, time.perf_counter() - t0, exc


# ═════════════════════════════════════════════════════════════════════════════
# Các bước test
# ═════════════════════════════════════════════════════════════════════════════

def run_loader(strategy: str, pdf: str, workdir: Path):
    from loader.directory_loader import PDFDocumentLoader
    kwargs = dict(
        language="both",
        pdf_strategy=strategy,
        extract_tables=True,
        deduplicate=True,
        describe_images=False,      # không gọi VLM trong smoke test
    )
    if strategy == "opendataloader":
        kwargs["odl_hybrid"] = None          # fast mode, không cần server
        kwargs["odl_struct_tree"] = False
    return PDFDocumentLoader(**kwargs).load(pdf)


def run_chunker(strategy: str, docs: list):
    from chunking.factory import get_chunker
    kwargs: dict = {"chunk_size": 500, "chunk_overlap": 100}
    if strategy == "token_based":
        kwargs["encoding_name"] = "cl100k_base"
    elif strategy == "semantic":
        kwargs["embedding_model_name"] = "sentence-transformers/all-MiniLM-L6-v2"
    elif strategy == "format_aware":
        kwargs["format_type"] = "auto"
    elif strategy == "hierarchical":
        kwargs["parent_chunk_size"] = 2000
    elif strategy == "contextual":
        kwargs["base_strategy"] = "recursive"
        # provider được chọn theo API key đang có
        if os.environ.get("ANTHROPIC_API_KEY"):
            kwargs["llm_provider"], kwargs["llm_model"] = "anthropic", "claude-haiku-4-5-20251001"
        elif os.environ.get("OPENAI_API_KEY"):
            kwargs["llm_provider"], kwargs["llm_model"] = "openai", "gpt-4o-mini"
        else:
            kwargs["llm_provider"], kwargs["llm_model"] = "google", "gemini-2.0-flash"
    return get_chunker(strategy, **kwargs).split(docs)


def run_embedder(provider: str, texts: list[str]):
    from embedding.factory import get_embedder
    model = EMBED_TEST_MODEL[provider]
    extra: dict = {}
    if provider == "cohere":
        extra["input_type"] = "search_document"
    elif provider == "ollama":
        extra["base_url"] = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    elif provider == "huggingface":
        extra["device"] = "cpu"
    emb = get_embedder(provider, model, **extra)
    return emb.embed_documents(texts[:3]), emb


def run_vdb(provider: str, chunks: list, embedder, workdir: Path):
    from vector_db.factory import get_vector_store
    kwargs: dict = {"collection_name": f"smoke_{provider}"}
    if provider in ("faiss", "chroma", "lancedb"):
        kwargs["persist_dir"] = str(workdir / f"vdb_{provider}")
    elif provider == "qdrant":
        kwargs["url"] = os.environ.get("QDRANT_URL", "http://localhost:6333")
        kwargs["api_key"] = os.environ.get("QDRANT_API_KEY")
    elif provider == "weaviate":
        kwargs["url"] = os.environ.get("WEAVIATE_URL", "http://localhost:8080")
        kwargs["api_key"] = os.environ.get("WEAVIATE_API_KEY")
    elif provider == "pgvector":
        kwargs["connection_string"] = os.environ.get("DATABASE_URL", "")
    elif provider == "pinecone":
        kwargs["cloud"] = "aws"
        kwargs["region"] = "us-east-1"

    store = get_vector_store(
        provider=provider, chunks=chunks, embedder=embedder,
        force_reindex=True, **kwargs,
    )
    # Kiểm tra store dùng được thật, không chỉ tạo xong là xong
    hits = store.similarity_search("test", k=1)
    return store, hits


# ═════════════════════════════════════════════════════════════════════════════
# Bộ chạy
# ═════════════════════════════════════════════════════════════════════════════

class Runner:
    def __init__(self, pdf: str, workdir: Path, offline: bool, verbose: bool):
        self.pdf      = pdf
        self.workdir  = workdir
        self.offline  = offline
        self.verbose  = verbose
        self.results: list[Result] = []
        self._doc_cache: dict[str, list] = {}
        self._chunk_cache: dict[tuple, list] = {}

    # ── báo cáo ────────────────────────────────────────────────────────────
    def record(self, r: Result) -> Result:
        self.results.append(r)
        color = _STATUS_COLOR[r.status]
        line = (f"  {color}{r.status:<4}{C_RESET}  {r.stage:<10} {r.name:<46} "
                f"{C_DIM}{r.seconds:6.2f}s{C_RESET}  {r.detail}")
        print(line, flush=True)
        if r.status == "FAIL":
            print(f"        {C_RED}{r.error}{C_RESET}", flush=True)
            if self.verbose and r.trace:
                print(C_DIM + "".join("        " + l for l in r.trace.splitlines(True)) + C_RESET)
        return r

    # ── LOADER ─────────────────────────────────────────────────────────────
    def test_loader(self, strategy: str) -> Result:
        skip = check_prereqs(LOADER_REQS[strategy], self.offline, strategy)
        if skip:
            return self.record(Result("loader", strategy, "SKIP", detail=skip))

        docs, secs, exc = timed(lambda: run_loader(strategy, self.pdf, self.workdir))
        if exc is not None:
            return self.record(Result("loader", strategy, "FAIL", secs,
                                      error=f"{type(exc).__name__}: {exc}",
                                      trace=traceback.format_exc()))
        chars = sum(len(d.page_content) for d in docs)
        self._doc_cache[strategy] = docs
        metrics = {"n_docs": len(docs), "n_chars": chars}
        if not docs:
            return self.record(Result("loader", strategy, "WARN", secs,
                                      detail="0 document", metrics=metrics))
        if chars == 0:
            return self.record(Result("loader", strategy, "WARN", secs,
                                      detail=f"{len(docs)} doc nhưng 0 ký tự "
                                             f"(PDF không có text layer?)", metrics=metrics))
        return self.record(Result("loader", strategy, "PASS", secs,
                                  detail=f"{len(docs)} doc · {chars:,} ký tự",
                                  metrics=metrics))

    # ── CHUNKING ───────────────────────────────────────────────────────────
    def test_chunker(self, strategy: str, docs: list, label: str | None = None) -> Result:
        name = label or strategy
        skip = check_prereqs(CHUNKER_REQS[strategy], self.offline, strategy)
        if skip:
            return self.record(Result("chunking", name, "SKIP", detail=skip))
        if not docs:
            return self.record(Result("chunking", name, "SKIP",
                                      detail="không có document đầu vào"))

        chunks, secs, exc = timed(lambda: run_chunker(strategy, docs))
        if exc is not None:
            return self.record(Result("chunking", name, "FAIL", secs,
                                      error=f"{type(exc).__name__}: {exc}",
                                      trace=traceback.format_exc()))
        self._chunk_cache[(strategy, id(docs))] = chunks
        metrics = {"n_chunks": len(chunks)}
        if not chunks:
            return self.record(Result("chunking", name, "WARN", secs,
                                      detail="0 chunk", metrics=metrics))
        avg = sum(len(c.page_content) for c in chunks) / len(chunks)
        return self.record(Result("chunking", name, "PASS", secs,
                                  detail=f"{len(chunks)} chunk · trung bình {avg:.0f} ký tự",
                                  metrics=metrics))

    # ── EMBEDDING ──────────────────────────────────────────────────────────
    def test_embedder(self, provider: str, texts: list[str], label: str | None = None):
        name = label or provider
        skip = check_prereqs(EMBEDDER_REQS[provider], self.offline, provider)
        if skip:
            return self.record(Result("embedding", name, "SKIP", detail=skip)), None
        if not texts:
            return self.record(Result("embedding", name, "SKIP",
                                      detail="không có text đầu vào")), None

        out, secs, exc = timed(lambda: run_embedder(provider, texts))
        if exc is not None:
            return self.record(Result("embedding", name, "FAIL", secs,
                                      error=f"{type(exc).__name__}: {exc}",
                                      trace=traceback.format_exc())), None
        vecs, emb = out
        if not vecs or not vecs[0]:
            return self.record(Result("embedding", name, "WARN", secs,
                                      detail="vector rỗng")), emb
        dim = len(vecs[0])
        if all(abs(x) < 1e-12 for x in vecs[0]):
            return self.record(Result("embedding", name, "WARN", secs,
                                      detail=f"{dim}d nhưng vector toàn 0")), emb
        return self.record(Result("embedding", name, "PASS", secs,
                                  detail=f"{len(vecs)} vector · {dim}d",
                                  metrics={"dim": dim})), emb

    # ── VECTOR DB ──────────────────────────────────────────────────────────
    def test_vdb(self, provider: str, chunks: list, embedder, label: str | None = None):
        name = label or provider
        skip = check_prereqs(VDB_REQS[provider], self.offline, provider)
        if skip:
            return self.record(Result("vector_db", name, "SKIP", detail=skip))
        if not chunks or embedder is None:
            return self.record(Result("vector_db", name, "SKIP",
                                      detail="không có chunk/embedder đầu vào"))

        out, secs, exc = timed(lambda: run_vdb(provider, chunks, embedder, self.workdir))
        if exc is not None:
            return self.record(Result("vector_db", name, "FAIL", secs,
                                      error=f"{type(exc).__name__}: {exc}",
                                      trace=traceback.format_exc()))
        _store, hits = out
        if not hits:
            return self.record(Result("vector_db", name, "WARN", secs,
                                      detail="index xong nhưng search trả 0 kết quả"))
        return self.record(Result("vector_db", name, "PASS", secs,
                                  detail=f"{len(chunks)} chunk · search OK"))

    # ── Chọn baseline chạy được, dùng làm đầu vào cho các bước sau ────────
    def baseline(self):
        """Trả về (docs, chunks, embedder) từ tổ hợp nhẹ nhất chạy được."""
        docs = None
        for s in ("pymupdf", "pypdf", "pdfplumber"):
            if s in self._doc_cache and self._doc_cache[s]:
                docs = self._doc_cache[s]
                break
        if docs is None:
            for v in self._doc_cache.values():
                if v:
                    docs = v
                    break
        if not docs:
            return None, None, None

        try:
            chunks = run_chunker("recursive", docs)
        except Exception:
            return docs, None, None
        if not chunks:
            return docs, None, None

        for p in ("fastembed", "huggingface", "openai", "ollama", "cohere"):
            if check_prereqs(EMBEDDER_REQS[p], self.offline, p):
                continue
            try:
                _, emb = run_embedder(p, [c.page_content for c in chunks])
                return docs, chunks, emb
            except Exception:
                continue
        return docs, chunks, None


# ═════════════════════════════════════════════════════════════════════════════
# Các chế độ
# ═════════════════════════════════════════════════════════════════════════════

def mode_single(rn: Runner):
    print(f"\n{C_BLUE}━━ 1/4 · LOADER — test riêng lẻ ━━{C_RESET}")
    for s in discover_loaders():
        rn.test_loader(s)

    docs, chunks, emb = rn.baseline()

    print(f"\n{C_BLUE}━━ 2/4 · CHUNKING — test riêng lẻ ━━{C_RESET}")
    if not docs:
        print(f"  {C_YELLOW}Không loader nào tạo được document — bỏ qua các bước sau.{C_RESET}")
        return
    for s in discover_chunkers():
        rn.test_chunker(s, docs)

    print(f"\n{C_BLUE}━━ 3/4 · EMBEDDING — test riêng lẻ ━━{C_RESET}")
    texts = [c.page_content for c in (chunks or [])] or [d.page_content for d in docs]
    texts = [t for t in texts if t.strip()] or ["Đây là một câu tiếng Việt để test embedding."]
    embedders: dict = {}
    for p in discover_embedders():
        _, e = rn.test_embedder(p, texts)
        if e is not None:
            embedders[p] = e

    print(f"\n{C_BLUE}━━ 4/4 · VECTOR DB — test riêng lẻ ━━{C_RESET}")
    use_emb = emb or next(iter(embedders.values()), None)
    vdb_chunks = chunks or docs
    for p in discover_vector_dbs():
        rn.test_vdb(p, vdb_chunks, use_emb)


def mode_pairs(rn: Runner, max_tests: int):
    """Test cặp liền kề — bắt lỗi tích hợp mà test riêng lẻ bỏ sót."""
    loaders  = discover_loaders()
    chunkers = discover_chunkers()
    embs     = discover_embedders()
    vdbs     = discover_vector_dbs()
    n = 0

    print(f"\n{C_BLUE}━━ 1/3 · LOADER × CHUNKING ━━{C_RESET}")
    for l in loaders:
        r = rn.test_loader(l)
        if r.status not in ("PASS", "WARN"):
            continue
        docs = rn._doc_cache.get(l) or []
        for c in chunkers:
            if n >= max_tests:
                return
            rn.test_chunker(c, docs, label=f"{l} → {c}")
            n += 1

    docs, chunks, _ = rn.baseline()
    if not chunks:
        print(f"  {C_YELLOW}Không tạo được chunk — dừng phần còn lại.{C_RESET}")
        return
    texts = [c.page_content for c in chunks if c.page_content.strip()] or ["test"]

    print(f"\n{C_BLUE}━━ 2/3 · CHUNKING × EMBEDDING ━━{C_RESET}")
    embedders: dict = {}
    for c in chunkers:
        r = rn.test_chunker(c, docs, label=f"chunk:{c}")
        if r.status not in ("PASS", "WARN"):
            continue
        c_chunks = rn._chunk_cache.get((c, id(docs))) or chunks
        c_texts  = [x.page_content for x in c_chunks if x.page_content.strip()] or texts
        for p in embs:
            if n >= max_tests:
                return
            _, e = rn.test_embedder(p, c_texts, label=f"{c} → {p}")
            if e is not None:
                embedders.setdefault(p, e)
            n += 1

    print(f"\n{C_BLUE}━━ 3/3 · EMBEDDING × VECTOR DB ━━{C_RESET}")
    for p in embs:
        e = embedders.get(p)
        if e is None:
            continue
        for v in vdbs:
            if n >= max_tests:
                return
            rn.test_vdb(v, chunks, e, label=f"{p} → {v}")
            n += 1


def mode_matrix(rn: Runner, max_tests: int):
    """Tích Descartes đầy đủ trên các option KHẢ DỤNG."""
    avail_l = [s for s in discover_loaders()
               if not check_prereqs(LOADER_REQS[s], rn.offline, s)]
    avail_c = [s for s in discover_chunkers()
               if not check_prereqs(CHUNKER_REQS[s], rn.offline, s)]
    avail_e = [s for s in discover_embedders()
               if not check_prereqs(EMBEDDER_REQS[s], rn.offline, s)]
    avail_v = [s for s in discover_vector_dbs()
               if not check_prereqs(VDB_REQS[s], rn.offline, s)]

    total = len(avail_l) * len(avail_c) * len(avail_e) * len(avail_v)
    print(f"\n{C_BLUE}━━ MATRIX ━━{C_RESET}")
    print(f"  Option khả dụng: loader={len(avail_l)} chunking={len(avail_c)} "
          f"embedding={len(avail_e)} vdb={len(avail_v)}")
    print(f"  Tổng tổ hợp: {total}  ·  giới hạn: {max_tests}\n")

    n = 0
    for l, c, p, v in itertools.product(avail_l, avail_c, avail_e, avail_v):
        if n >= max_tests:
            print(f"\n  {C_YELLOW}Đã đạt --max-tests={max_tests}, dừng.{C_RESET}")
            return
        n += 1
        combo = f"{l}|{c}|{p}|{v}"

        if l not in rn._doc_cache:
            rl = rn.test_loader(l)
            if rl.status == "FAIL":
                continue
        docs = rn._doc_cache.get(l) or []
        if not docs:
            rn.record(Result("combo", combo, "SKIP", detail=f"loader {l} không ra document"))
            continue

        chunks, secs, exc = timed(lambda: run_chunker(c, docs))
        if exc is not None:
            rn.record(Result("combo", combo, "FAIL", secs,
                             error=f"[chunking:{c}] {type(exc).__name__}: {exc}",
                             trace=traceback.format_exc()))
            continue
        if not chunks:
            rn.record(Result("combo", combo, "WARN", secs, detail=f"chunking {c} ra 0 chunk"))
            continue

        texts = [x.page_content for x in chunks if x.page_content.strip()] or ["test"]
        out, secs_e, exc = timed(lambda: run_embedder(p, texts))
        if exc is not None:
            rn.record(Result("combo", combo, "FAIL", secs + secs_e,
                             error=f"[embedding:{p}] {type(exc).__name__}: {exc}",
                             trace=traceback.format_exc()))
            continue
        _vecs, emb = out

        out, secs_v, exc = timed(lambda: run_vdb(v, chunks, emb, rn.workdir))
        if exc is not None:
            rn.record(Result("combo", combo, "FAIL", secs + secs_e + secs_v,
                             error=f"[vector_db:{v}] {type(exc).__name__}: {exc}",
                             trace=traceback.format_exc()))
            continue
        _store, hits = out
        total_s = secs + secs_e + secs_v
        if not hits:
            rn.record(Result("combo", combo, "WARN", total_s, detail="search trả 0 kết quả"))
        else:
            rn.record(Result("combo", combo, "PASS", total_s,
                             detail=f"{len(chunks)} chunk · search OK"))


# ═════════════════════════════════════════════════════════════════════════════
# Báo cáo
# ═════════════════════════════════════════════════════════════════════════════

def write_reports(results: list[Result], out_dir: Path, meta: dict):
    counts = {s: sum(1 for r in results if r.status == s)
              for s in ("PASS", "WARN", "SKIP", "FAIL")}

    (out_dir / "smoke_report.json").write_text(
        json.dumps({"meta": meta, "summary": counts,
                    "results": [asdict(r) for r in results]},
                   ensure_ascii=False, indent=2),
        encoding="utf-8")

    lines = [
        "# Smoke test — Indexing stage", "",
        f"- Chế độ: `{meta['mode']}`",
        f"- Python: `{meta['python']}`",
        f"- File test: `{meta['pdf']}`",
        f"- Thời điểm: {meta['timestamp']}", "",
        f"**PASS {counts['PASS']} · WARN {counts['WARN']} "
        f"· SKIP {counts['SKIP']} · FAIL {counts['FAIL']}**", "",
        "| Trạng thái | Bước | Option | Giây | Ghi chú |",
        "|---|---|---|---|---|",
    ]
    icon = {"PASS": "✅", "WARN": "⚠️", "SKIP": "⏭️", "FAIL": "❌"}
    for r in results:
        note = (r.detail or r.error).replace("|", "\\|").replace("\n", " ")[:160]
        lines.append(f"| {icon[r.status]} {r.status} | {r.stage} | `{r.name}` "
                     f"| {r.seconds:.2f} | {note} |")

    fails = [r for r in results if r.status == "FAIL"]
    if fails:
        lines += ["", "## Chi tiết lỗi", ""]
        for r in fails:
            lines += [f"### `{r.stage}` / `{r.name}`", "", "```", r.trace.strip(), "```", ""]

    (out_dir / "smoke_report.md").write_text("\n".join(lines), encoding="utf-8")
    return counts


def main() -> int:
    ap = argparse.ArgumentParser(description="Smoke test cho indexing stage")
    ap.add_argument("--pdf", required=True, help="File PDF hoặc thư mục PDF dùng để test")
    ap.add_argument("--mode", choices=["single", "pairs", "matrix"], default="single")
    ap.add_argument("--max-tests", type=int, default=500)
    ap.add_argument("--offline", action="store_true",
                    help="Chỉ test option chạy local, không gọi API trả tiền")
    ap.add_argument("--out", default=".", help="Thư mục ghi báo cáo")
    ap.add_argument("-v", "--verbose", action="store_true", help="In full traceback")
    ap.add_argument("--keep-workdir", action="store_true")
    args = ap.parse_args()

    pdf = Path(args.pdf)
    if not pdf.exists():
        print(f"{C_RED}Không tìm thấy: {pdf}{C_RESET}")
        return 2

    workdir = Path(tempfile.mkdtemp(prefix="rag_smoke_"))
    print(f"{C_BLUE}RAG-pipeline-visualizer · smoke test{C_RESET}")
    print(f"  python   : {sys.version.split()[0]}")
    print(f"  input    : {pdf}")
    print(f"  chế độ   : {args.mode}{'  (offline)' if args.offline else ''}")
    print(f"  workdir  : {workdir}")

    rn = Runner(str(pdf), workdir, args.offline, args.verbose)
    t0 = time.perf_counter()
    try:
        if args.mode == "single":
            mode_single(rn)
        elif args.mode == "pairs":
            mode_pairs(rn, args.max_tests)
        else:
            mode_matrix(rn, args.max_tests)
    except KeyboardInterrupt:
        print(f"\n{C_YELLOW}Bị ngắt bởi người dùng — vẫn ghi báo cáo.{C_RESET}")
    finally:
        if not args.keep_workdir:
            shutil.rmtree(workdir, ignore_errors=True)

    elapsed = time.perf_counter() - t0
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    counts = write_reports(rn.results, out_dir, {
        "mode": args.mode,
        "python": sys.version.split()[0],
        "pdf": str(pdf),
        "offline": args.offline,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_seconds": round(elapsed, 1),
    })

    print(f"\n{C_BLUE}━━ TỔNG KẾT ({elapsed:.1f}s) ━━{C_RESET}")
    print(f"  {C_GREEN}PASS {counts['PASS']}{C_RESET}  "
          f"{C_YELLOW}WARN {counts['WARN']}{C_RESET}  "
          f"{C_DIM}SKIP {counts['SKIP']}{C_RESET}  "
          f"{C_RED}FAIL {counts['FAIL']}{C_RESET}")
    print(f"  Báo cáo: {out_dir/'smoke_report.md'} · {out_dir/'smoke_report.json'}")

    if counts["FAIL"]:
        print(f"\n{C_RED}Các option lỗi:{C_RESET}")
        for r in rn.results:
            if r.status == "FAIL":
                print(f"  · {r.stage}/{r.name}: {r.error}")
    return 1 if counts["FAIL"] else 0


if __name__ == "__main__":
    sys.exit(main())
