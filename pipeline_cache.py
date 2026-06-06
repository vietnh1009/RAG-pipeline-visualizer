"""
pipeline_cache.py
=================
Cache cấp bước (step-level) cho RAG pipeline, lưu trên disk.

Cơ chế: Pipeline Fingerprint Chain
───────────────────────────────────
Mỗi bước có step_key = SHA256(prev_step_key + canonical_json(step_config)).

  input_hash ──► loader_key ──► chunking_key ──► embedding_key ──► vdb_key

Tính chất:
- Input + loader config không đổi              → loader cache HIT
- Input không đổi, chunking config thay đổi   → chunking MISS, loader HIT
- Input thay đổi                               → tất cả MISS

Cấu trúc thư mục:
  processed_data/
    <input_hash_12chars>/
      input_meta.json
      loader/<loader_key_12chars>/{docs.pkl, meta.json}
      chunking/<chunk_key_12chars>/{chunks.pkl, meta.json}
      embedding/<embed_key_12chars>/{dense.npz, sparse.pkl?, meta.json}
      vector_db/<vdb_key_12chars>/meta.json   # chỉ metadata, data ở persist_dir

Dùng tên 12 ký tự đầu của SHA256 (~69 nghìn tỷ khả năng) để đủ unique và dễ đọc.

Ví dụ sử dụng:
    cache = PipelineCache("processed_data")
    input_hash = cache.compute_input_hash(source_path)
    loader_key = cache.make_step_key(input_hash, loader_cfg)
    docs = cache.load_loader(input_hash, loader_key)
    if docs is None:
        docs = run_loader(...)
        cache.save_loader(input_hash, loader_key, docs, loader_cfg)
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import pickle
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
KEY_LEN       = 12        # hex chars to use from SHA256 (12 = 48 bits, ~281T combos)
CHUNK_SIZE    = 65_536    # bytes per read when hashing large files
STEP_NAMES    = ("loader", "chunking", "embedding", "vector_db")   # ordered — used for invalidation


# ── Helpers ───────────────────────────────────────────────────────────────────

def _sha256_file(path: Path) -> str:
    """SHA256 of a single file, streamed (memory-efficient for large files)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(CHUNK_SIZE):
            h.update(chunk)
    return h.hexdigest()


def _sha256_str(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _canonical_json(obj: Any) -> str:
    """Deterministic JSON — sorted keys, no whitespace, None→null."""
    def _clean(o: Any) -> Any:
        if isinstance(o, dict):
            return {str(k): _clean(v) for k, v in sorted(o.items())}
        if isinstance(o, (list, tuple)):
            return [_clean(i) for i in o]
        if isinstance(o, Path):
            return str(o)
        if o is None or isinstance(o, (bool, int, float, str)):
            return o
        return str(o)   # fallback — e.g. Enum, custom types
    return json.dumps(_clean(obj), ensure_ascii=False, separators=(",", ":"))


def _short(full_hash: str) -> str:
    return full_hash[:KEY_LEN]


# ── Main class ────────────────────────────────────────────────────────────────

class PipelineCache:
    """
    Disk-backed cache for RAG pipeline steps.

    Thread-safe for reads; writes use atomic rename to avoid partial files.
    """

    def __init__(self, base_dir: str | Path = "processed_data"):
        self.base = Path(base_dir)
        self.base.mkdir(parents=True, exist_ok=True)
        _write_gitignore(self.base)

    # ── Hash / key computation ─────────────────────────────────────────────

    def compute_input_hash(self, source_path: str) -> str:
        """
        Content-based SHA256 hash của toàn bộ file(s) từ source_path.

        Chỉ dùng tên file (p.name) + nội dung — KHÔNG dùng đường dẫn thư mục.
        Lý do: file upload vào Streamlit được lưu vào thư mục temp ngẫu nhiên
        (tmpXXXXXX) mỗi lần → nếu hash cả path thì cùng một file upload lại
        sẽ cho hash khác → cache miss sai.

        Tính chất:
          - Cùng file, upload lại           → hash GIỐNG (cache HIT ✅)
          - Cùng tên nhưng nội dung khác    → hash KHÁC  (cache MISS ✅)
          - Khác tên, cùng nội dung         → hash KHÁC  (cache MISS ✅)
          - Nhiều file: sort theo tên để deterministic

        source_path có thể là:
          - Đường dẫn file đơn
          - Đường dẫn thư mục (đệ quy)
          - Nhiều file cách nhau dấu phẩy  "a.pdf,b.docx"
        """
        h = hashlib.sha256()
        paths = _resolve_paths(source_path)
        # Sort theo tên file (không phải full path) để deterministic
        for p in sorted(paths, key=lambda x: x.name):
            if p.is_file():
                h.update(p.name.encode())           # tên file
                h.update(_sha256_file(p).encode())  # nội dung
        return h.hexdigest()

    def make_step_key(self, prev_key: str, step_cfg: dict) -> str:
        """
        step_key = SHA256(prev_key + canonical_json(step_cfg))

        Tính chất:
          - prev_key thay đổi → step_key thay đổi (propagation)
          - step_cfg thay đổi → step_key thay đổi
          - Cùng prev_key + cùng cfg → cùng step_key (deterministic)
        """
        combined = prev_key + _canonical_json(step_cfg)
        return _sha256_str(combined)

    # ── Directory helpers ─────────────────────────────────────────────────

    def _input_dir(self, input_hash: str) -> Path:
        return self.base / _short(input_hash)

    def _step_dir(self, input_hash: str, step: str, step_key: str) -> Path:
        return self._input_dir(input_hash) / step / _short(step_key)

    def has_step(self, input_hash: str, step: str, step_key: str) -> bool:
        """True nếu bước đã được cache."""
        d = self._step_dir(input_hash, step, step_key)
        return (d / "meta.json").exists()

    # ── Loader ────────────────────────────────────────────────────────────

    def save_loader(
        self,
        input_hash:    str,
        loader_key:    str,
        docs:          list,
        cfg:           dict,
        source_path:   str = "",
        input_display: str = "",
    ) -> None:
        d = self._step_dir(input_hash, "loader", loader_key)
        d.mkdir(parents=True, exist_ok=True)
        _atomic_pickle(docs, d / "docs.pkl")
        _write_meta(d, step="loader", cfg=cfg, stats={
            "n_docs": len(docs),
            "n_chars": sum(len(doc.page_content) for doc in docs),
            "source_path": source_path,
        }, parent_key=input_hash)
        # Lưu input meta — cập nhật input_display nếu có
        _write_input_meta(self._input_dir(input_hash), input_hash, source_path, input_display)
        _stamp_size(d)
        logger.debug("Cache SAVE loader %s", _short(loader_key))

    def load_loader(self, input_hash: str, loader_key: str) -> list | None:
        d = self._step_dir(input_hash, "loader", loader_key)
        pkl = d / "docs.pkl"
        if not pkl.exists():
            return None
        try:
            with open(pkl, "rb") as f:
                docs = pickle.load(f)
            logger.debug("Cache HIT  loader %s", _short(loader_key))
            return docs
        except Exception as e:
            logger.warning("Cache corrupt loader %s: %s — removing", _short(loader_key), e)
            shutil.rmtree(d, ignore_errors=True)
            return None

    # ── Chunking ──────────────────────────────────────────────────────────

    def save_chunking(
        self,
        input_hash: str,
        chunk_key: str,
        chunks: list,
        cfg: dict,
        loader_key: str = "",
    ) -> None:
        d = self._step_dir(input_hash, "chunking", chunk_key)
        d.mkdir(parents=True, exist_ok=True)
        _atomic_pickle(chunks, d / "chunks.pkl")
        _write_meta(d, step="chunking", cfg=cfg, stats={
            "n_chunks":    len(chunks),
            "avg_chars":   int(sum(len(c.page_content) for c in chunks) / max(len(chunks), 1)),
            "total_chars": sum(len(c.page_content) for c in chunks),
        }, parent_key=loader_key)
        _stamp_size(d)
        logger.debug("Cache SAVE chunking %s", _short(chunk_key))

    def load_chunking(self, input_hash: str, chunk_key: str) -> list | None:
        d = self._step_dir(input_hash, "chunking", chunk_key)
        pkl = d / "chunks.pkl"
        if not pkl.exists():
            return None
        try:
            with open(pkl, "rb") as f:
                chunks = pickle.load(f)
            logger.debug("Cache HIT  chunking %s", _short(chunk_key))
            return chunks
        except Exception as e:
            logger.warning("Cache corrupt chunking %s: %s — removing", _short(chunk_key), e)
            shutil.rmtree(d, ignore_errors=True)
            return None

    # ── Embedding ─────────────────────────────────────────────────────────

    def save_embedding(
        self,
        input_hash: str,
        embed_key: str,
        result: dict,
        cfg: dict,
        chunk_key: str = "",
    ) -> None:
        """
        result = {"dense": list[list[float]], "sparse": ..., "dims": int, ...}
        dense lưu dưới dạng .npz (float32) — tiết kiệm hơn pickle ~4×.
        """
        import numpy as np

        d = self._step_dir(input_hash, "embedding", embed_key)
        d.mkdir(parents=True, exist_ok=True)

        # Dense → .npz
        dense_arr = np.array(result["dense"], dtype=np.float32)
        np.savez_compressed(d / "dense.npz", dense=dense_arr)

        # Sparse → pickle (dict list, không dễ serialize sang npz)
        if result.get("sparse") is not None:
            _atomic_pickle(result["sparse"], d / "sparse.pkl")

        _write_meta(d, step="embedding", cfg=cfg, stats={
            "n_vectors":   result["n_embedded"],
            "dims":        result["dims"],
            "has_sparse":  result.get("sparse") is not None,
            "truncated":   result.get("truncated", False),
        }, parent_key=chunk_key)
        _stamp_size(d)
        logger.debug("Cache SAVE embedding %s", _short(embed_key))

    def load_embedding(self, input_hash: str, embed_key: str) -> dict | None:
        import numpy as np

        d = self._step_dir(input_hash, "embedding", embed_key)
        npz = d / "dense.npz"
        if not npz.exists():
            return None
        try:
            data      = np.load(npz)
            dense     = data["dense"].tolist()
            sparse    = None
            sparse_f  = d / "sparse.pkl"
            if sparse_f.exists():
                with open(sparse_f, "rb") as f:
                    sparse = pickle.load(f)
            meta      = _read_meta(d)
            stats     = meta.get("stats", {})
            logger.debug("Cache HIT  embedding %s", _short(embed_key))
            return {
                "dense":      dense,
                "sparse":     sparse,
                "dims":       stats.get("dims", len(dense[0]) if dense else 0),
                "n_embedded": stats.get("n_vectors", len(dense)),
                "truncated":  stats.get("truncated", False),
            }
        except Exception as e:
            logger.warning("Cache corrupt embedding %s: %s — removing", _short(embed_key), e)
            shutil.rmtree(d, ignore_errors=True)
            return None

    # ── Vector DB ─────────────────────────────────────────────────────────

    def save_vector_db(
        self,
        input_hash: str,
        vdb_key:    str,
        result:     dict,
        cfg:        dict,
        embed_key:  str = "",
    ) -> None:
        """
        Lưu metadata của bước Vector DB vào pipeline cache.

        Khác với các bước trước, KHÔNG lưu data thật (data đã nằm trong
        persist_dir hoặc remote DB). Chỉ lưu meta.json để đánh dấu "đã index".

        result phải chứa:
            {
                "provider":        str,    # "chroma" | "qdrant" | ...
                "collection_name": str,
                "n_vectors":       int,
            }
        """
        d = self._step_dir(input_hash, "vector_db", vdb_key)
        d.mkdir(parents=True, exist_ok=True)
        _write_meta(d, step="vector_db", cfg=cfg, stats={
            "provider":        result["provider"],
            "collection_name": result["collection_name"],
            "n_vectors":       result["n_vectors"],
            "persist_dir":     result.get("persist_dir", cfg.get("persist_dir", "")),
        }, parent_key=embed_key)
        _stamp_size(d)
        logger.debug("Cache SAVE vector_db %s", _short(vdb_key))

    def load_vector_db(self, input_hash: str, vdb_key: str) -> dict | None:
        """
        Trả về dict metadata nếu bước Vector DB đã được cache với vdb_key này.
        Trả về None nếu chưa có (cache miss → cần index lại).

        Dict trả về có dạng:
            {
                "provider":        str,
                "collection_name": str,
                "n_vectors":       int,
                "cfg":             dict,
                "saved_at":        str,
            }

        Lưu ý: hàm này chỉ kiểm tra xem bước đã chạy hay chưa.
        Việc reconnect đến vector store thật vẫn do app_visualizer.py xử lý
        (gọi get_vector_store với force_reindex=False).
        """
        d = self._step_dir(input_hash, "vector_db", vdb_key)
        if not (d / "meta.json").exists():
            return None
        try:
            meta = _read_meta(d)
            logger.debug("Cache HIT  vector_db %s", _short(vdb_key))
            return {
                **meta.get("stats", {}),
                "cfg":      meta.get("cfg", {}),
                "saved_at": meta.get("saved_at", ""),
            }
        except Exception as e:
            logger.warning("Cache corrupt vector_db %s: %s — removing", _short(vdb_key), e)
            shutil.rmtree(d, ignore_errors=True)
            return None

    # ── Cache info / management ───────────────────────────────────────────


    def list_complete_pipelines(self) -> list[dict]:
        """
        Liệt kê tất cả pipeline hoàn chỉnh (có đủ cả 4 bước).

        Mỗi pipeline = một tổ hợp cụ thể:
            input × loader_key × chunk_key × embed_key × vdb_key

        Reconstruction algorithm:
        1. Với mỗi input_dir, đọc tất cả key_dir của từng step.
        2. Với mỗi vdb_key_dir, đọc parent_key → phải khớp với một embed_key_dir.
        3. Truy ngược: embed → chunk → loader (kiểm tra parent_key chain).
        4. Chain hợp lệ → tạo một pipeline entry.

        Hỗ trợ backward compat: nếu parent_key không có (cache cũ),
        dùng heuristic: chỉ có 1 key per step → liên kết thẳng.

        Returns list[dict] — mỗi dict:
          {
            "pipeline_id":   str,            # vdb_key_short (unique)
            "input_short":   str,
            "source_path":   str,
            "created_at":    str,            # từ vdb meta (lúc hoàn tất)
            "total_size_mb": float,
            "loader":  {"key": str, "cfg": dict, "stats": dict},
            "chunking":{"key": str, "cfg": dict, "stats": dict},
            "embedding":{"key": str, "cfg": dict, "stats": dict},
            "vector_db":{"key": str, "cfg": dict, "stats": dict},
          }
        """
        pipelines = []
        if not self.base.exists():
            return pipelines

        for input_dir in sorted(self.base.iterdir()):
            if not input_dir.is_dir() or input_dir.name.startswith("."):
                continue

            meta_f = input_dir / "input_meta.json"
            input_meta = json.loads(meta_f.read_text("utf-8")) if meta_f.exists() else {}
            source_path = input_meta.get("source_path", "")

            # Collect all key dirs per step
            def _step_keys(step: str) -> list[tuple[str, dict]]:
                """Returns list of (key_short, meta_dict) for a step."""
                sdir = input_dir / step
                if not sdir.exists():
                    return []
                result = []
                for kd in sdir.iterdir():
                    if kd.is_dir():
                        m = _read_meta(kd)
                        result.append((kd.name, m))
                return result

            vdb_keys    = _step_keys("vector_db")
            embed_keys  = _step_keys("embedding")
            chunk_keys  = _step_keys("chunking")
            loader_keys = _step_keys("loader")

            if not (vdb_keys and embed_keys and chunk_keys and loader_keys):
                continue

            # Build lookup dicts: key_short → meta
            embed_by_key  = {k: m for k, m in embed_keys}
            chunk_by_key  = {k: m for k, m in chunk_keys}
            loader_by_key = {k: m for k, m in loader_keys}

            for vdb_ks, vdb_m in vdb_keys:
                # Priority 1: pipeline_chain.json manifest (written by app after indexing)
                chain_file = input_dir / "vector_db" / vdb_ks / "pipeline_chain.json"
                if chain_file.exists():
                    try:
                        chain = json.loads(chain_file.read_text("utf-8"))
                        _lks = chain.get("loader_key", "")
                        _cks = chain.get("chunk_key",  "")
                        _eks = chain.get("embed_key",  "")
                        if _lks in loader_by_key and _cks in chunk_by_key and _eks in embed_by_key:
                            lm = loader_by_key[_lks]; cm = chunk_by_key[_cks]; em = embed_by_key[_eks]
                            total_size = sum([lm.get("size_mb",0), cm.get("size_mb",0),
                                             em.get("size_mb",0), vdb_m.get("size_mb",0)])
                            sp  = chain.get("source_path", input_meta.get("source_path", ""))
                            cra = chain.get("created_at",  vdb_m.get("saved_at", ""))
                            _id = chain.get("input_display", "") or sp
                            pipelines.append({
                                "pipeline_id":   vdb_ks,
                                "input_short":   input_dir.name,
                                "source_path":   sp,
                                "input_display": _id,
                                "created_at":    cra,
                                "total_size_mb": round(total_size, 4),
                                "loader":   {"key": _lks, "cfg": lm.get("cfg",{}), "stats": lm.get("stats",{})},
                                "chunking": {"key": _cks, "cfg": cm.get("cfg",{}), "stats": cm.get("stats",{})},
                                "embedding":{"key": _eks, "cfg": em.get("cfg",{}), "stats": em.get("stats",{})},
                                "vector_db":{"key": vdb_ks,"cfg": vdb_m.get("cfg",{}),"stats": vdb_m.get("stats",{})},
                            })
                            continue
                    except Exception:
                        pass  # fall through to heuristics

                # Priority 2: parent_key chain (new format)
                vdb_parent = vdb_m.get("parent_key", "")
                embed_ks   = None

                if vdb_parent:
                    embed_ks_candidate = vdb_parent[:12]
                    if embed_ks_candidate in embed_by_key:
                        embed_ks = embed_ks_candidate
                else:
                    # Backward compat: no parent_key → if only 1 embed key, use it
                    if len(embed_by_key) == 1:
                        embed_ks = next(iter(embed_by_key))

                if embed_ks is None:
                    continue

                embed_m = embed_by_key[embed_ks]

                # Step 2: find chunking parent
                embed_parent = embed_m.get("parent_key", "")
                chunk_ks     = None

                if embed_parent:
                    chunk_ks_candidate = embed_parent[:12]
                    if chunk_ks_candidate in chunk_by_key:
                        chunk_ks = chunk_ks_candidate
                else:
                    if len(chunk_by_key) == 1:
                        chunk_ks = next(iter(chunk_by_key))

                if chunk_ks is None:
                    continue

                chunk_m = chunk_by_key[chunk_ks]

                # Step 3: find loader parent
                chunk_parent = chunk_m.get("parent_key", "")
                loader_ks    = None

                if chunk_parent:
                    loader_ks_candidate = chunk_parent[:12]
                    if loader_ks_candidate in loader_by_key:
                        loader_ks = loader_ks_candidate
                else:
                    if len(loader_by_key) == 1:
                        loader_ks = next(iter(loader_by_key))

                if loader_ks is None:
                    continue

                loader_m = loader_by_key[loader_ks]

                # Compute total size for this pipeline
                total_size = sum([
                    loader_m.get("size_mb", 0),
                    chunk_m.get("size_mb", 0),
                    embed_m.get("size_mb", 0),
                    vdb_m.get("size_mb", 0),
                ])

                pipelines.append({
                    "pipeline_id":   vdb_ks,
                    "input_short":   input_dir.name,
                    "source_path":   source_path,
                    "input_display": input_meta.get("input_display", "") or source_path,
                    "created_at":    vdb_m.get("saved_at", input_meta.get("created_at", "")),
                    "total_size_mb": round(total_size, 4),
                    "loader":        {"key": loader_ks, "cfg": loader_m.get("cfg", {}), "stats": loader_m.get("stats", {})},
                    "chunking":      {"key": chunk_ks,  "cfg": chunk_m.get("cfg",  {}), "stats": chunk_m.get("stats",  {})},
                    "embedding":     {"key": embed_ks,  "cfg": embed_m.get("cfg",  {}), "stats": embed_m.get("stats",  {})},
                    "vector_db":     {"key": vdb_ks,    "cfg": vdb_m.get("cfg",   {}), "stats": vdb_m.get("stats",   {})},
                })

        # Sort newest first
        pipelines.sort(key=lambda p: p["created_at"], reverse=True)
        return pipelines

    def list_entries(self) -> list[dict]:
        """
        Trả về list các entry trong cache.

        Hiệu năng:
        - Chỉ đọc meta.json (tiny JSON files) — KHÔNG scan file content.
        - Không gọi _dir_size_mb() per entry (tốn O(n_files) stat calls).
        - Size được lấy từ stats đã lưu trong meta.json lúc save.
        - O(n_entries × n_steps × n_keys) — thuần đọc JSON, rất nhanh.
        """
        entries = []
        if not self.base.exists():
            return entries
        for input_dir in sorted(self.base.iterdir()):
            if not input_dir.is_dir() or input_dir.name.startswith("."):
                continue
            meta_f = input_dir / "input_meta.json"
            # Đọc meta nếu có, không skip nếu thiếu (backward compat)
            meta = json.loads(meta_f.read_text("utf-8")) if meta_f.exists() else {}
            steps: dict[str, list[dict]] = {}
            total_size_mb = 0.0
            for step in STEP_NAMES:
                step_dir = input_dir / step
                if not step_dir.exists():
                    continue
                steps[step] = []
                for key_dir in sorted(step_dir.iterdir()):
                    if not key_dir.is_dir():
                        continue
                    m        = _read_meta(key_dir)
                    size_mb  = m.get("size_mb", 0.0)     # lưu lúc save
                    total_size_mb += size_mb
                    steps[step].append({
                        "key_short": key_dir.name,
                        "cfg":       m.get("cfg", {}),
                        "stats":     m.get("stats", {}),
                        "saved_at":  m.get("saved_at", ""),
                        "size_mb":   size_mb,
                    })
            # input_display: ưu tiên input_meta → pipeline_chain → fallback basename
            _id = meta.get("input_display", "")
            if not _id or _id == meta.get("source_path", ""):
                # Thử đọc từ bất kỳ pipeline_chain.json nào trong input_dir
                for _cf in input_dir.rglob("pipeline_chain.json"):
                    try:
                        _chain = json.loads(_cf.read_text("utf-8"))
                        _d = _chain.get("input_display", "")
                        _sp = _chain.get("source_path", "")
                        if _d and _d != _sp:
                            _id = _d
                            break
                    except Exception:
                        pass
            entries.append({
                "input_short":   input_dir.name,
                "full_hash":     meta.get("full_hash", ""),
                "source_path":   meta.get("source_path", ""),
                "input_display": _id,
                "created_at":    meta.get("created_at", ""),
                "steps":         steps,
                "total_size_mb": round(total_size_mb, 4),
            })
        return entries

    def total_size_mb(self) -> float:
        return _dir_size_mb(self.base)

    def clear_all(self) -> None:
        """Xoá toàn bộ cache."""
        if self.base.exists():
            shutil.rmtree(self.base)
        self.base.mkdir(parents=True, exist_ok=True)
        _write_gitignore(self.base)

    def clear_input(self, input_short: str) -> None:
        """Xoá cache của một input cụ thể."""
        d = self.base / input_short
        if d.exists():
            shutil.rmtree(d)

    def clear_step(self, input_short: str, step: str, key_short: str) -> None:
        """Xoá cache của một step+key cụ thể."""
        d = self.base / input_short / step / key_short
        if d.exists():
            shutil.rmtree(d)

    def prune_old(self, max_age_days: int = 30) -> int:
        """Xoá các entry cũ hơn max_age_days ngày. Trả về số entry đã xoá."""
        cutoff = time.time() - max_age_days * 86_400
        removed = 0
        for input_dir in list(self.base.iterdir()):
            if not input_dir.is_dir():
                continue
            if input_dir.stat().st_mtime < cutoff:
                shutil.rmtree(input_dir, ignore_errors=True)
                removed += 1
        return removed


# ── Private helpers ───────────────────────────────────────────────────────────

def _resolve_paths(source_path: str) -> list[Path]:
    """Trả về list[Path] từ source_path (file, dir, hoặc comma-separated)."""
    paths: list[Path] = []
    for part in source_path.split(","):
        p = Path(part.strip())
        if p.is_dir():
            paths.extend(p.rglob("*"))
        elif p.is_file():
            paths.append(p)
    return [p for p in paths if p.is_file()]


def _atomic_pickle(obj: Any, dest: Path) -> None:
    """Write pickle atomically: write to .tmp → rename."""
    tmp = dest.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(dest)


def _write_meta(d: Path, step: str, cfg: dict, stats: dict, parent_key: str = "") -> None:
    meta = {
        "step":       step,
        "cfg":        cfg,
        "stats":      stats,
        "saved_at":   datetime.now().isoformat(timespec="seconds"),
        "parent_key": parent_key,
    }
    data = json.dumps(meta, ensure_ascii=False, indent=2)
    dest = d / "meta.json"
    tmp  = dest.with_suffix(".tmp")
    tmp.write_text(data, encoding="utf-8")
    tmp.replace(dest)   # atomic rename — safe on crash


def _stamp_size(d: Path) -> None:
    """Cập nhật size_mb vào meta.json sau khi tất cả files đã được ghi.
    Gọi sau khi save_* hoàn tất — chỉ đọc meta rồi ghi lại với size thêm vào.
    """
    meta_f = d / "meta.json"
    if not meta_f.exists():
        return
    try:
        meta = json.loads(meta_f.read_text("utf-8"))
        meta["size_mb"] = round(
            sum(f.stat().st_size for f in d.rglob("*") if f.is_file()) / (1024**2), 4
        )
        tmp = meta_f.with_suffix(".tmp")
        tmp.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(meta_f)
    except Exception:
        pass   # non-critical — list_entries sẽ dùng 0.0 nếu không có


def _read_meta(d: Path) -> dict:
    f = d / "meta.json"
    if not f.exists():
        return {}
    try:
        return json.loads(f.read_text("utf-8"))
    except Exception:
        return {}


def _write_input_meta(
    input_dir:     Path,
    full_hash:     str,
    source_path:   str,
    input_display: str = "",
) -> None:
    f = input_dir / "input_meta.json"
    input_dir.mkdir(parents=True, exist_ok=True)
    # Merge với dữ liệu cũ nếu đã tồn tại (giữ created_at ban đầu)
    existing: dict = {}
    if f.exists():
        try:
            existing = json.loads(f.read_text("utf-8"))
        except Exception:
            existing = {}
    meta = {
        "full_hash":     full_hash,
        "short_hash":    full_hash[:KEY_LEN],
        "source_path":   source_path,
        "input_display": input_display or existing.get("input_display", "") or source_path,
        "created_at":    existing.get("created_at", datetime.now().isoformat(timespec="seconds")),
    }
    tmp = input_dir / "input_meta.tmp"
    tmp.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(f)


def _dir_size_mb(d: Path) -> float:
    if not d.exists():
        return 0.0
    total = sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
    return round(total / (1024 ** 2), 2)


def _write_gitignore(base: Path) -> None:
    gi = base / ".gitignore"
    if not gi.exists():
        gi.write_text(
            "# Auto-generated by pipeline_cache.py\n"
            "# Ignore large processed data files\n"
            "*.pkl\n"
            "*.npz\n",
            encoding="utf-8",
        )
