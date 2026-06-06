"""
pipeline/indexing_pipeline.py
==============================
IndexingPipeline — Stage 1 của RAG pipeline.

Chạy offline, thông thường một lần duy nhất trước khi deploy ứng dụng.

Luồng:
    source (file / thư mục)
        → Loader     (parse PDF → list[Document])
        → Chunker    (cắt thành chunk nhỏ)
        → Embedder   (dense + sparse vectors)
        → VectorStore (lưu index)

Có thể dùng trực tiếp từ CLI mà không cần Streamlit:

    from pipeline.indexing_pipeline import IndexingPipeline
    from pipeline_cache import PipelineCache

    pipe = IndexingPipeline(config_path="config.yaml")
    result = pipe.run(source_path="/data/papers/")

Hoặc:

    python -m pipeline.indexing_pipeline --source /data/papers/ --config config.yaml
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Kết quả từng bước & toàn bộ pipeline
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class StepResult:
    """Kết quả một bước indexing."""
    step:       str
    success:    bool
    from_cache: bool        = False
    n_items:    int         = 0
    meta:       dict        = field(default_factory=dict)
    error:      str | None  = None


@dataclass
class IndexingResult:
    """Kết quả toàn bộ IndexingPipeline.run()."""
    success:    bool
    steps:      list[StepResult]  = field(default_factory=list)
    docs:       list[Document]    = field(default_factory=list)
    chunks:     list[Document]    = field(default_factory=list)
    embed_result: dict            = field(default_factory=dict)
    vdb_result:   dict            = field(default_factory=dict)
    error:      str | None        = None

    @property
    def n_docs(self)    -> int: return len(self.docs)
    @property
    def n_chunks(self)  -> int: return len(self.chunks)
    @property
    def n_vectors(self) -> int: return self.embed_result.get("n_embedded", 0)


# ──────────────────────────────────────────────────────────────────────────────
# Config keys loại ra khỏi cache fingerprint
# (các param này không ảnh hưởng đến *giá trị* output)
# ──────────────────────────────────────────────────────────────────────────────

_LOADER_SKIP_CACHE_KEYS  = {"ollama_base_url"}
_CHUNK_SKIP_CACHE_KEYS   = {"ollama_base_url"}
_EMBED_SKIP_CACHE_KEYS   = {
    "skip", "dims", "max_preview", "ollama_base_url",
    "device", "torch_dtype_str", "batch_size",
}
_VDB_SKIP_CACHE_KEYS = {
    "skip", "force_reindex",
    "url", "uri", "redis_url", "connection_string",
    "endpoint", "api_key", "user", "password", "token",
}


# ──────────────────────────────────────────────────────────────────────────────
# IndexingPipeline
# ──────────────────────────────────────────────────────────────────────────────

class IndexingPipeline:
    """
    Stage 1 — Indexing Pipeline.

    Tham số
    -------
    cache_dir   : Thư mục lưu step-level cache (mặc định "./processed_data").
    on_progress : Callback ``fn(step, pct, msg)`` — dùng để update UI progress bar.
                  ``step``  ∈ {"loading","chunking","embedding","vector_db"}
                  ``pct``   float 0.0–1.0
                  ``msg``   str mô tả hiện tại
    stop_event  : threading.Event — set() để yêu cầu dừng giữa chừng.
    """

    def __init__(
        self,
        cache_dir:   str                        = "processed_data",
        on_progress: Callable[[str, float, str], None] | None = None,
        stop_event:  threading.Event | None     = None,
    ):
        from pipeline_cache import PipelineCache
        self.cache       = PipelineCache(cache_dir)
        self.on_progress = on_progress or (lambda *_: None)
        self.stop_event  = stop_event or threading.Event()

    # ── Public ────────────────────────────────────────────────────────────────

    def run(
        self,
        source_path:  str,
        loader_cfg:   dict,
        chunking_cfg: dict,
        embed_cfg:    dict,
        vdb_cfg:      dict,
    ) -> IndexingResult:
        """
        Chạy toàn bộ indexing pipeline.

        Tham số
        -------
        source_path  : Đường dẫn file PDF hoặc thư mục chứa PDF.
        loader_cfg   : Dict cấu hình loader (từ render_loader_settings hoặc config.yaml).
        chunking_cfg : Dict cấu hình chunking.
                       Phải có keys: strategy, chunk_size, chunk_overlap, extra_kwargs.
        embed_cfg    : Dict cấu hình embedding.
        vdb_cfg      : Dict cấu hình vector DB.

        Trả về
        ------
        IndexingResult với docs, chunks, embed_result, vdb_result.
        """
        result   = IndexingResult(success=False)
        cache    = self.cache
        stop     = self.stop_event

        # ── Tính cache fingerprint chain ──────────────────────────────────────
        input_hash = cache.compute_input_hash(source_path)

        loader_key = cache.make_step_key(
            input_hash,
            {k: v for k, v in loader_cfg.items() if k not in _LOADER_SKIP_CACHE_KEYS},
        )
        strategy     = chunking_cfg["strategy"]
        chunk_size   = chunking_cfg["chunk_size"]
        chunk_overlap = chunking_cfg["chunk_overlap"]
        extra_kwargs  = chunking_cfg.get("extra_kwargs", {})
        chunk_key = cache.make_step_key(
            loader_key,
            {
                "strategy": strategy, "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                **{k: v for k, v in extra_kwargs.items() if k not in _CHUNK_SKIP_CACHE_KEYS},
            },
        )
        embed_key = cache.make_step_key(
            chunk_key,
            {k: v for k, v in embed_cfg.items() if k not in _EMBED_SKIP_CACHE_KEYS},
        )
        vdb_key = cache.make_step_key(
            embed_key,
            {k: v for k, v in vdb_cfg.items() if k not in _VDB_SKIP_CACHE_KEYS},
        )

        # ── Bước 1: Loader ────────────────────────────────────────────────────
        step_load = self._run_loader(
            source_path, loader_cfg, input_hash, loader_key
        )
        result.steps.append(step_load)
        if not step_load.success:
            result.error = step_load.error
            return result
        result.docs = step_load.meta.get("docs", [])

        if stop.is_set():
            result.error = "Dừng sau bước Loading."
            return result

        # ── Bước 2: Chunking ──────────────────────────────────────────────────
        step_chunk = self._run_chunking(
            result.docs, chunking_cfg, input_hash, chunk_key
        )
        result.steps.append(step_chunk)
        if not step_chunk.success:
            result.error = step_chunk.error
            return result
        result.chunks = step_chunk.meta.get("chunks", [])

        if stop.is_set():
            result.error = "Dừng sau bước Chunking."
            return result

        # ── Bước 3: Embedding ─────────────────────────────────────────────────
        if embed_cfg.get("skip"):
            logger.info("Embedding bị skip.")
        else:
            step_embed = self._run_embedding(
                result.chunks, embed_cfg, input_hash, embed_key
            )
            result.steps.append(step_embed)
            if not step_embed.success:
                result.error = step_embed.error
                return result
            result.embed_result = step_embed.meta.get("embed_result", {})

            if stop.is_set():
                result.error = "Dừng sau bước Embedding."
                return result

            # ── Bước 4: Vector DB ─────────────────────────────────────────────
            if not vdb_cfg.get("skip"):
                step_vdb = self._run_vector_db(
                    result.chunks, embed_cfg, vdb_cfg,
                    input_hash, embed_key, vdb_key,
                )
                result.steps.append(step_vdb)
                if not step_vdb.success:
                    result.error = step_vdb.error
                    return result
                result.vdb_result = step_vdb.meta.get("vdb_result", {})

        result.success = True
        return result

    # ── Private step runners ──────────────────────────────────────────────────

    def _run_loader(
        self, source_path: str, loader_cfg: dict,
        input_hash: str, loader_key: str,
    ) -> StepResult:
        self.on_progress("loading", 0.0, "Đang kiểm tra cache loader...")

        cached = self.cache.load_loader(input_hash, loader_key)
        if cached is not None:
            self.on_progress("loading", 1.0, f"Loading từ cache ({len(cached)} docs)")
            logger.info("Loader cache HIT: %d docs", len(cached))
            return StepResult("loading", True, from_cache=True, n_items=len(cached),
                               meta={"docs": cached})

        self.on_progress("loading", 0.05, "Đang đọc tài liệu...")
        try:
            from app_visualizer import run_loader
            docs = run_loader(
                source_path      = source_path,
                pdf_strategy     = loader_cfg["pdf_strategy"],
                extract_tables   = loader_cfg["extract_tables"],
                language         = loader_cfg["language"],
                marker_device    = loader_cfg.get("marker_device", "cpu"),
                describe_images  = loader_cfg.get("describe_images", False),
                vision_model     = loader_cfg.get("vision_model", ""),
                vision_provider  = loader_cfg.get("vision_provider", "openai"),
                ollama_base_url  = loader_cfg.get("ollama_base_url", "http://localhost:11434"),
                odl_hybrid       = loader_cfg.get("odl_hybrid"),
                odl_struct_tree  = loader_cfg.get("odl_struct_tree", False),
            )
        except Exception as e:
            logger.exception("Loader lỗi")
            return StepResult("loading", False, error=str(e))

        self.cache.save_loader(input_hash, loader_key, docs,
                               {k: v for k, v in loader_cfg.items()
                                if k not in _LOADER_SKIP_CACHE_KEYS},
                               source_path)
        self.on_progress("loading", 1.0, f"Loading hoàn tất: {len(docs)} documents")
        logger.info("Loader: %d docs", len(docs))
        return StepResult("loading", True, n_items=len(docs), meta={"docs": docs})

    def _run_chunking(
        self, docs: list[Document], chunking_cfg: dict,
        input_hash: str, chunk_key: str,
    ) -> StepResult:
        self.on_progress("chunking", 0.0, "Đang kiểm tra cache chunking...")

        cached = self.cache.load_chunking(input_hash, chunk_key)
        if cached is not None:
            self.on_progress("chunking", 1.0, f"Chunking từ cache ({len(cached)} chunks)")
            return StepResult("chunking", True, from_cache=True, n_items=len(cached),
                               meta={"chunks": cached})

        self.on_progress("chunking", 0.1, "Đang chunking...")
        try:
            from app_visualizer import run_chunker
            chunks = run_chunker(
                docs          = docs,
                strategy      = chunking_cfg["strategy"],
                chunk_size    = chunking_cfg["chunk_size"],
                chunk_overlap = chunking_cfg["chunk_overlap"],
                extra_kwargs  = chunking_cfg.get("extra_kwargs", {}),
            )
        except Exception as e:
            logger.exception("Chunker lỗi")
            return StepResult("chunking", False, error=str(e))

        cfg_for_cache = {
            "strategy": chunking_cfg["strategy"],
            "chunk_size": chunking_cfg["chunk_size"],
            "chunk_overlap": chunking_cfg["chunk_overlap"],
            **{k: v for k, v in chunking_cfg.get("extra_kwargs", {}).items()
               if k not in _CHUNK_SKIP_CACHE_KEYS},
        }
        self.cache.save_chunking(input_hash, chunk_key, chunks, cfg_for_cache)
        self.on_progress("chunking", 1.0, f"Chunking hoàn tất: {len(chunks)} chunks")
        return StepResult("chunking", True, n_items=len(chunks), meta={"chunks": chunks})

    def _run_embedding(
        self, chunks: list[Document], embed_cfg: dict,
        input_hash: str, embed_key: str,
    ) -> StepResult:
        from app_visualizer import EMBED_PREVIEW_LIMIT
        self.on_progress("embedding", 0.0, "Đang kiểm tra cache embedding...")

        cached = self.cache.load_embedding(input_hash, embed_key)
        if cached is not None:
            self.on_progress("embedding", 1.0,
                             f"Embedding từ cache ({cached['n_embedded']} vectors)")
            return StepResult("embedding", True, from_cache=True,
                               n_items=cached["n_embedded"],
                               meta={"embed_result": cached})

        self.on_progress("embedding", 0.05, "Đang tính embedding vectors...")
        try:
            from app_visualizer import run_embedder
            embed_result = run_embedder(
                chunks               = chunks,
                provider             = embed_cfg["provider"],
                model_name           = embed_cfg["model_name"],
                enable_sparse        = embed_cfg["enable_sparse"],
                sparse_method        = embed_cfg["sparse_method"],
                dimensions           = embed_cfg.get("dimensions"),
                device               = embed_cfg.get("device", "cpu"),
                ollama_base_url      = embed_cfg.get("ollama_base_url", "http://localhost:11434"),
                input_type           = embed_cfg.get("input_type", "search_document"),
                query_instruction    = embed_cfg.get("query_instruction"),
                document_instruction = embed_cfg.get("document_instruction"),
                max_chunks           = embed_cfg.get("max_preview", EMBED_PREVIEW_LIMIT),
                torch_dtype_str      = embed_cfg.get("torch_dtype_str", "auto"),
                batch_size           = embed_cfg.get("batch_size", 32),
            )
        except Exception as e:
            logger.exception("Embedder lỗi")
            return StepResult("embedding", False, error=str(e))

        cfg_for_cache = {k: v for k, v in embed_cfg.items()
                         if k not in _EMBED_SKIP_CACHE_KEYS}
        self.cache.save_embedding(input_hash, embed_key, embed_result, cfg_for_cache)
        self.on_progress("embedding", 1.0,
                         f"Embedding hoàn tất: {embed_result['n_embedded']} vectors")
        return StepResult("embedding", True, n_items=embed_result["n_embedded"],
                           meta={"embed_result": embed_result})

    def _run_vector_db(
        self,
        chunks: list[Document],
        embed_cfg: dict,
        vdb_cfg: dict,
        input_hash: str,
        embed_key: str,
        vdb_key: str,
    ) -> StepResult:
        self.on_progress("vector_db", 0.0, "Đang kiểm tra cache Vector DB...")

        cached_vdb = self.cache.load_vector_db(input_hash, vdb_key)
        if cached_vdb is not None and not vdb_cfg.get("force_reindex"):
            # Cache hit: reconnect đến DB hiện có
            try:
                from app_visualizer import _embedder_kwargs_from_cfg, run_vector_db
                embedder_kwargs = _embedder_kwargs_from_cfg(embed_cfg)
                vdb_result = run_vector_db(
                    chunks          = chunks,
                    embedder_kwargs = embedder_kwargs,
                    sparse_index    = None,
                    vdb_cfg         = vdb_cfg,
                    force_reindex   = False,
                )
                self.on_progress("vector_db", 1.0,
                                 f"Vector DB từ cache ({vdb_result.get('n_vectors',0):,} vectors)")
                return StepResult("vector_db", True, from_cache=True,
                                   n_items=vdb_result.get("n_vectors", 0),
                                   meta={"vdb_result": vdb_result})
            except Exception as e:
                logger.warning("Vector DB reconnect thất bại, reindex: %s", e)

        self.on_progress("vector_db", 0.05, "Đang index vào Vector DB...")
        try:
            from app_visualizer import _embedder_kwargs_from_cfg, run_vector_db
            embedder_kwargs = _embedder_kwargs_from_cfg(embed_cfg)
            vdb_result = run_vector_db(
                chunks          = chunks,
                embedder_kwargs = embedder_kwargs,
                sparse_index    = None,
                vdb_cfg         = vdb_cfg,
                force_reindex   = vdb_cfg.get("force_reindex", False),
            )
        except Exception as e:
            logger.exception("Vector DB lỗi")
            return StepResult("vector_db", False, error=str(e))

        cfg_for_cache = {k: v for k, v in vdb_cfg.items()
                         if k not in _VDB_SKIP_CACHE_KEYS}
        self.cache.save_vector_db(input_hash, vdb_key, vdb_result, cfg_for_cache)
        self.on_progress("vector_db", 1.0,
                         f"Vector DB hoàn tất: {vdb_result.get('n_vectors',0):,} vectors")
        return StepResult("vector_db", True, n_items=vdb_result.get("n_vectors", 0),
                           meta={"vdb_result": vdb_result})

    # ── Cache helpers ─────────────────────────────────────────────────────────

    def make_cache_keys(
        self,
        source_path:  str,
        loader_cfg:   dict,
        chunking_cfg: dict,
        embed_cfg:    dict,
        vdb_cfg:      dict,
    ) -> dict[str, str]:
        """
        Trả về dict {step: cache_key} mà không chạy pipeline.
        Dùng để kiểm tra cache status trên UI.
        """
        cache = self.cache
        strategy     = chunking_cfg["strategy"]
        chunk_size   = chunking_cfg["chunk_size"]
        chunk_overlap = chunking_cfg["chunk_overlap"]
        extra_kwargs  = chunking_cfg.get("extra_kwargs", {})

        input_hash = cache.compute_input_hash(source_path)
        loader_key = cache.make_step_key(
            input_hash, {k: v for k, v in loader_cfg.items()
                         if k not in _LOADER_SKIP_CACHE_KEYS})
        chunk_key = cache.make_step_key(
            loader_key, {
                "strategy": strategy, "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                **{k: v for k, v in extra_kwargs.items()
                   if k not in _CHUNK_SKIP_CACHE_KEYS},
            })
        embed_key = cache.make_step_key(
            chunk_key, {k: v for k, v in embed_cfg.items()
                        if k not in _EMBED_SKIP_CACHE_KEYS})
        vdb_key = cache.make_step_key(
            embed_key, {k: v for k, v in vdb_cfg.items()
                        if k not in _VDB_SKIP_CACHE_KEYS})
        return {
            "input_hash": input_hash,
            "loader":     loader_key,
            "chunking":   chunk_key,
            "embedding":  embed_key,
            "vector_db":  vdb_key,
        }


# ──────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────────────────────

def _cli():
    import argparse, yaml, json, sys

    parser = argparse.ArgumentParser(
        description="RAG Indexing Pipeline — chạy offline để tạo index."
    )
    parser.add_argument("--source",   required=True, help="Đường dẫn file PDF hoặc thư mục")
    parser.add_argument("--config",   default="config.yaml", help="File cấu hình YAML")
    parser.add_argument("--verbose",  action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        print(f"❌ Không tìm thấy config: {cfg_path}", file=sys.stderr)
        sys.exit(1)

    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Build cfg dicts từ config.yaml
    loader_cfg = {
        "pdf_strategy":   cfg["loader"]["pdf_strategy"],
        "extract_tables": cfg["loader"].get("extract_tables", True),
        "language":       cfg["data"].get("language", "both"),
        "describe_images": cfg["loader"].get("describe_images", False),
        "vision_model":   cfg["loader"].get("vision_model", ""),
        "vision_provider": cfg["loader"].get("vision_provider", "openai"),
        "ollama_base_url": cfg["loader"].get("ollama_base_url", "http://localhost:11434"),
        "marker_device":  cfg["loader"].get("marker_device", "cpu"),
    }
    chunking_cfg = {
        "strategy":      cfg["chunking"]["strategy"],
        "chunk_size":    cfg["chunking"].get("chunk_size", 1000),
        "chunk_overlap": cfg["chunking"].get("chunk_overlap", 150),
        "extra_kwargs":  {},
    }
    embed_cfg = {
        "provider":       cfg["embedding"]["provider"],
        "model_name":     cfg["embedding"]["model_name"],
        "enable_sparse":  cfg["embedding"].get("sparse_method", "none") != "none",
        "sparse_method":  cfg["embedding"].get("sparse_method", "none"),
        "skip":           False,
    }
    vdb_cfg = {
        "provider":        cfg["vector_db"]["provider"],
        "collection_name": cfg["vector_db"].get("collection_name", "rag_docs"),
        "persist_directory": cfg["vector_db"].get("persist_directory", "./storage"),
        "skip":            False,
        "force_reindex":   False,
    }

    def _progress(step: str, pct: float, msg: str):
        bar = "█" * int(pct * 20) + "░" * (20 - int(pct * 20))
        print(f"  [{bar}] {pct*100:3.0f}% | {step:12s} | {msg}")

    pipe   = IndexingPipeline(on_progress=_progress)
    result = pipe.run(
        source_path  = args.source,
        loader_cfg   = loader_cfg,
        chunking_cfg = chunking_cfg,
        embed_cfg    = embed_cfg,
        vdb_cfg      = vdb_cfg,
    )

    print()
    if result.success:
        print("✅ Indexing hoàn tất!")
        print(f"   Documents : {result.n_docs}")
        print(f"   Chunks    : {result.n_chunks}")
        print(f"   Vectors   : {result.n_vectors}")
        for s in result.steps:
            cache_tag = " [cache]" if s.from_cache else ""
            print(f"   {s.step:12s}: {s.n_items} items{cache_tag}")
    else:
        print(f"❌ Indexing thất bại: {result.error}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    _cli()
