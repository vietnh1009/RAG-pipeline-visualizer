import os, sys, re, importlib, importlib.util, inspect, tempfile
from pathlib import Path
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
try:
    from dotenv import load_dotenv as _ld, dotenv_values as _dv
    _ld(override=True); _ENV = _dv()
except ImportError:
    _ENV = {}
import streamlit as st

from core.constants import EMBED_PREVIEW_LIMIT, VECTOR_DB_PROVIDER_META
from utils.env import _get_pipeline_cache, save_uploaded_files, _init_uploads_dir, _get_file_profile
from core.pipeline_runners import run_loader, run_chunker, run_embedder
from core.cache_helpers import (
    _cached_list_entries, _invalidate_list_entries_cache,
    _embedder_kwargs_from_cfg,
)
from pipeline_cache import PipelineCache
from ui.settings.loader    import render_loader_settings
from ui.settings.chunking  import render_chunking_settings
from ui.settings.embedding import render_embedding_settings
from ui.settings.vector_db import render_vector_db_settings
from ui.results.loader_results    import render_loader_results
from ui.results.chunking_results  import render_chunking_results
from ui.results.embedding_results import render_embedding_results
from ui.results.vdb_results       import render_vector_db_results, render_query_pipeline_results
from ui.components.pipeline_suggestions import get_pipeline_suggestions, render_pipeline_suggestions

def _write_pipeline_chain(cache, input_hash: str,
                           loader_key: str, chunk_key: str,
                           embed_key: str, vdb_key: str,
                           source_path: str,
                           input_display: str = "") -> None:
    """
    Ghi pipeline_chain.json vào vdb_key dir.
    Dùng bởi list_complete_pipelines() để reconstruct chain
    ngay cả khi meta.json không có parent_key (backward compat).
    """
    import json, datetime
    from pathlib import Path
    from pipeline_cache import _short
    vdb_dir = cache._step_dir(input_hash, "vector_db", vdb_key)
    chain = {
        "input_hash":  input_hash,
        "loader_key":  loader_key[:12],
        "chunk_key":   chunk_key[:12],
        "embed_key":   embed_key[:12],
        "vdb_key":     vdb_key[:12],
        "source_path": source_path,
        "input_display": input_display or source_path,
        "created_at":  datetime.datetime.now().isoformat(timespec="seconds"),
    }
    try:
        vdb_dir.mkdir(parents=True, exist_ok=True)
        tmp = vdb_dir / "pipeline_chain.tmp"
        dst = vdb_dir / "pipeline_chain.json"
        tmp.write_text(json.dumps(chain, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(dst)
    except Exception as _e:
        # Log to stderr nhưng không crash pipeline
        import sys
        print(f"[WARN] _write_pipeline_chain failed: {_e}", file=sys.stderr)


def _run_pipeline_and_show_progress() -> None:
    """
    Chạy indexing pipeline và hiển thị progress bars.
    Được gọi khi _pipeline_running=True.
    Đọc toàn bộ config từ session_state thay vì sidebar widgets
    (để tránh Streamlit rerun do widget re-render).
    """
    import threading, concurrent.futures
    import time as _time

    # Minimal sidebar: chỉ hiện thông tin + nút Stop
    with st.sidebar:
        st.warning("⏳ **Đang xử lý Indexing...**\nVui lòng chờ hoặc ấn Stop.", icon="⚠️")

    # Lấy config từ session_state (đã lưu khi Process được bấm)
    source_path   = st.session_state.get("_last_source_path")
    loader_cfg    = st.session_state.get("_last_loader_cfg", {})
    strategy      = st.session_state.get("_last_strategy", "recursive")
    chunk_size    = st.session_state.get("_last_chunk_size", 1000)
    chunk_overlap = st.session_state.get("_last_chunk_overlap", 150)
    extra_kwargs  = st.session_state.get("_last_extra_kwargs", {})
    emb_cfg       = st.session_state.get("_last_emb_cfg", {})
    vdb_cfg       = st.session_state.get("_last_vdb_cfg", {})

    if not source_path:
        st.session_state["_pipeline_running"] = False
        st.error("❌ Mất thông tin source path. Vui lòng chọn file và chạy lại.")
        st.rerun()
        return

    # Dummy process_btn = False (pipeline đã được trigger từ trước)
    process_btn = False

    if st.session_state.get("_pipeline_running") and not process_btn:
        import threading, concurrent.futures
        import time as _time

        # Guard: source_path có thể bị None nếu user chuyển tab rồi quay lại
        if source_path is None:
            source_path = st.session_state.get("_last_source_path")
        if source_path is None:
            st.session_state["_pipeline_running"] = False
            st.error("❌ Mất thông tin source path. Vui lòng chọn file và chạy lại.")
            st.rerun()

        # Restore configs từ session nếu bị mất khi chuyển tab
        if not loader_cfg.get("pdf_strategy"):
            loader_cfg = st.session_state.get("_last_loader_cfg", loader_cfg)
        if not strategy:
            strategy      = st.session_state.get("_last_strategy", strategy)
            chunk_size    = st.session_state.get("_last_chunk_size", chunk_size)
            chunk_overlap = st.session_state.get("_last_chunk_overlap", chunk_overlap)
            extra_kwargs  = st.session_state.get("_last_extra_kwargs", extra_kwargs)
        if not emb_cfg.get("provider"):
            emb_cfg = st.session_state.get("_last_emb_cfg", emb_cfg)
        if not vdb_cfg.get("provider"):
            vdb_cfg = st.session_state.get("_last_vdb_cfg", vdb_cfg)

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
            cache.save_loader(input_hash, loader_key, docs, loader_cfg_for_cache, source_path,
                               input_display=st.session_state.get("_last_source_display", ""))
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
                cache.save_chunking(input_hash, chunk_key, chunks, chunk_cfg_for_cache, loader_key=loader_key)
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
                # Trim dense preview khi load từ cache (nhất quán với fresh-embed path)
                from core.constants import EMBED_PREVIEW_LIMIT
                _cd = cached_embed.get("dense", [])
                _cs = cached_embed.get("sparse")
                st.session_state["embed_result"] = {
                    **{k: v for k, v in cached_embed.items() if k not in ("dense","sparse")},
                    "dense":  _cd[:EMBED_PREVIEW_LIMIT] if _cd else _cd,
                    "sparse": _cs[:EMBED_PREVIEW_LIMIT] if _cs else _cs,
                }
                st.session_state["emb_cfg_used"] = emb_cfg
                st.session_state["active_tab"] = 2
                st.success(
                    f"⚡ Bước 3/{_n_steps} — Embedding từ cache "
                    f"({cached_embed['n_embedded']} vectors · {cached_embed['dims']}d · "
                    f"{cache._step_dir(input_hash, 'embedding', embed_key).name})"
                )
            else:
                prog_embed = st.progress(0, text=f"Bước 3/{_n_steps} — Đang embedding...")
                stop_embed_placeholder = st.empty()

                def _do_embed():
                    from core.constants import EMBED_PREVIEW_LIMIT  # explicit import in thread scope
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
                            text=f"Bước 3/{_n_steps} — Đang embedding... ({tick}s)"
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
                cache.save_embedding(input_hash, embed_key, embed_result, embed_cfg_for_cache, chunk_key=chunk_key)
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

            # Guard: không có chunks → không thể index VDB
            _nonempty_chunks = [c for c in chunks_for_vdb if c.page_content.strip()]
            if not _nonempty_chunks:
                st.error(
                    f"❌ **0 chunks có nội dung** (tổng {len(chunks_for_vdb)} chunks đều rỗng) — "
                    "không thể index Vector DB.\n\n"
                    "**Nguyên nhân phổ biến với OpenDataLoader:**\n"
                    "- Document trả về nội dung rỗng (blank page_content)\n"
                    "- Server ODL hybrid chưa xử lý xong hoặc trả về lỗi\n"
                    "- File PDF bị scan/image-only, cần bật **Force OCR** trong ODL settings\n\n"
                    "**Cách fix:**\n"
                    "1. Kiểm tra tab **Loader** xem document có nội dung không\n"
                    "2. Thử lại với **Force OCR** bật (nếu dùng ODL hybrid)\n"
                    "3. Thử PDF strategy khác (docling, pypdf) để so sánh",
                    icon="❌",
                )
                st.session_state["_pipeline_running"] = False
                st.stop()

            # Pre-truncate cho Ollama/HF models để tránh context length error
            _hard_cap_vdb = 32000
            if emb_cfg.get("provider") in ("ollama", "huggingface", "fastembed"):
                from langchain_core.documents import Document as _Doc
                _n_truncated = sum(
                    1 for c in chunks_for_vdb if len(c.page_content) > _hard_cap_vdb
                )
                if _n_truncated > 0:
                    st.warning(
                        f"⚠️ **{_n_truncated}/{len(chunks_for_vdb)} chunk bị truncate** "
                        f"xuống {_hard_cap_vdb:,} ký tự (~{_hard_cap_vdb//4:,} tokens) "
                        f"do model **{emb_cfg.get('provider')} / {emb_cfg.get('model_name','').split('/')[-1]}** "
                        f"có giới hạn context window.\n\n"
                        f"💡 Để tránh mất thông tin: giảm **Chunk size** (≤ {_hard_cap_vdb//5:,} chars) "
                        f"hoặc chuyển sang model có context window lớn hơn (vd: `bge-m3` hỗ trợ 8192 tokens).",
                        icon="⚠️",
                    )
                chunks_for_vdb = [
                    _Doc(page_content=c.page_content[:_hard_cap_vdb], metadata=c.metadata)
                    if len(c.page_content) > _hard_cap_vdb else c
                    for c in chunks_for_vdb
                ]

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
                        "vector_store":         _store,  # alias for load path
                            "n_vectors":            cached_vdb.get("n_vectors", len(chunks_for_vdb)),
                            "collection_name":      cached_vdb.get("collection_name", vdb_cfg.get("collection_name", "rag")),
                            "loaded_from_existing": True,
                        }
                    except Exception as _e:
                        st.warning(f"⚠️ Không thể kết nối lại VDB cache: {_e}. Sẽ index lại.")
                        cached_vdb = None   # fall through to re-index below

                if cached_vdb is not None:
                    # Ghi chain manifest ngay cả khi cache hit
                    # → list_complete_pipelines luôn tìm thấy pipeline này
                    _write_pipeline_chain(
                        cache       = cache,
                        input_hash  = input_hash,
                        loader_key  = loader_key,
                        chunk_key   = chunk_key,
                        embed_key   = embed_key,
                        vdb_key     = vdb_key,
                        source_path = source_path,
                    )
                    st.session_state["vdb_result"]   = vdb_result
                    st.session_state["vdb_cfg_used"] = vdb_cfg
                    st.session_state["active_tab"] = 3
                    st.success(
                        f"⚡ Bước 4/{_n_steps} — VDB từ cache "
                        f"({vdb_result['n_vectors']:,} vectors · "
                        f"{vdb_cfg['provider']}:{vdb_cfg.get('collection_name','rag')} · "
                        f"{cache._step_dir(input_hash, 'vector_db', vdb_key).name})"
                    )
                    # Lưu pipeline meta cho "Index đang dùng" display
                    st.session_state["_current_pipeline_meta"] = {
                        "source_path":  source_path,
                        "input_display": st.session_state.get("_last_source_display", ""),
                        "created_at":   __import__("datetime").datetime.now().strftime("%Y-%m-%d %H:%M"),
                        "loader_cfg":   loader_cfg,
                        "chunking_cfg": {
                            "strategy":      strategy,
                            "chunk_size":    chunk_size,
                            "chunk_overlap": chunk_overlap,
                        },
                        "embed_cfg":    emb_cfg,
                        "vdb_cfg":      vdb_cfg,
                        "n_chunks":     len(st.session_state.get("chunks", [])),
                        "n_vectors":    vdb_result.get("n_vectors", 0),
                    }
                    _invalidate_list_entries_cache()

            # ── Cache miss hoặc force_reindex: index từ đầu ───────────────
            if cached_vdb is None or vdb_cfg.get("force_reindex"):
                prog_vdb = st.progress(0, text=f"Bước 4/{_n_steps} — Đang index vào Vector DB...")
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
                        chunks=chunks_for_vdb,  # đã truncate trước rồi
                        embedder=_embedder,
                        force_reindex=_force,
                        **_conn_kwargs,
                    )
                    return {
                        "store":                _store,
                        "vector_store":         _store,  # alias for load path
                        "n_vectors":            len(chunks_for_vdb),
                        "collection_name":      vdb_cfg.get("collection_name", "rag"),
                        "provider":             _provider,
                        "persist_dir":          vdb_cfg.get("persist_dir", ""),
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
                            text=f"Bước 4/{_n_steps} — Đang index vào Vector DB... ({tick}s)"
                        )
                        _time.sleep(0.5); tick += 1
                    try:
                        vdb_result = future.result()
                    except Exception as e:
                        prog_vdb.empty()
                        err_msg = str(e)
                        # Phát hiện lỗi dimension mismatch → hướng dẫn cụ thể
                        if "dimension" in err_msg.lower() and ("expecting" in err_msg.lower() or "got" in err_msg.lower()):
                            import re as _re
                            _m = _re.search(r"dimension of (\d+), got (\d+)", err_msg)
                            _old_dim = _m.group(1) if _m else "?"
                            _new_dim = _m.group(2) if _m else "?"
                            st.error(
                                f"❌ **Dimension mismatch**: Collection **{vdb_cfg.get('collection_name','rag')}** "
                                f"đã được tạo với embedding {_old_dim}d, "
                                f"nhưng model hiện tại tạo vector {_new_dim}d.\n\n"
                                f"**Cách fix** (chọn 1):\n"
                                f"1. ✅ Tick **Force reindex** trong sidebar → Process lại (xóa collection cũ, tạo mới)\n"
                                f"2. ✅ Đổi **Collection name** sang tên khác (vd: `rag_{_new_dim}d`)\n"
                                f"3. ✅ Chuyển lại embedding model cũ ({_old_dim}d)",
                                icon="❌",
                            )
                        elif "context length" in err_msg.lower() or "input length" in err_msg.lower():
                            st.error(
                                f"❌ **Context length exceeded**: Chunk quá dài cho model này.\n\n"
                                f"**Cách fix**: Giảm **Chunk size** xuống ≤ 4,000 chars trong sidebar.",
                                icon="❌",
                            )
                        else:
                            st.error(f"❌ Lỗi khi index Vector DB: {e}")
                            st.exception(e)
                        st.session_state["_pipeline_running"] = False
                        st.stop()

                stop_vdb_placeholder.empty()
                prog_vdb.progress(1.0, text=f"✅ Bước 4/{_n_steps} — Vector DB index hoàn tất!")

                # Lưu metadata vào pipeline cache
                cache.save_vector_db(
                    input_hash,
                    vdb_key,
                    {
                        "n_vectors":       vdb_result["n_vectors"],
                        "collection_name": vdb_result["collection_name"],
                        "provider":        vdb_result["provider"],
                        "persist_dir":     vdb_result.get("persist_dir", ""),
                    },
                    vdb_cfg_for_cache,
                    embed_key=embed_key,
                )
                # Write pipeline chain manifest so list_complete_pipelines
                # can reconstruct this run even without parent_key in meta
                _write_pipeline_chain(
                    cache        = cache,
                    input_hash   = input_hash,
                    loader_key   = loader_key,
                    chunk_key    = chunk_key,
                    embed_key    = embed_key,
                    vdb_key      = vdb_key,
                    source_path  = source_path,
                    input_display= st.session_state.get("_last_source_display", ""),
                )

                st.session_state["vdb_result"]   = vdb_result
                st.session_state["vdb_cfg_used"] = vdb_cfg
                st.session_state["active_tab"] = 3
                # Sau khi VDB xong, trim dense vectors xuống preview limit
                # Giữ đủ để hiển thị tab Embedding, nhưng không giữ toàn bộ (tiết kiệm RAM)
                if "embed_result" in st.session_state:
                    from core.constants import EMBED_PREVIEW_LIMIT
                    _er = st.session_state["embed_result"]
                    _dense = _er.get("dense", [])
                    _sparse = _er.get("sparse")
                    st.session_state["embed_result"] = {
                        **{k: v for k, v in _er.items() if k not in ("dense", "sparse")},
                        "dense":  _dense[:EMBED_PREVIEW_LIMIT] if _dense else [],
                        "sparse": _sparse[:EMBED_PREVIEW_LIMIT] if _sparse else None,
                    }
                # Lưu full pipeline config để "Index đang dùng" hiển thị đầy đủ
                st.session_state["_current_pipeline_meta"] = {
                    "source_path":  source_path,
                    "created_at":   __import__("datetime").datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "loader_cfg":   loader_cfg,
                    "chunking_cfg": {
                        "strategy":      strategy,
                        "chunk_size":    chunk_size,
                        "chunk_overlap": chunk_overlap,
                    },
                    "embed_cfg":    emb_cfg,
                    "vdb_cfg":      vdb_cfg,
                    "n_chunks":     len(st.session_state.get("chunks", [])),
                    "n_vectors":    vdb_result.get("n_vectors", 0),
                }
                st.success(
                    f"✅ Vector DB: {vdb_result['n_vectors']:,} vectors → "
                    f"{vdb_cfg['provider']}:{vdb_cfg.get('collection_name','rag')}"
                )

        # Rerun sạch để render results — tránh conflict giữa process widgets
        # và results widgets trong cùng 1 script run
        # Lưu config đã dùng vào đúng widget keys để sidebar giữ nguyên sau khi xong
        _lc = st.session_state.get("_last_loader_cfg", {})
        _ec = st.session_state.get("_last_emb_cfg", {})
        _vc = st.session_state.get("_last_vdb_cfg", {})
        for _k, _v in {
            "sel_pdf_strategy":      _lc.get("pdf_strategy"),
            "sel_chunking_strategy": st.session_state.get("_last_strategy"),
            "sel_emb_provider":      _ec.get("provider"),
            "sel_emb_model":         _ec.get("model_name"),
            "sel_vdb_provider":      _vc.get("provider"),
            "vdb_collection_name":   _vc.get("collection_name"),
        }.items():
            if _v is not None:
                st.session_state[_k] = _v
        # Cho phép auto-generate tên mới ở lần chạy tiếp theo
        # (reset flag để không bị lock vào tên cũ)
        st.session_state["_vdb_name_user_edited"] = False
        st.session_state.pop("_vdb_cfg_sig", None)

        st.session_state["_pipeline_running"] = False
        # Xóa các temp keys chỉ cần trong lúc chạy pipeline
        for _tmp_key in ("_last_extra_kwargs",):
            st.session_state.pop(_tmp_key, None)
        _invalidate_list_entries_cache()   # index mới sẵn sàng → bust cache ngay
        st.rerun()



def _page_indexing():
    """Page 1 — Indexing Stage (offline).
    Loader → Chunking → Embedding → Vector DB.
    """
    from pipeline.indexing_pipeline import IndexingPipeline

    st.title("🗃️ Indexing Pipeline")
    st.caption(
        "**Stage 1 — chạy offline một lần** để chuẩn bị index.  \n"
        "Upload tài liệu, cấu hình các bước bên trái, rồi ấn **▶️ Process**."
    )

    # ── Fast path: đang chạy pipeline → skip toàn bộ settings widgets ────────
    if st.session_state.get("_pipeline_running"):
        _run_pipeline_and_show_progress()
        return


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
        # ── Pre-read embedding selection để chunking hiển thị đúng ctx info ──
        # st.container() giữ đúng thứ tự visual (Chunking trên, Embedding dưới)
        # nhưng render embedding trước để lấy giá trị provider/model hiện tại.
        _chunk_container = st.container()
        st.markdown("---")

        # ── Embedding settings (render trước để lấy giá trị) ─────────────────
        _chunk_skipped = st.session_state.get("chunk_skip", False)
        emb_cfg = render_embedding_settings(
            local_only=local_only,
            force_skip=_chunk_skipped,
        )

        # ── Chunking settings (render vào container đã pre-allocate) ─────────
        with _chunk_container:
            strategy, chunk_size, chunk_overlap, extra_kwargs = render_chunking_settings(
                local_only=local_only,
            )

        st.markdown("---")

        # ── Vector DB settings ───────────────────────────────────────────────
        _emb_skipped = emb_cfg.get("skip", False)
        vdb_cfg = render_vector_db_settings(
            local_only=local_only,
            force_skip=_emb_skipped,
        )

        st.markdown("---")
        st.info(
            "⚡ Sau khi indexing xong, chuyển sang trang **💬 Generation** "
            "để đặt câu hỏi.",
            icon="💡",
        )
        st.markdown("---")
        _is_running = bool(st.session_state.get("_pipeline_running", False))
        process_btn = st.button(
            "⏳ Đang xử lý..." if _is_running else "▶️ Process",
            key="btn_process",
            type="primary",
            width="stretch",
            disabled=bool(source_path is None or _is_running),
        )
        # Guard: nếu đang chạy mà nút vẫn được nhấn (race), ignore
        if process_btn and _is_running:
            process_btn = False


        # ── Pipeline Cache management ─────────────────────────────────────────
        st.markdown("---")
        with st.expander("🗄️ Pipeline Cache", expanded=False):
            _pc = _get_pipeline_cache()
            _total_mb = _pc.total_size_mb()
            _entries  = _cached_list_entries("processed_data", st.session_state.get("_cache_list_version", 0))

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
                    _pc.clear_all(); _invalidate_list_entries_cache(); st.success("Đã xoá."); st.rerun()
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

    # ── Pipeline chạy (delegated to _run_pipeline_and_show_progress) ──────────
    _is_running = st.session_state.get("_pipeline_running", False)

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

        col_tab1, col_tab2, col_tab3, col_tab4, col_spacer, col_mode = st.columns([3, 3, 3, 3, 1, 2])
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






    # ── Hướng dẫn: chỉ hiện khi không đang chạy và chưa có kết quả ─────────
    if not _is_running and "loader_docs" not in st.session_state:
        # ── Main area: hướng dẫn khi chưa chọn file ─────────────────────────────
        if source_path is None:
            st.info(
                "👈 **Bắt đầu bằng cách chọn file hoặc nhập đường dẫn ở thanh bên trái**, "
                "sau đó cài đặt Loader, Chunking, Embedding, Vector DB, rồi ấn **▶️ Process**.",
                icon="🚀",
            )

        # ── Hướng dẫn Stage 1 ────────────────────────────────────────────────────
        with st.expander("📖 Hướng dẫn sử dụng — Stage 1: Indexing", expanded=(source_path is None)):
            st.markdown("""
**Indexing là gì?**
Stage 1 — chạy **offline, một lần duy nhất** trước khi người dùng sử dụng ứng dụng.
Kết quả được cache trên disk; lần sau chạy lại với cùng cấu hình sẽ reuse ngay lập tức.

---

**Bước 1 — Chọn file / thư mục** (đầu sidebar)
Upload trực tiếp hoặc nhập đường dẫn tuyệt đối.
Hỗ trợ: PDF, DOCX, PPTX, XLSX, CSV, TXT, MD, HTML, JSON, code files.

**Bước 2 — Loader** · *chọn cách đọc và trích xuất nội dung file*

**Bước 3 — Chunking** · *chọn cách cắt nhỏ văn bản thành chunk*

**Bước 4 — Embedding** · *chọn model để vector hoá chunk*

**Bước 5 — Vector DB** · *chọn nơi lưu trữ và index vector*

**Bước 6 — Ấn ▶️ Process** · chạy 4 bước indexing, kết quả hiện ở các tab bên dưới.

---
💡 **Sau khi Process xong**, chuyển sang trang **💬 Generation** để đặt câu hỏi.
""")

        with st.expander("⚙️ Loader — chọn PDF strategy nào?"):
            st.markdown("""| Strategy | Tốc độ | Bảng | Công thức | Khi nào dùng |
|----------|--------|------|-----------|--------------|
| `pypdf` | ⚡⚡⚡ Nhanh nhất | ❌ | ❌ | Prototype nhanh, PDF text thuần |
| `pymupdf` | ⚡⚡⚡ Rất nhanh | ⚠️ Cơ bản | ❌ | PDF layout đơn giản |
| `pdfplumber` | ⚡⚡ Nhanh | ✅ Tốt nhất | ❌ | PDF có nhiều bảng dạng text |
| `marker` ⭐ | ⚡ Chậm | ✅✅ Rất tốt | ✅ LaTeX | **PDF phức tạp: bảng, công thức, hình ảnh** |
| `docling` | ⚡ Chậm | ✅✅ Rất tốt | ✅ | Tài liệu học thuật, báo cáo phức tạp |
| `unstructured` | ⚡ Chậm | ✅✅ Rất tốt | ✅ OCR | PDF scan, cần OCR |
""")
            if st.checkbox("🔍 Xem giải thích chi tiết — Loader", key="detail_loader_idx"):
                st.markdown("""
---
**🟢 `pypdf` — Nhanh nhất, đơn giản nhất**

Thư viện Python thuần, đọc trực tiếp text layer trong PDF. Không cần cài thêm gì.

- ✅ Tốt cho PDF "text thuần" — tức là PDF được tạo từ Word, LaTeX, trình soạn thảo, có thể copy-paste text.
- ❌ Không đọc được bảng có cấu trúc phức tạp, không hiểu layout 2 cột.
- ❌ Hoàn toàn không dùng được với PDF scan (ảnh chụp).

---
**🔵 `pymupdf` — Nhanh, giữ layout tốt hơn pypdf**

Dùng thư viện MuPDF (C++) — cực nhanh. Trích xuất được vị trí text, font, màu sắc.

- ✅ Tốt hơn pypdf khi PDF có nhiều cột hoặc bảng đơn giản.
- ⚠️ Bảng phức tạp (merged cells, borders) vẫn có thể bị sai.
- ✅ Xuất được ảnh nhúng trong PDF.

---
**🔵 `pdfplumber` — Chuyên trích bảng**

Xây dựng trên pdfminer, tối ưu đặc biệt cho việc phát hiện và trích xuất bảng dạng text (line-based tables).

- ✅ Tốt nhất khi PDF có nhiều bảng số liệu, báo cáo tài chính.
- ❌ Chậm hơn pypdf/pymupdf khoảng 2–3x.
- ❌ Không xử lý được ảnh/công thức.

---
**🟡 `marker` ⭐ — Chuyển PDF sang Markdown chất lượng cao**

Dùng AI (Surya OCR + layout model) để phân tích cấu trúc PDF, chuyển thành Markdown sạch với heading, bảng, công thức LaTeX.

- ✅ Tốt nhất cho PDF học thuật, paper, báo cáo có hình ảnh và công thức.
- ✅ Output Markdown → kết hợp với `format_aware` chunking cho kết quả tối ưu.
- ❌ Chậm (khoảng 0.4s/trang với GPU, 2–5s/trang CPU).
- ❌ Cần cài `marker-pdf` và các model AI đi kèm.

---
**🟡 `docling` — Phân tích tài liệu học thuật**

Thư viện của IBM, chuyên xử lý tài liệu phức tạp: paper, báo cáo, slide. Hỗ trợ nhiều định dạng ngoài PDF.

- ✅ Rất tốt cho academic paper, technical document.
- ✅ Nhận diện figure caption, table of contents, footnote.
- ❌ Cần cài thêm và tốn RAM hơn marker.

---
**🔴 `unstructured` — Đa năng nhất, hỗ trợ OCR**

Framework xử lý nhiều loại file (PDF, Word, HTML, email...). Có thể gọi OCR engine bên ngoài.

- ✅ Dùng được với PDF scan (hình ảnh không có text layer).
- ✅ Xử lý được Word, HTML, email — không chỉ PDF.
- ❌ Cần cài nhiều dependency, cấu hình phức tạp hơn.
- 📐 Khi dùng OCR: cần Tesseract hoặc kết nối API OCR.
""")

        with st.expander("✂️ Chunking — chiến lược nào phù hợp với tôi?"):
            st.markdown("""| Strategy | Cơ chế | Best for |
|----------|--------|---------|
| `recursive` | Đệ quy: đoạn → dòng → câu → ký tự | Mặc định tốt nhất |
| `token_based` | Đếm BPE token | Khi embedding model có token limit chặt |
| `format_aware` ⭐ | Markdown heading, code block, HTML | **PDF qua Marker/Docling** |
| `sentence_aware` | Ranh giới câu (NLTK) | Q&A, FAQ, văn xuôi |
| `semantic` | Cosine similarity | PDF nhiều chủ đề khác nhau |
| `hierarchical` | Parent (lớn) + Child (nhỏ) | Corpus lớn, cần chính xác cao |
| `contextual` | Recursive + LLM context prefix | Production, có LLM budget |

    **Chunk size:** thường 512–1500 ký tự. **Overlap:** 10–15% chunk size.
""")
            if st.checkbox("🔍 Xem giải thích chi tiết — Chunking", key="detail_chunking_idx"):
                st.markdown("""
**🟢 `format_aware` ⭐** — Nhận diện `# Heading`, code block, HTML tag. Cắt tại ranh giới cấu trúc, không cắt giữa code. Tối ưu cho Marker/Docling output.

**🔵 `recursive`** — Chia theo thứ tự: đoạn → dòng → câu → ký tự. Không cần cài thêm thư viện phụ trợ, rất nhanh. Phù hợp khi không có yêu cầu đặc biệt.

**🟡 `semantic`** — Embed từng câu, cắt khi cosine similarity giảm đột ngột. Mỗi chunk = 1 chủ đề nhất quán. Chậm hơn do cần pre-embed.

**🟠 `hierarchical`** — Parent (~2000 chars) + Child (~400 chars). Retrieve bằng child, trả về parent cho LLM → chính xác + context đủ lớn.

**🔴 `contextual`** — LLM sinh context prefix cho mỗi chunk. Chất lượng cao nhất nhưng tốn LLM call per chunk.

**📐 Tham số quan trọng:**
- `chunk_size`: Số ký tự tối đa mỗi chunk. Nhỏ = tìm chính xác hơn; lớn = context nhiều hơn.
- `chunk_overlap`: Ký tự chồng lắp giữa 2 chunk liên tiếp. Giúp tránh cắt đứt ý quan trọng ở ranh giới.
""")

        with st.expander("🧮 Embedding — model và provider nào phù hợp với tôi?"):
            st.markdown("""| Provider | Model | Tiếng Việt | Chi phí | Best for |
|----------|-------|------------|---------|---------|
| **OpenAI** | `text-embedding-3-small` ⭐ | ⭐⭐⭐ | $0.02/1M | Mặc định tốt nhất |
| **OpenAI** | `text-embedding-3-large` | ⭐⭐⭐ | $0.13/1M | Độ chính xác cao nhất |
| **Cohere** | `embed-multilingual-v3.0` | ⭐⭐⭐⭐ | $0.10/1M | Corpus đa ngôn ngữ |
| **HuggingFace** | `BAAI/bge-m3` | ⭐⭐⭐⭐ | Miễn phí | Local, VI tốt |
| **HuggingFace** | `Qwen/Qwen3-Embedding` | ⭐⭐⭐⭐⭐ | Miễn phí | MTEB #1, VI tốt nhất |
| **FastEmbed** | `multilingual-e5-small` | ⭐⭐ | Miễn phí | CPU, không GPU |
| **Ollama** | `bge-m3` | ⭐⭐⭐⭐ | Miễn phí | Local server, privacy |

    **Sparse Embedding (Hybrid Retrieval):** Bật BM25 khi corpus tiếng Việt có nhiều thuật ngữ, tên riêng.
""")
            if st.checkbox("🔍 Xem giải thích chi tiết — Embedding", key="detail_embedding_gen"):
                st.markdown("""
---
**Bi-encoder (Dense Embedding)**

Mỗi text được encode độc lập thành một dense vector. Retrieval = tính cosine similarity query vs. tất cả doc vectors.

- 🔵 **OpenAI text-embedding-3-small:** MRL (Matryoshka) — có thể truncate về 512d mà chất lượng ~95%. Chi phí thấp nhất trong API models.
- 🟢 **BAAI/bge-m3:** Multilingual (100+ ngôn ngữ), context 8192 tokens, dense+sparse native. Tốt nhất cho tiếng Việt offline.
- 🟡 **Qwen3-Embedding:** MTEB benchmark #1 (tháng 6/2025), context 32K tokens. Nặng hơn bge-m3 (~4GB).
- 🟠 **FastEmbed multilingual-e5-small:** ONNX quantized, chạy tốt trên CPU, ~90MB. Chất lượng thấp hơn nhưng zero dependency GPU.

**Sparse Embedding (BM25)**

BM25 (Best Match 25) là thuật toán lexical scoring kinh điển: `TF × IDF × length normalization`. Tốt cho từ khoá chính xác, tên riêng, mã số, không bị ảnh hưởng bởi semantic gap.

- ✅ BM25 không cần GPU, fit nhanh trên CPU.
- 📐 **Tham số:** `k1` = saturation param (mặc định 1.5) · `b` = length norm (mặc định 0.75).
""")

        with st.expander("🗃️ Vector DB — chọn loại nào?"):
            st.markdown("""| Store | Scale | Hybrid | Best for |
|-------|-------|--------|---------|
| **Chroma** ⭐ | < 1M | ✅ | **Demo, dev, small corpus** — zero config |
| **FAISS** | < 100M | ❌ | Prototype nhanh nhất, in-memory |
| **Qdrant** | 1B+ | ✅ ACORN | Production, filtering phức tạp |
| **LanceDB** | < 1B | ✅ | Embedded, không cần server |
| **Weaviate** | ~1B | ✅ | Hybrid out-of-the-box |
| **PGVector** | < 100M | ✅ | Đã có PostgreSQL infrastructure |
| **Pinecone** | ~1B | ✅ | Zero ops, managed cloud |
""")
            if st.checkbox("🔍 Xem giải thích chi tiết — Vector DB", key="detail_vdb_idx"):
                st.markdown("""
---
**🟢 `Chroma` ⭐ — Lựa chọn mặc định cho dev/demo**

Chạy ngay không cần cài server riêng — dữ liệu lưu vào SQLite local. Hỗ trợ metadata filtering và hybrid search.

- ✅ Zero config: chạy được ngay sau `pip install chromadb`.
- ✅ Persist tự động, không cần code thêm.
- ✅ Hỗ trợ metadata filter (lọc theo nguồn, trang, ngày...).
- ❌ Không scale tốt trên 1–5 triệu vector.
- 📐 Phù hợp: prototype, RAG demo, corpus < 500K chunks.

---
**🔵 `FAISS` — Tìm kiếm vector nhanh nhất**

Thư viện của Meta, tối ưu cao cho ANN (Approximate Nearest Neighbor) search. Chạy in-memory, không cần server.

- ✅ Nhanh nhất trong các local option, đặc biệt với GPU.
- ❌ Không có hybrid search built-in, không có metadata filtering.
- ❌ Không persist tự động — phải save/load thủ công.
- 📐 Phù hợp: benchmark, prototype cần tốc độ, không cần filtering.

---
**🟡 `Qdrant` — Production-grade với filtering mạnh**

Vector database chuyên dụng, hỗ trợ ACORN filtering (filter trước khi search, không phải sau). Có thể chạy embedded (không cần server) hoặc cloud.

- ✅ Filtering hiệu quả nhất: không scan toàn bộ vector khi filter.
- ✅ Hỗ trợ hybrid search, quantization, multitenancy.
- ❌ Cần cài Qdrant server hoặc dùng Qdrant Cloud.
- 📐 Phù hợp: production với metadata phong phú, corpus lớn.

---
**🔵 `LanceDB` — Embedded, không cần server**

Database columnar dùng Apache Arrow, chạy embedded như SQLite. Tốt cho single-user hoặc pipeline offline.

- ✅ Không cần server, dữ liệu lưu local như file.
- ✅ Hỗ trợ hybrid search và versioning.
- ✅ Rất nhanh khi query với filter trên dataset lớn.
- ❌ Multiuser/concurrent write có giới hạn.
- 📐 Phù hợp: pipeline offline, single-user production.

---
**🔴 `Pinecone` / `Weaviate` / `PGVector` — Scale lớn / tích hợp hệ thống**

- **Pinecone**: Managed cloud, zero ops, phù hợp khi không muốn tự quản lý infrastructure.
- **Weaviate**: Hybrid search tốt nhất out-of-the-box, có GraphQL API.
- **PGVector**: Dùng khi đã có PostgreSQL — thêm vector search vào DB hiện tại.
""")

        if source_path is not None:
            suggestions = get_pipeline_suggestions(source_path, local_only)
            render_pipeline_suggestions(suggestions)

    if process_btn:
        # Tính _source_display_name TRƯỚC để _last_source_display dùng đúng giá trị mới
        if st.session_state.get("_upload_display_names"):
            _display = ", ".join(st.session_state["_upload_display_names"])
        else:
            from pathlib import Path as _P
            _display = _P(source_path).name if source_path else source_path
        st.session_state["_source_display_name"] = _display

        # Lưu TOÀN BỘ config trước khi rerun (sẽ không có widget nào trong pipeline runner)
        st.session_state["_last_source_path"]    = source_path
        st.session_state["_last_source_display"] = _display          # dùng giá trị vừa tính
        st.session_state["_last_loader_cfg"]     = loader_cfg
        st.session_state["_last_strategy"]       = strategy
        st.session_state["_last_chunk_size"]     = chunk_size
        st.session_state["_last_chunk_overlap"]  = chunk_overlap
        st.session_state["_last_extra_kwargs"]   = extra_kwargs
        st.session_state["_last_emb_cfg"]        = emb_cfg
        st.session_state["_last_vdb_cfg"]        = vdb_cfg
        st.session_state["_pipeline_running"]    = True
        st.rerun()
