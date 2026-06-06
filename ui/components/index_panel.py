
def _chunk_size_display(cc: dict, cs: dict) -> str:
    """Trả về chuỗi hiển thị chunk size/overlap chỉ khi strategy sử dụng chúng.
    format_aware, semantic, sentence_aware: không hiển thị chunk_size/overlap.
    """
    _NO_SIZE = {"format_aware", "semantic", "sentence_aware", "sentence", "late_chunking"}
    strategy = cc.get("strategy", "")
    # format_aware: chỉ hiện nếu split_large_sections = True
    if strategy == "format_aware":
        if not cc.get("split_large_sections"):
            return f"{cs.get('n_chunks','?')} chunks"
        sz = cc.get("chunk_size", "")
        ov = cc.get("chunk_overlap", "")
        return f"{sz}c / {ov}ov · {cs.get('n_chunks','?')} chunks" if sz else f"{cs.get('n_chunks','?')} chunks"
    if strategy in _NO_SIZE:
        return f"{cs.get('n_chunks','?')} chunks"
    sz = cc.get("chunk_size", "")
    ov = cc.get("chunk_overlap", "")
    n  = cs.get("n_chunks", "?")
    if sz and ov:
        return f"{sz}c / {ov}ov · {n} chunks"
    return f"{n} chunks"


def _input_display_name(entry: dict) -> str:
    """Lấy tên hiển thị của input.
    Ưu tiên:
      1. input_display field (tên file gốc lúc upload)
      2. pipeline_chain.json → input_display (index cũ chưa backfill)
      3. basename của source_path
    """
    import json as _json
    from pathlib import Path as _P

    src = entry.get("source_path", "") or ""

    disp = entry.get("input_display", "")
    # Bỏ qua nếu input_display chỉ là path temp (giống source_path)
    if disp and disp != src:
        return disp

    # Thử đọc từ pipeline_chain.json trong processed_data
    input_short = entry.get("input_short", "")
    if input_short:
        for base in [_P("processed_data"), _P.cwd() / "processed_data"]:
            input_dir = base / input_short
            if input_dir.exists():
                for chain_f in input_dir.rglob("pipeline_chain.json"):
                    try:
                        chain = _json.loads(chain_f.read_text("utf-8"))
                        d  = chain.get("input_display", "")
                        sp = chain.get("source_path", "")
                        if d and d != sp:
                            return d
                    except Exception:
                        pass
                break

    # Fallback: basename của source_path
    p = _P(src)
    return p.name or src


def _vdb_display_name(vs: dict, vc: dict = None) -> str:
    """persist_dir basename = faiss_mrkr_fmt_..._04Jun2026_220347"""
    import os
    vc = vc or {}
    persist_dir = vs.get("persist_dir") or vc.get("persist_dir", "")
    if persist_dir:
        basename = os.path.basename(persist_dir.rstrip("/\\"))
        if basename and basename not in ("faiss", "chroma", "lancedb", "storage"):
            return basename
    provider = vs.get("provider") or vc.get("provider", "?")
    coll     = vs.get("collection_name") or vc.get("collection_name", "?")
    return f"{provider}/{coll}"

import os, sys, re, importlib, importlib.util, inspect, tempfile
from pathlib import Path
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
try:
    from dotenv import load_dotenv as _ld, dotenv_values as _dv
    _ld(override=True); _ENV = _dv()
except ImportError:
    _ENV = {}
import streamlit as st

from core.cache_helpers import (
    _cached_list_entries, _cached_list_pipelines,
    _invalidate_list_entries_cache, _embedder_kwargs_from_cfg,
)
from pipeline_cache import PipelineCache

def _try_autoload_latest_index() -> None:
    """Tự động load index gần nhất khi vào Generation page lần đầu."""
    try:
        from pipeline_cache import PipelineCache
        pipelines = _cached_list_pipelines("processed_data", st.session_state.get("_cache_list_version", 0))
        if not pipelines:
            return
        # Thử lần lượt từng pipeline (mới nhất trước) cho đến khi có 1 load được
        for p in pipelines:
            try:
                _load_pipeline_into_session(p)
                return  # load thành công → dừng
            except Exception:
                continue  # pipeline này lỗi (file mất/đổi path) → thử cái tiếp
    except Exception:
        pass   # Fail silently — user sẽ tự chọn từ picker



def _pipeline_label(p: dict) -> str:
    """Short label for a pipeline entry."""
    src = _input_display_name(p)
    src = ("..." + src[-22:]) if len(src) > 25 else src
    dt    = p.get("created_at", "")[:16].replace("T", " ")
    lc    = p["loader"]["cfg"]
    cc    = p["chunking"]["cfg"]
    ec    = p["embedding"]["cfg"]
    vs    = p["vector_db"]["stats"]
    nv    = vs.get("n_vectors", "?")
    model = ec.get("model_name", "?").split("/")[-1]
    chunk = cc.get("strategy", "?")
    nv_s  = f"{nv:,}" if isinstance(nv, int) else str(nv)
    return f"{src}  |  {lc.get('pdf_strategy','?')}+{chunk}  |  {model}  |  {nv_s} vecs  |  {dt}"


def _load_pipeline_into_session(p: dict) -> None:
    """Load một pipeline hoàn chỉnh vào session_state."""
    from pipeline_cache import PipelineCache
    ec      = p["embedding"]["cfg"]
    vs      = p["vector_db"]["stats"]
    vc      = p["vector_db"]["cfg"]
    provider        = vs.get("provider",        vc.get("provider",        "chroma"))
    collection_name = vs.get("collection_name", vc.get("collection_name", "rag_docs"))
    persist_dir     = vc.get("persist_directory", vc.get("persist_dir", "./storage/chroma"))
    n_vectors       = vs.get("n_vectors", 0)

    full_emb_cfg = {
        "provider":        ec.get("provider",       "openai"),
        "model_name":      ec.get("model_name",     "text-embedding-3-small"),
        "enable_sparse":   ec.get("enable_sparse",  False),
        "sparse_method":   ec.get("sparse_method",  "none"),
        "dimensions":      ec.get("dimensions"),
        "ollama_base_url": ec.get("ollama_base_url","http://localhost:11434"),
        "input_type":      ec.get("input_type",     "search_document"),
    }
    vdb_cfg_full = {
        "provider":          provider,
        "collection_name":   collection_name,
        "persist_directory": persist_dir,
        "force_reindex":     False,
    }
    try:
        from embedding.factory import get_embedder
        embedder = get_embedder(**_embedder_kwargs_from_cfg(full_emb_cfg))
        vector_store = _load_vector_store_readonly(provider, embedder, {
            "collection_name": collection_name,
            "persist_dir":     persist_dir,
        })
        # Load chunks từ cache để BM25 sparse retrieval có documents
        _chunks = []
        try:
            from pathlib import Path as _P
            import pickle as _pk
            _input_short = p.get("input_short", "")
            _chunk_ks    = p.get("chunking", {}).get("key", "")
            # Try both relative and absolute paths
            for _base in [_P("processed_data"), _P.cwd() / "processed_data"]:
                _chunks_file = _base / _input_short / "chunking" / _chunk_ks / "chunks.pkl"
                if _chunks_file.exists():
                    with open(_chunks_file, "rb") as _f:
                        _chunks = _pk.load(_f)
                    break
        except Exception as _e:
            import sys
            print(f"[WARN] chunks load failed: {_e}", file=sys.stderr)
        st.session_state["chunks"] = _chunks

        st.session_state["vdb_result"] = {
            "vector_store":    vector_store,
            "sparse_index":    None,
            "provider":        provider,
            "collection_name": collection_name,
            "n_vectors":       n_vectors,
            "persist_dir":     persist_dir,
        }
        st.session_state["emb_cfg_used"] = full_emb_cfg
        st.session_state["vdb_cfg_used"] = vdb_cfg_full
        st.session_state["_loaded_pipeline_id"] = p.get("pipeline_id", "")
        # Save full pipeline meta for "Index đang dùng" display
        st.session_state["_current_pipeline_meta"] = {
            "source_path":  p.get("source_path", ""),
            "created_at":   p.get("created_at", "")[:16].replace("T", " "),
            "loader_cfg":   p["loader"]["cfg"],
            "chunking_cfg": p["chunking"]["cfg"],
            "embed_cfg":    full_emb_cfg,
            "vdb_cfg":      vdb_cfg_full,
            "n_chunks":     p["chunking"]["stats"].get("n_chunks", "?"),
            "n_vectors":    n_vectors,
        }
        nv_s = f"{n_vectors:,}" if isinstance(n_vectors, int) else str(n_vectors)
        st.toast(f"✅ Đã load: {collection_name} · {nv_s} vectors", icon="🗄️")
    except Exception as e:
        # Re-raise để caller quyết định có hiển thị lỗi không
        # (_try_autoload sẽ catch silently, Load button sẽ hiển thị lỗi)
        raise

def _render_index_switcher() -> None:
    """Dropdown chọn pipeline hoàn chỉnh trong generation sidebar."""
    _ver      = st.session_state.get("_cache_list_version", 0)
    pipelines = _cached_list_pipelines("processed_data", _ver)
    if not pipelines:
        st.caption("Chưa có pipeline nào hoàn tất trong cache.")
        return

    # Reset selectbox khi có pipeline mới
    if st.session_state.get("_switcher_ver") != _ver:
        st.session_state["gen_index_switcher"] = 0
        st.session_state["_switcher_ver"] = _ver

    sel = st.selectbox(
        "Pipeline",
        options=list(range(len(pipelines))),
        format_func=lambda i: _pipeline_label(pipelines[i]),
        key="gen_index_switcher",
        label_visibility="collapsed",
    )
    p    = pipelines[sel]
    lc   = p["loader"]["cfg"]
    cc   = p["chunking"]["cfg"];  cs = p["chunking"]["stats"]
    ec   = p["embedding"]["cfg"]
    vs   = p["vector_db"]["stats"]
    vc   = p["vector_db"]["cfg"]
    nv   = vs.get("n_vectors", 0)
    nv_s = f"{nv:,}" if isinstance(nv, int) else str(nv)
    sp   = f" + {ec.get('sparse_method','')}" if ec.get("enable_sparse") else ""
    _vlm_tag = ""
    if lc.get("describe_images"):
        _vlm_short = lc.get("vision_model", "VLM").split("/")[-1]
        _vlm_tag = f" <span style='color:#7c3aed;font-weight:600;'>🖼️ {_vlm_short}</span>"
    st.markdown(
        f"<small style='line-height:1.7;opacity:.8;'>"
        f"📄 {_input_display_name(p)}"
        f"  📅 {p.get('created_at','')[:16].replace('T',' ')}<br>"
        f"📄 <code>{lc.get('pdf_strategy','?')}</code>{_vlm_tag}"
        f" &rarr; <code>{cc.get('strategy','?')}</code>"
        f" · {_chunk_size_display(cc, cs)}<br>"
        f"🧮 <code>{ec.get('provider','?')}/{ec.get('model_name','?').split('/')[-1]}{sp}</code>"
        f" &rarr; <code>{_vdb_display_name(vs, vc)}</code>"
        f" · {nv_s} vecs</small>",
        unsafe_allow_html=True,
    )
    if st.button("✅ Dùng pipeline này", key="btn_switch_index",
                 type="primary", use_container_width=True):
        try:
            _load_pipeline_into_session(p)
            st.rerun()
        except Exception as _e:
            st.error(f"❌ Không thể load index: {_e}\n\n"
                     "File có thể đã bị xóa hoặc đổi đường dẫn. "
                     "Hãy chạy lại Indexing để tạo index mới.")


def _render_load_index_from_cache() -> None:
    """
    Hiển thị danh sách đầy đủ các index đã cached để người dùng chọn load.
    """
    entries = _cached_list_entries("processed_data", st.session_state.get("_cache_list_version", 0))
    valid   = [e for e in entries if "vector_db" in e.get("steps", {})]
    partial = [e for e in entries if e not in valid]

    st.markdown("### 📂 Chọn index đã có sẵn")
    st.caption(
        "Chưa có index nào được load. "
        "Chọn một index từ lần chạy trước, hoặc chuyển sang **🗃️ Indexing** để tạo mới."
    )

    if not valid:
        if partial:
            st.warning(
                f"Tìm thấy **{len(partial)} entry** trong cache nhưng chưa hoàn tất bước Vector DB "
                "(có thể bị dừng giữa chừng). Hãy chuyển sang **🗃️ Indexing** và chạy lại.",
                icon="⚠️",
            )
        else:
            st.info("Chưa có index nào trong cache. Hãy chuyển sang **🗃️ Indexing** để tạo trước.", icon="📭")
        return

    # ── Filter & Sort controls ─────────────────────────────────────────────
    ctrl1, ctrl2, ctrl3 = st.columns([2, 2, 2])
    with ctrl1:
        sort_by = st.selectbox(
            "Sắp xếp theo",
            ["Mới nhất", "Cũ nhất", "Dung lượng lớn nhất", "Dung lượng nhỏ nhất", "Nhiều vectors nhất"],
            key="cache_sort_by",
        )
    with ctrl2:
        filter_local = st.checkbox(
            "🔒 Chỉ local / self-hosted",
            value=False,
            key="cache_filter_local",
            help="Chỉ hiện index dùng embedding model local (HuggingFace, FastEmbed, Ollama) và vector DB local (Chroma, FAISS, LanceDB).",
        )
    with ctrl3:
        filter_provider = st.selectbox(
            "Lọc theo Embedding",
            ["Tất cả", "openai", "anthropic", "cohere", "huggingface", "fastembed", "ollama"],
            key="cache_filter_provider",
        )

    # Apply filters
    _LOCAL_EMB = {"huggingface", "fastembed", "ollama"}
    _LOCAL_VDB = {"chroma", "faiss", "lancedb"}

    def _get_emb_provider(e: dict) -> str:
        ec = (e["steps"].get("embedding") or [{}])[-1].get("cfg", {})
        return ec.get("provider", "?").lower()

    def _get_vdb_provider(e: dict) -> str:
        vs = (e["steps"].get("vector_db") or [{}])[-1].get("stats", {})
        return vs.get("provider", "?").lower()

    def _get_nvecs(e: dict) -> int:
        vs = (e["steps"].get("vector_db") or [{}])[-1].get("stats", {})
        n = vs.get("n_vectors", 0)
        return n if isinstance(n, int) else 0

    if filter_local:
        valid = [e for e in valid
                 if _get_emb_provider(e) in _LOCAL_EMB and _get_vdb_provider(e) in _LOCAL_VDB]
    if filter_provider != "Tất cả":
        valid = [e for e in valid if _get_emb_provider(e) == filter_provider]

    # Sort
    if sort_by == "Mới nhất":
        valid.sort(key=lambda e: e.get("created_at", ""), reverse=True)
    elif sort_by == "Cũ nhất":
        valid.sort(key=lambda e: e.get("created_at", ""))
    elif sort_by == "Dung lượng lớn nhất":
        valid.sort(key=lambda e: e.get("total_size_mb", 0), reverse=True)
    elif sort_by == "Dung lượng nhỏ nhất":
        valid.sort(key=lambda e: e.get("total_size_mb", 0))
    elif sort_by == "Nhiều vectors nhất":
        valid.sort(key=_get_nvecs, reverse=True)

    if not valid:
        st.warning("Không có index nào khớp với bộ lọc hiện tại.")
        return

    # Phân trang: hiện tối đa 10 entry mỗi trang để tránh render quá nhiều widget
    _PAGE_SIZE = 10
    _n_pages   = max(1, (len(valid) + _PAGE_SIZE - 1) // _PAGE_SIZE)
    _page      = st.session_state.get("cache_page", 0)
    _page      = min(_page, _n_pages - 1)

    if _n_pages > 1:
        pcol1, pcol2, pcol3 = st.columns([1, 3, 1])
        with pcol1:
            if st.button("◀", key="cache_prev", disabled=(_page == 0)):
                st.session_state["cache_page"] = _page - 1
                st.rerun()
        with pcol2:
            st.markdown(
                f"<div style='text-align:center;font-size:.9em;opacity:.7;'>"
                f"Trang {_page+1}/{_n_pages} · {len(valid)} index</div>",
                unsafe_allow_html=True,
            )
        with pcol3:
            if st.button("▶", key="cache_next", disabled=(_page == _n_pages - 1)):
                st.session_state["cache_page"] = _page + 1
                st.rerun()
    else:
        st.markdown(f"**{len(valid)} pipeline** — nhấn **Load** để dùng ngay.")
    st.markdown("")

    _start = _page * _PAGE_SIZE
    for entry in valid[_start: _start + _PAGE_SIZE]:
        steps       = entry.get("steps", {})
        input_short = entry.get("input_short", "")
        created     = entry.get("created_at", "")[:16].replace("T", " ")
        size_mb     = entry.get("total_size_mb", 0)
        source_path = entry.get("source_path", "-")

        def _latest_cfg(step: str) -> dict:
            return (steps.get(step) or [{}])[-1].get("cfg", {})
        def _latest_stats(step: str) -> dict:
            return (steps.get(step) or [{}])[-1].get("stats", {})

        lc = _latest_cfg("loader");  ls = _latest_stats("loader")
        cc = _latest_cfg("chunking"); cs = _latest_stats("chunking")
        ec = _latest_cfg("embedding"); es = _latest_stats("embedding")
        vs = _latest_stats("vector_db"); vc = _latest_cfg("vector_db")

        # Bổ sung describe_images/vision_model từ loader meta.json đúng pipeline
        # _latest_cfg("loader") lấy loader cuối cùng theo sort — không nhất thiết đúng.
        # Dùng loader_key từ pipeline_chain.json để đọc đúng meta.json.
        if not lc.get("describe_images"):
            import json as _j
            from pathlib import Path as _P2
            for _base in [_P2("processed_data"), _P2.cwd() / "processed_data"]:
                _idir = _base / input_short
                if _idir.exists():
                    for _cf in _idir.rglob("pipeline_chain.json"):
                        try:
                            _chain = _j.loads(_cf.read_text("utf-8"))
                            _lkey  = _chain.get("loader_key", "")
                            if not _lkey:
                                continue
                            _lmeta_f = _idir / "loader" / _lkey / "meta.json"
                            if _lmeta_f.exists():
                                _lmeta = _j.loads(_lmeta_f.read_text("utf-8"))
                                _lcfg  = _lmeta.get("cfg", {})
                                if _lcfg.get("describe_images"):
                                    lc = {**lc, **{
                                        k: v for k, v in _lcfg.items()
                                        if k in ("describe_images", "vision_model",
                                                 "vision_provider")
                                    }}
                        except Exception:
                            pass
                    break

        # Source display: dùng _input_display_name để ưu tiên tên file gốc
        src_display = _input_display_name(entry)
        if len(src_display) > 60:
            src_display = "..." + src_display[-57:]

        # Build detailed info strings
        pdf_strat = lc.get("pdf_strategy", "?")
        language  = lc.get("language", "?")
        n_docs    = ls.get("n_docs", "?")

        chunk_strategy = cc.get("strategy", "?")
        chunk_size     = cc.get("chunk_size", "?")
        chunk_overlap  = cc.get("chunk_overlap", "?")
        n_chunks       = cs.get("n_chunks", "?")
        chunk_detail   = _chunk_size_display(cc, cs)

        emb_provider = ec.get("provider", "?")
        emb_model    = ec.get("model_name", "?")
        enable_sparse = ec.get("enable_sparse", False)
        sparse_method = ec.get("sparse_method", "none")
        dimensions    = ec.get("dimensions")
        n_vecs        = es.get("n_vectors", vs.get("n_vectors", "?"))

        vdb_provider    = vs.get("provider", vc.get("provider", "?"))
        collection_name = vs.get("collection_name", vc.get("collection_name", "?"))
        n_vectors       = vs.get("n_vectors", "?")
        persist_dir     = vc.get("persist_directory", vc.get("persist_dir", "?"))

        # Build badge for local-only
        is_local = emb_provider in _LOCAL_EMB and vdb_provider in _LOCAL_VDB
        local_badge = " 🔒 local" if is_local else ""

        with st.container(border=True):
            hcol, bcol = st.columns([6, 1], vertical_alignment="top")

            with hcol:
                # Header row
                st.markdown(
                    f"**📄 {src_display}**{local_badge}"
                    f"<span style='color:var(--text-color);opacity:.5;font-size:.82em;margin-left:1em;'>"
                    f"{created} &nbsp;·&nbsp; {size_mb:.1f} MB</span>",
                    unsafe_allow_html=True,
                )
                # Detail rows — 4 columns
                c1, c2, c3, c4 = st.columns(4)

                with c1:
                    _describe_tag = ""
                    if lc.get("describe_images"):
                        _vlm = lc.get("vision_model", "")
                        _vlm_short = _vlm.split("/")[-1] if _vlm else "VLM"
                        _describe_tag = (
                            f"<br><span style='color:#7c3aed;font-size:.9em;font-weight:600;'>"
                            f"🖼️ VLM: {_vlm_short}</span>"
                        )
                    st.markdown(
                        f"<div style='font-size:.82em;line-height:1.6;'>"
                        f"<b>📂 Nguồn</b><br>"
                        f"<span style='opacity:.7;'>{n_docs} docs · {language}</span><br>"
                        f"<b>📄 Loader</b><br>"
                        f"<code>{pdf_strat}</code>"
                        f"{_describe_tag}"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

                with c2:
                    extra_chunk = ""
                    if chunk_strategy not in ("semantic", "contextual", "late"):
                        extra_chunk = f"<br><span style='opacity:.6;font-size:.9em;'>size={chunk_size} · overlap={chunk_overlap}</span>"
                    st.markdown(
                        f"<div style='font-size:.82em;line-height:1.6;'>"
                        f"<b>✂️ Chunking</b><br>"
                        f"<code>{chunk_strategy}</code>{extra_chunk}<br>"
                        f"<span style='opacity:.6;font-size:.9em;'>{n_chunks} chunks</span>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

                with c3:
                    # Build embedding detail
                    emb_short = emb_model.split("/")[-1]
                    sparse_tag = ""
                    if enable_sparse and sparse_method != "none":
                        sparse_tag = f"<br><span style='opacity:.65;font-size:.9em;'>+ {sparse_method} sparse</span>"
                    mrl_tag = ""
                    if dimensions:
                        mrl_tag = f"<br><span style='opacity:.65;font-size:.9em;'>MRL → {dimensions}d</span>"
                    st.markdown(
                        f"<div style='font-size:.82em;line-height:1.6;'>"
                        f"<b>🧮 Embedding</b><br>"
                        f"<code>{emb_provider}</code><br>"
                        f"<span style='opacity:.75;'>{emb_short}</span>"
                        f"{sparse_tag}{mrl_tag}"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

                with c4:
                    n_v_str = f"{n_vectors:,}" if isinstance(n_vectors, int) else str(n_vectors)
                    st.markdown(
                        f"<div style='font-size:.82em;line-height:1.6;'>"
                        f"<b>🗃️ Vector DB</b><br>"
                        f"<code>{_vdb_display_name(vs, vc)}</code><br>"
                        f"<span style='opacity:.6;font-size:.9em;'>{n_v_str} vectors</span>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

            with bcol:
                btn_key = f"load_idx_{input_short}"
                if st.button("Load →", key=btn_key, type="primary", use_container_width=True):
                    from pipeline_cache import PipelineCache
                    _load_index_into_session(entry, PipelineCache("processed_data"))
                    st.rerun()

    st.markdown("---")
    st.caption("Muốn tạo index mới? Chuyển sang trang **🗃️ Indexing** ở sidebar.")




def _get_chroma_client(persist_dir: str):
    """
    Trả về Chroma PersistentClient từ module-level lru_cache trong chroma_store.
    Dùng chung với ChromaStore._make_client → đảm bảo chỉ có MỘT instance
    per persist_dir trong toàn bộ process (tránh conflict settings).
    st.cache_resource không cần thiết vì lru_cache đã singleton ở process level.
    """
    from vector_db.chroma_store import _get_or_create_chroma_client
    return _get_or_create_chroma_client(persist_dir)

def _load_vector_store_readonly(provider: str, embedder, vdb_kwargs: dict):
    """
    Load một vector store đã tồn tại mà KHÔNG insert thêm document nào.
    Tránh crash Chroma.from_documents([]) khi chunks=[].
    """
    lc_embedder = embedder.embedder if hasattr(embedder, "embedder") else embedder

    if provider == "chroma":
        from langchain_chroma import Chroma
        import chromadb
        from chromadb.config import Settings
        persist_dir = vdb_kwargs.get("persist_dir") or vdb_kwargs.get("persist_directory", "./storage/chroma")
        collection_name = vdb_kwargs["collection_name"]
        # cache_resource giữ client sống suốt session, tránh GC destroy RustBindingsAPI
        chroma_client = _get_chroma_client(persist_dir)
        return Chroma(
            client             = chroma_client,
            collection_name    = collection_name,
            embedding_function = lc_embedder,
        )

    if provider == "faiss":
        from langchain_community.vectorstores import FAISS
        from pathlib import Path as _P
        persist_dir     = vdb_kwargs.get("persist_dir") or vdb_kwargs.get("persist_directory", "./storage/faiss")
        collection_name = vdb_kwargs["collection_name"]
        # persist_dir = ./storage/faiss_{collection_name} (đã bao gồm collection)
        idx_dir  = _P(persist_dir)
        idx_file = idx_dir / "index.faiss"
        if idx_file.exists():
            return FAISS.load_local(
                folder_path                     = str(idx_dir),
                embeddings                      = lc_embedder,
                allow_dangerous_deserialization = True,
            )
        raise FileNotFoundError(
            f"FAISS index not found: {idx_file}\n"
            f"Expected: {idx_dir}/index.faiss và {idx_dir}/index.pkl"
        )

    if provider == "lancedb":
        import lancedb
        from langchain_community.vectorstores import LanceDB
        persist_dir = vdb_kwargs.get("persist_dir") or vdb_kwargs.get("persist_directory", "./storage/lancedb")
        db = lancedb.connect(persist_dir)
        return LanceDB(
            connection      = db,
            embedding       = lc_embedder,
            table_name      = vdb_kwargs["collection_name"],
        )

    if provider == "qdrant":
        from langchain_qdrant import Qdrant
        return Qdrant.from_existing_collection(
            embedding       = lc_embedder,
            collection_name = vdb_kwargs["collection_name"],
            url             = vdb_kwargs.get("url", "http://localhost:6333"),
            api_key         = vdb_kwargs.get("api_key"),
        )

    if provider == "weaviate":
        import weaviate
        from langchain_weaviate import WeaviateVectorStore
        client = weaviate.connect_to_local(
            host    = vdb_kwargs.get("url", "http://localhost:8080").replace("http://","").split(":")[0],
            port    = int(vdb_kwargs.get("url","http://localhost:8080").rsplit(":",1)[-1]) if ":" in vdb_kwargs.get("url","") else 8080,
            api_key = weaviate.auth.ApiKey(vdb_kwargs["api_key"]) if vdb_kwargs.get("api_key") else None,
        )
        return WeaviateVectorStore(client=client, index_name=vdb_kwargs["collection_name"], text_key="text", embedding=lc_embedder)

    if provider == "pgvector":
        from langchain_postgres import PGVector
        return PGVector(
            embeddings         = lc_embedder,
            collection_name    = vdb_kwargs["collection_name"],
            connection         = vdb_kwargs.get("connection_string", ""),
        )

    if provider == "pinecone":
        from langchain_pinecone import PineconeVectorStore
        return PineconeVectorStore(
            index_name  = vdb_kwargs["collection_name"],
            embedding   = lc_embedder,
            api_key     = vdb_kwargs.get("api_key", ""),
        )

    raise ValueError(f"_load_vector_store_readonly: provider '{provider}' không được hỗ trợ.")

def _load_index_into_session(entry: dict, cache) -> None:
    """
    Load vector store từ cache entry vào st.session_state["vdb_result"].
    Đồng thời restore emb_cfg_used, vdb_cfg_used để Generation pipeline dùng đúng.
    """
    import json
    steps       = entry.get("steps", {})

    def _latest(step_key: str) -> dict:
        step_list = steps.get(step_key, [])
        return step_list[-1] if step_list else {}

    vdb_entry   = _latest("vector_db")
    vdb_stats   = vdb_entry.get("stats", {}) or {}
    vdb_config  = vdb_entry.get("cfg",   {}) or {}
    embed_entry = _latest("embedding")
    embed_cfg   = embed_entry.get("cfg", {}) or {}
    embed_stats = embed_entry.get("stats", {}) or {}

    provider        = vdb_stats.get("provider", vdb_config.get("provider", "chroma"))
    collection_name = vdb_stats.get("collection_name", vdb_config.get("collection_name", "rag_docs"))
    persist_dir     = vdb_config.get("persist_directory", "./storage/chroma")
    n_vectors       = vdb_stats.get("n_vectors", 0)

    # Restore emb_cfg_used (cần cho query embedding)
    full_emb_cfg = {
        "provider":       embed_cfg.get("provider", "openai"),
        "model_name":     embed_cfg.get("model_name", "text-embedding-3-small"),
        "enable_sparse":  embed_cfg.get("enable_sparse", False),
        "sparse_method":  embed_cfg.get("sparse_method", "none"),
        "dimensions":     embed_cfg.get("dimensions"),
        "ollama_base_url": embed_cfg.get("ollama_base_url", "http://localhost:11434"),
        "input_type":     embed_cfg.get("input_type", "search_document"),
    }

    # Reconnect vector store (load existing, không re-index)
    try:
        from embedding.factory import get_embedder
        from vector_db.factory import get_vector_store
        embedder_kwargs = _embedder_kwargs_from_cfg(full_emb_cfg)
        embedder = get_embedder(**embedder_kwargs)

        vdb_cfg_full = {
            "provider":          provider,
            "collection_name":   collection_name,
            "persist_directory": persist_dir,
            "force_reindex":     False,
        }

        # Build provider-specific kwargs for get_vector_store
        vdb_kwargs: dict = {"collection_name": collection_name}
        if provider in ("faiss", "chroma", "lancedb"):
            vdb_kwargs["persist_dir"] = persist_dir
        elif provider == "qdrant":
            import os as _os
            vdb_kwargs["url"]     = _os.environ.get("QDRANT_URL", "http://localhost:6333")
            vdb_kwargs["api_key"] = _os.environ.get("QDRANT_API_KEY", None)
        elif provider == "weaviate":
            import os as _os
            vdb_kwargs["url"]     = _os.environ.get("WEAVIATE_URL", "http://localhost:8080")
            vdb_kwargs["api_key"] = _os.environ.get("WEAVIATE_API_KEY", None)
        elif provider == "pgvector":
            import os as _os
            vdb_kwargs["connection_string"] = _os.environ.get("DATABASE_URL", "")
        elif provider == "pinecone":
            import os as _os
            vdb_kwargs["api_key"] = _os.environ.get("PINECONE_API_KEY", "")

        # Load-only: khởi tạo trực tiếp không qua get_or_create để tránh
        # Chroma.from_documents([]) crash khi chunks rỗng.
        vector_store = _load_vector_store_readonly(provider, embedder, vdb_kwargs)
        sparse_index = None  # BM25 index không serialize được — rebuild khi cần

        vdb_result = {
            "vector_store":    vector_store,
            "sparse_index":    sparse_index,
            "provider":        provider,
            "collection_name": collection_name,
            "n_vectors":       n_vectors,
            "persist_dir":     persist_dir,
        }

        st.session_state["vdb_result"]    = vdb_result
        st.session_state["emb_cfg_used"]  = full_emb_cfg
        st.session_state["vdb_cfg_used"]  = vdb_cfg_full
        n_v_str = f"{n_vectors:,}" if isinstance(n_vectors, int) else str(n_vectors)
        st.toast(f"✅ Load thành công: {collection_name} · {n_v_str} vectors", icon="🗄️")

    except Exception as e:
        st.error(f"❌ Không thể reconnect vector store: {e}")
        st.exception(e)



