import os, sys, re, importlib, importlib.util, inspect, tempfile
from pathlib import Path
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
try:
    from dotenv import load_dotenv as _ld, dotenv_values as _dv
    _ld(override=True); _ENV = _dv()
except ImportError:
    _ENV = {}
import streamlit as st

from pipeline_cache import PipelineCache

def _embedder_kwargs_from_cfg(emb_cfg: dict) -> dict:
    """
    Build kwargs cho get_embedder() từ emb_cfg dict.
    Mỗi provider chỉ nhận đúng kwargs của nó — không truyền thừa.
    """
    p  = emb_cfg["provider"]
    mn = emb_cfg["model_name"]
    ex: dict = {}
    if p == "openai":
        dims = emb_cfg.get("dimensions")
        if dims:
            ex["dimensions"] = dims
    elif p == "cohere":
        ex["input_type"] = emb_cfg.get("input_type", "search_document")
    elif p == "ollama":
        ex["base_url"] = emb_cfg.get("ollama_base_url", "http://localhost:11434")
    elif p == "huggingface":
        ex["device"] = emb_cfg.get("device", "cpu")
    return {"provider": p, "model_name": mn, **ex}


@st.cache_resource(show_spinner=False)
def _get_cached_embedder(provider: str, model_name: str, **kwargs):
    """
    Cache embedder theo provider+model — singleton per process.
    Tránh re-import torch / re-load tokenizer mỗi lần Load index.
    st.cache_resource tồn tại suốt vòng đời Streamlit server process.
    """
    from embedding.factory import get_embedder
    return get_embedder(provider=provider, model_name=model_name, **kwargs)



@st.cache_data(show_spinner=False, ttl=10)
def _cached_list_entries(_cache_dir: str = "processed_data", _version: int = 0) -> list:
    """Cache list_entries() theo version token. TTL=10s safety net."""
    from pipeline_cache import PipelineCache
    return PipelineCache(_cache_dir).list_entries()


@st.cache_data(show_spinner=False, ttl=10)
def _cached_list_pipelines(_cache_dir: str = "processed_data", _version: int = 0) -> list:
    """
    Cache list_complete_pipelines() — trả về các pipeline hoàn chỉnh.
    TTL=10s: safety net để không bao giờ hiển thị dữ liệu cũ quá 10 giây,
    ngay cả khi _version không được tăng đúng cách.
    """
    from pipeline_cache import PipelineCache
    return PipelineCache(_cache_dir).list_complete_pipelines()


def _invalidate_list_entries_cache() -> None:
    """Tăng version counter → _cached_list_entries sẽ re-scan disk ở lần gọi tiếp theo."""
    st.session_state["_cache_list_version"] = (
        st.session_state.get("_cache_list_version", 0) + 1
    )

def _try_autoload_latest_index() -> None:
    """
    Tự động load index gần nhất khi vào Generation page lần đầu.

    Lazy strategy: chỉ list pipelines (đọc JSON nhỏ, nhanh).
    Không khởi tạo embedder hay kết nối VDB ở đây — chỉ restore
    metadata vào session_state. Kết nối thực sự xảy ra khi user
    ấn Run (lazy connect).
    """
    try:
        pipelines = _cached_list_pipelines(
            "processed_data",
            st.session_state.get("_cache_list_version", 0),
        )
        if not pipelines:
            return
        p = pipelines[0]

        # Chỉ lưu metadata — KHÔNG gọi get_embedder() hay _load_vector_store_readonly()
        ec  = p["embedding"]["cfg"]
        vs  = p["vector_db"]["stats"]
        vc  = p["vector_db"]["cfg"]
        full_emb_cfg = {
            "provider":        ec.get("provider",       "openai"),
            "model_name":      ec.get("model_name",     "text-embedding-3-small"),
            "enable_sparse":   ec.get("enable_sparse",  False),
            "sparse_method":   ec.get("sparse_method",  "none"),
            "dimensions":      ec.get("dimensions"),
            "ollama_base_url": ec.get("ollama_base_url","http://localhost:11434"),
            "input_type":      ec.get("input_type",     "search_document"),
        }
        provider        = vs.get("provider",        vc.get("provider",        "chroma"))
        collection_name = vs.get("collection_name", vc.get("collection_name", "rag_docs"))
        persist_dir     = vc.get("persist_directory", vc.get("persist_dir",   "./storage/chroma"))
        n_vectors       = vs.get("n_vectors", 0)

        # Đánh dấu "pending connect" — vector_store chưa được khởi tạo
        st.session_state["_autoload_pending"] = {
            "pipeline": p,
            "full_emb_cfg": full_emb_cfg,
            "provider": provider,
            "collection_name": collection_name,
            "persist_dir": persist_dir,
            "n_vectors": n_vectors,
        }
        # Restore metadata ngay để UI hiển thị "Index đang dùng"
        st.session_state["emb_cfg_used"]  = full_emb_cfg
        st.session_state["vdb_cfg_used"]  = {
            "provider":          provider,
            "collection_name":   collection_name,
            "persist_directory": persist_dir,
            "force_reindex":     False,
        }
        st.session_state["_loaded_pipeline_id"]    = p.get("pipeline_id", "")
        st.session_state["_current_pipeline_meta"] = {
            "source_path":  p.get("source_path", ""),
            "input_display": p.get("input_display", ""),
            "created_at":   p.get("created_at", "")[:16].replace("T", " "),
            "loader_cfg":   p["loader"]["cfg"],
            "chunking_cfg": p["chunking"]["cfg"],
            "embed_cfg":    full_emb_cfg,
            "vdb_cfg":      st.session_state["vdb_cfg_used"],
            "n_chunks":     p["chunking"]["stats"].get("n_chunks", "?"),
            "n_vectors":    n_vectors,
        }
    except Exception:
        pass   # Fail silently


def _resolve_autoload_pending() -> None:
    """
    Hoàn tất kết nối VDB cho autoload pending (gọi khi user ấn Run).
    Chỉ chạy 1 lần — sau đó xoá "_autoload_pending" khỏi session_state.
    """
    pending = st.session_state.pop("_autoload_pending", None)
    if pending is None:
        return
    if "vdb_result" in st.session_state:
        return  # đã connect rồi

    try:
        p               = pending["pipeline"]
        full_emb_cfg    = pending["full_emb_cfg"]
        provider        = pending["provider"]
        collection_name = pending["collection_name"]
        persist_dir     = pending["persist_dir"]
        n_vectors       = pending["n_vectors"]

        kwargs = _embedder_kwargs_from_cfg(full_emb_cfg)
        embedder = _get_cached_embedder(**kwargs)
        vector_store = _load_vector_store_readonly(provider, embedder, {
            "collection_name": collection_name,
            "persist_dir":     persist_dir,
        })
        st.session_state["vdb_result"] = {
            "vector_store":    vector_store,
            "sparse_index":    None,
            "provider":        provider,
            "collection_name": collection_name,
            "n_vectors":       n_vectors,
            "persist_dir":     persist_dir,
        }
    except Exception as e:
        st.error(f"❌ Không thể kết nối index: {e}")



def _pipeline_label(p: dict) -> str:
    """Short label for a pipeline entry."""
    src   = p.get("source_path", "-").replace("\\", "/").split("/")[-1]
    src   = ("..." + src[-22:]) if len(src) > 25 else src
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
        kwargs   = _embedder_kwargs_from_cfg(full_emb_cfg)
        embedder = _get_cached_embedder(**kwargs)   # cache_resource — không re-init nếu đã có
        vector_store = _load_vector_store_readonly(provider, embedder, {
            "collection_name": collection_name,
            "persist_dir":     persist_dir,
        })
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
        st.session_state["_current_pipeline_meta"] = {
            "source_path":   p.get("source_path", ""),
            "input_display": p.get("input_display", ""),
            "created_at":    p.get("created_at", "")[:16].replace("T", " "),
            "loader_cfg":    p["loader"]["cfg"],
            "chunking_cfg":  p["chunking"]["cfg"],
            "embed_cfg":     full_emb_cfg,
            "vdb_cfg":       vdb_cfg_full,
            "n_chunks":      p["chunking"]["stats"].get("n_chunks", "?"),
            "n_vectors":     n_vectors,
        }
        nv_s = f"{n_vectors:,}" if isinstance(n_vectors, int) else str(n_vectors)
        st.toast(f"✅ Đã load: {collection_name} · {nv_s} vectors", icon="🗄️")
    except Exception as e:
        st.error(f"❌ Không thể load index: {e}")
        st.exception(e)

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
    nv   = vs.get("n_vectors", 0)
    nv_s = f"{nv:,}" if isinstance(nv, int) else str(nv)
    sp   = f" + {ec.get('sparse_method','')}" if ec.get("enable_sparse") else ""
    st.markdown(
        f"<small style='line-height:1.7;opacity:.8;'>"
        f"📄 {p.get('source_path','-').split('/')[-1]}"
        f"  📅 {p.get('created_at','')[:16].replace('T',' ')}<br>"
        f"📄 <code>{lc.get('pdf_strategy','?')}</code>"
        f" → ✂️ <code>{cc.get('strategy','?')}</code>"
        f" ({cc.get('chunk_size','?')} chars · {cs.get('n_chunks','?')} chunks)<br>"
        f"🧮 <code>{ec.get('provider','?')}/{ec.get('model_name','?').split('/')[-1]}{sp}</code>"
        f" → 🗃️ <code>{vs.get('provider','?')}/{vs.get('collection_name','?')}</code>"
        f" · {nv_s} vecs</small>",
        unsafe_allow_html=True,
    )
    if st.button("✅ Dùng pipeline này", key="btn_switch_index",
                 type="primary", use_container_width=True):
        _load_pipeline_into_session(p)
        st.rerun()


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
        st.markdown(f"**{len(valid)} index** — nhấn **Load** để dùng ngay.")
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

        # Source display: show folder name + file(s) if possible
        src_parts = source_path.replace("\\", "/").split(",")
        src_display = ", ".join(p.strip().split("/")[-1] for p in src_parts[:3])
        if len(src_parts) > 3:
            src_display += f" ... (+{len(src_parts)-3} files)"
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
                    st.markdown(
                        f"<div style='font-size:.82em;line-height:1.6;'>"
                        f"<b>📂 Nguồn</b><br>"
                        f"<span style='opacity:.7;'>{n_docs} docs · {language}</span><br>"
                        f"<b>📄 Loader</b><br>"
                        f"<code>{pdf_strat}</code>"
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
                        f"<code>{vdb_provider}</code><br>"
                        f"<span style='opacity:.75;'>{collection_name}</span><br>"
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
        from vector_db.factory import get_vector_store
        embedder_kwargs = _embedder_kwargs_from_cfg(full_emb_cfg)
        embedder = _get_cached_embedder(**embedder_kwargs)  # cache_resource

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



