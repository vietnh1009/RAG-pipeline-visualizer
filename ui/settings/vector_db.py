import os, sys, re, importlib, importlib.util, inspect, tempfile
from pathlib import Path
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
try:
    from dotenv import load_dotenv as _ld, dotenv_values as _dv
    _ld(override=True); _ENV = _dv()
except ImportError:
    _ENV = {}
import streamlit as st
from core.constants import VECTOR_DB_PROVIDER_META
from utils.env import _get_env
def render_vector_db_settings(local_only: bool = False, force_skip: bool = False) -> dict:
    """
    Hiển thị panel cài đặt Vector Database trong sidebar.
    Trả về dict cấu hình để truyền vào get_vector_store().
    """

    # ── Apply suggestion "Áp dụng cấu hình này" ─────────────────────────────
    # _want_* keys được set bởi pipeline_suggestions khi user click "Áp dụng"
    # Copy sang widget keys TRƯỚC khi widget render để Streamlit hiển thị đúng
    _want_map = {
        "_want_pdf_strategy":      "sel_pdf_strategy",
        "_want_chunking_strategy": "sel_chunking_strategy",
        "_want_format_type":       "sel_format_type",
        "_want_emb_provider":      "sel_emb_provider",
        "_want_emb_model":         "sel_emb_model",
        "_want_emb_sparse":        "emb_enable_sparse",
        "_want_emb_sparse_method": "emb_sparse_method",
        "_want_vdb_provider":      "sel_vdb_provider",
    }
    for _wk, _sk in _want_map.items():
        if _wk in st.session_state:
            st.session_state[_sk] = st.session_state.pop(_wk)
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

    # ── Auto-generate collection name ──────────────────────────────────────
    # Tạo tên unique từ: emb_provider + emb_model_short + chunk_strategy + timestamp
    # Chỉ auto-generate nếu user chưa chỉnh tay hoặc khi config thay đổi
    def _auto_collection_name() -> str:
        """Tạo collection name unique từ config hiện tại.
        Format: {loader}_{chunk}_{emb}_{dd}{Mon}{yyyy}_{HH}{mm}{ss}
        Ví dụ: odl_fmt_nomic_04Jun2026_105203
        """
        import datetime, re
        _MON = ["Jan","Feb","Mar","Apr","May","Jun",
                "Jul","Aug","Sep","Oct","Nov","Dec"]
        now  = datetime.datetime.now()
        ts   = f"{now.day:02d}{_MON[now.month-1]}{now.year}_{now.hour:02d}{now.minute:02d}{now.second:02d}"

        # Loader
        pdf_strat   = st.session_state.get("sel_pdf_strategy", "pypdf")
        loader_short = {
            "pypdf": "pypdf", "pymupdf": "mupdf", "pdfplumber": "plumb",
            "marker": "mrkr", "opendataloader": "odl", "docling": "dclg",
            "unstructured": "unst",
        }.get(pdf_strat, pdf_strat[:5])

        # Chunking
        chunk_strat  = st.session_state.get("sel_chunking_strategy", "recursive")
        chunk_short  = {
            "recursive": "rec", "format_aware": "fmt", "semantic": "sem",
            "contextual": "ctx", "sentence": "sent", "token": "tok",
            "late_chunking": "late",
        }.get(chunk_strat, chunk_strat[:4])

        # Embedding
        emb_prov  = st.session_state.get("sel_emb_provider", "")
        emb_model = st.session_state.get("sel_emb_model", "").split("/")[-1].split(":")[0]
        emb_clean = re.sub(r"[^a-z0-9]", "", emb_model.lower())[:12]
        if not emb_clean:
            emb_clean = re.sub(r"[^a-z0-9]", "", emb_prov.lower())[:6] or "emb"

        return f"{loader_short}_{chunk_short}_{emb_clean}_{ts}"

    # Config fingerprint: thay đổi khi user đổi bất kỳ thành phần nào
    _cfg_sig = (
        st.session_state.get("sel_pdf_strategy", ""),      # loader
        st.session_state.get("sel_chunking_strategy", ""), # chunking
        st.session_state.get("sel_emb_provider", ""),      # embedding provider
        st.session_state.get("sel_emb_model", ""),         # embedding model
        provider,                                           # vdb provider
    )
    # Auto-generate lần đầu hoặc khi config thay đổi
    if ("vdb_collection_name" not in st.session_state
            or st.session_state.get("_vdb_cfg_sig") != _cfg_sig
            and not st.session_state.get("_vdb_name_user_edited")):
        st.session_state["vdb_collection_name"] = _auto_collection_name()
        st.session_state["_vdb_cfg_sig"] = _cfg_sig

    # ── Common params ───────────────────────────────────────────────────────
    _prev_name = st.session_state.get("vdb_collection_name", "")
    collection_name = st.text_input(
        "Collection / Index name",
        key="vdb_collection_name",
        help=(
            "Tên collection/index. **Tự động tạo** từ embedding model + chunk strategy + timestamp "
            "để tránh trùng lặp. Bạn có thể chỉnh tay — tên sẽ được giữ nguyên khi config thay đổi."
        ),
    )
    # Theo dõi nếu user tự sửa tên
    if collection_name != _prev_name and _prev_name:
        st.session_state["_vdb_name_user_edited"] = True
    # Reset flag khi user xóa hết text (muốn auto lại)
    if not collection_name:
        st.session_state["_vdb_name_user_edited"] = False
        st.session_state["vdb_collection_name"] = _auto_collection_name()
        collection_name = st.session_state["vdb_collection_name"]

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

    # Local providers — persist_dir tự động, không cần user nhập thủ công
    if "persist_dir" in params:
        # Format: ./storage/{provider}_{collection_name}
        # Toàn bộ index nằm trong 1 thư mục duy nhất, dễ backup/xóa
        persist_dir = f"./storage/{provider}_{collection_name}"
        cfg["persist_dir"] = persist_dir
        st.caption(f"📁 Lưu tại: `{persist_dir}/`")

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



