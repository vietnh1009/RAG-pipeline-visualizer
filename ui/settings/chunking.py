import os, sys, re, importlib, importlib.util, inspect, tempfile
from pathlib import Path
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
try:
    from dotenv import load_dotenv as _ld, dotenv_values as _dv
    _ld(override=True); _ENV = _dv()
except ImportError:
    _ENV = {}
import streamlit as st

from core.constants import LLM_REQUIRED_STRATEGIES
from utils.env import _is_installed, get_chunking_strategies, _get_env

def render_chunking_settings(local_only: bool = False) -> tuple[str, int, int, dict]:
    """Hiển thị panel cài đặt chunking, trả về (strategy, chunk_size, overlap, extra)."""

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
        # Dùng giá trị được pass trực tiếp từ _page_indexing() sau khi
        # render_embedding_settings() đã chạy → luôn đúng, không bao giờ stale.
        _emb_provider = st.session_state.get("sel_emb_provider", "openai")
        _emb_model    = st.session_state.get("sel_emb_model", "text-embedding-3-small")
        _emb_skip     = st.session_state.get("emb_skip", False)

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
            # Ollama medium ctx (2048 token — đủ cho hầu hết chunk)
            "nomic-embed-text",
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
                "nomic-embed-text": "2,048 token",
                "Qwen/Qwen3-Embedding-0.6B": "32K token",
                "Qwen/Qwen3-Embedding-4B": "32K token",
                "Qwen/Qwen3-Embedding-8B": "32K token",
                "Alibaba-NLP/gte-Qwen2-7B-instruct": "131K token",
            }.get(_emb_model, "ctx window lớn")
            # Luôn dùng model/provider hiện tại từ session (có thể đã thay đổi)
            _cur_model    = st.session_state.get("sel_emb_model", _emb_model)
            _cur_provider = st.session_state.get("sel_emb_provider", _emb_provider)
            _cur_ctx_label = {
                "BAAI/bge-m3": "8192 token", "bge-m3": "8192 token",
                "nomic-embed-text": "2,048 token",
                "nomic-ai/nomic-embed-text-v1.5": "8192 token",
                "Qwen/Qwen3-Embedding-0.6B": "32K token",
                "Qwen/Qwen3-Embedding-4B": "32K token",
                "Qwen/Qwen3-Embedding-8B": "32K token",
                "Alibaba-NLP/gte-Qwen2-7B-instruct": "131K token",
                "text-embedding-3-small": "8,191 token",
                "text-embedding-3-large": "8,191 token",
                "text-embedding-ada-002": "8,191 token",
                "embed-multilingual-v3.0": "512 token (auto-truncate)",
                "embed-english-v3.0": "512 token (auto-truncate)",
            }.get(_cur_model, "ctx window lớn")
            _provider_label = {
                "openai": f"OpenAI/{_cur_model.split('/')[-1]} (8,191 token)",
                "cohere": f"Cohere/{_cur_model.split('/')[-1]} (512 token, auto-truncate)",
            }.get(_cur_provider, "")
            _display = _provider_label or f"{_cur_model} ({_cur_ctx_label})"
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


