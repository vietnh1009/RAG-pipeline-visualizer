import os, sys, re, importlib, importlib.util, inspect, tempfile
from pathlib import Path
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
try:
    from dotenv import load_dotenv as _ld, dotenv_values as _dv
    _ld(override=True); _ENV = _dv()
except ImportError:
    _ENV = {}
import streamlit as st

from core.constants import EMBED_PREVIEW_LIMIT, EMBEDDING_PROVIDER_META
from utils.env import _get_env, _is_installed
from core.pipeline_runners import _load_hf_embedder
from utils.env import _detect_gpu

def render_embedding_settings(local_only: bool = False, force_skip: bool = False) -> dict:
    """
    Hiển thị panel cài đặt Embedding trong sidebar.
    Trả về dict cấu hình để truyền vào run_embedder().
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
    st.session_state["_emb_provider_cur"] = provider

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
    st.session_state["_emb_model_cur"] = model_name

    dims_this_model = meta["model_dims"].get(model_name, 0)

    # ── Ctx window guidance (replaces chunking.py message — always in sync) ─
    _CTX = {
        "text-embedding-3-small": 8191, "text-embedding-3-large": 8191,
        "text-embedding-ada-002": 8191,
        "embed-multilingual-v3.0": 512, "embed-english-v3.0": 512, "embed-v4.0": 512,
        "nomic-embed-text": 2048, "bge-m3": 8192,
        "mxbai-embed-large": 512, "snowflake-arctic-embed": 512,
        "BAAI/bge-m3": 8192, "nomic-ai/nomic-embed-text-v1.5": 8192,
        "VinAI/phobert-large": 256,
        "intfloat/multilingual-e5-large": 512, "intfloat/multilingual-e5-small": 512,
        "BAAI/bge-small-en-v1.5": 512, "BAAI/bge-base-en-v1.5": 512,
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2": 512,
        "Qwen/Qwen3-Embedding-0.6B": 32000, "Qwen/Qwen3-Embedding-4B": 32000,
        "Qwen/Qwen3-Embedding-8B": 32000,
        "Alibaba-NLP/gte-Qwen2-7B-instruct": 131000,
    }
    _ctx = _CTX.get(model_name)
    if _ctx is not None:
        _ctx_k = f"{_ctx:,}"
        _ctx_chars = f"~{_ctx*3:,}"
        if _ctx <= 512:
            st.warning(
                f"⚠️ **`{model_name}`** ctx window chỉ **{_ctx_k} token** ({_ctx_chars} ký tự). "
                f"Chunk dài hơn sẽ bị truncate — nên bật **Chia nhỏ section** ở Chunking.",
                icon="⚠️",
            )
        else:
            st.caption(f"ℹ️ Context window: {_ctx_k} token ({_ctx_chars} ký tự). "
                       f"Chunk dài hơn sẽ tự động bị truncate.")

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
        if meta.get("install") and meta.get("pkg_probe") and not _is_installed(meta["pkg_probe"]):
            st.caption(f"📦 Cần cài: `{meta['install']}`")

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
        # ctx window info shown above (after model selectbox)

        ollama_base_url = st.text_input(
            "Ollama base URL",
            value=st.session_state.get("ollama_emb_url", "http://localhost:11434"),
            key="ollama_emb_url",
            help="URL Ollama server (không có /v1). Model tự pull nếu chưa có.",
        )
        # Auto-check khi mới chọn model (không cần bấm nút)
        try:
            import ollama as _ollama_auto
            _client_auto = _ollama_auto.Client(host=ollama_base_url)
            _pulled_auto = [m.model.split(":")[0] for m in _client_auto.list().models]
            _model_base  = model_name.split(":")[0]
            if _model_base in set(_pulled_auto):
                st.success(f"✅ `{model_name}` đã sẵn sàng.")
            else:
                st.warning(
                    f"⚠️ Kết nối OK nhưng `{model_name}` chưa pull.\n\n"
                    f"`ollama pull {model_name}`"
                )
        except Exception:
            pass  # Không hiển thị lỗi nếu Ollama chưa chạy (manual test button sẽ xử lý)

        if st.button("🔌 Test Ollama & kiểm tra model", key="test_ollama_emb"):
            with st.spinner("Đang kiểm tra..."):
                try:
                    import ollama as _ollama
                    client = _ollama.Client(host=ollama_base_url)
                    pulled = [m.model for m in client.list().models]
                    # Ollama thêm ':latest' vào tên model — cần normalize khi so sánh
                    pulled_normalized = {m.split(":")[0] for m in pulled}
                    model_base = model_name.split(":")[0]
                    if model_base in pulled_normalized:
                        st.success(f"✅ Kết nối OK · `{model_name}` đã sẵn sàng.")
                    else:
                        st.warning(
                            f"⚠️ Kết nối OK nhưng `{model_name}` chưa pull.\n\n"
                            f"`ollama pull {model_name}`"
                        )
                except Exception as exc:
                    st.error(f"❌ Không kết nối được: `{exc}`")

    elif provider == "fastembed":
        if meta.get("install") and meta.get("pkg_probe") and not _is_installed(meta["pkg_probe"]):
            st.caption(f"📦 Cần cài: `{meta['install']}`")

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
        if sparse_method == "splade" and not _is_installed("torch"):
            st.caption("💡 `pip install transformers torch` và cần GPU cho tốc độ hợp lý.")
        st.info(
            "ℹ️ BM25 bắt buộc phải **fit trên toàn bộ corpus** trước khi embed — "
            "bước này được thực hiện tự động khi ấn **Process**."
        )

    # ── Preview limit ────────────────────────────────────────────────────────
    st.markdown("")
    _embed_all = st.checkbox(
        "Embed toàn bộ chunk (production mode)",
        value=False,
        key="emb_embed_all",
        help=(
            "Bật để embed TẤT CẢ chunk — dùng khi đã sẵn sàng index thật cho production. "
            "⚠️ Có thể tốn nhiều API cost và thời gian với corpus lớn."
        ),
    )
    if _embed_all:
        st.warning(
            "⚠️ **Embed toàn bộ** đã bật. Tất cả chunk sẽ được embed — "
            "hãy đảm bảo bạn đã sẵn sàng chịu API cost tương ứng.",
            icon="💸",
        )
        max_preview = 999_999   # effectively unlimited
    else:
        max_preview = st.number_input(
            "Số chunk tối đa để embed (preview)",
            min_value=1,
            max_value=10_000,
            value=EMBED_PREVIEW_LIMIT,
            step=5,
            help=(
                f"Giới hạn số chunk gửi lên embedder để kiểm soát API cost và RAM. "
                f"Mặc định: {EMBED_PREVIEW_LIMIT}. "
                "Tăng dần để embed nhiều hơn. Bật 'Embed toàn bộ' để index production."
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



