import os, sys, re, importlib, importlib.util, inspect, tempfile
from pathlib import Path
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
try:
    from dotenv import load_dotenv as _ld, dotenv_values as _dv
    _ld(override=True); _ENV = _dv()
except ImportError:
    _ENV = {}
import streamlit as st

from core.constants import PDF_STRATEGY_DEPS
from utils.env import _get_env, _is_installed, _check_java_version, _is_pdf_strategy_installed, _detect_gpu, get_pdf_strategies
from utils.badges import file_type_badge

def _render_vlm_panel(key_prefix: str) -> None:
    """
    Panel chọn VLM để mô tả ảnh — dùng chung cho marker và unstructured.
    Kết quả được ghi vào session_state với key có prefix:
      _vlm_describe_{key_prefix}  : bool
      _vlm_provider_{key_prefix}  : "openai" | "ollama"
      _vlm_model_{key_prefix}     : str
      _vlm_url_{key_prefix}       : str
    """
    describe = st.checkbox(
        "Dùng VLM mô tả ảnh trong PDF",
        value=False,
        key=f"_vlm_describe_{key_prefix}",
        help="VLM đọc ảnh và sinh mô tả text để embedding. Cần API key hoặc Ollama local.",
    )

    if not describe:
        return

    vision_provider = st.radio(
        "VLM Provider",
        ["openai", "ollama"],
        horizontal=True,
        key=f"_vlm_provider_{key_prefix}",
        help="openai: cần OPENAI_API_KEY · ollama: chạy local, miễn phí",
    )

    if vision_provider == "openai":
        st.selectbox(
            "VLM Model",
            ["gpt-4o-mini", "gpt-4o"],
            key=f"_vlm_model_{key_prefix}",
            help="gpt-4o-mini: nhanh & rẻ · gpt-4o: chất lượng cao hơn",
        )
        st.caption("💡 `OPENAI_API_KEY` được load từ `.env` hoặc biến môi trường.")
    else:
        st.selectbox(
            "VLM Model (Ollama)",
            ["llava:7b", "llava:13b", "llava-llama3", "moondream",
             "minicpm-v", "qwen2-vl:7b", "qwen2.5vl:7b", "qwen2.5vl:3b", "glm4v"],
            key=f"_vlm_model_{key_prefix}",
            help="Cần pull model trước: `ollama pull <model>`. App tự pull nếu chưa có.",
        )
        st.text_input(
            "Ollama base URL",
            value="http://localhost:11434/v1",
            key=f"_vlm_url_{key_prefix}",
            help="Thay đổi nếu Ollama chạy ở host/port khác",
        )
        if st.button("🔌 Test kết nối Ollama", key=f"test_ollama_{key_prefix}"):
            _url = st.session_state.get(f"_vlm_url_{key_prefix}", "http://localhost:11434/v1")
            _mdl = st.session_state.get(f"_vlm_model_{key_prefix}", "llava:7b")
            with st.spinner("Đang kiểm tra..."):
                try:
                    from openai import OpenAI as _OAI
                    c = _OAI(base_url=_url, api_key="ollama")
                    models = [m.id for m in c.models.list().data]
                    if _mdl in set(models):
                        st.success(f"✅ Kết nối OK · `{_mdl}` sẵn sàng")
                    else:
                        st.warning(
                            f"⚠️ Kết nối OK nhưng `{_mdl}` chưa được pull\n\n"
                            f"Models có sẵn: `{'`, `'.join(models[:6])}`"
                        )
                        st.info("💡 Ấn **Process** — app sẽ tự động pull model trước khi mô tả ảnh.")
                except Exception as e:
                    st.error(f"❌ Không kết nối được Ollama: `{e}`")
                    st.caption("Đảm bảo Ollama đang chạy: `ollama serve`")


@st.cache_data(ttl=3, show_spinner=False)
def _check_odl_server_running(port: int) -> bool:
    """Kiểm tra ODL hybrid server có đang chạy không (cache 3 giây)."""
    import socket
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=1):
            return True
    except OSError:
        return False


def _start_odl_server(port: int, force_ocr: bool = False) -> None:
    """Khởi động opendataloader-pdf-hybrid server trong background."""
    import subprocess, sys, os
    cmd = [sys.executable, "-m", "opendataloader_pdf.hybrid_server",
           "--port", str(port)]
    if force_ocr:
        cmd += ["--force-ocr", "--ocr-lang", "vi,en"]
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        st.session_state["_odl_server_pid"] = proc.pid
        st.toast(f"✅ Đã khởi động server (PID {proc.pid}) tại port {port}", icon="🚀")
    except Exception as e:
        # Fallback: try shell command
        try:
            shell_cmd = f"opendataloader-pdf-hybrid --port {port}"
            if force_ocr:
                shell_cmd += ' --force-ocr --ocr-lang "vi,en"'
            proc = subprocess.Popen(
                shell_cmd, shell=True,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            st.session_state["_odl_server_pid"] = proc.pid
            st.toast(f"✅ Server khởi động tại port {port}", icon="🚀")
        except Exception as e2:
            st.error(f"❌ Không thể khởi động server: {e2}")



def _start_odl_server_with_feedback(port: int, force_ocr: bool = False) -> None:
    """
    Khởi động ODL server + set flag. Polling và st.status() được render
    trong sidebar nhưng NGOÀI st.columns() → chiếm full width của sidebar.
    """
    _start_odl_server(port, force_ocr=force_ocr)
    st.session_state["_odl_starting"]      = True
    st.session_state["_odl_start_port"]    = port
    st.session_state["_odl_start_elapsed"] = 0.0
    st.rerun()



def render_loader_settings() -> dict:
    """Hiển thị panel cài đặt loader, trả về dict các tham số."""

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
    st.subheader("⚙️ Cài đặt Loader")

    col1, col2 = st.columns(2)
    with col1:
        strategies = get_pdf_strategies()
        # Khởi tạo default một lần — tránh conflict khi Apply button ghi vào session state
        if "sel_pdf_strategy" not in st.session_state:
            st.session_state["sel_pdf_strategy"] = "pypdf"
        pdf_strategy = st.selectbox(
            "PDF Strategy",
            options=strategies,
            key="sel_pdf_strategy",
            help=(
                "**pypdf** — nhanh, chỉ text layer (không cần cài thêm)\n\n"
                "**pymupdf** — nhanh hơn, layout tốt hơn · `pip install pymupdf`\n\n"
                "**pdfplumber** — trích bảng tốt nhất · `pip install pdfplumber`\n\n"
                "**unstructured** — tốt nhất: OCR + bảng + hình · `pip install 'unstructured[pdf]' unstructured-inference`\n\n"
                "**docling** — IBM parser, Markdown output xuất sắc · `pip install docling`\n\n"
                "**marker** — Markdown chất lượng cao, bảng & LaTeX · `pip install marker-pdf`\n\n"
                "**opendataloader** — #1 benchmark (0.90), bounding box, no GPU · `pip install opendataloader-pdf` · **Java 11+ required**"
            ),
        )
        # Chỉ hiện cảnh báo khi thư viện chưa được cài
        if not _is_pdf_strategy_installed(pdf_strategy):
            entry = PDF_STRATEGY_DEPS.get(pdf_strategy)
            if entry:
                _, install_cmd = entry
                st.caption(f"⚠️ Cần cài thêm: `{install_cmd}`")
        extract_tables = st.checkbox("Trích xuất bảng → Markdown", value=True)

    with col2:
        # Các strategy có OCR tích hợp sẵn → disable OCR engine selector
        _BUILTIN_OCR: dict[str, str] = {
            "marker":         "Marker tích hợp **Surya OCR** — nhận dạng 90+ ngôn ngữ, không cần chọn thêm.",
            "docling":        "Docling tích hợp **RapidOCR** — không cần chọn thêm OCR engine.",
            "opendataloader": "OpenDataLoader tích hợp OCR trong hybrid mode — không cần chọn thêm OCR engine.",
            "pypdf":          "pypdf chỉ đọc text layer — OCR không áp dụng cho strategy này.",
            "pymupdf":        "PyMuPDF chỉ đọc text layer — OCR không áp dụng cho strategy này.",
            "pdfplumber":     "pdfplumber chỉ đọc text layer — OCR không áp dụng cho strategy này.",
        }
        ocr_disabled = pdf_strategy in _BUILTIN_OCR



        language = st.selectbox(
            "Ngôn ngữ corpus",
            options=["both", "vi", "en"],
            index=0,
            help="Dùng để chọn ngôn ngữ cho OCR và NLP tools"
        )

    # ── Device + VLM (full width, chỉ hiện khi marker) ───────────────────────
    marker_device   = "cpu"
    describe_images = False
    vision_provider = "openai"
    vision_model    = "gpt-4o-mini"
    ollama_base_url = "http://localhost:11434/v1"
    # ── OpenDataLoader options (chỉ hiện khi opendataloader) ─────────────────
    odl_hybrid      = None   # None = fast mode; "docling-fast" = hybrid mode
    odl_struct_tree = False

    if pdf_strategy == "marker":
        st.markdown("---")
        st.subheader("🖥️ Device cho Marker")
        auto_device, gpu_label = _detect_gpu()
        if auto_device != "cpu":
            st.success(f"✅ Phát hiện GPU: {gpu_label}")
        else:
            st.warning("⚠️ Không phát hiện GPU — Marker sẽ chạy trên CPU (~30-60s/trang)")

        device_options = [auto_device, "cpu"] if auto_device != "cpu" else ["cpu"]
        marker_device = st.radio(
            "Chọn device",
            options=device_options,
            index=0,
            horizontal=True,
            help="cuda: NVIDIA GPU · mps: Apple Silicon · cpu: chậm nhất",
        )

        st.markdown("---")
        _render_vlm_panel(key_prefix="marker")
        describe_images = st.session_state.get("_vlm_describe_marker", False)
        vision_provider  = st.session_state.get("_vlm_provider_marker", "openai")
        vision_model     = st.session_state.get("_vlm_model_marker", "gpt-4o-mini")
        ollama_base_url  = st.session_state.get("_vlm_url_marker", "http://localhost:11434/v1")

    elif pdf_strategy == "unstructured":
        st.markdown("---")
        st.subheader("🖼️ Mô tả ảnh bằng VLM")
        st.caption(
            "Unstructured detect được hình ảnh trong PDF (khi dùng `hi_res` strategy với OCR). "
            "Bật option này để dùng VLM sinh mô tả text cho từng ảnh, giúp embedding capture nội dung hình."
        )
        _render_vlm_panel(key_prefix="unstructured")
        describe_images = st.session_state.get("_vlm_describe_unstructured", False)
        vision_provider  = st.session_state.get("_vlm_provider_unstructured", "openai")
        vision_model     = st.session_state.get("_vlm_model_unstructured", "gpt-4o-mini")
        ollama_base_url  = st.session_state.get("_vlm_url_unstructured", "http://localhost:11434/v1")



    elif pdf_strategy == "opendataloader":
        st.markdown("---")
        st.subheader("⚙️ Cài đặt OpenDataLoader PDF")

        # ── Java availability check (cached) ─────────────────────────────────
        _java_ok, _java_msg = _check_java_version()
        if _java_ok:
            st.success(f"✅ Java đã sẵn sàng: `{_java_msg}`")
        elif _java_msg == "not_found":
            st.error(
                "❌ **Java chưa được cài đặt** — opendataloader-pdf yêu cầu Java 11+.\n\n"
                "Tải tại: https://adoptium.net/"
            )
        else:
            st.error("❌ Java không hoạt động. Cần Java 11+.")

        # Only show install hint if package is actually missing
        if not _is_pdf_strategy_installed("opendataloader"):
            st.caption("`pip install opendataloader-pdf`")

        # ── Mode: fast vs hybrid ──────────────────────────────────────────────
        st.markdown("")
        odl_mode = st.radio(
            "Mode",
            options=["fast", "hybrid"],
            index=0,
            horizontal=True,
            help=(
                "**fast** — Java local, deterministic, 0.05s/trang, accuracy 0.72. "
                "Không cần cài thêm.\n\n"
                "**hybrid** — AI backend routing, 0.43s/trang, accuracy **0.90 (#1 benchmark)**. "
                "Tốt cho bảng phức tạp, PDF scan, công thức. "
                "Cần: `pip install \"opendataloader-pdf[hybrid]\"` và server đang chạy."
            ),
            key="sel_odl_mode",
        )

        if odl_mode == "hybrid":
            odl_hybrid = "docling-fast"
            st.markdown("**Hybrid Server**")
            col_port, col_ocr = st.columns([2, 1])
            with col_port:
                _odl_port = st.number_input(
                    "Port",
                    min_value=1024, max_value=65535,
                    value=5002, step=1,
                    help="Port của opendataloader-pdf-hybrid server. Mặc định: 5002.",
                    key="odl_hybrid_port",
                )
            with col_ocr:
                _odl_ocr = st.checkbox(
                    "Force OCR (VI/scan)",
                    value=False,
                    key="odl_force_ocr",
                    help="Thêm --force-ocr --ocr-lang vi,en — cần cho PDF scan hoặc tiếng Việt.",
                )
            st.session_state["_odl_hybrid_port"] = int(_odl_port)

            # Check if server is already running
            _srv_running = _check_odl_server_running(int(_odl_port))

            if _srv_running:
                st.success(f"✅ Server đang chạy tại port **{int(_odl_port)}**")
            else:
                _is_starting = st.session_state.get("_odl_starting", False)
                st.warning(f"⚠️ Server chưa chạy tại port {int(_odl_port)}")

                # Nút + command nằm trong 2 cột
                col_a, col_b = st.columns(2)
                with col_a:
                    if _is_starting:
                        st.button("⏳ Đang khởi động...", key="btn_start_odl",
                                  type="primary", use_container_width=True, disabled=True)
                    elif st.button("▶️ Khởi động server", key="btn_start_odl",
                                   type="primary", use_container_width=True):
                        _start_odl_server_with_feedback(int(_odl_port), force_ocr=_odl_ocr)
                with col_b:
                    cmd = f"opendataloader-pdf-hybrid --port {int(_odl_port)}"
                    if _odl_ocr:
                        cmd += ' --force-ocr --ocr-lang "vi,en"'
                    st.code(cmd, language="bash")

                # Status box NGOÀI columns → full width của sidebar
                if _is_starting:
                    import time as _t
                    _port    = st.session_state.get("_odl_start_port", int(_odl_port))
                    _elapsed = st.session_state.get("_odl_start_elapsed", 0.0)
                    _max_wait = 25.0

                    with st.status(
                        f"⏳ Đang khởi động... ({_elapsed:.0f}s)",
                        expanded=True,
                    ) as _status:
                        st.write("🚀 Đã gửi lệnh khởi động")
                        st.write(f"⌛ Đợi JVM + model load (thường 5–15 giây)...")

                        _t.sleep(1.5)
                        _elapsed += 1.5
                        st.session_state["_odl_start_elapsed"] = _elapsed
                        _check_odl_server_running.clear()

                        if _check_odl_server_running(_port):
                            _status.update(label=f"✅ Server sẵn sàng!", state="complete", expanded=False)
                            st.session_state["_odl_starting"] = False
                            st.session_state.pop("_odl_start_elapsed", None)
                            st.rerun()
                        elif _elapsed >= _max_wait:
                            _status.update(label=f"⚠️ Timeout sau {_max_wait:.0f}s", state="error")
                            st.session_state["_odl_starting"] = False
                            st.warning("Thử chạy thủ công trong terminal.")
                        else:
                            st.rerun()
        else:
            odl_hybrid = None

        # ── use_struct_tree ───────────────────────────────────────────────────
        odl_struct_tree = st.checkbox(
            "Dùng native PDF structure tags (`use_struct_tree`)",
            value=False,
            help=(
                "Nếu PDF đã được tagged (Tagged PDF), bật để đọc layout trực tiếp từ "
                "structure tree của PDF — reading order chính xác nhất, không cần heuristic.\n\n"
                "Tắt (mặc định): dùng XY-Cut++ layout analysis — tốt cho hầu hết PDF thông thường."
            ),
            key="odl_struct_tree",
        )

    if pdf_strategy == "docling":
        st.markdown("---")
        st.info(
            "🖼️ **Docling tự động xử lý ảnh** — không cần VLM.\n\n"
            "Docling extract ảnh, encode thành base64 và nhúng trực tiếp vào Markdown output "
            "(`![Figure N](data:image/png;base64,...)`). App visualizer render được các ảnh này "
            "trong tab Loader.\n\n"
            "Để **mô tả nội dung ảnh bằng VLM** (sinh text caption để embedding), "
            "hãy dùng **Marker** hoặc **Unstructured** thay thế."
        )

    return {
        "pdf_strategy":    pdf_strategy,
        "extract_tables":  extract_tables,
        "language":        language,
        "marker_device":   marker_device,
        "describe_images": describe_images,
        "vision_model":    vision_model,
        "vision_provider": vision_provider,
        "ollama_base_url": ollama_base_url,
        "odl_hybrid":      odl_hybrid,
        "odl_struct_tree": odl_struct_tree,
    }


# ─── UI: Chunking settings panel ──────────────────────────────────────────────


