import os, sys, re, importlib, importlib.util, inspect, tempfile
from pathlib import Path
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
try:
    from dotenv import load_dotenv as _ld, dotenv_values as _dv
    _ld(override=True); _ENV = _dv()
except ImportError:
    _ENV = {}
import streamlit as st

from core.constants import _CHUNKING_PAGE_SIZE, _LOADER_PAGE_SIZE
from utils.badges import chunk_type_badge, file_type_badge

def _render_images_in_text(content: str, area_key: str, area_height: int):
    """
    Render plain-text mode: text_area cho text, st.image cho ảnh xen kẽ.
    Path ảnh trong markdown đã được MarkerPDFLoader convert sang tuyệt đối.
    """
    IMAGE_RE = re.compile(
        r'!\[([^\]]*)\]\((data:image/[^)]+|[^)]+\.(jpe?g|png|webp|gif|bmp))\)',
        re.IGNORECASE
    )

    if not IMAGE_RE.search(content):
        st.text_area(
            "Nội dung",
            value=content,
            height=area_height,
            key=area_key,
            disabled=True,
            label_visibility="collapsed",
        )
        return

    parts   = IMAGE_RE.split(content)
    matches = IMAGE_RE.findall(content)

    text_before = parts[0]
    if text_before.strip():
        st.text_area(
            "Nội dung",
            value=text_before,
            height=min(area_height, max(60, len(text_before) // 3)),
            key=f"{area_key}_t0",
            disabled=True,
            label_visibility="collapsed",
        )

    for m_idx, (alt, img_path, _ext) in enumerate(matches):
        caption = alt if alt else "Figure"

        if img_path.startswith("data:"):
            # Docling embedded mode: data URI → decode và render
            import base64 as _b64
            try:
                header, b64data = img_path.split(",", 1)
                img_bytes = _b64.b64decode(b64data)
                st.image(img_bytes, caption=caption, width="stretch")
            except Exception:
                st.caption(f"🖼️ *(không render được data URI)*")
        else:
            img_file = Path(img_path)
            if img_file.exists():
                st.image(str(img_file), caption=caption, width="stretch")
            else:
                st.caption(f"🖼️ `{img_path}` *(ảnh không tìm thấy trên disk)*")

        text_after_idx = 1 + m_idx * 4 + 3
        if text_after_idx < len(parts):
            text_after = parts[text_after_idx]
            if text_after.strip():
                st.text_area(
                    "Nội dung",
                    value=text_after,
                    height=min(area_height, max(60, len(text_after) // 3)),
                    key=f"{area_key}_t{m_idx+1}",
                    disabled=True,
                    label_visibility="collapsed",
                )


def _local_images_to_base64(content: str) -> str:
    """
    Scan Markdown content tìm ![](<local_path>), đọc file và
    chuyển thành data URI base64 để st.markdown() render được.
    URL (http/https) và path không tồn tại được giữ nguyên.
    """
    import base64, mimetypes

    IMAGE_RE = re.compile(r'!\[([^\]]*)\]\(([^)]+)\)')

    def _to_data_uri(m: re.Match) -> str:
        alt     = m.group(1)
        src     = m.group(2)
        # Bỏ qua URL và data URI (Docling embedded mode đã là base64 rồi)
        if src.startswith("http://") or src.startswith("https://") or src.startswith("data:"):
            return m.group(0)
        img_path = Path(src)
        if not img_path.exists():
            return m.group(0)   # giữ nguyên, sẽ hiện broken image
        mime, _ = mimetypes.guess_type(str(img_path))
        mime = mime or "image/jpeg"
        b64  = base64.b64encode(img_path.read_bytes()).decode()
        return f"![{alt}](data:{mime};base64,{b64})"

    return IMAGE_RE.sub(_to_data_uri, content)


def render_content_with_images(content: str, area_key: str, area_height: int,
                               display_mode: str = "Text"):
    """
    Hiển thị nội dung theo display_mode được truyền vào từ global toggle.

    - Text    : text_area thuần, ảnh local render inline bằng st.image
    - Markdown: st.markdown với ảnh được convert sang base64 data URI
    """
    st.caption("Nội dung")
    if display_mode == "Text":
        _render_images_in_text(content, area_key, area_height)
    else:
        # Nếu content có literal \n (2 ký tự, thường do JSON fallback), chuyển thành newline thật
        if r"\n" in content and "\n" not in content:
            content = content.replace(r"\n", "\n")
        md_with_embedded = _local_images_to_base64(content)
        st.markdown(md_with_embedded, unsafe_allow_html=True)


# ─── UI: Loader results ───────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def _compute_loader_stats(doc_contents: tuple[str, ...], doc_metas: tuple[str, ...]) -> dict:
    """Cache thống kê loader — chỉ tính lại khi docs thay đổi."""
    import json
    total_chars = sum(len(c) for c in doc_contents)
    file_types: dict[str, int] = {}
    for m_str in doc_metas:
        ft = json.loads(m_str).get("file_type", "unknown")
        file_types[ft] = file_types.get(ft, 0) + 1
    return {"total_chars": total_chars, "file_types": file_types}


def _compute_chunking_stats(chunk_contents: tuple[str, ...]) -> dict:
    """Cache thống kê chunking — chỉ tính lại khi chunks thay đổi."""
    if not chunk_contents:
        return {"total_chars": 0, "avg": 0, "min": 0, "max": 0, "sizes": ()}
    sizes = tuple(len(c) for c in chunk_contents)
    return {
        "total_chars": sum(sizes),
        "avg":  sum(sizes) // len(sizes),
        "min":  min(sizes),
        "max":  max(sizes),
        "sizes": sizes,
    }


_LOADER_PAGE_SIZE  = 20
_CHUNKING_PAGE_SIZE = 30


def render_loader_results(docs: list, display_mode: str = "Text"):
    """Hiển thị kết quả sau bước loading."""
    import json

    st.markdown("---")
    st.header("📂 Kết quả bước Loading")

    # --- Thống kê (cached) ---
    stats = _compute_loader_stats(
        tuple(d.page_content for d in docs),
        tuple(json.dumps(d.metadata, default=str, sort_keys=True) for d in docs),
    )
    total_chars = stats["total_chars"]
    file_types  = stats["file_types"]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("📄 Tổng documents", len(docs))
    col2.metric("🔤 Tổng ký tự", f"{total_chars:,}")
    col3.metric("📊 Ký tự TB / doc", f"{total_chars // max(len(docs), 1):,}")
    col4.metric("📁 Loại file", len(file_types))

    if file_types:
        st.markdown("**Phân bổ loại file:**")
        cols = st.columns(min(len(file_types), 6))
        for i, (ft, count) in enumerate(sorted(file_types.items())):
            cols[i % len(cols)].markdown(
                f"{file_type_badge(ft)} **×{count}**", unsafe_allow_html=True
            )
        st.markdown("")

    # --- Search + Pagination ---
    st.markdown(f"**Chi tiết {len(docs)} document(s):**")
    search_query = st.text_input("🔍 Tìm kiếm trong nội dung", placeholder="Nhập từ khoá…",
                                 key="loader_search")

    filtered = [
        (i, doc) for i, doc in enumerate(docs)
        if not search_query or search_query.lower() in doc.page_content.lower()
    ]

    total_pages = max(1, (len(filtered) + _LOADER_PAGE_SIZE - 1) // _LOADER_PAGE_SIZE)
    if "loader_page" not in st.session_state or search_query != st.session_state.get("_loader_search_prev"):
        st.session_state["loader_page"] = 0
        st.session_state["_loader_search_prev"] = search_query
    page = st.session_state["loader_page"]

    if total_pages > 1:
        col_prev, col_info, col_next = st.columns([1, 3, 1])
        with col_prev:
            if st.button("◀ Trước", key="loader_prev", disabled=(page == 0)):
                st.session_state["loader_page"] = max(0, page - 1)
        with col_info:
            st.caption(f"Trang {page+1}/{total_pages}  ·  {len(filtered)} kết quả")
        with col_next:
            if st.button("Sau ▶", key="loader_next", disabled=(page >= total_pages - 1)):
                st.session_state["loader_page"] = min(total_pages - 1, page + 1)

    start = page * _LOADER_PAGE_SIZE
    page_items = filtered[start: start + _LOADER_PAGE_SIZE]

    for i, doc in page_items:
        content = doc.page_content
        meta    = doc.metadata
        ft      = meta.get("file_type", "unknown")
        source  = Path(meta.get("source", "")).name or "unknown"
        page_n  = meta.get("page", "")
        chars   = len(content)
        page_info = f" · Trang {page_n}" if page_n else ""
        label = f"Doc {i+1}  |  {source}{page_info}  |  {chars:,} ký tự"

        with st.expander(label, expanded=(i == 0 and page == 0)):
            st.markdown(file_type_badge(ft), unsafe_allow_html=True)
            st.markdown("")
            render_content_with_images(
                content,
                area_key=f"doc_content_{i}",
                area_height=min(300, max(100, len(content) // 3)),
                display_mode=display_mode,
            )
            st.markdown("**Metadata:**")
            meta_clean = {k: v for k, v in meta.items() if v is not None and v != ""}
            st.json(meta_clean, expanded=False)


# ─── UI: Chunking results ─────────────────────────────────────────────────────


