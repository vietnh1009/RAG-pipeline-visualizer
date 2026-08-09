import os, sys, re, importlib, importlib.util, inspect, tempfile
from pathlib import Path
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
try:
    from dotenv import load_dotenv as _ld, dotenv_values as _dv
    _ld(override=True); _ENV = _dv()
except ImportError:
    _ENV = {}
import streamlit as st

from core.constants import _CHUNKING_PAGE_SIZE
from utils.badges import chunk_type_badge
from ui.results.loader_results import render_content_with_images


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

def render_chunking_results(chunks: list, strategy: str, display_mode: str = "Text"):
    """Hiển thị kết quả sau bước chunking."""
    st.markdown("---")
    st.header("✂️ Kết quả bước Chunking")

    if not chunks:
        st.warning("Không có chunk nào được tạo ra.")
        return

    # --- Thống kê (cached) ---
    stats = _compute_chunking_stats(tuple(c.page_content for c in chunks))
    total_chars = stats["total_chars"]
    sizes       = stats["sizes"]

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("🧩 Tổng chunks",  len(chunks))
    col2.metric("🔤 Tổng ký tự",   f"{total_chars:,}")
    col3.metric("📊 TB / chunk",    f"{stats['avg']:,}")
    col4.metric("⬇️ Nhỏ nhất",     f"{stats['min']:,}")
    col5.metric("⬆️ Lớn nhất",     f"{stats['max']:,}")

    st.markdown("**Phân bổ kích thước chunk:**")
    size_buckets = {"<200": 0, "200-500": 0, "500-1000": 0, "1000-2000": 0, ">2000": 0}
    for s in sizes:
        if s < 200:   size_buckets["<200"]      += 1
        elif s < 500: size_buckets["200-500"]    += 1
        elif s < 1000:size_buckets["500-1000"]   += 1
        elif s < 2000:size_buckets["1000-2000"]  += 1
        else:         size_buckets[">2000"]      += 1

    bucket_cols = st.columns(5)
    for i, (label, count) in enumerate(size_buckets.items()):
        pct = count / len(chunks) * 100
        bucket_cols[i].metric(label, f"{count} ({pct:.0f}%)")

    levels = set(c.metadata.get("chunk_level", "") for c in chunks)
    levels.discard("")
    if levels:
        st.info(f"🏗️ Hierarchical chunks: {', '.join(sorted(levels))}")

    st.markdown(f"**Chi tiết {len(chunks)} chunk(s):**")

    # Bộ lọc
    col_f1, col_f2 = st.columns([3, 1])
    with col_f1:
        chunk_search = st.text_input(
            "🔍 Tìm trong chunk", placeholder="Nhập từ khoá…", key="chunk_search"
        )
    with col_f2:
        if levels:
            level_filter = st.selectbox("Lọc level", ["Tất cả"] + sorted(levels))
        else:
            level_filter = "Tất cả"

    # Lọc trước khi paginate
    filtered_chunks = [
        (i, c) for i, c in enumerate(chunks)
        if (not chunk_search or chunk_search.lower() in c.page_content.lower())
        and (level_filter == "Tất cả" or c.metadata.get("chunk_level", "") == level_filter)
    ]

    if not filtered_chunks:
        st.info("Không có chunk nào khớp với bộ lọc.")
        return

    total_pages = max(1, (len(filtered_chunks) + _CHUNKING_PAGE_SIZE - 1) // _CHUNKING_PAGE_SIZE)
    _prev_search = st.session_state.get("_chunk_search_prev")
    _prev_level  = st.session_state.get("_chunk_level_prev")
    if chunk_search != _prev_search or level_filter != _prev_level:
        st.session_state["chunk_page"] = 0
        st.session_state["_chunk_search_prev"] = chunk_search
        st.session_state["_chunk_level_prev"]  = level_filter
    if "chunk_page" not in st.session_state:
        st.session_state["chunk_page"] = 0
    page = st.session_state["chunk_page"]

    if total_pages > 1:
        col_prev, col_info, col_next = st.columns([1, 3, 1])
        with col_prev:
            if st.button("◀ Trước", key="chunk_prev", disabled=(page == 0)):
                st.session_state["chunk_page"] = max(0, page - 1)
        with col_info:
            st.caption(f"Trang {page+1}/{total_pages}  ·  {len(filtered_chunks)} kết quả")
        with col_next:
            if st.button("Sau ▶", key="chunk_next", disabled=(page >= total_pages - 1)):
                st.session_state["chunk_page"] = min(total_pages - 1, page + 1)

    start = page * _CHUNKING_PAGE_SIZE
    page_items = filtered_chunks[start: start + _CHUNKING_PAGE_SIZE]

    for i, chunk in page_items:
        content     = chunk.page_content
        meta        = chunk.metadata
        char_count  = len(content)
        source      = Path(meta.get("source", "")).name or ""
        chunk_level = meta.get("chunk_level", "")

        source_info = f" · {source}" if source else ""
        level_info  = f" · {chunk_level}" if chunk_level else ""
        label = f"Chunk {i+1}{level_info}{source_info}  |  {char_count:,} ký tự"

        with st.expander(label, expanded=(i < 3 and page == 0)):
            badge_html = chunk_type_badge(i, chunk_level)
            st.markdown(badge_html, unsafe_allow_html=True)
            st.markdown("")
            render_content_with_images(
                content,
                area_key=f"chunk_{i}",
                area_height=min(250, max(80, char_count // 2)),
                display_mode=display_mode,
            )
            with st.container():
                meta_display = {
                    k: v for k, v in meta.items()
                    if v is not None and v != "" and k not in ("late_embedding",)
                }
                st.markdown("**Metadata:**")
                st.json(meta_display, expanded=False)

    st.caption(f"Hiển thị {len(page_items)} / {len(filtered_chunks)} chunks (trang {page+1})")


