import os, sys, re, importlib, importlib.util, inspect, tempfile
from pathlib import Path
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
try:
    from dotenv import load_dotenv as _ld, dotenv_values as _dv
    _ld(override=True); _ENV = _dv()
except ImportError:
    _ENV = {}
import streamlit as st
from core.constants import EMBEDDING_PROVIDER_META
@st.cache_data(show_spinner=False)
def _compute_cosine_heatmap_html(dense_preview: tuple, n_heat: int) -> str:
    """Tính cosine similarity matrix và sinh HTML heatmap. Cache theo vectors."""
    import numpy as np

    vecs  = np.array(dense_preview[:n_heat], dtype=np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1e-9
    vecs_norm  = vecs / norms
    sim_matrix = vecs_norm @ vecs_norm.T

    def _cell_color(val: float) -> str:
        t = (val + 1) / 2
        if t < 0.5:
            r = int(255 * 2 * t); g = r; b = 255
        else:
            t2 = (t - 0.5) * 2
            r = 255; g = int(255 * (1 - t2)); b = g
        return f"rgb({r},{g},{b})"

    def _text_color(val: float) -> str:
        t = (val + 1) / 2
        return "#111" if 0.25 < t < 0.75 else ("#fff" if t <= 0.25 else "#111")

    labels = [f"C{i+1}" for i in range(n_heat)]
    header = (
        "<tr><th style='background:#1a252f;color:#fff;padding:6px 10px;'></th>"
        + "".join(f"<th style='background:#1a252f;color:#fff;padding:6px 10px;font-size:0.8rem'>{l}</th>" for l in labels)
        + "</tr>"
    )
    rows = []
    for i in range(n_heat):
        cells = [f"<td style='background:#1a252f;color:#fff;padding:6px 10px;font-size:0.8rem;font-weight:600'>{labels[i]}</td>"]
        for j in range(n_heat):
            v  = float(sim_matrix[i, j])
            bg = _cell_color(v); fg = _text_color(v)
            cells.append(
                f"<td style='background:{bg};color:{fg};padding:6px 10px;"
                f"text-align:center;font-size:0.82rem;font-weight:{'700' if i==j else '400'}'>"
                f"{v:.2f}</td>"
            )
        rows.append("<tr>" + "".join(cells) + "</tr>")

    return (
        "<div style='overflow-x:auto'>"
        "<table style='border-collapse:collapse;border-radius:8px;overflow:hidden;font-family:monospace'>"
        f"<thead>{header}</thead><tbody>{''.join(rows)}</tbody>"
        "</table></div>"
    )



# ─── UI: Embedding results ────────────────────────────────────────────────────

def render_embedding_results(chunks: list, embed_result: dict, emb_cfg: dict):
    """Hiển thị kết quả sau bước embedding."""
    import math

    st.markdown("---")
    st.header("🧮 Kết quả bước Embedding")

    if not embed_result or embed_result.get("dense") is None:
        st.warning("Không có vector nào được tạo ra.")
        return

    dense        = embed_result["dense"]
    sparse       = embed_result.get("sparse")
    dims         = embed_result["dims"]
    n_embedded   = embed_result["n_embedded"]
    truncated    = embed_result["truncated"]
    provider     = emb_cfg["provider"]
    model_name   = emb_cfg["model_name"]
    meta         = EMBEDDING_PROVIDER_META.get(provider, {})

    # ── Summary metrics ──────────────────────────────────────────────────────
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("🧮 Vectors tạo ra",   n_embedded)
    col2.metric("📐 Số chiều (dim)",   f"{dims:,}")
    col3.metric("💾 RAM / vector",      f"{dims * 4 / 1024:.1f} KB")
    col4.metric("📦 Tổng RAM (dense)", f"{n_embedded * dims * 4 / (1024**2):.2f} MB")
    col5.metric("🔀 Sparse",           "✅ BẬT" if sparse else "❌ TẮT")

    if truncated:
        st.warning(
            f"⚠️ Chỉ embed **{n_embedded}/{len(chunks)}** chunks đầu tiên (giới hạn preview). "
            "Tăng **Số chunk tối đa** trong sidebar để embed thêm."
        )

    # ── Provider + model info ────────────────────────────────────────────────
    st.markdown(
        f"**Provider:** {meta.get('icon', '')} `{provider}` &nbsp;&nbsp; "
        f"**Model:** `{model_name}` &nbsp;&nbsp; "
        f"**VI Quality:** {meta.get('vi_quality', '—')}",
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # ── Cosine similarity heatmap ─────────────────────────────────────────────
    n_heat = min(10, len(dense))
    if n_heat >= 2:
        st.markdown(f"**🌡️ Cosine Similarity giữa {n_heat} chunks đầu**")
        st.caption(
            "Giá trị gần 1.0 → hai chunk rất giống nhau về ngữ nghĩa. "
            "Diagonal luôn = 1.0 (self-similarity). Màu đậm → similarity cao."
        )
        table_html = _compute_cosine_heatmap_html(
            tuple(tuple(v) for v in dense[:n_heat]), n_heat
        )
        st.markdown(table_html, unsafe_allow_html=True)
        st.markdown("")

    # ── Per-chunk vector preview ──────────────────────────────────────────────
    st.markdown(f"**📋 Chi tiết {n_embedded} vector(s)**")
    search_emb = st.text_input(
        "🔍 Tìm trong nội dung chunk", placeholder="Nhập từ khoá…", key="emb_search"
    )

    PREVIEW_DIMS = 8

    for i in range(n_embedded):
        if i >= len(chunks) or i >= len(dense):
            break
        chunk   = chunks[i]
        content = chunk.page_content
        vec     = dense[i]
        norm    = math.sqrt(sum(v * v for v in vec))

        if search_emb and search_emb.lower() not in content.lower():
            continue

        preview_vals = [f"{v:+.4f}" for v in vec[:PREVIEW_DIMS]]
        vec_preview  = "  ".join(preview_vals)
        if dims > PREVIEW_DIMS:
            vec_preview += f"  … (+{dims - PREVIEW_DIMS} chiều)"

        _typical_norm = math.sqrt(dims) * 0.3
        bar_pct = min(int(norm / _typical_norm * 100), 100) if _typical_norm > 0 else 50

        label = (
            f"Chunk {i+1}  |  {len(content):,} ký tự  |  "
            f"{dims}d  |  ‖v‖ = {norm:.4f}"
        )
        with st.expander(label, expanded=(i < 2)):
            st.markdown(
                f"<div style='margin-bottom:6px'>"
                f"<div style='background:#eee;border-radius:4px;height:8px;width:100%'>"
                f"<div style='background:#2980b9;border-radius:4px;height:8px;width:{bar_pct}%'></div>"
                f"</div>"
                f"<span style='font-size:0.75rem;color:#666'>Norm: {norm:.4f}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )
            st.code(f"[{vec_preview}]", language=None)
            st.caption("Nội dung chunk:")
            st.text_area(
                "content",
                value=content[:600] + ("…" if len(content) > 600 else ""),
                height=100,
                key=f"emb_content_{i}",
                disabled=True,
                label_visibility="collapsed",
            )
            if sparse and i < len(sparse):
                sv = sparse[i]
                if sv:
                    top_tokens = sorted(sv.items(), key=lambda x: -x[1])[:10]
                    st.markdown("**Sparse (top-10 tokens):**")
                    token_badges = " &nbsp; ".join(
                        f'<span style="background:#16a085;color:#fff;padding:2px 8px;'
                        f'border-radius:10px;font-size:0.78rem;font-family:monospace">'
                        f'{tok} <b>{score:.2f}</b></span>'
                        for tok, score in top_tokens
                    )
                    st.markdown(token_badges, unsafe_allow_html=True)
                    st.caption(f"Tổng {len(sv):,} token non-zero")

    # ── Tips ─────────────────────────────────────────────────────────────────
    with st.expander("💡 Mẹo chọn embedding model"):
        st.markdown("""
| Tình huống | Gợi ý |
|---|---|
| Tiếng Việt, có GPU | `BAAI/bge-m3` (HuggingFace) hoặc `Cohere embed-multilingual-v3.0` |
| Tiếng Việt, chất lượng cao nhất | `Qwen/Qwen3-Embedding-0.6B` (MTEB #1) |
| Hoàn toàn local, không GPU | `FastEmbed · intfloat/multilingual-e5-small` |
| Local + privacy, có GPU | `Ollama · bge-m3` |
| Budget API thấp | `FastEmbed multilingual-e5-small` (free, CPU) |
| OpenAI ecosystem | `text-embedding-3-small` (default) |
| Hybrid retrieval | Bật **BM25** (nhanh, không GPU) hoặc **SPLADE** (chất lượng cao hơn) |
""")



