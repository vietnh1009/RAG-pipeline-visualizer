import os, sys, re, importlib, importlib.util, inspect, tempfile
from pathlib import Path
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
try:
    from dotenv import load_dotenv as _ld, dotenv_values as _dv
    _ld(override=True); _ENV = _dv()
except ImportError:
    _ENV = {}
import streamlit as st
def _page_welcome():
    """Page Landing — giới thiệu tổng quát và overview 2 stage."""

    # ── Hero ─────────────────────────────────────────────────────────────────
    st.markdown("""
<div style="padding:2.5rem 0 1.5rem 0;">
  <h1 style="font-size:2.6rem;font-weight:800;margin:0 0 .5rem 0;
             background:linear-gradient(135deg,#6366f1 0%,#8b5cf6 40%,#06b6d4 100%);
             -webkit-background-clip:text;-webkit-text-fill-color:transparent;
             background-clip:text;">
    🔬 RAG Pipeline Visualizer
  </h1>
  <p style="font-size:1.2rem;color:var(--text-color);opacity:.75;margin:0;max-width:640px;line-height:1.6;">
    Công cụ trực quan hóa, cấu hình và chạy toàn bộ RAG pipeline —
    từ bước load tài liệu PDF đến bước sinh câu trả lời từ LLM.
  </p>
</div>
""", unsafe_allow_html=True)

    # ── 2-stage cards ─────────────────────────────────────────────────────────
    st.markdown("""
<div style="margin:0 0 .6rem 0;">
  <span style="font-size:1rem;font-weight:700;text-transform:uppercase;
               letter-spacing:.1em;color:var(--text-color);opacity:.5;">
    Pipeline được chia thành 2 stage độc lập
  </span>
</div>
""", unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="medium")

    # ── Stage 1 card ──────────────────────────────────────────────────────────
    with col1:
        st.markdown("""
<div style="border:1px solid rgba(99,102,241,.25);border-top:3px solid #6366f1;
            border-radius:12px;padding:1.5rem 1.6rem 1.4rem;height:100%;
            background:linear-gradient(160deg,rgba(99,102,241,.04) 0%,transparent 60%);">

  <div style="display:flex;align-items:center;gap:.6rem;margin-bottom:.75rem;">
    <span style="font-size:1.7rem;">🗃️</span>
    <div>
      <div style="font-size:1.25rem;font-weight:700;color:var(--text-color);">Stage 1 — Indexing</div>
      <div style="font-size:1rem;font-weight:600;color:#6366f1;margin-top:1px;">
        OFFLINE &nbsp;·&nbsp; Chạy một lần duy nhất
      </div>
    </div>
  </div>

  <p style="font-size:1.25rem;color:var(--text-color);opacity:.7;margin:0 0 1.25rem;line-height:1.55;">
    Chuẩn bị index từ tài liệu PDF.
    Kết quả được cache — lần sau reuse ngay tức thì.
  </p>

  <div style="display:flex;flex-direction:column;gap:0;">
    <div style="display:flex;align-items:center;gap:.75rem;padding:.55rem .75rem;
                border-radius:8px;background:rgba(99,102,241,.07);">
      <span style="font-size:1rem;">📄</span>
      <span style="font-size:1.25rem;font-weight:600;color:var(--text-color);">File PDF / Thư mục</span>
    </div>
    <div style="display:flex;justify-content:center;padding:.2rem 0;color:#6366f1;font-size:1.25rem;">↓</div>
    <div style="display:flex;align-items:center;gap:.75rem;padding:.55rem .75rem;
                border-radius:8px;background:rgba(99,102,241,.05);">
      <span style="display:inline-flex;align-items:center;justify-content:center;
                   width:22px;height:22px;border-radius:6px;font-size:1.25rem;font-weight:700;
                   background:#6366f1;color:white;">1</span>
      <span style="font-size:1.25rem;color:var(--text-color);">
        <b>Loader</b> <span style="opacity:.6;font-size:1rem;">— parse PDF → Documents</span>
      </span>
    </div>
    <div style="display:flex;justify-content:center;padding:.2rem 0;color:#6366f1;font-size:1.25rem;">↓</div>
    <div style="display:flex;align-items:center;gap:.75rem;padding:.55rem .75rem;
                border-radius:8px;background:rgba(99,102,241,.05);">
      <span style="display:inline-flex;align-items:center;justify-content:center;
                   width:22px;height:22px;border-radius:6px;font-size:1.25rem;font-weight:700;
                   background:#6366f1;color:white;">2</span>
      <span style="font-size:1.25rem;color:var(--text-color);">
        <b>Chunking</b> <span style="opacity:.6;font-size:1rem;">— cắt nhỏ thành chunks</span>
      </span>
    </div>
    <div style="display:flex;justify-content:center;padding:.2rem 0;color:#6366f1;font-size:1.25rem;">↓</div>
    <div style="display:flex;align-items:center;gap:.75rem;padding:.55rem .75rem;
                border-radius:8px;background:rgba(99,102,241,.05);">
      <span style="display:inline-flex;align-items:center;justify-content:center;
                   width:22px;height:22px;border-radius:6px;font-size:1.25rem;font-weight:700;
                   background:#6366f1;color:white;">3</span>
      <span style="font-size:1.25rem;color:var(--text-color);">
        <b>Embedding</b> <span style="opacity:.6;font-size:1rem;">— vector hoá chunks</span>
      </span>
    </div>
    <div style="display:flex;justify-content:center;padding:.2rem 0;color:#6366f1;font-size:1.25rem;">↓</div>
    <div style="display:flex;align-items:center;gap:.75rem;padding:.55rem .75rem;
                border-radius:8px;background:rgba(99,102,241,.05);">
      <span style="display:inline-flex;align-items:center;justify-content:center;
                   width:22px;height:22px;border-radius:6px;font-size:1.25rem;font-weight:700;
                   background:#6366f1;color:white;">4</span>
      <span style="font-size:1.25rem;color:var(--text-color);">
        <b>Vector DB</b> <span style="opacity:.6;font-size:1rem;">— lưu index xuống disk</span>
      </span>
    </div>
  </div>

  <div style="margin-top:1.2rem;padding:.6rem .9rem;border-radius:8px;
              background:rgba(99,102,241,.1);text-align:center;">
    <span style="font-size:1rem;font-weight:600;color:#6366f1;">
      → Trang 🗃️ Indexing (sidebar trái)
    </span>
  </div>
</div>
""", unsafe_allow_html=True)

    # ── Stage 2 card ──────────────────────────────────────────────────────────
    with col2:
        st.markdown("""
<div style="border:1px solid rgba(6,182,212,.25);border-top:3px solid #06b6d4;
            border-radius:12px;padding:1.5rem 1.6rem 1.4rem;height:100%;
            background:linear-gradient(160deg,rgba(6,182,212,.04) 0%,transparent 60%);">

  <div style="display:flex;align-items:center;gap:.6rem;margin-bottom:.75rem;">
    <span style="font-size:1.7rem;">💬</span>
    <div>
      <div style="font-size:1.25rem;font-weight:700;color:var(--text-color);">Stage 2 — Generation</div>
      <div style="font-size:1rem;font-weight:600;color:#06b6d4;margin-top:1px;">
        ONLINE &nbsp;·&nbsp; Chạy mỗi khi có câu hỏi
      </div>
    </div>
  </div>

  <p style="font-size:1.25rem;color:var(--text-color);opacity:.7;margin:0 0 1.25rem;line-height:1.55;">
    Truy vấn index và sinh câu trả lời từ LLM.
    Cần hoàn thành Stage 1 trước.
  </p>

  <div style="display:flex;flex-direction:column;gap:0;">
    <div style="display:flex;align-items:center;gap:.75rem;padding:.55rem .75rem;
                border-radius:8px;background:rgba(6,182,212,.07);">
      <span style="font-size:1rem;">❓</span>
      <span style="font-size:1.25rem;font-weight:600;color:var(--text-color);">Câu hỏi người dùng</span>
    </div>
    <div style="display:flex;justify-content:center;padding:.2rem 0;color:#06b6d4;font-size:1.25rem;">↓</div>
    <div style="display:flex;align-items:center;gap:.75rem;padding:.55rem .75rem;
                border-radius:8px;background:rgba(6,182,212,.05);">
      <span style="display:inline-flex;align-items:center;justify-content:center;
                   width:22px;height:22px;border-radius:6px;font-size:1.25rem;font-weight:700;
                   background:#06b6d4;color:white;">7</span>
      <span style="font-size:1.25rem;color:var(--text-color);">
        <b>Pre-retrieval</b>
        <span style="opacity:.5;font-size:.95rem;margin-left:.3rem;">tuỳ chọn</span>
      </span>
    </div>
    <div style="display:flex;justify-content:center;padding:.2rem 0;color:#06b6d4;font-size:1.25rem;">↓</div>
    <div style="display:flex;align-items:center;gap:.75rem;padding:.55rem .75rem;
                border-radius:8px;background:rgba(6,182,212,.05);">
      <span style="display:inline-flex;align-items:center;justify-content:center;
                   width:22px;height:22px;border-radius:6px;font-size:1.25rem;font-weight:700;
                   background:#06b6d4;color:white;">8</span>
      <span style="font-size:1.25rem;color:var(--text-color);">
        <b>Retrieval</b> <span style="opacity:.6;font-size:1rem;">— tìm chunk liên quan</span>
      </span>
    </div>
    <div style="display:flex;justify-content:center;padding:.2rem 0;color:#06b6d4;font-size:1.25rem;">↓</div>
    <div style="display:flex;align-items:center;gap:.75rem;padding:.55rem .75rem;
                border-radius:8px;background:rgba(6,182,212,.05);">
      <span style="display:inline-flex;align-items:center;justify-content:center;
                   width:22px;height:22px;border-radius:6px;font-size:1.25rem;font-weight:700;
                   background:#06b6d4;color:white;">9</span>
      <span style="font-size:1.25rem;color:var(--text-color);">
        <b>Post-retrieval</b>
        <span style="opacity:.5;font-size:.95rem;margin-left:.3rem;">tuỳ chọn</span>
      </span>
    </div>
    <div style="display:flex;justify-content:center;padding:.2rem 0;color:#06b6d4;font-size:1.25rem;">↓</div>
    <div style="display:flex;align-items:center;gap:.75rem;padding:.55rem .75rem;
                border-radius:8px;background:rgba(6,182,212,.05);">
      <span style="display:inline-flex;align-items:center;justify-content:center;
                   width:22px;height:22px;border-radius:6px;font-size:1.25rem;font-weight:700;
                   background:#06b6d4;color:white;">10</span>
      <span style="font-size:1.25rem;color:var(--text-color);">
        <b>Prompt</b> <span style="opacity:.6;font-size:1rem;">— xây dựng prompt</span>
      </span>
    </div>
    <div style="display:flex;justify-content:center;padding:.2rem 0;color:#06b6d4;font-size:1.25rem;">↓</div>
    <div style="display:flex;align-items:center;gap:.75rem;padding:.55rem .75rem;
                border-radius:8px;background:rgba(6,182,212,.05);">
      <span style="display:inline-flex;align-items:center;justify-content:center;
                   width:22px;height:22px;border-radius:6px;font-size:1.25rem;font-weight:700;
                   background:#06b6d4;color:white;">11</span>
      <span style="font-size:1.25rem;color:var(--text-color);">
        <b>Generation</b> <span style="opacity:.6;font-size:1rem;">— LLM sinh câu trả lời</span>
      </span>
    </div>
    <div style="display:flex;justify-content:center;padding:.2rem 0;color:#06b6d4;font-size:1.25rem;">↓</div>
    <div style="display:flex;align-items:center;gap:.75rem;padding:.55rem .75rem;
                border-radius:8px;background:rgba(6,182,212,.07);">
      <span style="font-size:1rem;">💬</span>
      <span style="font-size:1.25rem;font-weight:600;color:var(--text-color);">Câu trả lời có trích dẫn nguồn</span>
    </div>
  </div>

  <div style="margin-top:1.2rem;padding:.6rem .9rem;border-radius:8px;
              background:rgba(6,182,212,.1);text-align:center;">
    <span style="font-size:1rem;font-weight:600;color:#06b6d4;">
      → Trang 💬 Generation (sidebar trái)
    </span>
  </div>
</div>
""", unsafe_allow_html=True)

    st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)

    # ── Full pipeline flow diagram ────────────────────────────────────────────
    st.markdown("""
<div style="margin:1rem 0 .5rem;font-size:1.25rem;font-weight:700;color:var(--text-color);">
  🔗 Full Pipeline Flow
</div>
""", unsafe_allow_html=True)

    st.markdown("""
<div style="border-radius:14px;overflow:hidden;border:1px solid rgba(99,102,241,.2);">

  <!-- Stage 1 row -->
  <div style="background:linear-gradient(90deg,rgba(99,102,241,.12) 0%,rgba(99,102,241,.05) 100%);
              padding:1rem 1.4rem;border-bottom:1px solid rgba(99,102,241,.15);">
    <div style="font-size:1.25rem;font-weight:700;text-transform:uppercase;letter-spacing:.12em;
                color:#6366f1;margin-bottom:.7rem;">
      ⬛ Stage 1 — Indexing &nbsp;(offline)
    </div>
    <div style="display:flex;align-items:center;flex-wrap:wrap;gap:.35rem;">
      <span style="padding:.3rem .7rem;border-radius:6px;font-size:1rem;font-weight:600;
                   background:rgba(99,102,241,.15);color:#6366f1;border:1px solid rgba(99,102,241,.3);">
        📄 PDF
      </span>
      <span style="color:#6366f1;font-weight:700;">→</span>
      <span style="padding:.3rem .7rem;border-radius:6px;font-size:1rem;
                   background:rgba(99,102,241,.08);color:var(--text-color);
                   border:1px solid rgba(99,102,241,.15);">1 · Loader</span>
      <span style="color:#6366f1;font-weight:700;">→</span>
      <span style="padding:.3rem .7rem;border-radius:6px;font-size:1rem;
                   background:rgba(99,102,241,.08);color:var(--text-color);
                   border:1px solid rgba(99,102,241,.15);">2 · Chunking</span>
      <span style="color:#6366f1;font-weight:700;">→</span>
      <span style="padding:.3rem .7rem;border-radius:6px;font-size:1rem;
                   background:rgba(99,102,241,.08);color:var(--text-color);
                   border:1px solid rgba(99,102,241,.15);">3 · Embedding</span>
      <span style="color:#6366f1;font-weight:700;">→</span>
      <span style="padding:.3rem .8rem;border-radius:6px;font-size:1rem;font-weight:600;
                   background:rgba(99,102,241,.15);color:#6366f1;border:1px solid rgba(99,102,241,.3);">
        4 · Vector DB 🗄️
      </span>
    </div>
  </div>

  <!-- Connector -->
  <div style="display:flex;justify-content:flex-end;padding:.4rem 1.4rem;
              background:rgba(255,255,255,.02);border-bottom:1px solid rgba(6,182,212,.1);">
    <div style="display:flex;align-items:center;gap:.4rem;">
      <span style="font-size:.95rem;color:var(--text-color);opacity:.5;">index</span>
      <span style="font-size:1.25rem;
                   background:linear-gradient(90deg,#6366f1,#06b6d4);
                   -webkit-background-clip:text;-webkit-text-fill-color:transparent;">⟶</span>
      <span style="font-size:.95rem;color:var(--text-color);opacity:.5;">query time</span>
    </div>
  </div>

  <!-- Stage 2 row -->
  <div style="background:linear-gradient(90deg,rgba(6,182,212,.08) 0%,rgba(6,182,212,.03) 100%);
              padding:1rem 1.4rem;">
    <div style="font-size:1.25rem;font-weight:700;text-transform:uppercase;letter-spacing:.12em;
                color:#06b6d4;margin-bottom:.7rem;">
      ⬛ Stage 2 — Generation &nbsp;(online, per query)
    </div>
    <div style="display:flex;align-items:center;flex-wrap:wrap;gap:.35rem;">
      <span style="padding:.3rem .7rem;border-radius:6px;font-size:1rem;font-weight:600;
                   background:rgba(6,182,212,.15);color:#06b6d4;border:1px solid rgba(6,182,212,.3);">
        ❓ Query
      </span>
      <span style="color:#06b6d4;font-weight:700;">→</span>
      <span style="padding:.3rem .7rem;border-radius:6px;font-size:1rem;
                   background:rgba(6,182,212,.06);color:var(--text-color);
                   border:1px solid rgba(6,182,212,.12);">
        7 · Pre-retrieval
        <span style="opacity:.5;font-size:1.25rem;"> opt</span>
      </span>
      <span style="color:#06b6d4;font-weight:700;">→</span>
      <span style="padding:.3rem .7rem;border-radius:6px;font-size:1rem;
                   background:rgba(6,182,212,.08);color:var(--text-color);
                   border:1px solid rgba(6,182,212,.2);font-weight:600;">
        8 · Retrieval
      </span>
      <span style="color:#06b6d4;font-weight:700;">→</span>
      <span style="padding:.3rem .7rem;border-radius:6px;font-size:1rem;
                   background:rgba(6,182,212,.06);color:var(--text-color);
                   border:1px solid rgba(6,182,212,.12);">
        9 · Post-retrieval
        <span style="opacity:.5;font-size:1.25rem;"> opt</span>
      </span>
      <span style="color:#06b6d4;font-weight:700;">→</span>
      <span style="padding:.3rem .7rem;border-radius:6px;font-size:1rem;
                   background:rgba(6,182,212,.08);color:var(--text-color);
                   border:1px solid rgba(6,182,212,.2);">10 · Prompt</span>
      <span style="color:#06b6d4;font-weight:700;">→</span>
      <span style="padding:.3rem .7rem;border-radius:6px;font-size:1rem;
                   background:rgba(6,182,212,.08);color:var(--text-color);
                   border:1px solid rgba(6,182,212,.2);">11 · Generation</span>
      <span style="color:#06b6d4;font-weight:700;">→</span>
      <span style="padding:.3rem .8rem;border-radius:6px;font-size:1rem;font-weight:600;
                   background:rgba(6,182,212,.15);color:#06b6d4;border:1px solid rgba(6,182,212,.3);">
        💬 Câu trả lời
      </span>
    </div>
  </div>

</div>
""", unsafe_allow_html=True)

    st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)

    # ── Quick start ───────────────────────────────────────────────────────────
    st.markdown("""
<div style="margin:0 0 .5rem;font-size:1.25rem;font-weight:700;color:var(--text-color);">
  ⚡ Quick Start
</div>
""", unsafe_allow_html=True)

    st.markdown("""
<div style="display:flex;gap:1rem;margin-bottom:1.25rem;flex-wrap:wrap;">

  <div style="flex:1;min-width:260px;padding:1.1rem 1.3rem;border-radius:10px;
              border:1px solid rgba(99,102,241,.25);
              background:linear-gradient(135deg,rgba(99,102,241,.06),transparent);">
    <div style="font-size:1.25rem;font-weight:700;text-transform:uppercase;letter-spacing:.1em;
                color:#6366f1;margin-bottom:.6rem;">Step 1</div>
    <div style="font-size:1.25rem;color:var(--text-color);line-height:1.55;">
      Chuyển sang <b>🗃️ Indexing</b> ở sidebar<br>
      → Upload PDF → Cấu hình → Ấn <b>▶️ Process</b>
    </div>
  </div>

  <div style="display:flex;align-items:center;justify-content:center;
              font-size:1.3rem;color:var(--text-color);opacity:.3;">→</div>

  <div style="flex:1;min-width:260px;padding:1.1rem 1.3rem;border-radius:10px;
              border:1px solid rgba(6,182,212,.25);
              background:linear-gradient(135deg,rgba(6,182,212,.06),transparent);">
    <div style="font-size:1.25rem;font-weight:700;text-transform:uppercase;letter-spacing:.1em;
                color:#06b6d4;margin-bottom:.6rem;">Step 2</div>
    <div style="font-size:1.25rem;color:var(--text-color);line-height:1.55;">
      Chuyển sang <b>💬 Generation</b> ở sidebar<br>
      → Cấu hình LLM → Nhập câu hỏi → Ấn <b>▶️ Run</b>
    </div>
  </div>

</div>
""", unsafe_allow_html=True)

    # Recommended config table
    with st.expander("📋 Cấu hình khuyến nghị cho người mới bắt đầu", expanded=False):
        st.markdown("""
| Bước | Setting | Lý do |
|------|---------|-------|
| **Loader** | `marker` | Markdown chất lượng cao — bảng, công thức, hình ảnh |
| **Chunking** | `format_aware` | Nhận diện Markdown heading từ Marker output |
| **Embedding** | OpenAI `text-embedding-3-small` + BM25 | Cân bằng chi phí / chất lượng, hybrid retrieval |
| **Vector DB** | `chroma` | Zero config, persist tự động, phù hợp dev |
| **Retrieval** | `hybrid` · Top-K = 15 | Dense + Sparse RRF — tốt nhất khi bật BM25 |
| **Post-retrieval** | `cross_encoder` · Top-N = 5 · `sandwich` | Rerank không cần GPU, giảm lost-in-middle |
| **Prompt** | `citation` | LLM bắt buộc trích dẫn [NGUỒN N] |
| **Generation** | `gpt-4.1-mini` · Temperature = 0 · Streaming = On | Rẻ, nhanh, deterministic |
""")

    st.markdown("<div style='height:.5rem'></div>", unsafe_allow_html=True)
    st.caption(
        "RAG-pipeline-visualizer · Built with Streamlit + LangChain · "
        "[GitHub](https://github.com/your-username/RAG-pipeline-visualizer)"
    )



