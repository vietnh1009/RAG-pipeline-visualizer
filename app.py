"""
app.py — RAG-pipeline-visualizer entry point.

Khởi động:
    streamlit run app.py

Kiến trúc module:
    app.py                       ← entry point (file này)
    ui/
      pages/welcome.py           ← _page_welcome
      pages/indexing.py          ← _page_indexing + _write_pipeline_chain
      pages/generation.py        ← _page_generation
      settings/loader.py         ← render_loader_settings
      settings/chunking.py       ← render_chunking_settings
      settings/embedding.py      ← render_embedding_settings
      settings/vector_db.py      ← render_vector_db_settings
      results/loader_results.py  ← render_loader_results
      results/chunking_results.py← render_chunking_results
      results/embedding_results.py← render_embedding_results
      results/vdb_results.py     ← render_vector_db_results
      components/
        index_panel.py           ← index switcher, load from cache
        pipeline_suggestions.py  ← get/render pipeline suggestions
        badges.py                ← badges, friendly errors
    core/
      pipeline_runners.py        ← run_loader, run_chunker, run_embedder
      cache_helpers.py           ← _cached_list_*, _invalidate, load helpers
    utils/
      env.py                     ← _is_installed, _get_env, upload, GPU detect
      badges.py                  ← file_type_badge, chunk_type_badge
"""

import os
import sys
from pathlib import Path

from core.constants import PAGE_TITLE
# Đảm bảo project root trong sys.path để tất cả module import được nhau
_ROOT = Path(__file__).parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# CUDA allocator phải set trước khi bất kỳ torch import nào
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

try:
    from dotenv import load_dotenv as _ld
    _ld(override=True)
except ImportError:
    pass

import streamlit as st

from ui.pages.welcome    import _page_welcome
from ui.pages.indexing   import _page_indexing
from ui.pages.generation import _page_generation


PAGE_TITLE = "RAG-pipeline-visualizer"


def main() -> None:
    st.set_page_config(
        page_title=PAGE_TITLE,
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # ── Global CSS ────────────────────────────────────────────────────────
    st.markdown("""
        <style>
        html, body, [class*="css"] { font-size: 17px !important; }
        section[data-testid="stSidebar"] * { font-size: 15px !important; }
        section[data-testid="stSidebar"] label,
        section[data-testid="stSidebar"] .stSelectbox label,
        section[data-testid="stSidebar"] .stRadio label,
        section[data-testid="stSidebar"] .stCheckbox label {
            font-size: 15px !important; font-weight: 500 !important;
        }
        .stSelectbox label, .stRadio label, .stCheckbox label,
        .stTextInput label, .stNumberInput label, .stSlider label {
            font-size: 16px !important; font-weight: 500 !important;
        }
        .stSelectbox div[data-baseweb="select"] { font-size: 16px !important; }
        .stButton button { font-size: 16px !important; }
        .streamlit-expanderHeader,
        [data-testid="stExpander"] summary {
            font-size: 16px !important; font-weight: 600 !important;
        }
        .stCaption, .stCaption p { font-size: 14px !important; }
        .stMarkdown p, .stMarkdown li, .stMarkdown td, .stMarkdown th {
            font-size: 16px !important; line-height: 1.8 !important;
        }
        .stMarkdown h1 { font-size: 2rem   !important; }
        .stMarkdown h2 { font-size: 1.6rem !important; }
        .stMarkdown h3 { font-size: 1.3rem !important; }
        .stMarkdown img { max-width:100%!important; height:auto!important; }
        .stMarkdown code {
            background:#fff3cd!important; color:#856404!important;
            padding:2px 7px!important; border-radius:3px!important;
            font-size:15px!important;
        }
        .stAlert p { font-size: 16px !important; }
        .stNumberInput input, .stTextInput input, .stTextArea textarea {
            font-size: 16px !important;
        }
        .stRadio div[role="radiogroup"] label { font-size: 15px !important; }
        [data-testid="stMetricLabel"]  { font-size: 14px !important; }
        [data-testid="stMetricValue"]  { font-size: 1.8rem !important; }
        .stJson { font-size: 14px !important; }
        .stMarkdown { font-size: 16px; }
        </style>
    """, unsafe_allow_html=True)

    # ── Navigation ────────────────────────────────────────────────────────
    # Khi đang indexing: force về trang indexing, không cho chuyển trang
    _running = st.session_state.get("_pipeline_running", False)

    if _running:
        # Chạy thẳng _page_indexing, bỏ qua st.navigation hoàn toàn
        # → sidebar navigation không render → user không click được
        _page_indexing()
        return

    page = st.navigation(
        [
            st.Page(_page_welcome,    title="Trang chủ",  icon="🏠",
                    url_path="welcome",  default=True),
            st.Page(_page_indexing,   title="Indexing",   icon="🗃️",
                    url_path="indexing"),
            st.Page(_page_generation, title="Generation", icon="💬",
                    url_path="generation"),
        ],
        position="sidebar",
    )
    page.run()


if __name__ == "__main__":
    main()
