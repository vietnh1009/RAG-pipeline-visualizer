
def _input_display_name(entry: dict) -> str:
    """Lấy tên hiển thị của input.
    Ưu tiên: input_display > tên file trong thư mục temp > basename source_path
    """
    from pathlib import Path as _P
    disp = entry.get("input_display", "")
    if disp:
        return disp
    src = entry.get("source_path", "") or ""
    p   = _P(src)
    # Nếu là thư mục temp uploads → list file bên trong
    if p.is_dir():
        files = [f.name for f in p.iterdir() if f.is_file()]
        if files:
            return ", ".join(sorted(files))
    # Fallback về basename
    return p.name or src

import os, sys, re, importlib, importlib.util, inspect, tempfile
from pathlib import Path
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
try:
    from dotenv import load_dotenv as _ld, dotenv_values as _dv
    _ld(override=True); _ENV = _dv()
except ImportError:
    _ENV = {}
import streamlit as st

from utils.env import _get_pipeline_cache, _get_env
from core.cache_helpers import (
    _cached_list_pipelines, _invalidate_list_entries_cache,
    _embedder_kwargs_from_cfg, _resolve_autoload_pending,
)
from ui.components.index_panel import (
    _try_autoload_latest_index, _render_load_index_from_cache,
    _render_index_switcher, _load_pipeline_into_session,
)
from ui.settings.loader    import render_loader_settings
from ui.settings.chunking  import render_chunking_settings
from ui.settings.embedding import render_embedding_settings
from ui.settings.vector_db import render_vector_db_settings
from ui.results.vdb_results import render_query_pipeline_results

def _vdb_display_name(vs: dict, vc: dict = None) -> str:
    """persist_dir basename = faiss_mrkr_fmt_..._04Jun2026_220347"""
    import os
    vc = vc or {}
    persist_dir = vs.get("persist_dir") or vc.get("persist_dir", "")
    if persist_dir:
        basename = os.path.basename(persist_dir.rstrip("/\\"))
        if basename and basename not in ("faiss", "chroma", "lancedb", "storage"):
            return basename
    provider = vs.get("provider") or vc.get("provider", "?")
    coll     = vs.get("collection_name") or vc.get("collection_name", "?")
    return f"{provider}/{coll}"


def _chunk_size_display(cc: dict, cs: dict) -> str:
    """Trả về chuỗi hiển thị chunk size/overlap chỉ khi strategy sử dụng chúng.
    format_aware, semantic, sentence_aware: không hiển thị chunk_size/overlap.
    """
    _NO_SIZE = {"format_aware", "semantic", "sentence_aware", "sentence", "late_chunking"}
    strategy = cc.get("strategy", "")
    # format_aware: chỉ hiện nếu split_large_sections = True
    if strategy == "format_aware":
        if not cc.get("split_large_sections"):
            return f"{cs.get('n_chunks','?')} chunks"
        sz = cc.get("chunk_size", "")
        ov = cc.get("chunk_overlap", "")
        return f"{sz}c / {ov}ov · {cs.get('n_chunks','?')} chunks" if sz else f"{cs.get('n_chunks','?')} chunks"
    if strategy in _NO_SIZE:
        return f"{cs.get('n_chunks','?')} chunks"
    sz = cc.get("chunk_size", "")
    ov = cc.get("chunk_overlap", "")
    n  = cs.get("n_chunks", "?")
    if sz and ov:
        return f"{sz}c / {ov}ov · {n} chunks"
    return f"{n} chunks"


def _page_generation():
    """Page 2 — Generation Stage (online, per-query).
    Pre-retrieval → Retrieval → Post-retrieval → Prompt → Generation.
    """
    from pipeline.generation_pipeline import GenerationPipeline

    st.title("💬 Generation Pipeline")
    st.caption(
        "**Stage 2 — chạy mỗi khi có câu hỏi mới.**  \n"
        "Cần hoàn thành **🗃️ Indexing** trước. Cấu hình pipeline ở sidebar trái, "
        "nhập câu hỏi và ấn **▶️ Run**."
    )

    # ── Tự động load index gần nhất nếu chưa có ───────────────────────────
    if "vdb_result" not in st.session_state:
        _try_autoload_latest_index()
    if "vdb_result" not in st.session_state:
        _render_load_index_from_cache()
        return

    with st.sidebar:
        st.header("⚙️ Cấu hình Generation")
        vdb_info = st.session_state.get("vdb_result", {})
        n_vecs   = vdb_info.get("n_vectors", 0)
        provider = vdb_info.get("provider", "?")
        collection = vdb_info.get("collection_name", "")
        st.success(f"✅ Index: **{collection}** · {provider} · {n_vecs:,} vectors")

        # ── Chọn index khác ─────────────────────────────────────────────────
        with st.expander("🔄 Đổi index...", expanded=False):
            _render_index_switcher()
        st.markdown("---")

        # Transfer pending suggestion values
        _PENDING_GEN = [
            ("_want_ret_strategy",   "ret_strategy"),
            ("_want_pre_transforms", "pre_ret_strategies"),
            ("_want_post_reranker",  "post_ret_reranker"),
        ]
        for want_key, widget_key in _PENDING_GEN:
            if want_key in st.session_state:
                st.session_state[widget_key] = st.session_state.pop(want_key)

        local_only = st.session_state.get("local_only", False)
        emb_cfg    = st.session_state.get("emb_cfg_used", {})
        vdb_cfg    = st.session_state.get("vdb_cfg_used", {})

        # ── Pre-retrieval settings ──────────────────────────────────────────
        with st.expander("🔄 Pre-retrieval — biến đổi query  *(tuỳ chọn)*", expanded=True):
            st.info(
                "**Tuỳ chọn.** Mặc định `none` — query được truyền thẳng vào Retrieval.  \n"
                "Bật khi: query có lỗi chính tả (`rewrite`), muốn tăng recall (`multi_query`), "
                "query quá ngắn/trừu tượng (`hyde`), hoặc corpus đa domain (`route`).",
                icon="ℹ️",
            )
            pre_strategies = ["none", "rewrite", "expand",
                              "step_back", "multi_query", "decompose", "self_query", "route"]
            pre_sel = st.multiselect(
                "Transformations (theo thứ tự áp dụng)",
                options=pre_strategies,
                default=["none"],
                key="pre_ret_strategies",
                help="none = passthrough. Các bước áp dụng tuần tự.",
            )
            pre_llm_provider = st.selectbox(
                "LLM Provider",
                ["openai", "anthropic", "ollama"],
                key="pre_ret_llm_provider",
                help="openai/anthropic: API · ollama: local (không cần internet)",
            )
            _PRE_MODEL_PRESETS = {
                "openai":    ["gpt-4.1-mini", "gpt-4o-mini", "gpt-4o"],
                "anthropic": ["claude-haiku-4-5-20251001", "claude-sonnet-4-6"],
                "ollama":    ["qwen2.5:7b", "llama3.2:3b", "mistral:7b", "gemma2:2b", "phi4-mini"],
            }
            _PRE_MODEL_NOTES = {
                "gpt-4.1-mini":              "⭐ $0.40/$1.60/1M — tốt nhất cho query transform",
                "gpt-4o-mini":               "$0.15/$0.60/1M — nhanh nhất, rẻ nhất",
                "gpt-4o":                    "$2.50/$10/1M — chất lượng cao nhất",
                "claude-haiku-4-5-20251001": "⭐ $0.80/$4/1M — rẻ nhất Anthropic",
                "claude-sonnet-4-6":         "$3/$15/1M — balanced",
                "qwen2.5:7b":                "⭐ Local · tốt nhất tiếng Việt · ~4.7GB",
                "llama3.2:3b":               "Local · nhẹ nhất · ~2GB · CPU OK",
                "mistral:7b":                "Local · tiếng Anh · ~4.1GB",
                "gemma2:2b":                 "Local · siêu nhẹ · ~1.6GB",
                "phi4-mini":                 "Local · reasoning tốt · ~2.5GB",
            }
            _pre_presets = _PRE_MODEL_PRESETS.get(pre_llm_provider, ["gpt-4.1-mini"])
            # Reset model khi đổi provider
            if st.session_state.get("_pre_provider_prev") != pre_llm_provider:
                st.session_state["pre_ret_llm_model"] = _pre_presets[0]
                st.session_state["_pre_provider_prev"] = pre_llm_provider
            pre_llm_model = st.selectbox(
                "LLM Model",
                _pre_presets,
                key="pre_ret_llm_model",
                format_func=lambda m: f"{m}  —  {_PRE_MODEL_NOTES[m]}" if m in _PRE_MODEL_NOTES else m,
            )
            if pre_llm_provider == "ollama":
                _pre_ollama_url = st.text_input(
                    "Ollama base URL", value="http://localhost:11434",
                    key="pre_ret_ollama_url",
                )
            else:
                _pre_ollama_url = "http://localhost:11434"
            if "multi_query" in pre_sel:
                pre_n_queries = st.slider("Số sub-queries", 2, 6, 3, key="pre_ret_n_queries")
            else:
                pre_n_queries = 3
            st.session_state["query_pre_cfg"] = {
                "transformations":   pre_sel or ["none"],
                "llm_provider":      pre_llm_provider,
                "llm_model":         pre_llm_model,
                "ollama_base_url":   _pre_ollama_url,
                "multi_query_count": pre_n_queries,
            }

        with st.expander("🔎 Retrieval — chiến lược tìm kiếm", expanded=True):
            ret_strategies = ["hybrid", "dense", "sparse", "multi_query",
                              "parent_document", "sentence_window", "multi_hop", "contextual"]
            ret_strategy = st.selectbox(
                "Strategy",
                ret_strategies,
                key="ret_strategy",
                help="hybrid = dense + BM25 (khuyến nghị mặc định).",
            )
            ret_top_k = st.slider("Top-K kết quả", 3, 30, 10, key="ret_top_k")
            if ret_strategy == "hybrid":
                ret_fusion = st.selectbox(
                    "Fusion method", ["rrf", "weighted", "dbsf"],
                    key="ret_fusion",
                    help="rrf = Reciprocal Rank Fusion (mặc định, robust nhất).",
                )
                ret_alpha = st.slider(
                    "Alpha (dense weight)", 0.0, 1.0, 0.5, 0.05,
                    key="ret_alpha",
                    help="1.0 = pure dense · 0.0 = pure sparse. Chỉ dùng với weighted.",
                )
            else:
                ret_fusion, ret_alpha = "rrf", 0.5
            st.session_state["query_ret_cfg"] = {
                "strategy":      ret_strategy,
                "top_k":         ret_top_k,
                "fusion_method": ret_fusion,
                "hybrid_alpha":  ret_alpha,
            }

        with st.expander("⚙️ Post-retrieval — reranking & filtering  *(tuỳ chọn)*", expanded=True):
            st.info(
                "**Tuỳ chọn.** Mặc định `none` — kết quả retrieval được dùng trực tiếp.  \n"
                "Bật `cross_encoder` reranker khi cần precision cao hơn (PDF kỹ thuật, bảng phức tạp).  \n"
                "Bật `context ordering` (sandwich) luôn có ích — giảm lost-in-the-middle khi có nhiều chunks.",
                icon="ℹ️",
            )
            reranker_opts = ["none", "cross_encoder", "cohere", "llm"]
            reranker = st.selectbox(
                "Reranker",
                reranker_opts,
                key="post_ret_reranker",
                help="cross_encoder: BAAI/bge-reranker-v2-m3 — tốt nhất cho VI, không cần GPU.\ncohere: API, tốt nhất về chất lượng.\nllm: không cần GPU, chậm hơn.",
            )
            post_top_n = st.slider("Top-N sau reranking", 1, 15, 5, key="post_ret_top_n")
            if reranker == "cross_encoder":
                ce_model = st.text_input(
                    "Cross-encoder model",
                    value="BAAI/bge-reranker-v2-m3",
                    key="post_ret_ce_model",
                )
            else:
                ce_model = "BAAI/bge-reranker-v2-m3"

            col_a, col_b = st.columns(2)
            with col_a:
                apply_redundancy = st.checkbox("Semantic dedup", value=True, key="post_apply_redundancy")
                apply_mmr        = st.checkbox("MMR diversity",  value=False, key="post_apply_mmr")
            with col_b:
                apply_compress   = st.checkbox("Compress context", value=False, key="post_apply_compress")
                apply_llm_filter = st.checkbox("LLM filter",     value=False, key="post_apply_llm_filter")

            ordering = st.selectbox(
                "Context ordering",
                ["sandwich", "relevance", "reverse", "original"],
                key="post_ret_ordering",
                help="sandwich = most relevant first + last (tốt nhất cho lost-in-middle).",
            )
            st.session_state["query_post_cfg"] = {
                "reranker":           reranker,
                "top_n":              post_top_n,
                "cross_encoder_model": ce_model,
                "apply_redundancy":   apply_redundancy,
                "apply_mmr":          apply_mmr,
                "apply_compression":  apply_compress,
                "apply_llm_filter":   apply_llm_filter,
                "context_ordering":   ordering,
            }

        with st.expander("📝 Prompt — xây dựng prompt  *(chạy lúc query)*", expanded=True):
            # ── Domain presets ─────────────────────────────────────────────
            _PROMPT_PRESETS: dict[str, dict] = {
                "— Tuỳ chọn thủ công —": {},
                "🔬 Nghiên cứu khoa học": {
                    "template": "citation",
                    "language": "en",
                    "rules": ["grounded_only", "cite_sources", "no_outside_knowledge",
                              "acknowledge_uncertainty", "technical_precision"],
                    "system_extra": "You are a scientific research assistant. Be precise, use technical terminology correctly, and always cite specific sections or figures.",
                },
                "🏦 Ngân hàng & Tài chính": {
                    "template": "citation",
                    "language": "vi",
                    "rules": ["grounded_only", "cite_sources", "no_outside_knowledge",
                              "no_financial_advice", "escalate_to_human"],
                    "system_extra": "You are a banking assistant. Do not give specific investment advice. For complex questions, guide customers to contact a specialist.",
                },
                "🏥 Y tế & Sức khoẻ": {
                    "template": "citation",
                    "language": "vi",
                    "rules": ["grounded_only", "cite_sources", "no_outside_knowledge",
                              "no_medical_advice", "escalate_to_human", "acknowledge_uncertainty"],
                    "system_extra": "You are a medical information assistant. Do not diagnose conditions or prescribe treatments. Always recommend consulting a qualified doctor for specific health concerns.",
                },
                "⚖️ Pháp lý & Luật": {
                    "template": "citation",
                    "language": "vi",
                    "rules": ["grounded_only", "cite_sources", "cite_article_number",
                              "no_legal_advice", "acknowledge_uncertainty", "escalate_to_human"],
                    "system_extra": "You are a legal reference assistant. Always cite specific articles and clauses (e.g., Article 15, Clause 2). Do not provide legal advice — recommend consulting a licensed attorney.",
                },
                "🎓 Giáo dục & Đào tạo": {
                    "template": "conversational",
                    "language": "vi",
                    "rules": ["grounded_only", "explain_step_by_step", "use_examples",
                              "encourage_curiosity", "acknowledge_uncertainty"],
                    "system_extra": "You are a friendly tutor. Explain step by step using clear, simple language and concrete examples. Encourage learners to ask follow-up questions.",
                },
                "🛒 Thương mại điện tử & Hỗ trợ KH": {
                    "template": "citation",
                    "language": "vi",
                    "rules": ["grounded_only", "cite_sources", "polite_tone",
                              "no_competitor_mention", "escalate_to_human"],
                    "system_extra": "You are a customer support assistant. Be polite, concise, and solution-focused. For complex issues, offer to connect the customer with a human agent.",
                },
                "🏭 Kỹ thuật & Công nghiệp": {
                    "template": "citation",
                    "language": "vi",
                    "rules": ["grounded_only", "cite_sources", "technical_precision",
                              "use_specs_and_numbers", "safety_first"],
                    "system_extra": "You are a technical documentation assistant. Be precise with specifications, part numbers, and measurement units. Always mention relevant safety warnings.",
                },
                "⚽ Thể thao & Giải trí": {
                    "template": "basic",
                    "language": "vi",
                    "rules": ["grounded_only", "engaging_tone", "use_examples"],
                    "system_extra": "You are an enthusiastic sports expert. Respond with energy and include statistics and data from the context when available.",
                },
            }

            _preset_sel = st.selectbox(
                "Domain preset",
                options=list(_PROMPT_PRESETS.keys()),
                key="prompt_preset_sel",
                help="Chọn domain để tự động điền template + rules phù hợp. Chọn 'Tuỳ chọn thủ công' để cấu hình từng mục.",
            )
            _preset = _PROMPT_PRESETS[_preset_sel]

            # Auto-apply preset khi user đổi selection (không cần nút bấm)
            if _preset and st.session_state.get("_last_preset_sel") != _preset_sel:
                st.session_state["prompt_template"]          = _preset.get("template", "citation")
                st.session_state["prompt_language"]          = _preset.get("language", "both")
                st.session_state["prompt_rules_sel"]         = _preset.get("rules", [])
                st.session_state["prompt_system_extra"]      = _preset.get("system_extra", "")
                # Sync widget keys ngay để Streamlit không dùng giá trị cũ
                st.session_state["prompt_template"]          = _preset.get("template", "citation")
                st.session_state["prompt_system_extra_input"]= _preset.get("system_extra", "")
                st.session_state["_last_preset_sel"]         = _preset_sel
                st.rerun()
            st.session_state["_last_preset_sel"] = _preset_sel

            st.markdown("---")

            # ── Template & Language ────────────────────────────────────────
            prompt_template = st.selectbox(
                "Template",
                ["citation", "basic", "conversational", "structured"],
                key="prompt_template",
                help="citation: yêu cầu [NGUỒN N] inline.\nbasic: tối giản.\nconversational: multi-turn.\nstructured: JSON output.",
            )
            prompt_language = st.selectbox(
                "Ngôn ngữ prompt",
                ["both", "vi", "en"],
                key="prompt_language",
                help="vi: instruction tiếng Việt · en: tiếng Anh · both: dùng vi (an toàn cho corpus song ngữ)",
            )
            prompt_max_ctx = st.number_input(
                "Giới hạn context (ký tự, 0 = không giới hạn)",
                min_value=0, max_value=100_000, value=0, step=1000,
                key="prompt_max_ctx",
            )
            if prompt_template == "conversational":
                prompt_max_hist = st.slider("Số lượt lịch sử tối đa", 1, 10, 5, key="prompt_max_hist")
            else:
                prompt_max_hist = 5

            # ── Rule Builder ───────────────────────────────────────────────
            st.markdown("**⚙️ Rules (gắn vào system prompt)**")

            _ALL_RULES: dict[str, dict] = {
                # Grounding
                "grounded_only":         {"label": "🔒 Chỉ dùng context",        "group": "Grounding",
                    "vi": "Chỉ trả lời dựa trên context được cung cấp. Không dùng kiến thức bên ngoài.",
                    "en": "Answer ONLY using the provided context. Do not use outside knowledge or make assumptions."},
                "cite_sources":          {"label": "📚 Trích dẫn nguồn",          "group": "Grounding",
                    "vi": "Trích dẫn nguồn cho mọi thông tin bằng [NGUỒN N] hoặc (nguồn: tên file, trang N).",
                    "en": "Cite the source for every factual claim using [SOURCE N] or (source: filename, page N)."},
                "cite_article_number":   {"label": "⚖️ Trích dẫn điều khoản",     "group": "Grounding",
                    "vi": "Khi trích dẫn văn bản pháp lý, ghi rõ số điều khoản (ví dụ: Điều 15, Khoản 2).",
                    "en": "When referencing legal text, always cite the exact article number (e.g., Điều 15, Khoản 2)."},
                "no_outside_knowledge":  {"label": "🚫 Không dùng kiến thức ngoài","group": "Grounding",
                    "vi": "Không dùng thông tin từ internet, dữ liệu huấn luyện, hoặc bất kỳ nguồn nào ngoài tài liệu được cung cấp.",
                    "en": "Do NOT use web information, training data, or any knowledge outside the provided documents."},
                # Uncertainty
                "acknowledge_uncertainty":{"label": "🤔 Thừa nhận không chắc",    "group": "Uncertainty",
                    "vi": "Nếu context không trả lời rõ câu hỏi, hãy nói: \"Tôi không tìm thấy thông tin này trong tài liệu được cung cấp.\" Không đoán mò.",
                    "en": "If the context does not clearly answer the question, say: \"I cannot find this information in the provided documents.\" Do not guess."},
                "partial_answer_ok":     {"label": "✂️ Trả lời một phần",         "group": "Uncertainty",
                    "vi": "Nếu chỉ trả lời được một phần, hãy cung cấp những gì biết từ context và nêu rõ phần còn thiếu.",
                    "en": "If you can only partially answer, provide what you know from the context and clearly state what is missing."},
                # Safety / Compliance
                "no_financial_advice":   {"label": "💰 Không tư vấn tài chính",   "group": "Safety & Compliance",
                    "vi": "Không đưa ra tư vấn đầu tư, giao dịch, hay tài chính cụ thể. Luôn khuyến nghị tham khảo chuyên gia tài chính có chứng chỉ.",
                    "en": "Do NOT provide specific investment, trading, or financial advice. Always recommend consulting a licensed financial advisor."},
                "no_medical_advice":     {"label": "🏥 Không tư vấn y tế",        "group": "Safety & Compliance",
                    "vi": "Không chẩn đoán bệnh hoặc kê đơn thuốc. Luôn khuyến nghị người dùng tham khảo bác sĩ hoặc chuyên gia y tế.",
                    "en": "Do NOT diagnose conditions or prescribe treatments. Always recommend consulting a qualified healthcare professional."},
                "no_legal_advice":       {"label": "⚖️ Không tư vấn pháp lý",     "group": "Safety & Compliance",
                    "vi": "Không đưa ra tư vấn pháp lý cụ thể. Luôn khuyến nghị tham khảo luật sư cho các vấn đề pháp lý.",
                    "en": "Do NOT provide specific legal advice. Always recommend consulting a licensed attorney for legal matters."},
                "escalate_to_human":     {"label": "👤 Chuyển nhân viên khi cần",  "group": "Safety & Compliance",
                    "vi": "Với các vấn đề phức tạp, khẩn cấp hoặc nhạy cảm, chủ động đề xuất người dùng liên hệ nhân viên hỗ trợ.",
                    "en": "For complex, urgent, or sensitive issues, proactively suggest contacting a human representative."},
                "safe_messaging":        {"label": "🛡️ Safe messaging",           "group": "Safety & Compliance",
                    "vi": "Tuân thủ hướng dẫn truyền thông an toàn. Không cung cấp thông tin có thể gây hại. Cung cấp đường dây hỗ trợ khủng hoảng khi phù hợp.",
                    "en": "Follow safe messaging guidelines. Do not provide detailed information that could enable self-harm. Provide crisis helpline information when appropriate."},
                # Quality
                "technical_precision":   {"label": "🔬 Chính xác kỹ thuật",       "group": "Quality",
                    "vi": "Dùng thuật ngữ kỹ thuật chính xác. Ghi rõ giá trị, đơn vị, số model và phiên bản khi có trong context.",
                    "en": "Use precise technical terminology. Include specific values, units, model numbers, and version numbers when available in the context."},
                "use_specs_and_numbers": {"label": "📊 Dùng số liệu cụ thể",      "group": "Quality",
                    "vi": "Luôn kèm theo số liệu đo lường, thông số kỹ thuật, ngày tháng và dữ liệu định lượng từ context khi liên quan.",
                    "en": "Always include specific measurements, specifications, dates, and quantitative data from the context when relevant."},
                "explain_step_by_step":  {"label": "📝 Giải thích từng bước",     "group": "Quality",
                    "vi": "Chia nhỏ câu trả lời phức tạp thành các bước đánh số rõ ràng. Dùng ngôn ngữ đơn giản, tránh thuật ngữ không cần thiết.",
                    "en": "Break down complex answers into clear numbered steps. Use simple language and avoid unnecessary jargon."},
                "use_examples":          {"label": "💡 Dùng ví dụ minh hoạ",      "group": "Quality",
                    "vi": "Minh hoạ các khái niệm trừu tượng bằng ví dụ cụ thể lấy từ context được cung cấp khi có thể.",
                    "en": "Illustrate abstract concepts with concrete examples drawn from the provided context when possible."},
                "concise_answer":        {"label": "✂️ Ngắn gọn, súc tích",       "group": "Quality",
                    "vi": "Giữ câu trả lời ngắn gọn và tập trung. Tránh lặp lại không cần thiết. Trả lời thẳng vào vấn đề trước, rồi mới giải thích thêm.",
                    "en": "Keep answers concise and focused. Avoid unnecessary repetition. Lead with the direct answer, then provide supporting details."},
                "structured_output_md":  {"label": "📋 Dùng Markdown formatting",  "group": "Quality",
                    "vi": "Định dạng câu trả lời bằng Markdown: dùng **in đậm** cho từ khoá, danh sách gạch đầu dòng cho nhiều mục, bảng cho so sánh.",
                    "en": "Format your response with Markdown: use **bold** for key terms, bullet lists for multiple items, and tables for comparisons."},
                # Tone
                "polite_tone":           {"label": "😊 Lịch sự, thân thiện",      "group": "Tone",
                    "vi": "Giữ thái độ lịch sự, chuyên nghiệp và thân thiện trong suốt cuộc trò chuyện.",
                    "en": "Maintain a polite, professional, and friendly tone throughout the conversation."},
                "engaging_tone":         {"label": "🎯 Năng động, cuốn hút",      "group": "Tone",
                    "vi": "Dùng giọng văn sinh động, nhiệt tình. Làm cho thông tin trở nên thú vị và dễ tiếp cận.",
                    "en": "Use an engaging, enthusiastic tone. Make information interesting and accessible."},
                "formal_tone":           {"label": "👔 Trang trọng, chuyên nghiệp","group": "Tone",
                    "vi": "Dùng ngôn ngữ trang trọng, chuyên nghiệp phù hợp với bối cảnh kinh doanh hoặc học thuật.",
                    "en": "Use formal, professional language appropriate for business or academic contexts."},
                "empathetic_tone":       {"label": "💙 Đồng cảm, thấu hiểu",     "group": "Tone",
                    "vi": "Thể hiện sự đồng cảm và thấu hiểu. Ghi nhận tình huống của người dùng trước khi cung cấp thông tin.",
                    "en": "Show empathy and understanding. Acknowledge the user's situation before providing information."},
                # Multilingual
                "respond_in_query_language": {"label": "🌐 Trả lời cùng ngôn ngữ câu hỏi", "group": "Language",
                    "vi": "Nhận diện ngôn ngữ câu hỏi của người dùng và trả lời bằng ngôn ngữ đó, bất kể ngôn ngữ của tài liệu nguồn.",
                    "en": "Detect the language of the user's question and respond in the same language, regardless of the language of the source documents."},
                "force_vietnamese":      {"label": "🇻🇳 Luôn trả lời tiếng Việt","group": "Language",
                    "vi": "Luôn trả lời bằng Tiếng Việt, kể cả khi tài liệu nguồn bằng tiếng Anh.",
                    "en": "Always respond in Vietnamese (Tiếng Việt), even if the source documents are in English."},
                "force_english":         {"label": "🇬🇧 Luôn trả lời tiếng Anh", "group": "Language",
                    "vi": "Luôn trả lời bằng tiếng Anh, kể cả khi tài liệu nguồn bằng tiếng Việt.",
                    "en": "Always respond in English, even if the source documents are in Vietnamese."},
                # Interaction
                "ask_clarifying":        {"label": "❓ Hỏi lại khi mơ hồ",        "group": "Interaction",
                    "vi": "Nếu câu hỏi mơ hồ hoặc có thể hiểu nhiều cách, hãy hỏi lại để làm rõ trước khi trả lời.",
                    "en": "If the question is ambiguous or could be interpreted multiple ways, ask a clarifying question before answering."},
                "no_competitor_mention": {"label": "🚫 Không đề cập đối thủ",     "group": "Interaction",
                    "vi": "Không đề cập, so sánh hoặc đánh giá sản phẩm hay dịch vụ của đối thủ cạnh tranh.",
                    "en": "Do not mention, compare, or evaluate competitor products or services."},
                "encourage_curiosity":   {"label": "🌱 Khuyến khích khám phá",    "group": "Interaction",
                    "vi": "Kết thúc câu trả lời bằng một câu hỏi gợi mở hoặc gợi ý để giúp người dùng khám phá thêm về chủ đề.",
                    "en": "End responses with a related follow-up question or suggestion to help users explore the topic further."},
                "summarize_long":        {"label": "📄 Tóm tắt câu trả lời dài",  "group": "Interaction",
                    "vi": "Với câu trả lời dài (>5 gạch đầu dòng hoặc >300 từ), thêm phần tóm tắt ngắn gọn ở đầu.",
                    "en": "For long answers (>5 bullet points or >300 words), add a brief TL;DR summary at the top."},
                "safety_first":          {"label": "⚠️ Ưu tiên cảnh báo an toàn","group": "Interaction",
                    "vi": "Luôn đề cập đến cảnh báo an toàn, biện pháp phòng ngừa hoặc yêu cầu quy định liên quan khi chúng xuất hiện trong context.",
                    "en": "Always mention relevant safety warnings, precautions, or regulatory requirements when they appear in the context."},
            }

            # Group rules for display
            _groups: dict[str, list] = {}
            for rk, rv in _ALL_RULES.items():
                g = rv["group"]
                _groups.setdefault(g, []).append(rk)

            current_rules = list(st.session_state.get("prompt_rules_sel", [
                "grounded_only", "cite_sources", "no_outside_knowledge", "acknowledge_uncertainty"
            ]))

            new_rules = list(current_rules)
            for grp, rule_keys in _groups.items():
                st.markdown(f"*{grp}*")
                for rk in rule_keys:
                    rv      = _ALL_RULES[rk]
                    checked = rk in current_rules
                    val = st.checkbox(
                        rv["label"],
                        value=checked,
                        key=f"rule_{rk}",
                        help=rv.get("en", rv.get("vi", "")),
                    )
                    if val and rk not in new_rules:
                        new_rules.append(rk)
                    elif not val and rk in new_rules:
                        new_rules.remove(rk)

            st.session_state["prompt_rules_sel"] = new_rules

            # system_extra đến từ preset (không hiển thị widget riêng)
            # User có thể chỉnh sửa trực tiếp trong System message bên dưới
            prompt_system_extra = st.session_state.get("prompt_system_extra", "")

            # ── Build final system instruction ────────────────────────────
            # Thứ tự: system_extra (vai trò) trước, rules sau
            # system_extra đặt trước base template text để định nghĩa vai trò rõ ràng
            # Rules luôn dùng tiếng Anh — LLM hiểu instruction EN tốt hơn
            _rule_texts = [_ALL_RULES[r].get("en", _ALL_RULES[r].get("vi", ""))
                           for r in new_rules if r in _ALL_RULES]
            _rules_str = "\n".join([f"{i+1}) {t}" for i, t in enumerate(_rule_texts)])

            # Thứ tự: system_extra (vai trò domain) → rules
            # Nếu không có gì → dùng default generic role
            _DEFAULT_ROLE = (
                "You are a helpful assistant. Answer questions based on the provided context. "
                "Be accurate, concise, and cite your sources when possible."
            )
            _parts = []
            if prompt_system_extra.strip():
                _parts.append(prompt_system_extra.strip())
            elif not _rules_str:
                # Không có preset và không có rules → dùng default
                _parts.append(_DEFAULT_ROLE)
            if _rules_str:
                _parts.append(_rules_str)
            _combined_instruction = "\n\n".join(_parts)

            st.session_state["query_prompt_cfg"] = {
                "template":           prompt_template,
                "language":           prompt_language,
                "max_context_chars":  int(prompt_max_ctx),
                "max_history_turns":  prompt_max_hist,
                "system_instruction": _combined_instruction,
            }

            st.markdown("---")
            st.markdown("**👁️ System message** *(tự động cập nhật)*")
            if True:  # always show

                try:
                    from prompt.factory import get_prompt_builder
                    _builder = get_prompt_builder(
                        prompt_template,
                        language=prompt_language,
                        max_context_chars=int(prompt_max_ctx),
                        system_instruction=_combined_instruction,
                        **({"max_history_turns": prompt_max_hist} if prompt_template == "conversational" else {}),
                    )
                    _r = _builder.build("", [])
                    _sys_content = next((m["content"] for m in _r.messages if m["role"] == "system"), "")
                    import hashlib as _hl
                    _cfg_hash = _hl.md5(_sys_content.encode()).hexdigest()[:8]
                    st.markdown("**System message** *(gửi cho LLM mỗi lần query)*:")
                    _sys_area = st.text_area(
                        "System", value=_sys_content, height=200,
                        key=f"prompt_preview_system_{_cfg_hash}",
                        label_visibility="collapsed",
                    )
                    st.caption(
                        "ℹ️ **User message** sẽ được tạo tự động khi Run — "
                        "chứa các đoạn văn bản tìm được từ index + câu hỏi của bạn."
                    )
                    if _sys_area != _sys_content:
                        st.session_state["_prompt_override"] = {"system": _sys_area}
                        st.info("✏️ Bạn đã chỉnh sửa system message — bản tuỳ chỉnh sẽ được dùng khi Run.")
                    else:
                        st.session_state.pop("_prompt_override", None)
                except Exception as _pe:
                    st.warning(f"Không thể tạo preview: {_pe}")

        with st.expander("🤖 Generation — LLM sinh câu trả lời  *(chạy lúc query)*", expanded=True):
            st.info(
                "Chọn LLM để sinh câu trả lời cuối cùng từ prompt đã xây dựng.",
                icon="ℹ️",
            )
            # Provider selection
            gen_provider = st.selectbox(
                "Provider",
                ["openai", "anthropic", "google", "ollama", "cohere"],
                key="gen_provider",
                help="openai: GPT-4.1-mini (rẻ, nhanh) · anthropic: Claude · google: Gemini (free tier) · ollama: local · cohere: RAG-optimised",
            )

            # Model presets per provider
            _GEN_MODEL_PRESETS: dict[str, list[str]] = {
                "openai":    ["gpt-4.1-mini", "gpt-4o-mini", "gpt-4o", "o3-mini"],
                "anthropic": ["claude-haiku-4-5-20251001", "claude-sonnet-4-6", "claude-opus-4-6"],
                "google":    ["gemini-2.0-flash", "gemini-2.0-flash-lite", "gemini-1.5-pro"],
                "ollama":    ["qwen2.5:7b", "llama3.2:3b", "mistral:7b", "gemma2:2b", "phi4-mini"],
                "cohere":    ["command-r-plus", "command-r"],
            }
            _GEN_MODEL_NOTES: dict[str, str] = {
                "gpt-4.1-mini":              "⭐ $0.40/$1.60 per 1M — mặc định tốt nhất",
                "gpt-4o-mini":               "$0.15/$0.60 per 1M — nhanh nhất, rẻ nhất",
                "gpt-4o":                    "$2.50/$10 per 1M — chất lượng cao nhất",
                "o3-mini":                   "Reasoning model, câu hỏi phức tạp",
                "claude-haiku-4-5-20251001": "⭐ $0.80/$4 per 1M — rẻ nhất Anthropic",
                "claude-sonnet-4-6":         "$3/$15 per 1M — balanced",
                "claude-opus-4-6":           "$15/$75 per 1M — mạnh nhất",
                "gemini-2.0-flash":          "⭐ Free tier · nhanh · đa ngôn ngữ",
                "gemini-2.0-flash-lite":     "Free tier · rẻ nhất Gemini",
                "gemini-1.5-pro":            "2M context window · mạnh nhất Gemini",
                "qwen2.5:7b":                "⭐ Local · tốt nhất tiếng Việt · ~4.7GB",
                "llama3.2:3b":               "Local · nhẹ nhất · ~2GB · CPU OK",
                "mistral:7b":                "Local · tiếng Anh · ~4.1GB",
                "gemma2:2b":                 "Local · siêu nhẹ · ~1.6GB",
                "phi4-mini":                 "Local · reasoning tốt · ~2.5GB",
                "command-r-plus":            "⭐ RAG-optimised · 128K ctx · $2.5/$10/1M",
                "command-r":                 "$0.15/$0.60/1M — đủ tốt cho Q&A",
            }
            presets = _GEN_MODEL_PRESETS.get(gen_provider, [])
            # Reset model selection khi đổi provider
            if st.session_state.get("_gen_provider_prev") != gen_provider:
                st.session_state["gen_model"] = presets[0] if presets else ""
                st.session_state["_gen_provider_prev"] = gen_provider

            def _gen_model_label(m: str) -> str:
                note = _GEN_MODEL_NOTES.get(m, "")
                return f"{m}  —  {note}" if note else m

            gen_model = st.selectbox(
                "Model",
                presets,
                key="gen_model",
                format_func=_gen_model_label,
            )

            if gen_provider == "ollama":
                gen_ollama_url = st.text_input(
                    "Ollama URL", value="http://localhost:11434",
                    key="gen_ollama_url",
                    help="URL Ollama server. Model tự pull nếu chưa có.",
                )
                gen_auto_pull = st.checkbox("Tự động pull model nếu chưa có", value=True, key="gen_auto_pull")
            else:
                gen_ollama_url = "http://localhost:11434"
                gen_auto_pull  = True

            col_gen1, col_gen2 = st.columns(2)
            with col_gen1:
                gen_temperature = st.slider(
                    "Temperature", 0.0, 1.0, 0.0, 0.05,
                    key="gen_temperature",
                    help="0 = deterministic. Tăng để câu trả lời đa dạng hơn.",
                )
            with col_gen2:
                gen_max_tokens = st.number_input(
                    "Max tokens", min_value=256, max_value=8192, value=2048, step=256,
                    key="gen_max_tokens",
                    help="Số token tối đa trong câu trả lời.",
                )
            gen_streaming = st.checkbox(
                "Streaming (hiển thị từng token)",
                value=True,
                key="gen_streaming",
                help="Bật để xem câu trả lời xuất hiện dần — trải nghiệm tốt hơn với câu trả lời dài.",
            )

            # API key check
            _GEN_ENV_KEYS = {
                "openai":    "OPENAI_API_KEY",
                "anthropic": "ANTHROPIC_API_KEY",
                "google":    "GOOGLE_API_KEY",
                "cohere":    "COHERE_API_KEY",
            }
            _env_key = _GEN_ENV_KEYS.get(gen_provider)
            if _env_key:
                if _get_env(_env_key):
                    st.success(f"✅ `{_env_key}` đã được cấu hình.")
                else:
                    st.warning(f"⚠️ Cần `{_env_key}` trong file `.env` hoặc biến môi trường.")

            st.session_state["query_gen_cfg"] = {
                "provider":    gen_provider,
                "model_name":  gen_model,
                "temperature": gen_temperature,
                "max_tokens":  int(gen_max_tokens),
                "streaming":   gen_streaming,
                "base_url":    gen_ollama_url,
                "auto_pull":   gen_auto_pull,
            }

        # ── Apply Settings button ────────────────────────────────────────────
        st.markdown("---")
        _cfg_confirmed = st.session_state.get("_gen_cfg_confirmed", False)
        if st.button(
            "✅ Áp dụng cấu hình" if not _cfg_confirmed else "✅ Đã áp dụng — cập nhật lại",
            key="btn_apply_gen_cfg",
            type="primary",
            use_container_width=True,
            help="Xác nhận cấu hình hiện tại. Sau đó nhập câu hỏi ở trang chính để chạy.",
        ):
            st.session_state["_gen_cfg_confirmed"] = True
            st.session_state["_gen_cfg_snapshot"] = {
                "pre":    st.session_state.get("query_pre_cfg",  {}),
                "ret":    st.session_state.get("query_ret_cfg",  {}),
                "post":   st.session_state.get("query_post_cfg", {}),
                "prompt": st.session_state.get("query_prompt_cfg", {}),
                "gen":    st.session_state.get("query_gen_cfg",  {}),
            }
            st.rerun()

        if _cfg_confirmed:
            _snap = st.session_state.get("_gen_cfg_snapshot", {})
            _snap_gen    = _snap.get("gen", {})
            _snap_ret    = _snap.get("ret", {})
            _snap_prompt = _snap.get("prompt", {})
            st.success(
                f"**Pipeline đã sẵn sàng.**  \n"
                f"Retrieval: `{_snap_ret.get('strategy','?')}` · "
                f"Top-K: `{_snap_ret.get('top_k','?')}` · "
                f"Prompt: `{_snap_prompt.get('template','?')}` · "
                f"LLM: `{_snap_gen.get('provider','?')}/{_snap_gen.get('model_name','?').split('/')[-1]}`  \n"
                f"➡️ Nhập câu hỏi ở trang chính và ấn **▶️ Run**."
            )
        else:
            st.info("⬆️ Cấu hình xong thì ấn **Áp dụng** để kích hoạt.", icon="💡")

    # ── Query input & results ────────────────────────────────────────────────
    # Transfer pending
    _PENDING_QUERY = [
        ("_want_ret_strategy",   "ret_strategy"),
        ("_want_pre_transforms", "pre_ret_strategies"),
        ("_want_post_reranker",  "post_ret_reranker"),
    ]
    for want_key, widget_key in _PENDING_QUERY:
        if want_key in st.session_state:
            st.session_state[widget_key] = st.session_state.pop(want_key)

    # Nút xóa lịch sử
    if st.session_state.get("_query_history"):
        if st.button("🗑️ Xóa lịch sử hội thoại", key="btn_clear_hist", type="secondary"):
            st.session_state["_query_history"] = []
            st.rerun()

    # ── Index hiện tại + chọn index khác ─────────────────────────────────────
    _vdb     = st.session_state.get("vdb_result", {})
    _emb_cfg = st.session_state.get("emb_cfg_used", {})

    with st.container(border=True):
        _ci, _cb = st.columns([5, 1], vertical_alignment="center")
        with _ci:
            _n_vecs   = _vdb.get("n_vectors", 0)
            _provider = _vdb.get("provider", "?")
            _coll     = _vdb.get("collection_name", "?")
            _em       = _emb_cfg.get("model_name", "?").split("/")[-1]
            _ep       = _emb_cfg.get("provider", "?")
            _sp       = _emb_cfg.get("enable_sparse", False)
            _sm       = _emb_cfg.get("sparse_method", "none")

            # Try to get full pipeline meta (saved after indexing or loaded from pipeline)
            _pmeta = st.session_state.get("_current_pipeline_meta", {})
            _lc    = _pmeta.get("loader_cfg", {})
            _cc    = _pmeta.get("chunking_cfg", {})
            _src   = _pmeta.get("source_path", "")
            _dt    = _pmeta.get("created_at", "")
            _nchk  = _pmeta.get("n_chunks", "?")

            # Source display — dùng _input_display_name để hiện tên file gốc
            _src_disp = _input_display_name(_pmeta)
            if not _src_disp or _src_disp in ("-", ""):
                _src_disp = _coll  # fallback: collection name

            # Build tag strings
            _sp_tag = f" <code>+ {_sm}</code>" if _sp and _sm != "none" else ""
            _dim    = _emb_cfg.get("dimensions")
            _mrl    = f" <code>MRL→{_dim}d</code>" if _dim else ""
            _pdf_s  = _lc.get("pdf_strategy", "?")
            _cks    = _cc.get("strategy", "?")
            _NO_SZ = {"format_aware", "semantic", "sentence_aware", "sentence", "late_chunking"}
            _csz   = _cc.get("chunk_size", "")
            _cov   = _cc.get("chunk_overlap", "")
            _split = _cc.get("split_large_sections", False)
            if _cks in _NO_SZ and not (_cks == "format_aware" and _split):
                _chunk_detail = ""
            elif _csz and _cov:
                _chunk_detail = f" ({_csz}c / {_cov}ov)"
            else:
                _chunk_detail = ""

            _nv_s = f"{_n_vecs:,}" if isinstance(_n_vecs, int) else str(_n_vecs)

            st.markdown(
                f"<div style='font-weight:700;font-size:1em;margin-bottom:.4rem;'>"
                f"🗃️ Index đang dùng"
                f"  <span style='background:#1a3a2a;color:#7ee787;border-radius:6px;"
                f"padding:2px 10px;font-size:.8em;'>✅ {_nv_s} vectors</span>"
                f"{'  ' + _dt if _dt else ''}"
                f"</div>"
                f"<div style='font-size:.84em;line-height:1.75;opacity:.82;'>"
                f"<b>📄 Input:</b> <code>{_src_disp}</code><br>"
                f"<b>📂 Loader:</b> <code>{_pdf_s}</code>"
                f" &nbsp;·&nbsp; <b>✂️ Chunk:</b> <code>{_cks}</code>{_chunk_detail}"
                f" &nbsp;·&nbsp; {_nchk} chunks<br>"
                f"<b>🧮 Embed:</b> <code>{_ep}/{_em}</code>{_sp_tag}{_mrl}"
                f" &nbsp;·&nbsp; <b>🗄️ VDB:</b> <code>{_vdb_display_name(_vdb)}</code>"
                f"</div>",
                unsafe_allow_html=True,
            )
        with _cb:
            if st.button("🔄 Đổi", key="btn_main_change_index",
                         type="secondary", use_container_width=True,
                         help="Chọn index khác từ cache"):
                st.session_state["_show_index_picker"] = not st.session_state.get("_show_index_picker", False)

    # ── Bảng chọn index (toggle) ───────────────────────────────────────────
    if st.session_state.get("_show_index_picker", False):
        _valid = _cached_list_pipelines(
            "processed_data", st.session_state.get("_cache_list_version", 0)
        )
        if not _valid:
            st.info("Chưa có index nào trong cache.", icon="📭")
        else:
            st.markdown(f"**{len(_valid)} index có sẵn** — nhấn ▶️ để load:")
            for _ie, _ent in enumerate(_valid):
                # list_complete_pipelines format: _ent["loader"]["cfg"] directly
                _disp3 = _ent.get("input_display", "")
                _src  = _disp3 if _disp3 else _ent.get("source_path", "-").replace("\\", "/").split("/")[-1]
                _dt   = _ent.get("created_at", "")[:16].replace("T", " ")
                _sz   = _ent.get("total_size_mb", 0)
                _lc   = _ent.get("loader",    {}).get("cfg",   {})
                _cc   = _ent.get("chunking",  {}).get("cfg",   {})
                _ec   = _ent.get("embedding", {}).get("cfg",   {})
                _vs   = _ent.get("vector_db", {}).get("stats", {})
                _vc   = _ent.get("vector_db", {}).get("cfg",   {})
                _nv   = _vs.get("n_vectors", "?")
                _nvs  = f"{_nv:,}" if isinstance(_nv, int) else str(_nv)
                _mod  = _ec.get("model_name", "?").split("/")[-1]
                _prv  = _ec.get("provider", "?")
                _cks  = _cc.get("strategy", "?")
                _pdfs = _lc.get("pdf_strategy", "?")
                _vp   = _vs.get("provider", "?")
                _vco  = _vs.get("collection_name", "?")
                _spe  = _ec.get("enable_sparse", False)
                _spn  = _ec.get("sparse_method", "none")
                _spt  = f" + {_spn}" if _spe and _spn != "none" else ""
                _curr = (_ent.get("pipeline_id") == st.session_state.get("_loaded_pipeline_id"))
                _bc   = "rgba(40,167,69,.4)" if _curr else "rgba(48,54,61,.5)"
                _vlm_badge = ""
                if _lc.get("describe_images"):
                    _vlm_m = _lc.get("vision_model", "VLM").split("/")[-1]
                    _vlm_badge = (f" <span style='color:#7c3aed;font-size:.8em;"
                                  f"font-weight:600;'>🖼️ {_vlm_m}</span>")

                _ra, _rb = st.columns([5, 1], vertical_alignment="center")
                with _ra:
                    _curr_tag = '  <span style="color:#7ee787;font-size:.8em;">✅ đang dùng</span>' if _curr else ""
                    st.markdown(
                        f"<div style='border:1px solid {_bc};border-radius:8px;"
                        f"padding:.45rem .85rem;font-size:.84em;line-height:1.6;'>"
                        f"<b>📄 {_src}</b>{_curr_tag}"
                        f"<span style='opacity:.5;font-size:.8em;margin-left:.6rem;'>"
                        f"{_dt} · {_sz:.1f} MB</span><br>"
                        f"<code>{_pdfs}</code>{_vlm_badge} &rarr; <code>{_cks}</code> &rarr; "
                        f"<code>{_prv}/{_mod}{_spt}</code> &rarr; "
                        f"<code>{_vdb_display_name(_vs, _vc)}</code> · {_nvs} vecs"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                with _rb:
                    if not _curr:
                        if st.button("▶️ Load", key=f"main_load_{_ie}",
                                     type="primary", use_container_width=True):
                            _load_pipeline_into_session(_ent)
                            st.session_state["_show_index_picker"] = False
                            st.rerun()
                    else:
                        st.caption("đang dùng")
        st.markdown("---")

    # ── Hướng dẫn Stage 2 ────────────────────────────────────────────────────
    with st.expander("📖 Hướng dẫn sử dụng — Stage 2: Generation", expanded=False):
        st.markdown("""
**Generation là gì?**
Stage 2 — chạy **online, mỗi khi người dùng đặt câu hỏi**.
Pipeline lấy query → tìm chunk liên quan từ index → xây prompt → LLM sinh câu trả lời.

---

**Bước 7 — Pre-retrieval** *(tuỳ chọn)*
Biến đổi query trước khi tìm kiếm. Mặc định `none` — dùng query thẳng.

**Bước 8 — Retrieval**
Tìm top-K chunk liên quan từ Vector DB. `hybrid` (dense + BM25) là tốt nhất khi đã bật sparse embedding.

**Bước 9 — Post-retrieval** *(tuỳ chọn)*
Rerank, filter, compress chunk sau khi retrieved. `cross_encoder` reranker khuyến nghị.

**Bước 10 — Prompt**
Xây dựng system + user message gửi lên LLM. Template `citation` yêu cầu LLM trích dẫn [NGUỒN N].

**Bước 11 — Generation**
LLM sinh câu trả lời. Chọn provider và model phù hợp budget và yêu cầu tiếng Việt.

---
💡 **Tip:** Cấu hình các bước ở sidebar trái, sau đó nhập câu hỏi và ấn **▶️ Run**.
""")

    with st.expander("🔄 Pre-retrieval — khi nào nên dùng?"):
        st.markdown("""| Strategy | Cơ chế | Khi nào dùng |
|----------|--------|-------------|
| `none` ⭐ | Giữ nguyên query | Query đã rõ ràng và chi tiết |
| `rewrite` | LLM viết lại chuẩn hơn | Query ngắn, thiếu context |
| `expand` | Thêm từ đồng nghĩa | Corpus có nhiều thuật ngữ tương đương |
| `step_back` | Tổng quát hóa query | Query quá cụ thể |
| `multi_query` | Tạo N query con → merge RRF | Query phức tạp, nhiều khía cạnh |
| `decompose` | Chia sub-query độc lập | Multi-hop reasoning |
| `self_query` | Parse → metadata filter + semantic | Corpus có metadata phong phú |
| `route` | Phân loại → chọn strategy phù hợp | Corpus đa dạng |
""")

    with st.expander("🔎 Retrieval — chiến lược nào?"):
        st.markdown("""| Strategy | Cơ chế | Khi nào dùng |
|----------|--------|-------------|
| `hybrid` ⭐ | Dense + Sparse RRF | **Mặc định tốt nhất** — cần bật BM25 ở Indexing |
| `dense` | Cosine similarity | Câu hỏi ngữ nghĩa |
| `sparse` | BM25 / SPLADE | Từ khoá, tên riêng, số liệu |
| `multi_query` | N phiên bản query → merge | Query mơ hồ |
| `parent_document` | Retrieve child, return parent | Cần thêm context |
| `sentence_window` | N câu xung quanh câu hit | Corpus văn xuôi |
| `multi_hop` | Multi-step reasoning | Multi-hop Q&A |

**Top-K:** thường 10–20 trước reranking.
""")
        if st.checkbox("🔍 Xem giải thích chi tiết — Retrieval", key="detail_retrieval_gen"):
            st.markdown("""
---
**🟢 `hybrid` — Hybrid Retrieval (Reciprocal Rank Fusion)**

Chạy cả dense ANN search và BM25 lexical search song song, merge bằng RRF:
`score(d) = Σ 1 / (rank_dense(d) + 60) + 1 / (rank_sparse(d) + 60)`

- ✅ Bổ trợ nhau: dense tốt cho paraphrase, BM25 tốt cho exact match.
- ❌ Cần BM25 index (bật Sparse Embedding khi indexing).
- 📐 `top_k` = số candidate mỗi retriever trước merge.

---
**🔵 `multi_query` — Multi-Query Retrieval**

LLM sinh ra N biến thể của query gốc (paraphrase, góc nhìn khác), chạy retrieval cho mỗi biến thể, deduplicate kết quả.

- ✅ Tăng recall đáng kể với query ngắn/mơ hồ.
- ❌ N lần retrieval cost · cần LLM call.

---
**🟠 `parent_document` — Parent Document Retrieval**

Index child chunks nhỏ (chính xác), nhưng khi hit, trả về parent chunk lớn hơn cho LLM.

- ✅ Tìm kiếm chính xác (child) + context đủ (parent) → giảm lost-in-middle.
- ❌ Cần hierarchical chunking ở Indexing stage.

---
**🔴 `sentence_window` — Sentence Window Retrieval**

Retrieve câu hit, nhưng trả về window ±N câu xung quanh để có context.

- ✅ Tốt cho văn xuôi, narrative, sách — context tự nhiên.
- 📐 `window_size` = số câu mỗi bên (thường 2–3).
""")

    with st.expander("⚙️ Post-retrieval — reranking & filtering"):
        st.markdown("""**Reranker:**

| Reranker | Khi nào dùng |
|----------|-------------|
| `none` | Retrieval đã tốt, latency thấp |
| `cross_encoder` ⭐ | **Tốt nhất VI, không cần GPU** — BAAI/bge-reranker-v2-m3 |
| `cohere` | Best API quality, 100+ ngôn ngữ |
| `llm` | Không có reranker model |

**Context ordering `sandwich` ⭐:** Most relevant ở đầu và cuối prompt — giảm lost-in-the-middle.

**Pipeline order:** MetadataFilter → RedundancyFilter → Reranker → LLMFilter → MMRFilter → Compressor → Orderer
""")
        if st.checkbox("🔍 Xem giải thích chi tiết — Post-retrieval", key="detail_post_gen"):
            st.markdown("""
---
**Post-retrieval là gì?**

Sau khi Retrieval tìm được K chunks (thường 10–20), bước này lọc và sắp xếp lại trước khi đưa vào LLM.
Mục tiêu: giảm nhiễu, đưa thông tin quan trọng lên trước, tránh LLM bị "phân tâm" bởi chunk không liên quan.

---
**🟢 Reranker — Xếp hạng lại kết quả**

Retrieval (vector search) tìm theo "gần về nghĩa" nhưng không đánh giá chính xác mức độ liên quan với câu hỏi cụ thể.
Reranker dùng mô hình phức tạp hơn để chấm điểm lại từng chunk:

**`cross_encoder` ⭐** — Mô hình chạy local, xử lý cả query + chunk cùng lúc (không phải từng cái riêng như embedding). Chậm hơn nhưng chính xác hơn hẳn.
- Dùng `BAAI/bge-reranker-v2-m3` — hỗ trợ tiếng Việt tốt, không cần GPU.
- Giảm latency: chỉ rerank Top-K nhỏ (5–10) thay vì toàn bộ corpus.

**`cohere`** — API của Cohere, hỗ trợ 100+ ngôn ngữ. Tiện khi không muốn cài model local.

**`llm`** — Dùng LLM để chấm điểm. Chính xác nhất nhưng tốn nhiều LLM call nhất.

---
**🔵 Các filter khác trong pipeline**

Pipeline post-retrieval chạy theo thứ tự: `MetadataFilter → RedundancyFilter → Reranker → LLMFilter → MMRFilter → Compressor → Orderer`

- **MetadataFilter**: Lọc theo metadata (ví dụ: chỉ giữ chunks từ tài liệu năm 2024).
- **RedundancyFilter**: Loại bỏ các chunks gần giống nhau (cosine similarity > threshold) — tránh lặp thông tin.
- **LLMFilter**: LLM đánh giá từng chunk có thực sự liên quan không — chất lượng cao nhất, tốn LLM call.
- **MMRFilter** (Maximal Marginal Relevance): Cân bằng giữa relevance và diversity — tránh tất cả chunks nói cùng 1 điều.
- **Compressor**: Rút gọn nội dung từng chunk, chỉ giữ phần liên quan đến câu hỏi.
- **Orderer**: Sắp xếp thứ tự chunks trong prompt.

---
**🟡 Context Ordering — Sandwich**

Nghiên cứu cho thấy LLM có xu hướng "quên" thông tin ở giữa context dài (*lost-in-the-middle*).
Strategy `sandwich` đặt chunk quan trọng nhất ở **đầu và cuối** prompt, chunk ít quan trọng ở giữa.

- ✅ Cải thiện chất lượng trả lời rõ rệt khi có nhiều chunks.
- ✅ Không tốn thêm LLM call.
""")

    with st.expander("📝 Prompt — template nào phù hợp với tôi?"):
        st.markdown("""| Template | Output | Khi nào dùng |
|----------|--------|-------------|
| `citation` ⭐ | Text + [NGUỒN N] | Production, cần verify fact |
| `basic` | Plain text | Prototype nhanh |
| `conversational` | Text + lịch sử | Chatbot, follow-up questions |
| `structured` | JSON (claims+sources+confidence) | Downstream code cần parse |

**Ngôn ngữ prompt:** `vi` = instruction tiếng Việt · `both` = an toàn cho corpus song ngữ.
""")

    with st.expander("🤖 Generation — LLM nào phù hợp với tôi?"):
        st.markdown("""| Tình huống | Provider | Model | Ghi chú |
|-----------|---------|-------|---------|
| 🏆 Chất lượng + Tiết kiệm | OpenAI | `gpt-4.1-mini` | $0.40/$1.60 per 1M |
| 💰 Rẻ nhất API | OpenAI | `gpt-4o-mini` | $0.15/$0.60 per 1M |
| 🇻🇳 VI tốt nhất (API) | Anthropic | `claude-haiku-4-5` | $0.80/$4 per 1M |
| 🆓 Miễn phí | Google | `gemini-2.0-flash` | Free tier |
| 🔒 Offline / Privacy | Ollama | `qwen2.5:7b` | ~4.7 GB RAM |
| 🪶 Máy yếu, offline | Ollama | `llama3.2:3b` | ~2 GB RAM |

**Temperature = 0.0** ⭐ cho RAG — deterministic, ít hallucinate nhất.
""")

    st.markdown("---")

    # ── Guard: nhắc Apply nếu chưa confirm ───────────────────────────────────
    _cfg_ready = st.session_state.get("_gen_cfg_confirmed", False)
    if not _cfg_ready:
        st.warning(
            "⚙️ Chưa áp dụng cấu hình — vui lòng kiểm tra các tuỳ chọn ở **sidebar trái** "
            "rồi ấn **✅ Áp dụng cấu hình** trước khi đặt câu hỏi.",
            icon="⚠️",
        )

    _query_input = st.text_input(
        "Câu hỏi",
        placeholder="Ví dụ: Kết quả thực nghiệm của phương pháp hybrid retrieval là gì?",
        key="query_pipeline_input",
        disabled=not _cfg_ready,
    )
    _run_query = st.button(
        "▶️ Run",
        key="btn_run_query",
        type="primary",
        disabled=not _cfg_ready,
    )

    if _run_query and _query_input.strip():
        # Hoàn tất kết nối VDB nếu đang ở trạng thái autoload pending
        _resolve_autoload_pending()
        # Dùng snapshot đã được confirm thay vì đọc thẳng session_state live
        _snap       = st.session_state.get("_gen_cfg_snapshot", {})
        _prompt_cfg = _snap.get("prompt", st.session_state.get("query_prompt_cfg", {"template": "citation", "language": "both"}))
        # Nếu user đã chỉnh sửa prompt trong preview → gắn override vào cfg
        if st.session_state.get("_prompt_override"):
            _prompt_cfg = {**_prompt_cfg, "prompt_override": st.session_state["_prompt_override"]}
        render_query_pipeline_results(
            query      = _query_input.strip(),
            pre_cfg    = _snap.get("pre",  st.session_state.get("query_pre_cfg",  {"transformations": ["none"]})),
            ret_cfg    = _snap.get("ret",  st.session_state.get("query_ret_cfg",  {"strategy": "dense", "top_k": 10})),
            post_cfg   = _snap.get("post", st.session_state.get("query_post_cfg", {"reranker": "none", "top_n": 5})),
            vdb_result = st.session_state.get("vdb_result", {}),
            emb_cfg    = st.session_state.get("emb_cfg_used",   {}),
            prompt_cfg = _prompt_cfg,
            gen_cfg    = _snap.get("gen",  st.session_state.get("query_gen_cfg",  {
                "provider": "openai", "model_name": "gpt-4.1-mini",
                "temperature": 0.0, "max_tokens": 2048, "streaming": True,
            })),
            history = st.session_state.get("_query_history")
                if _prompt_cfg.get("template") == "conversational"
                else None,
        )
    elif _cfg_ready and not _query_input.strip():
        st.info("Nhập câu hỏi ở trên rồi nhấn **▶️ Run**.")





