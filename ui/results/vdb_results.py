import os, sys, re, importlib, importlib.util, inspect, tempfile
from pathlib import Path
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
try:
    from dotenv import load_dotenv as _ld, dotenv_values as _dv
    _ld(override=True); _ENV = _dv()
except ImportError:
    _ENV = {}
import streamlit as st
from core.constants import EMBEDDING_PROVIDER_META, VECTOR_DB_PROVIDER_META

from core.cache_helpers import _embedder_kwargs_from_cfg

def render_vector_db_results(vdb_result: dict, vdb_cfg: dict):
    """
    Hiển thị kết quả sau khi đã index vào vector DB.
    """
    provider = vdb_cfg.get("provider", "unknown")
    meta     = VECTOR_DB_PROVIDER_META.get(provider, {})

    # ── Summary metrics ─────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Provider",    f"{meta.get('icon','')} {meta.get('label', provider)}")
    col2.metric("Vectors",     f"{vdb_result.get('n_vectors', 0):,}")
    col3.metric("Collection",  vdb_result.get("collection_name", "—"))
    col4.metric("Status",      "✅ Loaded" if vdb_result.get("loaded_from_existing") else "✅ Indexed")

    st.markdown("---")

    # ── Provider info ───────────────────────────────────────────────────────
    info_col, detail_col = st.columns([1, 1])
    with info_col:
        st.markdown("**📋 Thông tin Provider**")
        st.markdown(f"- **Tier:** {meta.get('tier_icon','')} {meta.get('tier','')}")
        st.markdown(f"- **Mode:** {meta.get('mode','')}")
        st.markdown(f"- **Scale:** {meta.get('scale','')}")
        st.markdown(f"- **Hybrid search:** {'✅' if meta.get('hybrid') else '❌'}")
        st.markdown(f"- **Filtering:** {meta.get('filtering','')}")

    with detail_col:
        st.markdown("**⚙️ Cấu hình đã dùng**")
        ignore = {"skip", "provider"}
        for k, v in vdb_cfg.items():
            if k not in ignore and v is not None and v != "":
                # Mask sensitive values
                display_v = "••••••" if any(s in k for s in ("key", "password", "token", "conn")) and v else v
                st.markdown(f"- **{k}:** `{display_v}`")

    # ── Quick test search ───────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("**🔎 Test search nhanh**")
    test_query = st.text_input(
        "Nhập câu hỏi thử:",
        placeholder="Ví dụ: What is the main topic of this document?",
        key="vdb_test_query",
    )
    top_k = st.slider("Top-K", min_value=1, max_value=10, value=3, key="vdb_test_k")

    if st.button("🔍 Search", key="vdb_test_btn") and test_query:
        vector_store = vdb_result.get("vector_store") or vdb_result.get("store")
        if vector_store is not None:
            try:
                with st.spinner("Đang search..."):
                    results = vector_store.similarity_search(test_query, k=top_k)
                st.success(f"Tìm thấy {len(results)} kết quả:")
                for i, doc in enumerate(results, 1):
                    src = doc.metadata.get("source", "")
                    src_label = f"  ·  `{src}`" if src else ""
                    with st.expander(f"**Kết quả #{i}**{src_label}", expanded=(i == 1)):
                        st.text(doc.page_content[:800] + ("…" if len(doc.page_content) > 800 else ""))
                        if doc.metadata:
                            st.json({k: v for k, v in doc.metadata.items() if k != "source"})
            except Exception as e:
                st.error(f"❌ Search thất bại: {e}")
        else:
            st.warning("Vector store chưa khả dụng.")


# ─── UI: Pipeline suggestion cards ───────────────────────────────────────────

def render_pipeline_suggestions(suggestions: list[dict]):
    """Hiển thị tối đa 3 gợi ý pipeline. Click 'Áp dụng' để tự điền sidebar."""
    if not suggestions:
        return

    RANK_META = {
        1: ("#27ae60", "#27ae60",  "🏆 Tốt nhất"),
        2: ("#2980b9", "#2980b9",  "⚡ Thay thế tốt"),
        3: ("#e67e22", "#e67e22",  "💡 Phương án khác"),
    }

    st.subheader("💡 Gợi ý cấu hình Indexing cho input của bạn")
    st.caption(
        "Dựa trên loại file đã phát hiện — chỉ bao gồm các bước **Stage 1: Indexing** "
        "(Loader, Chunking, Embedding, Vector DB). "
        "Cấu hình Stage 2 (Retrieval, Prompt, Generation) ở trang **💬 Generation**. "
        "Click **✅ Áp dụng** để tự điền sidebar, rồi ấn **▶️ Process**."
    )

    cols = st.columns(len(suggestions))
    for col, sug in zip(cols, suggestions):
        rank              = sug["rank"]
        color, _, rlbl    = RANK_META.get(rank, ("#95a5a6", "#95a5a6", f"#{rank}"))
        pdf_strat         = sug["pdf_strategy"]
        chunk_strat       = sug["chunking_strategy"]
        fmt_type          = sug.get("chunking_extra", {}).get("format_type", "")
        fmt_badge         = f" ({fmt_type})" if fmt_type else ""

        emb_provider      = sug.get("emb_provider", "")
        emb_model         = sug.get("emb_model", "")
        emb_sparse        = sug.get("emb_sparse", False)
        emb_reason        = sug.get("emb_reason", "")
        vdb_provider      = sug.get("vdb_provider", "")
        vdb_reason        = sug.get("vdb_reason", "")

        ret_strategy   = sug.get("ret_strategy", "")
        pre_transforms = sug.get("pre_transforms", "none")
        post_reranker  = sug.get("post_reranker", "none")
        ret_reason     = sug.get("ret_reason", "")
        pre_reason     = sug.get("pre_reason", "")
        post_reason    = sug.get("post_reason", "")

        emb_meta          = EMBEDDING_PROVIDER_META.get(emb_provider, {})
        emb_icon          = emb_meta.get("icon", "🧮")
        emb_short         = emb_model.split("/")[-1]
        sparse_badge      = " + BM25" if emb_sparse else ""

        vdb_meta          = VECTOR_DB_PROVIDER_META.get(vdb_provider, {})
        vdb_icon          = vdb_meta.get("icon", "🗃️")
        vdb_label         = vdb_meta.get("label", vdb_provider)

        # ── Inline markdown → HTML ────────────────────────────────────────────
        def _md(text: str) -> str:
            text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)
            text = re.sub(r'`([^`]+)`',
                          r'<code style="background:var(--color-background-tertiary);'
                          r'padding:1px 5px;border-radius:4px;font-size:0.82em;'
                          r'font-family:var(--font-mono);">\1</code>', text)
            return text

        # ── Shared style constants ────────────────────────────────────────────
        _B = (
            "display:inline-block;padding:4px 10px;border-radius:6px;"
            "font-size:0.78rem;font-family:var(--font-mono);font-weight:500;"
            "margin:0 4px 4px 0;white-space:nowrap;"
        )
        _NOTE = (
            "margin-top:10px;padding:9px 11px;"
            "border-left:3px solid {c};border-radius:0 6px 6px 0;"
            "background:{bg};font-size:0.82rem;"
            "color:var(--color-text-primary);line-height:1.6;"
        )

        # ── Tech badges ───────────────────────────────────────────────────────
        s_ret  = (f'<span style="{_B}background:rgba(41,128,185,0.12);color:#2471a3;'
                  f'border:1px solid rgba(41,128,185,0.35);">🔎 {ret_strategy}</span>'
                  ) if ret_strategy else ""
        s_pre  = (f'<span style="{_B}background:rgba(142,68,173,0.10);color:#7d3c98;'
                  f'border:1px solid rgba(142,68,173,0.30);">🔄 {pre_transforms}</span>'
                  ) if pre_transforms and pre_transforms != "none" else ""
        s_post = (f'<span style="{_B}background:rgba(192,57,43,0.10);color:#c0392b;'
                  f'border:1px solid rgba(192,57,43,0.30);">⚙️ {post_reranker}</span>'
                  ) if post_reranker and post_reranker != "none" else ""
        s_loader  = (f'<span style="{_B}background:rgba(26,107,58,0.12);color:#1a6b3a;'
                     f'border:1px solid rgba(26,107,58,0.35);">📄 {pdf_strat}</span>')
        s_chunker = (f'<span style="{_B}background:rgba(180,83,9,0.12);color:#b45309;'
                     f'border:1px solid rgba(180,83,9,0.35);">✂️ {chunk_strat}{fmt_badge}</span>')
        s_emb = (f'<span style="{_B}background:rgba(124,58,237,0.12);color:#7c3aed;'
                 f'border:1px solid rgba(124,58,237,0.35);">{emb_icon} {emb_short}{sparse_badge}</span>'
                 ) if emb_provider else ""
        s_vdb = (f'<span style="{_B}background:rgba(11,110,79,0.12);color:#0b6e4f;'
                 f'border:1px solid rgba(11,110,79,0.35);">{vdb_icon} {vdb_label}</span>'
                 ) if vdb_provider else ""

        # ── Note boxes ────────────────────────────────────────────────────────
        loader_reason     = sug.get("loader_reason", sug.get("reason", ""))
        chunking_reason   = sug.get("chunking_reason", "")

        # ── Note boxes ────────────────────────────────────────────────────────
        _loader_note = (
            f'<div style="{_NOTE.format(c="#1a6b3a", bg="rgba(26,107,58,0.06)")}">'
            f'<span style="font-weight:600;color:#1a6b3a;">📄 Loader</span>'
            f'<div style="margin-top:3px;color:var(--color-text-secondary);">{_md(loader_reason)}</div>'
            f'</div>'
        ) if loader_reason else ""

        _chunking_note = (
            f'<div style="{_NOTE.format(c="#b45309", bg="rgba(180,83,9,0.06)")}">'
            f'<span style="font-weight:600;color:#b45309;">✂️ Chunking</span>'
            f'<div style="margin-top:3px;color:var(--color-text-secondary);">{_md(chunking_reason)}</div>'
            f'</div>'
        ) if chunking_reason else ""

        _emb_note = (
            f'<div style="{_NOTE.format(c="#7c3aed", bg="rgba(124,58,237,0.06)")}">'
            f'<span style="font-weight:600;color:#7c3aed;">🧮 Embedding</span>'
            f'<div style="margin-top:3px;color:var(--color-text-secondary);">{_md(emb_reason)}</div>'
            f'</div>'
        ) if emb_reason else ""

        _vdb_note = (
            f'<div style="{_NOTE.format(c="#0b6e4f", bg="rgba(11,110,79,0.06)")}">'
            f'<span style="font-weight:600;color:#0b6e4f;">🗃️ Vector DB</span>'
            f'<div style="margin-top:3px;color:var(--color-text-secondary);">{_md(vdb_reason)}</div>'
            f'</div>'
        ) if vdb_reason else ""



        def _make_note(color, bg, label, body):
            if not body:
                return ""
            style = ("margin-top:8px;padding:8px 10px;"
                     f"border-left:3px solid {color};"
                     "border-radius:0 6px 6px 0;"
                     f"font-size:0.82rem;background:{bg};")
            return (
                f'<div style="{style}">'
                f'<span style="font-weight:600;color:{color};">{label}</span>'
                f'<div style="margin-top:3px;color:var(--color-text-secondary);">{_md(body)}</div>'
                f'</div>'
            )

        _ret_note  = _make_note("#2471a3", "rgba(41,128,185,0.06)",  "Retrieval",                ret_reason)
        _pre_note  = _make_note("#7d3c98", "rgba(142,68,173,0.05)", "Pre-retrieval (optional)",  pre_reason)
        _post_note = _make_note("#c0392b", "rgba(192,57,43,0.05)",  "Post-retrieval (optional)", post_reason)


        # ── Card HTML ─────────────────────────────────────────────────────────
        card_html = (
            f'<div style="border:1.5px solid {color}40;border-top:3px solid {color};'
            f'border-radius:10px;padding:16px 18px;'
            f'background:var(--color-background-secondary);">'

            # rank pill
            f'<span style="display:inline-block;background:{color}18;color:{color};'
            f'border:1px solid {color}40;padding:2px 10px;border-radius:20px;'
            f'font-size:0.72rem;font-weight:700;letter-spacing:0.3px;margin-bottom:10px;">'
            f'{rlbl}</span>'

            # title
            f'<div style="font-weight:600;font-size:0.92rem;line-height:1.45;'
            f'color:var(--color-text-primary);margin-bottom:10px;">{sug["title"]}</div>'

            # tech badges — Indexing stage only (Stage 2 steps not shown here)
            f'<div style="display:flex;flex-wrap:wrap;gap:4px;margin-bottom:12px;">'
            f'{s_loader}{s_chunker}{s_emb}{s_vdb}</div>'

            # Indexing stage note boxes only
            f'{_loader_note}{_chunking_note}{_emb_note}{_vdb_note}'

            f'</div>'
        )

        with col:
            st.markdown(card_html, unsafe_allow_html=True)
            st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
            if st.button(
                "✅ Áp dụng cấu hình này",
                key=f"apply_sug_{rank}",
                width="stretch",
                type="secondary",
            ):
                st.session_state["_want_pdf_strategy"]      = pdf_strat
                st.session_state["_want_chunking_strategy"] = chunk_strat
                if fmt_type:
                    st.session_state["_want_format_type"]   = fmt_type
                if emb_provider:
                    st.session_state["_want_emb_provider"]  = emb_provider
                    st.session_state["_want_emb_model"]     = emb_model
                if emb_sparse:
                    st.session_state["_want_emb_sparse"]        = True
                    st.session_state["_want_emb_sparse_method"] = "bm25"
                else:
                    st.session_state["_want_emb_sparse"]        = False
                if vdb_provider:
                    st.session_state["_want_vdb_provider"]  = vdb_provider
                if ret_strategy:
                    st.session_state["_want_ret_strategy"]  = ret_strategy
                if pre_transforms and pre_transforms != "none":
                    st.session_state["_want_pre_transforms"] = [pre_transforms]
                if post_reranker and post_reranker != "none":
                    st.session_state["_want_post_reranker"] = post_reranker
                st.rerun()

    st.markdown("")

# ─── UI: Query pipeline results ───────────────────────────────────────────────



def render_query_pipeline_results(
    query:      str,
    pre_cfg:    dict,
    ret_cfg:    dict,
    post_cfg:   dict,
    vdb_result: dict,
    emb_cfg:    dict,
    prompt_cfg: dict | None = None,
    gen_cfg:    dict | None = None,
    history:    list[dict] | None = None,
):
    """
    Chạy toàn bộ query pipeline (pre → retrieval → post → prompt → generation)
    và hiển thị kết quả từng bước để người dùng debug.
    """
    prompt_cfg = prompt_cfg or {"template": "citation", "language": "both"}
    gen_cfg    = gen_cfg    or {"provider": "openai", "model_name": "gpt-4.1-mini",
                                "temperature": 0.0, "max_tokens": 2048, "streaming": True}
    # "vector_store" key: set by _load_pipeline_into_session (load từ cache)
    # "store" key: set by fresh indexing run
    vector_store = vdb_result.get("vector_store") or vdb_result.get("store")
    if vector_store is None:
        st.warning("Vector store chưa khả dụng — chạy Process trước.")
        return

    # ── Bước 1: Pre-retrieval ──────────────────────────────────────────────
    st.markdown("#### 🔄 Bước 7 — Pre-retrieval")
    pre_transforms = pre_cfg.get("transformations", ["none"])

    with st.spinner("Đang chạy pre-retrieval..."):
        try:
            from pre_retrieval import build_pipeline
            pre_pipeline = build_pipeline(
                transformations=pre_transforms,
                llm_model=pre_cfg.get("llm_model", "gpt-4.1-mini"),
                llm_provider=pre_cfg.get("llm_provider", "openai"),
            )
            transform_result = pre_pipeline.transform(query)
        except Exception as e:
            st.error(f"Pre-retrieval lỗi: {e}")
            return

    if pre_transforms == ["none"] or pre_transforms == ["none"]:
        st.caption("_Passthrough — query không được biến đổi._")
    else:
        for i, q in enumerate(transform_result.queries, 1):
            label = "Query gốc" if q == query else f"Query #{i}"
            st.markdown(
                f'<div style="padding:6px 12px;margin:4px 0;background:var(--color-background-secondary);'
                f'border-left:3px solid #7d3c98;border-radius:0 6px 6px 0;font-size:0.87rem;">'
                f'<b style="color:#7d3c98;">{label}:</b> {q}</div>',
                unsafe_allow_html=True,
            )
        if transform_result.metadata_filter:
            st.json({"metadata_filter": transform_result.metadata_filter})
        if transform_result.intent:
            st.caption(f"Intent: `{transform_result.intent}`")
        if transform_result.retrieval_path:
            st.caption(f"Route: `{transform_result.retrieval_path}`")

    st.markdown("---")

    # ── Bước 2: Retrieval ──────────────────────────────────────────────────
    st.markdown("#### 🔎 Bước 8 — Retrieval")
    ret_strategy = ret_cfg.get("strategy", "dense")
    top_k        = ret_cfg.get("top_k", 10)

    with st.spinner(f"Đang retrieval ({ret_strategy}, top-{top_k})..."):
        try:
            from retrieval import get_retriever
            chunks = st.session_state.get("chunks", [])
            # Build retriever kwargs — only pass params relevant to strategy
            _effective_strategy = ret_strategy
            _retriever_kwargs = {"top_k": top_k}

            if _effective_strategy in ("hybrid", "sparse"):
                if chunks:
                    _retriever_kwargs["fusion_method"] = ret_cfg.get("fusion_method", "rrf")
                    _retriever_kwargs["alpha"]         = ret_cfg.get("hybrid_alpha", 0.5)
                else:
                    st.warning(
                        f"⚠️ Strategy `{ret_strategy}` cần BM25 index nhưng chunks chưa được load. "
                        "Tự động chuyển sang `dense`."
                    )
                    _effective_strategy = "dense"

            retriever = get_retriever(
                strategy=_effective_strategy,
                vector_store=vector_store,
                documents=chunks,
                **_retriever_kwargs,
            )
            retrieved_docs = retriever.retrieve(transform_result)
        except Exception as e:
            st.error(f"Retrieval lỗi: {e}")
            return

    st.caption(f"Tìm được **{len(retrieved_docs)}** documents với strategy `{ret_strategy}`")
    for i, doc in enumerate(retrieved_docs, 1):
        src   = doc.metadata.get("source", "")
        page  = doc.metadata.get("page", "")
        score = (doc.metadata.get("rrf_score")
                 or doc.metadata.get("relevance_score")
                 or doc.metadata.get("bm25_score", 0))
        score_str = f" · score={score:.3f}" if score else ""
        label = f"#{i}  {src}" + (f" p.{page}" if page else "") + score_str
        with st.expander(label, expanded=(i <= 3)):
            st.text(doc.page_content[:600] + ("…" if len(doc.page_content) > 600 else ""))

    st.markdown("---")

    # ── Bước 3: Post-retrieval ─────────────────────────────────────────────
    st.markdown("#### ⚙️ Bước 9 — Post-retrieval")
    reranker = post_cfg.get("reranker", "none")

    if reranker == "none" and not post_cfg.get("apply_redundancy") and not post_cfg.get("apply_mmr"):
        st.caption("_Post-retrieval bị tắt — kết quả giữ nguyên từ Retrieval._")
        final_docs = retrieved_docs
    else:
        with st.spinner("Đang xử lý post-retrieval..."):
            try:
                from post_retrieval import build_pipeline as build_post
                post_pipeline = build_post(
                    reranker=reranker,
                    top_n=post_cfg.get("top_n", 5),
                    cross_encoder_model=post_cfg.get("cross_encoder_model", "BAAI/bge-reranker-v2-m3"),
                    apply_redundancy=post_cfg.get("apply_redundancy", True),
                    apply_mmr=post_cfg.get("apply_mmr", False),
                    apply_compression=post_cfg.get("apply_compression", False),
                    apply_llm_filter=post_cfg.get("apply_llm_filter", False),
                    context_ordering=post_cfg.get("context_ordering", "sandwich"),
                )
                final_docs = post_pipeline.process(query, retrieved_docs)
            except Exception as e:
                st.error(f"Post-retrieval lỗi: {e}")
                return

        st.caption(
            f"Sau post-retrieval: **{len(retrieved_docs)}** → **{len(final_docs)}** documents"
            + (f" (reranker: `{reranker}`)" if reranker != "none" else "")
        )

    st.markdown("**📋 Kết quả cuối (đưa vào LLM):**")
    for i, doc in enumerate(final_docs, 1):
        src  = doc.metadata.get("source", "")
        page = doc.metadata.get("page", "")
        rs   = (doc.metadata.get("rerank_score")
                or doc.metadata.get("rrf_score")
                or doc.metadata.get("relevance_score", 0))
        rs_str = f" · rerank={rs:.3f}" if rs else ""
        label  = f"#{i}  {src}" + (f" p.{page}" if page else "") + rs_str
        with st.expander(label, expanded=(i <= 2)):
            st.text(doc.page_content[:600] + ("…" if len(doc.page_content) > 600 else ""))

    st.markdown("---")

    # ── Bước 10: Prompt ───────────────────────────────────────────────────────
    st.markdown("#### 📝 Bước 10 — Prompt")

    try:
        from prompt import get_prompt_builder
        prompt_builder = get_prompt_builder(
            template           = prompt_cfg.get("template", "citation"),
            language           = prompt_cfg.get("language", "both"),
            max_context_chars  = prompt_cfg.get("max_context_chars", 0),
            system_instruction = prompt_cfg.get("system_instruction", ""),
            **({"max_history_turns": prompt_cfg["max_history_turns"]}
               if prompt_cfg.get("template") == "conversational" else {}),
            **({"validate_citations": True}
               if prompt_cfg.get("template") == "citation" else {}),
        )
        prompt_result = prompt_builder.build(
            query   = query,
            docs    = final_docs,
            history = history or [],
        )
        # Nếu user đã chỉnh sửa prompt trong preview → dùng bản tuỳ chỉnh
        _override = prompt_cfg.get("prompt_override")
        if _override:
            _new_msgs = []
            for _m in prompt_result.messages:
                if _m["role"] == "system" and _override.get("system"):
                    _new_msgs.append({"role": "system", "content": _override["system"]})
                elif _m["role"] == "user" and _override.get("user"):
                    _new_msgs.append({"role": "user", "content": _override["user"]})
                else:
                    _new_msgs.append(_m)
            prompt_result.messages = _new_msgs
    except Exception as e:
        st.error(f"Prompt builder lỗi: {e}")
        return

    template_name = prompt_cfg.get("template", "citation")
    st.caption(
        f"Template: `{template_name}` · Ngôn ngữ: `{prompt_cfg.get('language','both')}` · "
        f"{prompt_result.n_sources} nguồn trong context"
    )

    with st.expander("👁️ Xem prompt đầy đủ gửi lên LLM", expanded=False):
        # Hiển thị từng message riêng biệt
        for msg in prompt_result.messages:
            role_label = {"system": "⚙️ System", "user": "👤 User", "assistant": "🤖 Assistant"}.get(
                msg["role"], msg["role"].capitalize()
            )
            st.markdown(f"**{role_label}**")
            st.text_area(
                label=f"msg_{msg['role']}",
                value=msg["content"],
                height=min(300, max(80, msg["content"].count("\n") * 20 + 80)),
                disabled=True,
                label_visibility="collapsed",
                key=f"prompt_msg_{msg['role']}_{id(msg)}",
            )

    st.markdown("---")

    # ── Bước 11: Generation ───────────────────────────────────────────────────
    st.markdown("#### 🤖 Bước 11 — Generation")

    provider   = gen_cfg.get("provider",   "openai")
    model_name = gen_cfg.get("model_name", "gpt-4.1-mini")
    streaming  = gen_cfg.get("streaming",  True)

    st.caption(f"Provider: `{provider}` · Model: `{model_name}` · Streaming: `{streaming}`")

    try:
        from generation import get_generator
        generator = get_generator(
            provider   = provider,
            model_name = model_name,
            temperature = gen_cfg.get("temperature", 0.0),
            max_tokens  = gen_cfg.get("max_tokens",  2048),
            streaming   = streaming,
            **({"base_url":   gen_cfg.get("base_url", "http://localhost:11434"),
                "auto_pull":  gen_cfg.get("auto_pull", True)}
               if provider == "ollama" else {}),
        )
    except Exception as e:
        st.error(f"Khởi tạo generator lỗi: {e}")
        return

    st.markdown("**💬 Câu trả lời:**")
    answer_placeholder = st.empty()

    try:
        if streaming:
            # ── Streaming mode ──────────────────────────────────────────────
            full_answer = ""
            with st.spinner(""):
                for chunk in generator.stream(prompt_result):
                    full_answer += chunk
                    answer_placeholder.markdown(full_answer + "▌")
            answer_placeholder.markdown(full_answer)

            # Post-process để lấy citations
            from generation.base import GenerationResult
            from prompt.citation import CitationPromptBuilder
            from prompt.structured_output import StructuredOutputPromptBuilder

            cited: list[int] = []
            structured = None
            if template_name == "citation":
                cited = CitationPromptBuilder.extract_cited_indices(full_answer)
                cited = [i for i in cited if 1 <= i <= prompt_result.n_sources]
            elif template_name == "structured":
                structured = StructuredOutputPromptBuilder.parse_response(full_answer)

            gen_result = GenerationResult(
                answer        = full_answer,
                provider      = provider,
                model_name    = model_name,
                cited_sources = cited,
                structured    = structured,
            )
        else:
            # ── Non-streaming mode ──────────────────────────────────────────
            with st.spinner(f"Đang sinh câu trả lời ({model_name})..."):
                gen_result = generator.generate(prompt_result)
            answer_placeholder.markdown(gen_result.answer)

    except Exception as e:
        st.error(f"Generation lỗi: {e}")
        st.exception(e)
        return

    # ── Metadata: token usage + citations ────────────────────────────────────
    meta_cols = st.columns(4)
    meta_cols[0].metric("Provider",  f"{provider}")
    meta_cols[1].metric("Model",     model_name.split("/")[-1].split(":")[0])
    meta_cols[2].metric("Input tok", f"{gen_result.input_tokens:,}"  if gen_result.input_tokens  else "—")
    meta_cols[3].metric("Output tok", f"{gen_result.output_tokens:,}" if gen_result.output_tokens else "—")

    # ── Citation analysis ─────────────────────────────────────────────────────
    if template_name == "citation" and prompt_result.n_sources > 0:
        st.markdown("")
        if gen_result.cited_sources:
            cited_docs = [
                final_docs[i - 1]
                for i in gen_result.cited_sources
                if i <= len(final_docs)
            ]
            st.markdown(f"**📚 Nguồn được trích dẫn: {gen_result.cited_sources}**")
            for i, idx in enumerate(gen_result.cited_sources):
                if idx <= len(final_docs):
                    doc = final_docs[idx - 1]
                    src = doc.metadata.get("source", "")
                    pg  = doc.metadata.get("page", "")
                    label = f"[NGUỒN {idx}] {src}" + (f" p.{pg}" if pg else "")
                    with st.expander(label, expanded=False):
                        st.text(doc.page_content[:400] + ("…" if len(doc.page_content) > 400 else ""))
        else:
            st.caption("ℹ️ Câu trả lời không trích dẫn nguồn cụ thể nào.")

    # ── Structured output display ─────────────────────────────────────────────
    if template_name == "structured" and gen_result.structured:
        st.markdown("")
        st.markdown("**📊 Structured Output:**")
        parsed = gen_result.structured
        if parsed.get("claims"):
            st.markdown("**Claims:**")
            for claim in parsed["claims"]:
                st.markdown(f"- {claim}")
        if parsed.get("sources"):
            st.markdown(f"**Sources:** {parsed['sources']}")
        if parsed.get("confidence"):
            conf_color = {"high": "🟢", "medium": "🟡", "low": "🔴"}.get(parsed["confidence"], "⚪")
            st.markdown(f"**Confidence:** {conf_color} {parsed['confidence']}")
        if parsed.get("unanswered"):
            st.info(f"**Chưa trả lời được:** {parsed['unanswered']}")

    # Lưu vào session_state để có thể reuse trong conversational mode
    _hist = st.session_state.get("_query_history", [])
    _hist.append({"role": "user",      "content": query})
    _hist.append({"role": "assistant", "content": gen_result.answer})
    st.session_state["_query_history"] = _hist[-20:]  # giữ tối đa 10 lượt


# ─── Helper: render text+images mixed content ─────────────────────────────────


