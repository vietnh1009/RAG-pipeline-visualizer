"""
pipeline/generation_pipeline.py
================================
GenerationPipeline — Stage 2 của RAG pipeline.

Chạy online, mỗi khi người dùng đặt câu hỏi.

Luồng:
    query (string)
        → Pre-retrieval  (biến đổi query — tuỳ chọn)
        → Retrieval      (tìm chunk liên quan từ Vector DB)
        → Post-retrieval (rerank, filter, compress — tuỳ chọn)
        → Prompt         (xây dựng prompt với context)
        → Generation     (LLM sinh câu trả lời)

Có thể dùng trực tiếp từ code mà không cần Streamlit:

    from pipeline.generation_pipeline import GenerationPipeline

    pipe   = GenerationPipeline()
    result = pipe.run(
        query      = "Hybrid retrieval cải thiện bao nhiêu so với dense-only?",
        vdb_result = indexing_result.vdb_result,
        emb_cfg    = embed_cfg,
        pre_cfg    = {"transformations": ["none"]},
        ret_cfg    = {"strategy": "hybrid", "top_k": 15},
        post_cfg   = {"reranker": "cross_encoder", "top_n": 5},
        prompt_cfg = {"template": "citation", "language": "both"},
        gen_cfg    = {"provider": "openai", "model_name": "gpt-4.1-mini",
                      "temperature": 0.0, "max_tokens": 2048, "streaming": False},
    )
    print(result.answer)
    print("Sources:", result.cited_sources)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Iterator

from langchain_core.documents import Document

from prompt.base import PromptResult
from generation.base import GenerationResult

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Kết quả từng bước & toàn bộ pipeline
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class GenerationStepResult:
    """Kết quả một bước trong generation pipeline."""
    step:    str
    success: bool
    meta:    dict        = field(default_factory=dict)
    error:   str | None  = None


@dataclass
class GenerationPipelineResult:
    """
    Kết quả toàn bộ GenerationPipeline.run().

    Attributes
    ----------
    answer        : Câu trả lời đầy đủ từ LLM.
    query_used    : Query thực sự đưa vào retrieval (sau pre-retrieval transform).
    retrieved     : Chunks trả về từ retrieval (trước post-retrieval).
    final_docs    : Chunks sau post-retrieval (đã rerank, filter, order).
    prompt_result : PromptResult đã build (messages, full_prompt, ...).
    gen_result    : GenerationResult (token counts, cited_sources, structured, ...).
    steps         : List kết quả từng bước.
    error         : Mô tả lỗi nếu pipeline thất bại.
    """
    success:       bool
    answer:        str                    = ""
    query_used:    str | list[str]        = ""
    retrieved:     list[Document]         = field(default_factory=list)
    final_docs:    list[Document]         = field(default_factory=list)
    prompt_result: PromptResult | None    = None
    gen_result:    GenerationResult | None = None
    steps:         list[GenerationStepResult] = field(default_factory=list)
    error:         str | None             = None

    @property
    def cited_sources(self) -> list[int]:
        return self.gen_result.cited_sources if self.gen_result else []

    @property
    def input_tokens(self) -> int:
        return self.gen_result.input_tokens if self.gen_result else 0

    @property
    def output_tokens(self) -> int:
        return self.gen_result.output_tokens if self.gen_result else 0


# ──────────────────────────────────────────────────────────────────────────────
# GenerationPipeline
# ──────────────────────────────────────────────────────────────────────────────

class GenerationPipeline:
    """
    Stage 2 — Generation Pipeline (Query-time).

    Stateless: mỗi lần gọi run() là độc lập, không lưu trạng thái giữa các lần.
    Thread-safe: có thể khởi tạo một lần và gọi run() đồng thời.
    """

    # ── API công khai ─────────────────────────────────────────────────────────

    def run(
        self,
        query:      str,
        vdb_result: dict,
        emb_cfg:    dict,
        pre_cfg:    dict | None = None,
        ret_cfg:    dict | None = None,
        post_cfg:   dict | None = None,
        prompt_cfg: dict | None = None,
        gen_cfg:    dict | None = None,
        history:    list[dict] | None = None,
    ) -> GenerationPipelineResult:
        """
        Chạy toàn bộ generation pipeline cho một query.

        Tham số
        -------
        query      : Câu hỏi của người dùng.
        vdb_result : Kết quả từ IndexingPipeline (chứa vector_store).
        emb_cfg    : Embedding config — dùng để embed query.
        pre_cfg    : Pre-retrieval config. Mặc định: {"transformations": ["none"]}.
        ret_cfg    : Retrieval config. Mặc định: {"strategy": "dense", "top_k": 10}.
        post_cfg   : Post-retrieval config. Mặc định: {"reranker": "none", "top_n": 5}.
        prompt_cfg : Prompt config. Mặc định: {"template": "citation", "language": "both"}.
        gen_cfg    : Generation config. Mặc định: OpenAI gpt-4.1-mini.
        history    : Lịch sử hội thoại cho conversational template.

        Trả về
        ------
        GenerationPipelineResult với answer, cited_sources, token usage, ...
        """
        pre_cfg    = pre_cfg    or {"transformations": ["none"]}
        ret_cfg    = ret_cfg    or {"strategy": "dense", "top_k": 10}
        post_cfg   = post_cfg   or {"reranker": "none", "top_n": 5}
        prompt_cfg = prompt_cfg or {"template": "citation", "language": "both"}
        gen_cfg    = gen_cfg    or {
            "provider": "openai", "model_name": "gpt-4.1-mini",
            "temperature": 0.0, "max_tokens": 2048, "streaming": False,
        }

        result = GenerationPipelineResult(success=False)

        # ── Bước 7: Pre-retrieval ─────────────────────────────────────────────
        step_pre, queries = self._run_pre_retrieval(query, pre_cfg)
        result.steps.append(step_pre)
        if not step_pre.success:
            result.error = step_pre.error
            return result
        result.query_used = queries

        # ── Bước 8: Retrieval ─────────────────────────────────────────────────
        step_ret, retrieved = self._run_retrieval(queries, vdb_result, emb_cfg, ret_cfg)
        result.steps.append(step_ret)
        if not step_ret.success:
            result.error = step_ret.error
            return result
        result.retrieved = retrieved

        # ── Bước 9: Post-retrieval ────────────────────────────────────────────
        step_post, final_docs = self._run_post_retrieval(query, retrieved, emb_cfg, post_cfg)
        result.steps.append(step_post)
        if not step_post.success:
            result.error = step_post.error
            return result
        result.final_docs = final_docs

        # ── Bước 10: Prompt ───────────────────────────────────────────────────
        step_prompt, prompt_result = self._run_prompt(query, final_docs, prompt_cfg, history)
        result.steps.append(step_prompt)
        if not step_prompt.success:
            result.error = step_prompt.error
            return result
        result.prompt_result = prompt_result

        # ── Bước 11: Generation ───────────────────────────────────────────────
        step_gen, gen_result = self._run_generation(prompt_result, gen_cfg)
        result.steps.append(step_gen)
        if not step_gen.success:
            result.error = step_gen.error
            return result
        result.gen_result = gen_result
        result.answer     = gen_result.answer
        result.success    = True
        return result

    def stream(
        self,
        query:      str,
        vdb_result: dict,
        emb_cfg:    dict,
        pre_cfg:    dict | None = None,
        ret_cfg:    dict | None = None,
        post_cfg:   dict | None = None,
        prompt_cfg: dict | None = None,
        gen_cfg:    dict | None = None,
        history:    list[dict] | None = None,
    ) -> Iterator[str]:
        """
        Stream câu trả lời token-by-token.

        Chạy pre-retrieval → retrieval → post-retrieval → prompt đồng bộ,
        sau đó stream generation. Dùng trong Streamlit với st.write_stream()
        hoặc vòng lặp for.

        Yields
        ------
        str — từng chunk text nhỏ từ LLM.

        Lưu ý: Sau khi stream xong, gọi thêm run(..., streaming=False) nếu cần
        full GenerationPipelineResult (token counts, citations, ...).
        """
        pre_cfg    = pre_cfg    or {"transformations": ["none"]}
        ret_cfg    = ret_cfg    or {"strategy": "dense", "top_k": 10}
        post_cfg   = post_cfg   or {"reranker": "none", "top_n": 5}
        prompt_cfg = prompt_cfg or {"template": "citation", "language": "both"}
        gen_cfg    = {**(gen_cfg or {}), "streaming": True,
                      "provider":   (gen_cfg or {}).get("provider", "openai"),
                      "model_name": (gen_cfg or {}).get("model_name", "gpt-4.1-mini"),
                      "temperature": (gen_cfg or {}).get("temperature", 0.0),
                      "max_tokens":  (gen_cfg or {}).get("max_tokens", 2048)}

        _, queries    = self._run_pre_retrieval(query, pre_cfg)
        _, retrieved  = self._run_retrieval(queries, vdb_result, emb_cfg, ret_cfg)
        _, final_docs = self._run_post_retrieval(query, retrieved, emb_cfg, post_cfg)
        _, prompt_res = self._run_prompt(query, final_docs, prompt_cfg, history)

        from generation import get_generator
        generator = get_generator(
            provider    = gen_cfg["provider"],
            model_name  = gen_cfg["model_name"],
            temperature = gen_cfg.get("temperature", 0.0),
            max_tokens  = gen_cfg.get("max_tokens", 2048),
            streaming   = True,
            **({"base_url":  gen_cfg.get("base_url", "http://localhost:11434"),
                "auto_pull": gen_cfg.get("auto_pull", True)}
               if gen_cfg["provider"] == "ollama" else {}),
        )
        yield from generator.stream(prompt_res)

    # ── Hàm chạy từng bước ────────────────────────────────────────────────────

    def _run_pre_retrieval(
        self, query: str, pre_cfg: dict,
    ) -> tuple[GenerationStepResult, str | list[str]]:
        transforms = pre_cfg.get("transformations", ["none"])
        if not transforms or transforms == ["none"]:
            return (
                GenerationStepResult("pre_retrieval", True,
                                     meta={"original_query": query, "query_used": query}),
                query,
            )
        try:
            from pre_retrieval import build_pipeline
            pipeline = build_pipeline(
                transformations = transforms,
                llm_provider    = pre_cfg.get("llm_provider", "openai"),
                llm_model       = pre_cfg.get("llm_model", "gpt-4.1-mini"),
                ollama_base_url = pre_cfg.get("ollama_base_url", "http://localhost:11434"),
                multi_query_count   = pre_cfg.get("multi_query_count", 3),
                num_expansions      = pre_cfg.get("num_expansions", 3),
                rewrite_language    = pre_cfg.get("language", "both"),
                step_back_language  = pre_cfg.get("language", "both"),
            )
            queries = pipeline.transform(query)
        except Exception as e:
            logger.exception("Pre-retrieval lỗi")
            return GenerationStepResult("pre_retrieval", False, error=str(e)), query

        return (
            GenerationStepResult("pre_retrieval", True,
                                 meta={"original_query": query, "query_used": queries}),
            queries,
        )

    def _run_retrieval(
        self,
        queries:    str | list[str],
        vdb_result: dict,
        emb_cfg:    dict,
        ret_cfg:    dict,
    ) -> tuple[GenerationStepResult, list[Document]]:
        try:
            # Dùng get_retriever (API linh hoạt) chứ KHÔNG dùng
            # build_retriever_from_config: hàm đó chỉ nhận (cfg, vector_store,
            # documents) và còn đọc cfg["query_pipeline"]["generation"], nên gọi
            # với sparse_index=/embedder_kwargs= sẽ ném TypeError và thiếu
            # section "generation" thì ném KeyError. Đây là đường mà UI đang
            # dùng và đã chạy tốt (xem ui/results/vdb_results.py).
            from retrieval import get_retriever

            vector_store = vdb_result.get("vector_store")
            if vector_store is None:
                return (
                    GenerationStepResult("retrieval", False,
                                         error="Chưa có Vector DB. Hãy chạy Indexing trước."),
                    [],
                )
            # Corpus đầy đủ — bắt buộc cho sparse/hybrid (BM25) và
            # parent_document / sentence_window (mở rộng cửa sổ).
            documents = (vdb_result.get("chunks")
                         or vdb_result.get("documents")
                         or [])

            strategy = ret_cfg.get("strategy", "dense")
            retriever_kwargs = {"top_k": ret_cfg.get("top_k", 10)}

            if strategy in ("sparse", "hybrid", "parent_document", "sentence_window"):
                if documents:
                    if strategy == "hybrid":
                        retriever_kwargs["fusion_method"] = ret_cfg.get("fusion_method", "rrf")
                        retriever_kwargs["alpha"]         = ret_cfg.get("hybrid_alpha", 0.5)
                else:
                    # Không có corpus thì BM25/window không dựng được → lùi về dense
                    # thay vì ném lỗi giữa chừng.
                    logger.warning(
                        "Strategy '%s' cần corpus nhưng vdb_result không có chunks — "
                        "tự động chuyển sang 'dense'.", strategy,
                    )
                    strategy = "dense"

            retriever = get_retriever(
                strategy     = strategy,
                vector_store = vector_store,
                documents    = documents,
                **retriever_kwargs,
            )
            # Retriever của project nhận TransformResult, KHÔNG phải str, và
            # phương thức là .retrieve() chứ không phải .invoke() (xem
            # retrieval/base.py). _run_pre_retrieval có thể trả về str (đường
            # passthrough) hoặc TransformResult (khi có transformation) — chuẩn
            # hoá cả hai về TransformResult tại đây.
            from pre_retrieval.base import TransformResult

            if isinstance(queries, TransformResult):
                transform_result = queries
            elif isinstance(queries, str):
                transform_result = TransformResult(original_query=queries,
                                                   queries=[queries])
            else:                                    # list[str]
                q_list = list(queries)
                transform_result = TransformResult(
                    original_query=q_list[0] if q_list else "", queries=q_list,
                )

            all_docs: list[Document] = []
            seen: set[str] = set()
            for doc in retriever.retrieve(transform_result):
                key = doc.page_content[:120]
                if key not in seen:
                    seen.add(key)
                    all_docs.append(doc)
        except Exception as e:
            logger.exception("Retrieval lỗi")
            return GenerationStepResult("retrieval", False, error=str(e)), []

        return (
            GenerationStepResult("retrieval", True,
                                 meta={"n_retrieved": len(all_docs),
                                       "strategy": ret_cfg.get("strategy", "dense")}),
            all_docs,
        )

    def _run_post_retrieval(
        self,
        query:     str,
        docs:      list[Document],
        emb_cfg:   dict,
        post_cfg:  dict,
    ) -> tuple[GenerationStepResult, list[Document]]:
        reranker = post_cfg.get("reranker", "none")
        if reranker == "none" and not any([
            post_cfg.get("semantic_dedup"),
            post_cfg.get("mmr_diversity"),
            post_cfg.get("compress_context"),
            post_cfg.get("apply_llm_filter"),
        ]):
            # Không có gì cần làm — trả nguyên
            return (
                GenerationStepResult("post_retrieval", True,
                                     meta={"n_input": len(docs), "n_output": len(docs)}),
                docs,
            )
        try:
            # Dùng build_pipeline (nhận kwargs trực tiếp) chứ KHÔNG dùng
            # build_pipeline_from_config: hàm đó còn đọc
            # cfg["query_pipeline"]["generation"] và cfg["data"]["language"] —
            # hai section mà chỗ này không có → KeyError: 'generation'.
            # Hệ quả: mọi post_cfg có reranker/filter đều hỏng ở đường headless.
            from post_retrieval import build_pipeline

            pipeline = build_pipeline(
                reranker             = post_cfg.get("reranker",             "none"),
                top_n                = post_cfg.get("top_n",                5),
                apply_mmr            = post_cfg.get("apply_mmr",            False),
                mmr_lambda           = post_cfg.get("mmr_lambda",           0.5),
                apply_compression    = post_cfg.get("apply_compression",    False),
                compression_mode     = post_cfg.get("compression_mode",     "extract"),
                apply_llm_filter     = post_cfg.get("apply_llm_filter",     False),
                apply_redundancy     = post_cfg.get("apply_redundancy",     True),
                redundancy_threshold = post_cfg.get("redundancy_threshold", 0.92),
                context_ordering     = post_cfg.get("context_ordering",     "sandwich"),
                metadata_conditions  = post_cfg.get("metadata_conditions",  None),
                cross_encoder_model  = post_cfg.get("cross_encoder_model",  "BAAI/bge-reranker-v2-m3"),
                cohere_rerank_model  = post_cfg.get("cohere_rerank_model",  "rerank-v3.5"),
                # LLM dùng cho llm_reranker / llm_filter / compressor.
                llm_provider         = post_cfg.get("llm_provider",         "openai"),
                llm_model            = post_cfg.get("llm_model",            "gpt-4.1-mini"),
                language             = post_cfg.get("language",             "both"),
            )
            final_docs = pipeline.process(query=query, docs=docs)
        except Exception as e:
            logger.exception("Post-retrieval lỗi")
            return (
                GenerationStepResult("post_retrieval", False, error=str(e)),
                docs,  # fallback: trả về docs gốc
            )

        return (
            GenerationStepResult("post_retrieval", True,
                                 meta={"n_input": len(docs), "n_output": len(final_docs)}),
            final_docs,
        )

    def _run_prompt(
        self,
        query:      str,
        docs:       list[Document],
        prompt_cfg: dict,
        history:    list[dict] | None,
    ) -> tuple[GenerationStepResult, PromptResult]:
        try:
            from prompt import get_prompt_builder
            template = prompt_cfg.get("template", "citation")
            extra_kw: dict = {}
            if template == "conversational":
                extra_kw["max_history_turns"] = prompt_cfg.get("max_history_turns", 5)
            elif template == "citation":
                extra_kw["validate_citations"] = prompt_cfg.get("validate_citations", True)
            elif template == "structured":
                extra_kw["include_confidence"] = prompt_cfg.get("include_confidence", True)

            builder = get_prompt_builder(
                template          = template,
                language          = prompt_cfg.get("language", "both"),
                max_context_chars = prompt_cfg.get("max_context_chars", 0),
                **extra_kw,
            )
            prompt_result = builder.build(query=query, docs=docs, history=history or [])
        except Exception as e:
            logger.exception("Prompt builder lỗi")
            # Fallback: tạo PromptResult tối giản để không crash hoàn toàn
            from prompt.base import PromptResult
            prompt_result = PromptResult(
                messages=[
                    {"role": "system", "content": "Trả lời dựa trên ngữ cảnh."},
                    {"role": "user",   "content": f"Ngữ cảnh:\n{chr(10).join(d.page_content[:500] for d in docs)}\n\nCâu hỏi: {query}"},
                ],
                full_prompt   = query,
                context_docs  = docs,
                n_sources     = len(docs),
                template_name = "fallback",
            )
            return (
                GenerationStepResult("prompt", False, error=str(e),
                                     meta={"prompt_result": prompt_result}),
                prompt_result,
            )

        return (
            GenerationStepResult("prompt", True,
                                 meta={"template": prompt_cfg.get("template", "citation"),
                                       "n_sources": prompt_result.n_sources}),
            prompt_result,
        )

    def _run_generation(
        self,
        prompt_result: PromptResult,
        gen_cfg: dict,
    ) -> tuple[GenerationStepResult, GenerationResult]:
        provider   = gen_cfg.get("provider",   "openai")
        model_name = gen_cfg.get("model_name", "gpt-4.1-mini")
        try:
            from generation import get_generator
            generator = get_generator(
                provider    = provider,
                model_name  = model_name,
                temperature = gen_cfg.get("temperature", 0.0),
                max_tokens  = gen_cfg.get("max_tokens",  2048),
                streaming   = False,  # generate() luôn dùng non-streaming
                **({"base_url":  gen_cfg.get("base_url", "http://localhost:11434"),
                    "auto_pull": gen_cfg.get("auto_pull", True)}
                   if provider == "ollama" else {}),
            )
            gen_result = generator.generate(prompt_result)
        except Exception as e:
            logger.exception("Generation lỗi")
            return (
                GenerationStepResult("generation", False, error=str(e)),
                GenerationResult(answer="", provider=provider, model_name=model_name),
            )

        return (
            GenerationStepResult("generation", True,
                                 meta={"model": model_name, "provider": provider,
                                       "input_tokens": gen_result.input_tokens,
                                       "output_tokens": gen_result.output_tokens}),
            gen_result,
        )


# ──────────────────────────────────────────────────────────────────────────────
# Điểm vào CLI
# ──────────────────────────────────────────────────────────────────────────────

def _cli():
    import argparse, pickle, sys
    from pathlib import Path

    # Console Windows mặc định là cp1252 → in câu trả lời tiếng Việt sẽ ném
    # UnicodeEncodeError. Chỉ ép ở đường CLI, không đụng tiến trình Streamlit.
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    parser = argparse.ArgumentParser(
        description="RAG Generation Pipeline — hỏi đáp với index đã tạo."
    )
    parser.add_argument("--query",      required=True,  help="Câu hỏi")
    parser.add_argument("--vdb-result", required=True,  help="File .pkl chứa vdb_result từ indexing")
    parser.add_argument("--provider",   default="openai")
    parser.add_argument("--model",      default="gpt-4.1-mini")
    parser.add_argument("--verbose",    action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    vdb_path = Path(args.vdb_result)
    if not vdb_path.exists():
        print(f"❌ Không tìm thấy: {vdb_path}", file=sys.stderr)
        sys.exit(1)

    with open(vdb_path, "rb") as f:
        vdb_result = pickle.load(f)

    pipe = GenerationPipeline()
    result = pipe.run(
        query      = args.query,
        vdb_result = vdb_result,
        emb_cfg    = {"provider": "openai", "model_name": "text-embedding-3-small",
                      "enable_sparse": False, "sparse_method": "none"},
        gen_cfg    = {"provider": args.provider, "model_name": args.model,
                      "temperature": 0.0, "max_tokens": 2048, "streaming": False},
    )

    print()
    if result.success:
        print("=" * 60)
        print(result.answer)
        print("=" * 60)
        if result.cited_sources:
            print(f"Nguồn trích dẫn: {result.cited_sources}")
        print(f"Tokens: {result.input_tokens} in / {result.output_tokens} out")
    else:
        print(f"❌ Generation thất bại: {result.error}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    _cli()
