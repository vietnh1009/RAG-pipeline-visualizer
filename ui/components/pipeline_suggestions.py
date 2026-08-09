import os
import sys
import re
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

try:
    from dotenv import load_dotenv as _load_dotenv, dotenv_values as _dotenv_values
    _load_dotenv(override=True)
    _ENV = _dotenv_values()
except ImportError:
    _ENV = {}

import streamlit as st
from core.constants import EMBEDDING_PROVIDER_META, VECTOR_DB_PROVIDER_META

from utils.env import _get_file_profile
from utils.badges import file_type_badge, chunk_type_badge

def get_pipeline_suggestions(source_path: str, local_only: bool = False) -> list[dict]:
    """
    Trả về tối đa 3 gợi ý pipeline (loader + chunking + embedding) dựa trên loại file.
    Mỗi gợi ý: {rank, title, pdf_strategy, chunking_strategy, chunking_extra,
                 emb_provider, emb_model, emb_sparse, emb_reason, reason}
    Khi local_only=True: chỉ gợi ý các công cụ chạy hoàn toàn local.
    """
    p = _get_file_profile(source_path)
    pdf_r, md_r, html_r, code_r = p["pdf_r"], p["md_r"], p["html_r"], p["code_r"]
    suggestions: list[dict] = []

    # ── PDF dominant ──────────────────────────────────────────────────────────
    if pdf_r > 0.4:
        suggestions.append({
            "rank": 1,
            "title": "PDF phức tạp — bảng, layout nhiều cột, cần độ chính xác cao",
            "pdf_strategy": "opendataloader",
            "chunking_strategy": "format_aware",
            "chunking_extra": {"format_type": "markdown"},
            "loader_reason": (
                "**opendataloader** — #1 benchmark (0.90 overall, 0.93 table). "
                "Không cần GPU, Java backend local. Fast mode: 0.05s/trang. "
                "Hybrid mode: 0.43s/trang với AI backend cho bảng phức tạp & scan."
            ),
            "chunking_reason": (
                "**Format-aware (markdown)** cắt theo heading → chunk khớp section gốc. "
                "Lý tưởng với output Markdown từ opendataloader."
            ),
            # Embedding: PDF tiếng Việt thường có nhiều bảng, danh mục, số liệu
            # → cần model mạnh về VI + sparse để match chính xác mã/số/tên riêng
            "emb_provider": "huggingface" if local_only else "cohere",
            "emb_model":    "BAAI/bge-m3" if local_only else "embed-multilingual-v3.0",
            "emb_sparse":   True,
            "emb_reason": (
                "**Cohere embed-multilingual-v3.0** ⭐⭐⭐⭐ VI — top API choice cho tài liệu "
                "PDF tiếng Việt, 1024d, 108 ngôn ngữ. "
                "**BM25** bắt buộc: PDF thường chứa tên riêng, mã số, số liệu cần khớp chính xác."
            ) if not local_only else (
                "**BAAI/bge-m3** ⭐⭐⭐⭐ VI — local, 1024d, 8192 ctx, hỗ trợ "
                "dense+sparse+multivec. Tốt nhất cho PDF đa ngôn ngữ chạy offline. "
                "**BM25** để match chính xác số liệu, mã bảng, tên riêng trong PDF."
            ),
        "vdb_provider": "qdrant" if local_only else "chroma",
        "vdb_reason": (
            "🎯 Qdrant** phù hợp nhất: PDF thường cần filter theo trang, section, metadata tài liệu. ACORN filtering giữ recall cao khi kết hợp dense+BM25."
        ) if local_only else (
            "🎨 **Chroma** — persistent local, filtering native theo metadata (source, page). Zero config, phù hợp prototype offline."
        ),
            "ret_strategy":  "hybrid",
            "pre_transforms": "rewrite",
            "post_reranker":  "cross_encoder",
            "ret_reason":     "**hybrid** (dense+BM25) — bắt buộc khi dùng BM25 embedding. RRF fusion cân bằng semantic recall và exact match. **rewrite** pre-retrieval: chuẩn hoá query trước khi tìm kiếm.",
            "pre_reason":     "**rewrite** — chuẩn hoá lỗi chính tả, đại từ mơ hồ. Tùy chọn: **multi_query** để tăng recall.",
            "post_reason":    "**cross_encoder** reranker (BAAI/bge-reranker-v2-m3) — re-score sau hybrid merge, cải thiện precision đáng kể. Không cần GPU.",
        })
        suggestions.append({
            "rank": 2,
            "title": "PDF phức tạp — text, bảng, ảnh & công thức",
            "pdf_strategy": "marker",
            "chunking_strategy": "format_aware",
            "chunking_extra": {"format_type": "auto"},
            "loader_reason": (
                "**Marker** chuyển PDF sang Markdown chuẩn, giữ nguyên bảng, công thức LaTeX "
                "và caption ảnh. Tốt nhất cho PDF kỹ thuật, báo cáo khoa học."
            ),
            "chunking_reason": (
                "**Format-aware (auto)** cắt theo Markdown heading → chunk khớp đúng "
                "với section gốc. Recall cao nhất khi dùng với loader ra Markdown."
            ),
            # Marker output là Markdown → chunk thường dài (section-level)
            # → cần model có ctx window lớn + hiểu cả công thức/ký hiệu
            "emb_provider": "huggingface",
            "emb_model":    "BAAI/bge-m3",
            "emb_sparse":   True,
            "emb_reason": (
                "**BAAI/bge-m3** — 8192 ctx lý tưởng cho chunk section-level dài từ Marker. "
                "Hỗ trợ cả dense + sparse natively. "
                "**BM25** giúp match công thức, ký hiệu, tên riêng trong tài liệu kỹ thuật."
            ),
        "vdb_provider": "chroma" if local_only else "qdrant",
        "vdb_reason": (
            "🎨 **Chroma** — chunk section-level từ Marker thường lớn. Chroma lưu metadata heading để filter theo cấu trúc tài liệu."
        ) if local_only else (
            "🎯 **Qdrant** — BM25+dense hybrid. ACORN filtering đặc biệt hiệu quả với chunk Markdown section có metadata phong phú từ Marker."
        ),
            "ret_strategy":  "hybrid",
            "pre_transforms": "rewrite",
            "post_reranker":  "cross_encoder",
            "ret_reason":     "**hybrid** (dense+BM25) — bắt buộc khi dùng BM25 embedding. RRF fusion cân bằng semantic recall và exact match. **rewrite** pre-retrieval: chuẩn hoá query trước khi tìm kiếm.",
            "pre_reason":     "**rewrite** — chuẩn hoá lỗi chính tả, đại từ mơ hồ. Tùy chọn: **multi_query** để tăng recall.",
            "post_reason":    "**cross_encoder** reranker (BAAI/bge-reranker-v2-m3) — re-score sau hybrid merge, cải thiện precision đáng kể. Không cần GPU.",
        })
        suggestions.append({
            "rank": 3,
            "title": "PDF có bảng — không cần xử lý ảnh/công thức",
            "pdf_strategy": "pdfplumber",
            "chunking_strategy": "recursive",
            "chunking_extra": {},
            "loader_reason": (
                "**pdfplumber** trích bảng tốt nhất trong nhóm rule-based, "
                "nhanh hơn Marker đáng kể. Không cần Java. Output text + Markdown table sạch."
            ),
            "chunking_reason": (
                "**Recursive** — lựa chọn mặc định an toàn, cân bằng tốc độ và chất lượng. "
                "Cắt ưu tiên: đoạn văn → dòng → câu → ký tự."
            ),
            # Chunk ngắn hơn (recursive) → model nhỏ hơn đủ dùng
            "emb_provider": "fastembed" if local_only else "openai",
            "emb_model":    "intfloat/multilingual-e5-small" if local_only else "text-embedding-3-small",
            "emb_sparse":   False,
            "emb_reason": (
                "**text-embedding-3-small** — cân bằng tốt giữa tốc độ, chi phí ($0.02/1M) "
                "và chất lượng. Chunk ngắn (recursive) không cần ctx window lớn. "
                "Sparse không cần thiết cho PDF text thuần."
            ) if not local_only else (
                "**FastEmbed · multilingual-e5-small** — ONNX, CPU-only, không cần GPU. "
                "Đủ nhanh cho prototype với chunk ngắn từ recursive splitting."
            ),
        "vdb_provider": "faiss" if local_only else "chroma",
        "vdb_reason": (
            "💾 **FAISS** — chunk ngắn (recursive), không cần filtering phức tạp. Nhanh nhất cho prototype local, zero config."
        ) if local_only else (
            "🎨 **Chroma** — chunk ngắn, use-case đơn giản. Persistent, dễ reset khi thay đổi embedding model."
        ),
            "ret_strategy":  "dense",
            "pre_transforms": "none",
            "post_reranker":  "none",
            "ret_reason":     "**dense** — đủ cho corpus text thuần, query ngữ nghĩa. Nếu corpus có nhiều tên riêng hoặc mã số, hãy chuyển sang **hybrid**.",
            "pre_reason":     "Pre-retrieval không bắt buộc cho use-case đơn giản. Tuỳ chọn: **rewrite** để chuẩn hoá query.",
            "post_reason":    "Reranker không bắt buộc cho use-case đơn giản. Tuỳ chọn: **cross_encoder** nếu cần precision cao hơn.",
        })

    # ── Markdown dominant ─────────────────────────────────────────────────────
    elif md_r > 0.4:
        suggestions.append({
            "rank": 1,
            "title": "Markdown có cấu trúc heading (#, ##, ###)",
            "pdf_strategy": "pypdf",
            "chunking_strategy": "format_aware",
            "chunking_extra": {"format_type": "markdown"},
            "loader_reason": (
                "**pymupdf** / **pdfplumber** đọc PDF text sạch và nhanh. "
                "Output plain text đủ cho chunking không cần Markdown."
            ),
            "chunking_reason": (
                "**Format-aware (markdown)** cắt theo heading, mỗi chunk là một section hoàn "
                "chỉnh. Metadata heading giữ lại → filtering theo chủ đề chính xác."
            ),
            # Section-level chunk → cần ctx window lớn + hiểu ngữ nghĩa sâu
            "emb_provider": "huggingface",
            "emb_model":    "BAAI/bge-m3",
            "emb_sparse":   False,
            "emb_reason": (
                "**BAAI/bge-m3** — 8192 ctx phù hợp với section chunk lớn từ format_aware. "
                "Dense capture ngữ nghĩa section tốt hơn sparse. "
                "Không cần BM25 khi chunk đã khớp cấu trúc heading."
            ),
        "vdb_provider": "chroma" if local_only else "chroma",
        "vdb_reason": (
            "🎨 **Chroma** — section chunk từ format_aware có metadata heading đầy đủ. Filtering theo heading level để retrieve đúng section."
        ) if local_only else (
            "🎨 **Chroma** — đủ cho tài liệu Markdown structured. Filtering theo heading metadata; dense-only, không cần hybrid."
        ),
            "ret_strategy":  "dense",
            "pre_transforms": "none",
            "post_reranker":  "none",
            "ret_reason":     "**dense** — đủ cho corpus text thuần, query ngữ nghĩa. Nếu corpus có nhiều tên riêng hoặc mã số, hãy chuyển sang **hybrid**.",
            "pre_reason":     "Pre-retrieval không bắt buộc cho use-case đơn giản. Tuỳ chọn: **rewrite** để chuẩn hoá query.",
            "post_reason":    "Reranker không bắt buộc cho use-case đơn giản. Tuỳ chọn: **cross_encoder** nếu cần precision cao hơn.",
        })
        suggestions.append({
            "rank": 2,
            "title": "Markdown hỗn hợp, ít heading nhất quán",
            "pdf_strategy": "pypdf",
            "chunking_strategy": "recursive",
            "chunking_extra": {},
            "loader_reason": (
                "**pypdf** — nhanh nhất, không cần dep extra. "
                "Phù hợp PDF text thuần, không bảng phức tạp."
            ),
            "chunking_reason": (
                "**Recursive** — an toàn khi Markdown không có heading nhất quán. "
                "Cắt theo đoạn văn → dòng → câu → ký tự."
            ),
            "emb_provider": "fastembed" if local_only else "openai",
            "emb_model":    "intfloat/multilingual-e5-small" if local_only else "text-embedding-3-small",
            "emb_sparse":   False,
            "emb_reason": (
                "**text-embedding-3-small** — default tốt cho chunk paragraph-level. "
                "MRL support: có thể cắt xuống 256d để tiết kiệm ~83% RAM nếu cần scale."
            ) if not local_only else (
                "**FastEmbed · multilingual-e5-small** — đủ tốt cho Markdown đa ngôn ngữ, "
                "không cần GPU, tự download ONNX model."
            ),
        "vdb_provider": "faiss" if local_only else "chroma",
        "vdb_reason": (
            "💾 **FAISS** — chunk paragraph đơn giản, không cần filtering. Nhanh nhất local."
        ) if local_only else (
            "🎨 **Chroma** — chunk ngắn (recursive), use-case đơn giản. Dễ dùng nhất."
        ),
            "ret_strategy":  "dense",
            "pre_transforms": "none",
            "post_reranker":  "none",
            "ret_reason":     "**dense** — đủ cho corpus text thuần, query ngữ nghĩa. Nếu corpus có nhiều tên riêng hoặc mã số, hãy chuyển sang **hybrid**.",
            "pre_reason":     "Pre-retrieval không bắt buộc cho use-case đơn giản. Tuỳ chọn: **rewrite** để chuẩn hoá query.",
            "post_reason":    "Reranker không bắt buộc cho use-case đơn giản. Tuỳ chọn: **cross_encoder** nếu cần precision cao hơn.",
        })
        suggestions.append({
            "rank": 3,
            "title": "Tài liệu Markdown dài — nhiều section, cần context rộng",
            "pdf_strategy": "pypdf",
            "chunking_strategy": "hierarchical",
            "chunking_extra": {"parent_chunk_size": 2000},
            "loader_reason": (
                "**marker** / **docling** — output Markdown có heading rõ ràng, "
                "lý tưởng để hierarchical chunking phát hiện ranh giới section."
            ),
            "chunking_reason": (
                "**Hierarchical** lưu cả parent (section lớn) và child (đoạn nhỏ). "
                "Retrieve child → trả parent cho LLM → đủ context, giảm hallucination."
            ),
            # Hierarchical: embed child chunk nhỏ để search, parent chunk lớn cho LLM
            # → model cần hiểu cả 2 granularity → BM25 giúp match term chính xác trong child
            "emb_provider": "huggingface",
            "emb_model":    "BAAI/bge-m3",
            "emb_sparse":   True,
            "emb_reason": (
                "**BAAI/bge-m3** — ctx 8192 xử lý được cả parent (2000 chars) và child chunk. "
                "**BM25** đặc biệt hiệu quả với child chunk ngắn: exact term matching bù đắp "
                "cho dense khi chunk thiếu ngữ cảnh đầy đủ."
            ),
        "vdb_provider": "chroma" if local_only else "qdrant",
        "vdb_reason": (
            "🎨 **Chroma** — hỗ trợ filtering theo metadata để phân biệt parent/child chunk khi retrieve."
        ) if local_only else (
            "🎯 **Qdrant** — ACORN filtering cần thiết để phân biệt parent chunk vs child chunk. Retrieve child → filter trả về parent đúng cặp."
        ),
            "ret_strategy":  "hybrid",
            "pre_transforms": "rewrite",
            "post_reranker":  "cross_encoder",
            "ret_reason":     "**hybrid** (dense+BM25) — bắt buộc khi dùng BM25 embedding. RRF fusion cân bằng semantic recall và exact match. **rewrite** pre-retrieval: chuẩn hoá query trước khi tìm kiếm.",
            "pre_reason":     "**rewrite** — chuẩn hoá lỗi chính tả, đại từ mơ hồ. Tùy chọn: **multi_query** để tăng recall.",
            "post_reason":    "**cross_encoder** reranker (BAAI/bge-reranker-v2-m3) — re-score sau hybrid merge, cải thiện precision đáng kể. Không cần GPU.",
        })

    # ── Code dominant ─────────────────────────────────────────────────────────
    elif code_r > 0.4:
        suggestions.append({
            "rank": 1,
            "title": "Source code — Python, JS, TS, Java, Go…",
            "pdf_strategy": "pypdf",
            "chunking_strategy": "format_aware",
            "chunking_extra": {"format_type": "code"},
            "loader_reason": (
                "**pypdf** / **pdfplumber** đủ cho tài liệu kỹ thuật text. "
                "Chọn pdfplumber nếu có bảng API reference, pypdf nếu cần tốc độ."
            ),
            "chunking_reason": (
                "**Format-aware (code)** dùng AST-based splitting — ranh giới chunk luôn nằm "
                "giữa top-level definition (class, function). Không bao giờ cắt giữa thân hàm."
            ),
            # Code search: identifier name matching cực kỳ quan trọng → sparse bắt buộc
            "emb_provider": "huggingface" if local_only else "cohere",
            "emb_model":    "BAAI/bge-m3" if local_only else "text-embedding-3-small",
            "emb_sparse":   True,
            "emb_reason": (
                "**text-embedding-3-small** — balances cost and quality for "
                "code search. OpenAI ecosystem, tốt cho documentation + code hỗn hợp."
            ) if not local_only else (
                "**BAAI/bge-m3** local — dense+sparse natively, 8192 ctx. "
                "**BM25** bắt buộc: tên hàm, class cần exact match."
            ),
        "vdb_provider": "chroma" if local_only else "qdrant",
        "vdb_reason": (
            "🎨 **Chroma** — filter theo metadata: ngôn ngữ lập trình, tên file, loại definition (class/function)."
        ) if local_only else (
            "🎯 **Qdrant** — ACORN filtering để filter theo ngôn ngữ, file path, definition type. Hybrid dense+BM25 đặc biệt mạnh cho code identifier search."
        ),
            "ret_strategy":  "hybrid",
            "pre_transforms": "rewrite",
            "post_reranker":  "cross_encoder",
            "ret_reason":     "**hybrid** (dense+BM25) — bắt buộc khi dùng BM25 embedding. RRF fusion cân bằng semantic recall và exact match. **rewrite** pre-retrieval: chuẩn hoá query trước khi tìm kiếm.",
            "pre_reason":     "**rewrite** — chuẩn hoá lỗi chính tả, đại từ mơ hồ. Tùy chọn: **multi_query** để tăng recall.",
            "post_reason":    "**cross_encoder** reranker (BAAI/bge-reranker-v2-m3) — re-score sau hybrid merge, cải thiện precision đáng kể. Không cần GPU.",
        })
        suggestions.append({
            "rank": 2,
            "title": "Code hỗn hợp ngôn ngữ hoặc script ngắn",
            "pdf_strategy": "pypdf",
            "chunking_strategy": "recursive",
            "chunking_extra": {},
            "loader_reason": (
                "**pypdf** / **pymupdf** đủ cho PDF văn bản. "
                "Không cần parser nặng nếu không có bảng phức tạp."
            ),
            "chunking_reason": (
                "**Recursive** dùng separator đặc trưng của code (def, class, …) "
                "trước khi fallback về ký tự. Đơn giản hơn AST."
            ),
            "emb_provider": "huggingface" if local_only else "openai",
            "emb_model":    "BAAI/bge-m3" if local_only else "text-embedding-3-small",
            "emb_sparse":   True,
            "emb_reason": (
                "**text-embedding-3-small** — đủ tốt cho code hỗn hợp. "
                "**BM25** vẫn nên bật: identifier, API name, error code cần exact match."
            ) if not local_only else (
                "**BAAI/bge-m3** local + **BM25**: combo đủ mạnh cho code search offline. "
                "Identifier matching qua BM25 là không thể thiếu với code."
            ),
        "vdb_provider": "chroma" if local_only else "qdrant",
        "vdb_reason": (
            "🎨 **Chroma** — filter theo file extension, ngôn ngữ. Đủ cho code hỗn hợp ít file."
        ) if local_only else (
            "🎯 **Qdrant** — hybrid BM25+dense với ACORN filtering theo ngôn ngữ và file path. Phù hợp khi codebase lớn hơn."
        ),
            "ret_strategy":  "hybrid",
            "pre_transforms": "rewrite",
            "post_reranker":  "cross_encoder",
            "ret_reason":     "**hybrid** (dense+BM25) — bắt buộc khi dùng BM25 embedding. RRF fusion cân bằng semantic recall và exact match. **rewrite** pre-retrieval: chuẩn hoá query trước khi tìm kiếm.",
            "pre_reason":     "**rewrite** — chuẩn hoá lỗi chính tả, đại từ mơ hồ. Tùy chọn: **multi_query** để tăng recall.",
            "post_reason":    "**cross_encoder** reranker (BAAI/bge-reranker-v2-m3) — re-score sau hybrid merge, cải thiện precision đáng kể. Không cần GPU.",
        })

    # ── HTML dominant ─────────────────────────────────────────────────────────
    elif html_r > 0.4:
        suggestions.append({
            "rank": 1,
            "title": "HTML có cấu trúc heading rõ ràng",
            "pdf_strategy": "pypdf",
            "chunking_strategy": "format_aware",
            "chunking_extra": {"format_type": "html"},
            "loader_reason": (
                "**pymupdf** / **pdfplumber** đủ cho PDF. "
                "Strip boilerplate (header, nav, footer) trước khi load nếu có thể."
            ),
            "chunking_reason": (
                "**Format-aware (html)** cắt theo thẻ semantic <h1>…<h6>, phân tách "
                "đúng theo cấu trúc trang. Tốt cho web scraping hoặc documentation."
            ),
            # HTML thường là web content → đa ngôn ngữ, VI là phổ biến
            "emb_provider": "huggingface" if local_only else "cohere",
            "emb_model":    "BAAI/bge-m3" if local_only else "embed-multilingual-v3.0",
            "emb_sparse":   False,
            "emb_reason": (
                "**Cohere embed-multilingual-v3.0** ⭐⭐⭐⭐ VI — tốt nhất cho web content "
                "tiếng Việt qua API, 108 ngôn ngữ. Dense đủ mạnh khi chunk đã khớp heading."
            ) if not local_only else (
                "**BAAI/bge-m3** local — đa ngôn ngữ, hiểu web content tốt. "
                "Dense đủ khi chunk đã được cắt sạch theo heading HTML."
            ),
        "vdb_provider": "chroma" if local_only else "weaviate",
        "vdb_reason": (
            "🎨 **Chroma** — filter theo URL/domain metadata từ web scraping. Dense-only đủ khi chunk đã clean."
        ) if local_only else (
            "🕸️ **Weaviate** — native hybrid search (dense + BM25) không cần config. Phù hợp web content đa ngôn ngữ cần search cả keyword lẫn ngữ nghĩa."
        ),
            "ret_strategy":  "dense",
            "pre_transforms": "none",
            "post_reranker":  "none",
            "ret_reason":     "**dense** — đủ cho corpus text thuần, query ngữ nghĩa. Nếu corpus có nhiều tên riêng hoặc mã số, hãy chuyển sang **hybrid**.",
            "pre_reason":     "Pre-retrieval không bắt buộc cho use-case đơn giản. Tuỳ chọn: **rewrite** để chuẩn hoá query.",
            "post_reason":    "Reranker không bắt buộc cho use-case đơn giản. Tuỳ chọn: **cross_encoder** nếu cần precision cao hơn.",
        })
        suggestions.append({
            "rank": 2,
            "title": "HTML có nhiều boilerplate / nav / footer",
            "pdf_strategy": "pypdf",
            "chunking_strategy": "recursive",
            "chunking_extra": {},
            "loader_reason": (
                "**pypdf** / **pymupdf** đủ cho PDF text. "
                "Không cần parser nặng nếu HTML/PDF đã được strip sạch."
            ),
            "chunking_reason": (
                "**Recursive** — ổn ngay cả khi HTML không có heading nhất quán. "
                "An toàn hơn khi content đã được strip boilerplate."
            ),
            "emb_provider": "fastembed" if local_only else "openai",
            "emb_model":    "intfloat/multilingual-e5-small" if local_only else "text-embedding-3-small",
            "emb_sparse":   False,
            "emb_reason": (
                "**text-embedding-3-small** — lựa chọn an toàn cho web content hỗn hợp, "
                "không cần sparse khi content đã được strip boilerplate."
            ) if not local_only else (
                "**FastEmbed · multilingual-e5-small** — ONNX nhẹ, không cần GPU, "
                "đủ cho HTML content đã strip."
            ),
        "vdb_provider": "faiss" if local_only else "chroma",
        "vdb_reason": (
            "💾 **FAISS** — HTML đã clean, chunk đơn giản. FAISS đủ nhanh và đơn giản nhất."
        ) if local_only else (
            "🎨 **Chroma** — dễ dùng, persistent, đủ cho HTML content nhỏ-vừa."
        ),
            "ret_strategy":  "dense",
            "pre_transforms": "none",
            "post_reranker":  "none",
            "ret_reason":     "**dense** — đủ cho corpus text thuần, query ngữ nghĩa. Nếu corpus có nhiều tên riêng hoặc mã số, hãy chuyển sang **hybrid**.",
            "pre_reason":     "Pre-retrieval không bắt buộc cho use-case đơn giản. Tuỳ chọn: **rewrite** để chuẩn hoá query.",
            "post_reason":    "Reranker không bắt buộc cho use-case đơn giản. Tuỳ chọn: **cross_encoder** nếu cần precision cao hơn.",
        })

    # ── Mixed / generic ───────────────────────────────────────────────────────
    else:
        suggestions.append({
            "rank": 1,
            "title": "Văn bản hỗn hợp — lựa chọn mặc định tốt nhất",
            "pdf_strategy": "pymupdf",
            "chunking_strategy": "recursive",
            "chunking_extra": {},
            "loader_reason": (
                "**pymupdf** — nhanh, ổn định, layout tốt cho hầu hết PDF. "
                "Lựa chọn mặc định an toàn cho corpus hỗn hợp."
            ),
            "chunking_reason": (
                "**Recursive** — mặc định được khuyến nghị, cân bằng tốt giữa "
                "tốc độ và chất lượng ngữ nghĩa. Không phụ thuộc cấu trúc."
            ),
            "emb_provider": "huggingface" if local_only else "openai",
            "emb_model":    "BAAI/bge-m3" if local_only else "text-embedding-3-small",
            "emb_sparse":   False,
            "emb_reason": (
                "**text-embedding-3-small** — default phổ biến nhất, đủ tốt cho văn bản "
                "hỗn hợp. MRL support: trim xuống 512d để giảm RAM nếu cần."
            ) if not local_only else (
                "**BAAI/bge-m3** local — đa ngôn ngữ, dense chất lượng cao, "
                "không cần API key."
            ),
        "vdb_provider": "chroma" if local_only else "chroma",
        "vdb_reason": (
            "🎨 **Chroma** — lựa chọn mặc định an toàn. Persistent, filtering cơ bản, zero config. Phù hợp cho hầu hết use case."
        ) if local_only else (
            "🎨 **Chroma** — lựa chọn mặc định tốt nhất. Dễ migrate sang provider khác sau khi prototype xong."
        ),
            "ret_strategy":  "dense",
            "pre_transforms": "none",
            "post_reranker":  "none",
            "ret_reason":     "**dense** — đủ cho corpus text thuần, query ngữ nghĩa. Nếu corpus có nhiều tên riêng hoặc mã số, hãy chuyển sang **hybrid**.",
            "pre_reason":     "Pre-retrieval không bắt buộc cho use-case đơn giản. Tuỳ chọn: **rewrite** để chuẩn hoá query.",
            "post_reason":    "Reranker không bắt buộc cho use-case đơn giản. Tuỳ chọn: **cross_encoder** nếu cần precision cao hơn.",
        })
        suggestions.append({
            "rank": 2,
            "title": "Tài liệu đa ngôn ngữ (Việt + Anh)",
            "pdf_strategy": "pymupdf",
            "chunking_strategy": "token_based",
            "chunking_extra": {"encoding_name": "cl100k_base"},
            "loader_reason": (
                "**pymupdf** / **pdfplumber** đọc PDF text sạch. "
                "Tốt cho corpus tiếng Việt, không bị ảnh hưởng bởi encoding đặc thù."
            ),
            "chunking_reason": (
                "**Token-based** đếm token thay vì ký tự — quan trọng với tiếng Việt. "
                "Tránh chunk bị cắt lệch so với ngưỡng token của embedding model."
            ),
            # Đa ngôn ngữ VI+EN → cần model tốt nhất về VI
            "emb_provider": "huggingface" if local_only else "cohere",
            "emb_model":    "BAAI/bge-m3" if local_only else "embed-multilingual-v3.0",
            "emb_sparse":   True,
            "emb_reason": (
                "**Cohere embed-multilingual-v3.0** ⭐⭐⭐⭐ VI — top API choice cho corpus "
                "song ngữ Việt–Anh. **BM25** giúp match từ khoá tiếng Việt không dấu hoặc "
                "tên riêng mà dense có thể bỏ sót khi cross-lingual."
            ) if not local_only else (
                "**BAAI/bge-m3** ⭐⭐⭐⭐ VI local — tốt nhất cho song ngữ Việt–Anh offline. "
                "**BM25** match chính xác từ khoá tiếng Việt đặc thù."
            ),
        "vdb_provider": "chroma" if local_only else "qdrant",
        "vdb_reason": (
            "🎨 **Chroma** — filtering theo ngôn ngữ (VI/EN) trong metadata. Phù hợp corpus song ngữ local."
        ) if local_only else (
            "🎯 **Qdrant** — ACORN filtering theo ngôn ngữ trong metadata. Hybrid dense+BM25 đặc biệt hiệu quả cho corpus song ngữ Việt–Anh."
        ),
            "ret_strategy":  "hybrid",
            "pre_transforms": "rewrite",
            "post_reranker":  "cross_encoder",
            "ret_reason":     "**hybrid** (dense+BM25) — bắt buộc khi dùng BM25 embedding. RRF fusion cân bằng semantic recall và exact match. **rewrite** pre-retrieval: chuẩn hoá query trước khi tìm kiếm.",
            "pre_reason":     "**rewrite** — chuẩn hoá lỗi chính tả, đại từ mơ hồ. Tùy chọn: **multi_query** để tăng recall.",
            "post_reason":    "**cross_encoder** reranker (BAAI/bge-reranker-v2-m3) — re-score sau hybrid merge, cải thiện precision đáng kể. Không cần GPU.",
        })

    # Lọc strategy cần external LLM API nếu local_only
    if local_only:
        NON_LOCAL = {"contextual"}
        suggestions = [s for s in suggestions if s["chunking_strategy"] not in NON_LOCAL]

    return suggestions[:3]


# ─── UI: Loader settings panel ────────────────────────────────────────────────




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


