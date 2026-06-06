"""
core/constants.py
=================
Tất cả hằng số dùng chung toàn app.
Import ở bất kỳ module nào cần dùng:
    from core.constants import EMBED_PREVIEW_LIMIT, CHUNK_DESCRIPTIONS, ...
"""

PAGE_TITLE = "RAG-pipeline-visualizer"

# Số chunk tối đa embed trong preview mode (tránh tốn API cost)
EMBED_PREVIEW_LIMIT = 20

# Màu nhãn theo loại file
FILE_TYPE_COLORS = {
    "pdf":      "#e74c3c",
    "docx":     "#2980b9",
    "xlsx":     "#27ae60",
    "csv":      "#16a085",
    "markdown": "#8e44ad",
    "html":     "#e67e22",
    "code":     "#2c3e50",
    "txt":      "#7f8c8d",
    "email":    "#c0392b",
    "epub":     "#d35400",
    "image":    "#f39c12",
    "json":     "#1abc9c",
}

# Mô tả ngắn cho từng chiến lược chunking
CHUNK_DESCRIPTIONS = {
    "recursive":        "Cắt theo thứ tự ưu tiên: đoạn văn → dòng → câu → ký tự. Phù hợp nhất khi không có yêu cầu đặc biệt.",
    "token_based":      "Đếm token thay vì ký tự — quan trọng với tiếng Việt & đa ngôn ngữ.",
    "sentence_aware":   "Ranh giới chunk luôn trùng với cuối câu — tốt cho FAQ, Q&A.",
    "semantic":         "Phát hiện ranh giới chủ đề qua cosine similarity — tốt cho văn bản đa chủ đề.",
    "contextual":       "LLM sinh context prefix cho mỗi chunk — cải thiện recall đáng kể (Anthropic 2024).",
    "hierarchical":     "Tạo cặp parent (lớn) + child (nhỏ) — search child, trả parent cho LLM.",
    "format_aware":     "Cắt theo cấu trúc tài liệu: Markdown heading, AST code, HTML tag.",
}

# Chiến lược cần LLM (sẽ hiện cảnh báo)
LLM_REQUIRED_STRATEGIES = {"contextual"}

# ── UI pagination sizes ──────────────────────────────────────────────────────
_LOADER_PAGE_SIZE  = 20
_CHUNKING_PAGE_SIZE = 30

# ── PDF strategy dependencies ────────────────────────────────────────────────
PDF_STRATEGY_DEPS = {
    "pypdf":        None,
    "pymupdf":      ("fitz",          "pip install pymupdf"),
    "pdfplumber":   ("pdfplumber",    "pip install pdfplumber tabulate"),
    "unstructured": ("unstructured",  "pip install 'unstructured[pdf]' unstructured-inference"),
    "docling":      ("docling",       "pip install docling"),
    "marker":       ("marker",        "pip install marker-pdf"),
    "opendataloader": ("opendataloader_pdf", "pip install opendataloader-pdf  # Java 11+ required"),
}

# ── Embedding provider metadata ─────────────────────────────────────────────
EMBEDDING_PROVIDER_META: dict[str, dict] = {
    "openai": {
        "label":       "OpenAI",
        "icon":        "🤖",
        "models":      ["text-embedding-3-small", "text-embedding-3-large", "text-embedding-ada-002"],
        "default":     "text-embedding-3-small",
        "model_dims":  {"text-embedding-3-small": 1536, "text-embedding-3-large": 3072, "text-embedding-ada-002": 1536},
        "requires_env": "OPENAI_API_KEY",
        "local":       False,
        "mrl":         True,   # hỗ trợ truncation qua dimensions=
        "vi_quality":  "⭐⭐⭐",
        "note":        "$0.02/1M (3-small) · $0.13/1M (3-large). 3-small là mặc định tốt nhất cho tiếng Anh/đa ngôn ngữ cơ bản.",
        "install":     "pip install langchain-openai",
        "pkg_probe":   "langchain_openai",
    },
    "cohere": {
        "label":       "Cohere",
        "icon":        "🌀",
        "models":      ["embed-multilingual-v3.0", "embed-english-v3.0", "embed-v4.0"],
        "default":     "embed-multilingual-v3.0",
        "model_dims":  {"embed-multilingual-v3.0": 1024, "embed-english-v3.0": 1024, "embed-v4.0": 1536},
        "requires_env": "COHERE_API_KEY",
        "local":       False,
        "mrl":         False,
        "vi_quality":  "⭐⭐⭐⭐",
        "note":        "$0.10/1M · 108 ngôn ngữ · embed-multilingual-v3.0 là lựa chọn hàng đầu cho tiếng Việt qua API.",
        "install":     "pip install langchain-cohere",
        "pkg_probe":   "langchain_cohere",
        "asymmetric":  True,    # cần input_type khác cho doc vs query
        "doc_input_type": "search_document",
        "query_input_type": "search_query",
    },
    "huggingface": {
        "label":       "HuggingFace (local)",
        "icon":        "🤗",
        "models":      [
            "BAAI/bge-m3",
            "Qwen/Qwen3-Embedding-0.6B",
            "Qwen/Qwen3-Embedding-4B",
            "Qwen/Qwen3-Embedding-8B",
            "intfloat/multilingual-e5-large",
            "intfloat/e5-mistral-7b-instruct",
            "nomic-ai/nomic-embed-text-v1.5",
            "Alibaba-NLP/gte-Qwen2-7B-instruct",
            "VinAI/phobert-large",
        ],
        "default":     "BAAI/bge-m3",
        "model_dims":  {
            "BAAI/bge-m3": 1024,
            "Qwen/Qwen3-Embedding-0.6B": 1024,
            "Qwen/Qwen3-Embedding-4B": 2560,
            "Qwen/Qwen3-Embedding-8B": 4096,
            "intfloat/multilingual-e5-large": 1024,
            "intfloat/e5-mistral-7b-instruct": 4096,
            "nomic-ai/nomic-embed-text-v1.5": 768,
            "Alibaba-NLP/gte-Qwen2-7B-instruct": 3584,
            "VinAI/phobert-large": 768,
        },
        "model_notes": {
            "BAAI/bge-m3":                       "1024d · 8192 ctx · Dense+sparse+multivec · ⭐⭐⭐⭐ VI",
            "Qwen/Qwen3-Embedding-0.6B":         "1024d · 32K ctx · MTEB #1 nhẹ nhất · ⭐⭐⭐⭐⭐ VI",
            "Qwen/Qwen3-Embedding-4B":           "2560d · 32K ctx · MTEB #1 cân bằng · ⭐⭐⭐⭐⭐ VI",
            "Qwen/Qwen3-Embedding-8B":           "4096d · 32K ctx · MTEB #1 tốt nhất · ⭐⭐⭐⭐⭐ VI",
            "intfloat/multilingual-e5-large":    "1024d · 512 ctx · 100 ngôn ngữ · ⭐⭐⭐ VI",
            "intfloat/e5-mistral-7b-instruct":   "4096d · 32K ctx · Instruction-tuned · ⭐⭐⭐ VI",
            "nomic-ai/nomic-embed-text-v1.5":    "768d · 8192 ctx · MRL support · ⭐⭐⭐ VI",
            "Alibaba-NLP/gte-Qwen2-7B-instruct": "3584d · 131K ctx · Long context · ⭐⭐⭐⭐ VI",
            "VinAI/phobert-large":               "768d · 256 ctx · Vietnamese-specific · ⭐⭐⭐⭐ VI",
        },
        "requires_env": None,
        "local":       True,
        "mrl":         False,
        "vi_quality":  "⭐⭐⭐⭐⭐",
        "note":        "Chạy hoàn toàn local. Model tự động tải từ HuggingFace Hub lần đầu dùng.",
        "install":     "pip install langchain-huggingface sentence-transformers",
        "pkg_probe":   "sentence_transformers",
    },
    "ollama": {
        "label":       "Ollama (local)",
        "icon":        "🦙",
        "models":      ["nomic-embed-text", "mxbai-embed-large", "bge-m3", "snowflake-arctic-embed"],
        "default":     "nomic-embed-text",
        "model_dims":  {"nomic-embed-text": 768, "mxbai-embed-large": 1024, "bge-m3": 1024, "snowflake-arctic-embed": 1024},
        "model_notes": {
            "nomic-embed-text":      "768d · ctx 2048 token · Fast · English",
            "mxbai-embed-large":     "1024d · ctx 512 token · Strong general purpose",
            "bge-m3":                "1024d · ctx 8192 token · Multilingual · tốt nhất cho tiếng Việt",
            "snowflake-arctic-embed":"1024d · ctx 512 token · Strong English retrieval",
        },
        "requires_env": None,
        "local":       True,
        "mrl":         False,
        "vi_quality":  "⭐⭐⭐",
        "note":        "Cần Ollama server đang chạy. Pull model trước: `ollama pull <model>`.",
        "install":     "pip install langchain-ollama",
        "pkg_probe":   "langchain_ollama",
    },
    "fastembed": {
        "label":       "FastEmbed (local · CPU)",
        "icon":        "⚡",
        "models":      [
            "BAAI/bge-small-en-v1.5",
            "BAAI/bge-base-en-v1.5",
            "intfloat/multilingual-e5-small",
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        ],
        "default":     "BAAI/bge-small-en-v1.5",
        "model_dims":  {
            "BAAI/bge-small-en-v1.5": 384,
            "BAAI/bge-base-en-v1.5": 768,
            "intfloat/multilingual-e5-small": 384,
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2": 384,
        },
        "model_notes": {
            "BAAI/bge-small-en-v1.5": "384d · Nhanh nhất · English",
            "BAAI/bge-base-en-v1.5":  "768d · Chất lượng tốt hơn · English",
            "intfloat/multilingual-e5-small": "384d · Đa ngôn ngữ (VI ok) · ONNX",
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2": "384d · Đa ngôn ngữ · Nhẹ",
        },
        "requires_env": None,
        "local":       True,
        "mrl":         False,
        "vi_quality":  "⭐⭐⭐",
        "note":        "CPU-optimised ONNX. Không cần GPU. Nhanh hơn sentence-transformers trên CPU.",
        "install":     "pip install fastembed langchain-community",
        "pkg_probe":   "fastembed",
    },
}

# ── Vector DB provider metadata ─────────────────────────────────────────────
VECTOR_DB_PROVIDER_META: dict[str, dict] = {
    "faiss": {
        "icon": "💾", "label": "FAISS",
        "tier": "free", "tier_icon": "🆓",
        "mode": "local", "local": True,
        "scale": "< 10M vectors",
        "hybrid": False, "filtering": "post-filter",
        "requires_env": None,
        "pkg_probe": "langchain_openai",
        "note": (
            "**Meta AI.** Chạy hoàn toàn in-process — không server, không API key. "
            "Lý tưởng cho prototype, offline pipeline, CI/CD. Index được lưu vào disk.\n\n"
            "⚠️ Không có metadata filtering native — filter sau khi retrieve."
        ),
        "install": "pip install faiss-cpu langchain-community",
        "pkg_probe": "faiss",
        "params": ["persist_dir"],
    },
    "chroma": {
        "icon": "🎨", "label": "Chroma",
        "tier": "free", "tier_icon": "🆓",
        "mode": "local", "local": True,
        "scale": "< 10M vectors",
        "hybrid": False, "filtering": "native",
        "requires_env": None,
        "pkg_probe": "langchain_cohere",
        "note": (
            "**Chroma.** SQLite backend, Python-native, hỗ trợ metadata filtering. "
            "Dễ dùng nhất cho dev và demo. Collection persist giữa các lần chạy.\n\n"
            "✅ Metadata filtering · 🆓 Zero config · 📦 Tích hợp LangChain tốt nhất"
        ),
        "install": "pip install langchain-chroma",
        "pkg_probe": "chromadb",
        "params": ["persist_dir"],
    },
    "lancedb": {
        "icon": "🏹", "label": "LanceDB",
        "tier": "free", "tier_icon": "🆓",
        "mode": "local / cloud", "local": True,
        "scale": "~ 1B vectors",
        "hybrid": False, "filtering": "native",
        "requires_env": None,
        "pkg_probe": "sentence_transformers",
        "note": (
            "**LanceDB.** Columnar format (Lance ≈ Parquet), embedded mode không cần server. "
            "Tích hợp tốt với pandas/Arrow/DuckDB. Hỗ trợ LanceDB Cloud (serverless).\n\n"
            "✅ Không server · ✅ Columnar = đọc nhanh · ✅ DuckDB SQL trên vectors"
        ),
        "install": "pip install lancedb langchain-community",
        "pkg_probe": "lancedb",
        "params": ["persist_dir", "distance"],
    },
    "qdrant": {
        "icon": "🎯", "label": "Qdrant",
        "tier": "free / paid", "tier_icon": "🔓",
        "mode": "self-host / cloud", "local": False,
        "scale": "~ 1B+ vectors",
        "hybrid": True, "filtering": "native (ACORN)",
        "requires_env": "QDRANT_URL",
        "note": (
            "**Qdrant.** Vector DB tốt nhất cho metadata filtering phức tạp nhờ thuật toán "
            "**ACORN** (filter trong graph traversal, không phải sau). Hỗ trợ hybrid dense+sparse. "
            "Có thể self-host bằng Docker hoặc dùng Qdrant Cloud.\n\n"
            "✅ ACORN filtering · ✅ Hybrid search · ✅ Self-host hoặc Cloud"
        ),
        "install": "pip install qdrant-client langchain-qdrant",
        "pkg_probe": "qdrant_client",
        "params": ["url", "api_key", "distance", "on_disk"],
    },
    "weaviate": {
        "icon": "🕸️", "label": "Weaviate",
        "tier": "free / paid", "tier_icon": "🔓",
        "mode": "self-host / cloud", "local": False,
        "scale": "~ 1B vectors",
        "hybrid": True, "filtering": "native",
        "requires_env": "WEAVIATE_URL",
        "note": (
            "**Weaviate.** Lưu cả dense vector lẫn BM25 term freq natively → hybrid search "
            "out-of-the-box không cần config thêm. GraphQL API cho filtering phức tạp.\n\n"
            "✅ Hybrid search mặc định · ✅ GraphQL API · ✅ Schema linh hoạt"
        ),
        "install": "pip install weaviate-client langchain-weaviate",
        "pkg_probe": "weaviate",
        "params": ["url", "api_key"],
    },
    "pgvector": {
        "icon": "🐘", "label": "pgvector (PostgreSQL)",
        "tier": "free", "tier_icon": "🆓",
        "mode": "self-host", "local": False,
        "scale": "< 50M vectors",
        "hybrid": False, "filtering": "native (SQL)",
        "requires_env": "DATABASE_URL",
        "note": (
            "**pgvector.** Extension PostgreSQL — ACID, JOINs với các bảng khác, SQL filtering "
            "đầy đủ. Best choice nếu đã có PostgreSQL trong stack.\n\n"
            "✅ ACID transactions · ✅ SQL filtering · ✅ Join với bảng khác · "
            "⚠️ Không scale > 50M vectors tốt"
        ),
        "install": "pip install langchain-postgres psycopg",
        "pkg_probe": "psycopg",
        "params": ["connection_string", "distance_strategy"],
    },
    "pinecone": {
        "icon": "🌲", "label": "Pinecone",
        "tier": "paid", "tier_icon": "💳",
        "mode": "managed cloud", "local": False,
        "scale": "~ 1B vectors",
        "hybrid": True, "filtering": "native",
        "requires_env": "PINECONE_API_KEY",
        "note": (
            "**Pinecone Serverless.** Fully managed, auto-scaling, zero ops. "
            "API key là bắt buộc. Tốt cho startup muốn time-to-market nhanh.\n\n"
            "✅ Zero ops · ✅ Auto-scale · 💳 Paid (free tier giới hạn) · ⚠️ Vendor lock-in"
        ),
        "install": "pip install pinecone langchain-pinecone",
        "pkg_probe": "pinecone",
        "params": ["cloud", "region"],
    },
}
