<p align="center">
  <h1 align="center">🔬 RAG-pipeline-visualizer</h1>
</p>

<p align="center">
  <b>RAG-pipeline-visualizer</b> là ứng dụng Streamlit giúp bạn xây dựng, cấu hình và <b>trực quan hóa từng bước</b> trong pipeline RAG (Retrieval-Augmented Generation) — từ bước load tài liệu PDF đến bước sinh câu trả lời từ LLM.
</p>

---

<p align="center">
  <a href="https://www.youtube.com/watch?v=YPcXqbcZqaE">
    <img src="https://img.youtube.com/vi/YPcXqbcZqaE/maxresdefault.jpg" width=800>
  </a><br/>
  <i>▶️ Indexing Pipeline — Load · Chunk · Embed · Store</i>
</p>

<p align="center">
  <a href="https://www.youtube.com/watch?v=2THKb3pnFIk">
    <img src="https://img.youtube.com/vi/2THKb3pnFIk/maxresdefault.jpg" width=800>
  </a><br/>
  <i>▶️ Generation Pipeline — Retrieve · Rerank · Prompt · Generate</i>
</p>

> **Mục tiêu:** Cho phép người dùng thử nghiệm và quan sát tác động của từng lựa chọn cấu hình (chunking strategy, embedding model, vector DB, retrieval strategy, ...) đến chất lượng câu trả lời cuối cùng — không cần viết một dòng code.

---

## 📋 Mục lục

1. [Tổng quan Pipeline](#1-tổng-quan-pipeline)
2. [Cấu trúc Project](#2-cấu-trúc-project)
3. [Yêu cầu hệ thống](#3-yêu-cầu-hệ-thống)
4. [Tạo môi trường & Cài đặt](#4-tạo-môi-trường--cài-đặt)
5. [Cấu hình API Keys](#5-cấu-hình-api-keys)
6. [Khởi động ứng dụng](#6-khởi-động-ứng-dụng)
7. [Hướng dẫn sử dụng — Indexing Pipeline](#7-hướng-dẫn-sử-dụng--indexing-pipeline)
8. [Hướng dẫn sử dụng — Generation Pipeline](#8-hướng-dẫn-sử-dụng--generation-pipeline)
9. [Demo End-to-End: PDF phức tạp → Câu trả lời có trích dẫn](#9-demo-end-to-end)
10. [Pipeline Cache](#10-pipeline-cache)
11. [config.yaml](#11-configyaml)
12. [Troubleshooting](#12-troubleshooting)

---

## 1. Tổng quan Pipeline

Pipeline gồm hai stage độc lập, tương ứng với hai page trong ứng dụng:

**STAGE 1 — INDEXING (offline)**

```
[File PDF]  →  [1. Loader]  →  [2. Chunking]  →  [3. Embedding]  →  [4. Vector DB]
```

**STAGE 2 — GENERATION (online)**

```
[Query]  →  [7. Pre-retrieval]  →  [8. Retrieval]  →  [9. Post-retrieval]
                                                    →  [10. Prompt]  →  [11. Generation]  →  [Câu trả lời]
```

| Bước | Tên | Mô tả | Stage | Bắt buộc |
|------|-----|--------|-------|---------|
| 1 | **Loader** | Parse tài liệu PDF → `list[Document]` | Indexing | ✅ |
| 2 | **Chunking** | Cắt Document thành chunk nhỏ hơn | Indexing | ✅ |
| 3 | **Embedding** | Chuyển chunk thành dense/sparse vectors | Indexing | ✅ |
| 4 | **Vector DB** | Lưu vectors vào index có thể tìm kiếm | Indexing | ✅ |
| 7 | **Pre-retrieval** | Biến đổi query trước khi tìm kiếm | Generation | ❌ Tuỳ chọn |
| 8 | **Retrieval** | Tìm chunk liên quan từ Vector DB | Generation | ✅ |
| 9 | **Post-retrieval** | Rerank, filter, nén context | Generation | ❌ Tuỳ chọn |
| 10 | **Prompt** | Xây dựng prompt đưa vào LLM | Generation | ✅ |
| 11 | **Generation** | LLM sinh câu trả lời cuối cùng | Generation | ✅ |

---

## 2. Cấu trúc Project

```
RAG-pipeline-visualizer/
├── app.py                          # Entry point — chạy bằng: streamlit run app.py
├── pipeline_cache.py               # Step-level disk cache (SHA-256 fingerprint)
├── config.yaml                     # Cấu hình mặc định cho toàn bộ pipeline
├── requirements.txt                # Dependencies
├── .env                            # API keys (tạo thủ công, KHÔNG commit lên Git)
│
├── pipeline/                       # Hai pipeline class chính
│   ├── indexing_pipeline.py        # IndexingPipeline (Stage 1)
│   └── generation_pipeline.py      # GenerationPipeline (Stage 2)
│
├── loader/                         # Bước 1 — Load PDF
│   ├── pdf_loader.py               # 7 PDF loader strategies
│   ├── directory_loader.py         # Scan thư mục, dispatch đến pdf_loader
│   └── utils.py
│
├── chunking/                       # Bước 2 — Chunking
│   ├── recursive.py                # RecursiveCharacterTextSplitter
│   ├── token_based.py              # Tiktoken-based
│   ├── format_aware.py             # Markdown / code / HTML boundaries
│   ├── sentence_aware.py           # NLTK sentence boundaries
│   ├── semantic.py                 # Cosine similarity breakpoints
│   ├── hierarchical.py             # Parent + Child chunks
│   ├── contextual.py               # LLM-generated context prefix
│   └── deduplication.py            # MinHash deduplication
│
├── embedding/                      # Bước 3 — Embedding
│   ├── openai_embedder.py
│   ├── cohere_embedder.py
│   ├── huggingface_embedder.py
│   ├── ollama_embedder.py
│   ├── fastembed_embedder.py
│   ├── sparse_embedder.py          # BM25 / SPLADE
│   └── pipeline.py                 # Dense + sparse combination
│
├── vector_db/                      # Bước 4 — Vector Store
│   ├── chroma_store.py
│   ├── faiss_store.py
│   ├── qdrant_store.py
│   ├── lancedb_store.py
│   ├── weaviate_store.py
│   ├── pgvector_store.py
│   └── pinecone_store.py
│
├── pre_retrieval/                  # Bước 7 — Query Transformation
│   ├── pipeline.py                 # Chain nhiều transformers
│   └── factory.py
│
├── retrieval/                      # Bước 8 — Retrieval
│   ├── dense.py
│   ├── sparse.py
│   ├── hybrid.py                   # RRF / Weighted / DBSF fusion
│   ├── multi_query.py
│   ├── parent_document.py
│   ├── sentence_window.py
│   └── multi_hop.py
│
├── post_retrieval/                 # Bước 9 — Post-processing
│   ├── pipeline.py
│   ├── cross_encoder_reranker.py
│   ├── cohere_reranker.py
│   ├── llm_reranker.py
│   ├── mmr_filter.py
│   ├── redundancy_filter.py
│   ├── context_compressor.py
│   └── context_orderer.py
│
├── prompt/                         # Bước 10 — Prompt Builder
│   ├── basic.py
│   ├── citation.py
│   ├── conversational.py
│   └── structured_output.py
│
├── generation/                     # Bước 11 — LLM Generation
│   ├── openai_generator.py
│   ├── anthropic_generator.py
│   ├── google_generator.py
│   ├── ollama_generator.py
│   └── cohere_generator.py
│
├── ui/
│   ├── pages/
│   │   ├── welcome.py              # 🏠 Trang chủ
│   │   ├── indexing.py             # 🗃️ Indexing pipeline UI
│   │   └── generation.py          # 💬 Generation pipeline UI
│   ├── settings/                   # Sidebar config panels
│   ├── results/                    # Result visualization panels
│   └── components/                 # Shared components
│
├── core/
│   ├── pipeline_runners.py         # run_loader, run_chunker, run_embedder
│   ├── cache_helpers.py            # Cache helpers
│   └── constants.py                # App-wide constants & metadata
│
└── utils/
    ├── env.py                      # _is_installed, API key helpers, GPU detect
    └── badges.py                   # File type / chunk type badges
```

---

## 3. Yêu cầu hệ thống

| Yêu cầu | Tối thiểu | Khuyến nghị |
|---------|-----------|-------------|
| **Python** | 3.10 | **3.11** |
| **RAM** | 4 GB | 8 GB+ |
| **Disk** | 2 GB | 5 GB+ *(model cache lần đầu)* |
| **GPU** | Không bắt buộc | NVIDIA CUDA *(tăng tốc embedding & reranker)* |
| **OS** | Windows 10 / macOS 12 / Ubuntu 20.04 | Bất kỳ |
| **Internet** | Cần để gọi API (OpenAI, Anthropic, ...) | — |

> 💡 **Ollama (LLM local):** Khuyến nghị RAM ≥ 8 GB và SSD. GPU không bắt buộc nhưng tăng tốc đáng kể.

---

## 4. Tạo môi trường & Cài đặt

### 4.1 — Clone repository

```bash
git clone https://github.com/<your-username>/RAG-pipeline-visualizer.git
cd RAG-pipeline-visualizer
```

### 4.2 — Tạo virtual environment

**Dùng `venv` (Python built-in):**

```bash
# macOS / Linux
python3.11 -m venv .venv
source .venv/bin/activate

# Windows (Command Prompt)
python -m venv .venv
.venv\Scripts\activate.bat

# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**Hoặc dùng `conda`:**

```bash
conda create -n rag_visualizer python=3.11 -y
conda activate rag_visualizer
```

> ✅ Kiểm tra Python version: `python --version` → phải trả về `Python 3.11.x`

### 4.3 — Upgrade pip

```bash
pip install --upgrade pip
```

### 4.4 — Cài đặt dependencies bắt buộc

```bash
pip install -r requirements.txt
```

> ⏱️ Lần đầu mất khoảng **3–7 phút** tuỳ tốc độ mạng. Các lần sau sẽ nhanh hơn nhờ cache pip.

### 4.5 — Cài đặt thêm tuỳ tính năng muốn dùng

#### Loader nâng cao

**Marker PDF** *(khuyến nghị cho PDF phức tạp: bảng, công thức, hình ảnh)*:

```bash
pip install marker-pdf==1.6.2
```

> ⚠️ Marker tải các model AI (~1.5 GB) trong lần chạy đầu tiên. Cần kết nối internet.

**Docling** *(IBM Research, chất lượng Markdown rất cao)*:

```bash
pip install docling==2.28.0
```

**Unstructured** *(OCR tổng hợp — tốt nhất cho PDF scan)*:

```bash
pip install 'unstructured[pdf]==0.16.12'
```

#### Embedding providers khác

```bash
# Cohere (multilingual, tốt cho tiếng Việt)
pip install cohere==5.13.12 langchain-cohere==0.4.3

# Google Gemini
pip install google-generativeai==0.8.3 langchain-google-genai==2.0.11

# Ollama (local)
pip install ollama==0.4.7 langchain-ollama==0.2.3
```

#### Generation (LLM) providers khác

```bash
# Anthropic Claude
pip install anthropic==0.43.0

# Ollama (đã cài ở trên nếu dùng cả embedding)
```

#### Vector DB khác

```bash
# Qdrant (production, filtering phức tạp)
pip install qdrant-client==1.13.3 langchain-qdrant==0.2.0

# LanceDB (embedded, không cần server)
pip install lancedb==0.17.0

# Weaviate (hybrid out-of-the-box)
pip install weaviate-client==4.10.4 langchain-weaviate==0.0.3

# PGVector (nếu đã có PostgreSQL)
pip install 'psycopg[binary]==3.2.4' langchain-postgres==0.0.13

# Pinecone (managed cloud)
pip install pinecone==5.0.1 langchain-pinecone==0.2.0
```

#### SPLADE sparse embedding *(tốt hơn BM25, cần GPU khuyến nghị)*

```bash
pip install transformers==4.47.1 torch==2.5.1
```

---

## 5. Cấu hình API Keys

Tạo file `.env` ở thư mục gốc của project:

```bash
# macOS / Linux
cp .env.example .env    # nếu có file mẫu
# hoặc tạo mới:
touch .env

# Windows PowerShell
New-Item .env -ItemType File
```

Mở `.env` và điền API keys cần thiết:

```dotenv
# ─── LLM & Embedding Providers ────────────────────────────────────
# Bắt buộc nếu dùng OpenAI embedding hoặc generation
OPENAI_API_KEY=sk-proj-...

# Anthropic Claude (nếu dùng)
ANTHROPIC_API_KEY=sk-ant-...

# Google Gemini (nếu dùng)
GOOGLE_API_KEY=AIza...

# Cohere (nếu dùng)
COHERE_API_KEY=...

# ─── Vector Databases ─────────────────────────────────────────────
# Qdrant (để trống nếu dùng local Docker)
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=

# Weaviate (để trống nếu dùng local)
WEAVIATE_URL=http://localhost:8080
WEAVIATE_API_KEY=

# Pinecone
PINECONE_API_KEY=...

# PGVector
DATABASE_URL=postgresql://user:password@localhost:5432/ragdb

# ─── Local Storage ────────────────────────────────────────────────
CHROMA_PERSIST_DIR=./storage/chroma_db
FAISS_PERSIST_DIR=./storage/faiss_index
```

> ⚠️ **Quan trọng:** Thêm `.env` vào `.gitignore` để tránh lộ API key khi push lên GitHub.

```bash
echo ".env" >> .gitignore
```

---

## 6. Khởi động ứng dụng

```bash
# Đảm bảo đang ở thư mục gốc project và đã activate môi trường ảo
streamlit run app.py
```

Mặc định app chạy tại: **[http://localhost:8501](http://localhost:8501)**

Nếu port 8501 đã bị chiếm:

```bash
streamlit run app.py --server.port 8502
```

Mở rộng kích thước upload file (mặc định Streamlit giới hạn 200 MB):

```bash
streamlit run app.py --server.maxUploadSize 500
```

Ứng dụng có **3 page** điều hướng qua sidebar:

| Page | Icon | Mô tả |
|------|------|-------|
| **Trang chủ** | 🏠 | Giới thiệu tổng quan, hướng dẫn nhanh |
| **Indexing** | 🗃️ | Stage 1: Load → Chunk → Embed → Lưu vào Vector DB |
| **Generation** | 💬 | Stage 2: Query → Retrieve → Rerank → Generate |

---

## 7. Hướng dẫn sử dụng — Indexing Pipeline

Chuyển sang page **🗃️ Indexing** từ sidebar.

---

### Bước 0 — Tải tài liệu lên

Ở đầu sidebar, chọn cách nạp tài liệu:

**Cách 1 — Upload file trực tiếp:**
Nhấn ô **"Kéo thả hoặc browse file"** → chọn một hoặc nhiều file PDF. Phù hợp khi xử lý 1–5 file.

**Cách 2 — Nhập đường dẫn thư mục:**
Nhập đường dẫn tuyệt đối đến thư mục chứa PDF. App tự động tìm toàn bộ `.pdf` trong thư mục đó (bao gồm subfolder).

> 💡 Dùng **Upload** cho demo nhanh; dùng **Thư mục** khi xử lý dataset lớn hoặc muốn chạy lại thường xuyên.

---

### Bước 1 — Loader (PDF Strategy)

Mở expander **"1️⃣ Loader"** trong sidebar.

Chọn **PDF Strategy** phù hợp với tài liệu:

| Strategy | Tốc độ | Bảng | Công thức | Khi nào dùng |
|----------|--------|------|-----------|--------------|
| `pypdf` | ⚡⚡⚡ | ❌ | ❌ | Prototype nhanh, PDF text thuần |
| `pymupdf` | ⚡⚡⚡ | ⚠️ Cơ bản | ❌ | PDF layout đơn giản |
| `pdfplumber` | ⚡⚡ | ✅ Tốt nhất | ❌ | PDF nhiều bảng dạng text layer |
| `marker` ⭐ | ⚡ Chậm | ✅✅ | ✅ LaTeX | **PDF phức tạp: bảng, công thức, hình** |
| `docling` | ⚡ Chậm | ✅✅ | ✅ | Tài liệu học thuật, báo cáo |
| `unstructured` | ⚡ Chậm | ✅✅ | ✅ OCR | PDF scan, cần OCR |

**OCR Engine** *(nếu PDF là bản scan hoặc chứa ảnh có chữ)*:

| Engine | Khi nào dùng |
|--------|-------------|
| `none` | PDF có text layer *(mặc định)* |
| `tesseract` | Văn bản in rõ trên nền trắng, 100+ ngôn ngữ |
| `paddleocr` | Tiếng Việt, CJK, layout phức tạp |
| `easyocr` | Đa ngôn ngữ, dễ cài, ổn định trên CPU |
| `surya` | Đa ngôn ngữ chất lượng cao, transformer-based |

Các tuỳ chọn thêm:
- **Trích xuất bảng**: Chuyển bảng trong PDF thành Markdown table *(dùng với pdfplumber, unstructured)*.
- **Mô tả hình ảnh bằng VLM**: Dùng GPT-4o-mini hoặc Ollama vision model để mô tả hình ảnh trong PDF thành text. Hữu ích khi PDF chứa biểu đồ, sơ đồ quan trọng.

**Kết quả** (tab **"📄 Loader"** ở vùng chính): Số document, tổng ký tự, bảng thống kê theo file, preview nội dung từng document.

---

### Bước 2 — Chunking

Mở expander **"2️⃣ Chunking"** trong sidebar.

| Strategy | Cơ chế | Best for |
|----------|--------|---------|
| `recursive` ⭐ | Cắt theo thứ tự: đoạn→dòng→câu→ký tự | **Mặc định tốt nhất cho hầu hết trường hợp** |
| `token_based` | Đếm BPE token (tiktoken) thay vì ký tự | Tiếng Việt, tránh silent truncation |
| `format_aware` | Nhận diện cấu trúc: Markdown heading, code block, HTML tag | **PDF qua Marker/Docling** ← khuyến nghị kết hợp |
| `sentence_aware` | Ranh giới chunk = cuối câu (NLTK) | FAQ, Q&A ngắn, văn học |
| `semantic` | Phát hiện ranh giới chủ đề qua cosine similarity | Văn bản đa chủ đề |
| `hierarchical` | Tạo cặp parent (lớn) + child (nhỏ) | Corpus lớn, cần độ chính xác cao |
| `contextual` | LLM sinh context prefix cho mỗi chunk | Production, có LLM budget |

Tham số quan trọng:
- **Chunk size**: Số ký tự tối đa mỗi chunk. Thông thường **512–1500**.
- **Chunk overlap**: Số ký tự chồng lấp giữa 2 chunk liên tiếp. Thông thường **10–15% của chunk size**.

**Kết quả** (tab **"✂️ Chunking"**): Histogram phân phối chunk size, metrics min/max/mean/median, preview từng chunk với metadata.

---

### Bước 3 — Embedding

Mở expander **"3️⃣ Embedding"** trong sidebar.

| Provider | Model tiêu biểu | Tiếng Việt | Chi phí | Best for |
|----------|-----------------|-----------|---------|---------|
| **OpenAI** | `text-embedding-3-small` ⭐ | ⭐⭐⭐ | $0.02/1M | Mặc định cân bằng nhất |
| **OpenAI** | `text-embedding-3-large` | ⭐⭐⭐ | $0.13/1M | Độ chính xác tối đa |
| **Cohere** | `embed-multilingual-v3.0` | ⭐⭐⭐⭐ | $0.10/1M | Corpus tiếng Việt + đa ngôn ngữ |
| **HuggingFace** | `BAAI/bge-m3` | ⭐⭐⭐⭐ | Miễn phí | Local, tốt cho VI |
| **HuggingFace** | `Qwen/Qwen3-Embedding-4B` | ⭐⭐⭐⭐⭐ | Miễn phí | MTEB #1, tốt nhất cho VI |
| **FastEmbed** | `multilingual-e5-small` | ⭐⭐⭐ | Miễn phí | CPU-only, không cần GPU |
| **Ollama** | `bge-m3` | ⭐⭐⭐⭐ | Miễn phí | Hoàn toàn local, bảo mật cao |

**Sparse Embedding (Hybrid Retrieval):**
- Checkbox **"Bật sparse embedding (hybrid retrieval)"**: Kết hợp dense vector với BM25/SPLADE để cải thiện recall, đặc biệt cho từ khoá, tên riêng, số liệu.
- `BM25` — Lexical matching, không cần GPU, khuyến nghị bật mặc định.
- `SPLADE` — Sparse neural, hiệu quả hơn BM25, cần `transformers` + `torch`.

**MRL (Matryoshka Representation Learning):** Áp dụng cho OpenAI `text-embedding-3-*` và một số HuggingFace model. Cho phép cắt ngắn chiều vector (ví dụ từ 1536 → 512) mà không mất nhiều chất lượng, tiết kiệm storage.

**Kết quả** (tab **"🧮 Embedding"**): Embedding matrix shape, cosine similarity heatmap, nearest neighbors của từng chunk.

---

### Bước 4 — Vector DB

Mở expander **"4️⃣ Vector DB"** trong sidebar.

| Provider | Mode | Scale | Hybrid | Best for |
|----------|------|-------|--------|---------|
| **Chroma** ⭐ | Local | < 10M | ❌* | Demo, dev, small corpus |
| **FAISS** | Local | < 100M | ❌ | Prototype cực nhanh |
| **Qdrant** | Self-host / Cloud | 1B+ | ✅ ACORN | Production, filtering phức tạp |
| **LanceDB** | Local / Cloud | ~1B | ✅ | Embedded, columnar, DuckDB SQL |
| **Weaviate** | Self-host / Cloud | ~1B | ✅ | Hybrid native out-of-the-box |
| **PGVector** | Self-host | < 50M | ❌* | Đã có PostgreSQL infrastructure |
| **Pinecone** | Managed cloud | ~1B | ✅ | Zero-ops, startup |

*\* Hybrid thực hiện qua BM25 sparse index riêng kết hợp ở tầng retrieval.*

Tham số cần nhập:
- **Collection name**: Tên collection/index để lưu (ví dụ: `rag_docs`).
- **Persist directory**: Thư mục lưu index trên disk (với Chroma, FAISS, LanceDB).

---

### Bước 5 — Chạy Indexing

Sau khi cấu hình xong, nhấn nút **▶️ Process** ở cuối sidebar.

App chạy tuần tự 4 bước với progress bar. Kết quả mỗi bước được **cache trên disk** theo fingerprint SHA-256 — nếu chạy lại với cùng cấu hình, kết quả được tái sử dụng ngay lập tức mà không cần tính lại.

- Nhấn **🛑 Stop** để dừng bất cứ lúc nào.
- Khi indexing xong, chuyển sang **Generation** page để bắt đầu hỏi đáp.

---

## 8. Hướng dẫn sử dụng — Generation Pipeline

Chuyển sang page **💬 Generation** từ sidebar.

---

### Chọn Index đã build

Ở đầu sidebar (hoặc vùng chính), chọn pipeline/index đã được build ở bước Indexing. App hiển thị danh sách tất cả index đã tạo, kèm thông tin: file nguồn, loader strategy, embedding model, vector DB.

---

### Bước 7 — Pre-retrieval *(tuỳ chọn)*

Mở expander **"🔄 Pre-retrieval"** trong sidebar.

| Strategy | Cơ chế | Khi nào dùng |
|----------|--------|-------------|
| `none` ⭐ | Giữ nguyên query | Query đã rõ ràng, muốn latency thấp |
| `rewrite` | LLM viết lại chuẩn hơn, sửa lỗi chính tả | Query ngắn, thiếu ngữ cảnh |
| `expand` | Thêm từ đồng nghĩa và thuật ngữ liên quan | Corpus nhiều thuật ngữ tương đương |
| `step_back` | Tổng quát hóa câu hỏi để tìm kiến thức nền | Query quá cụ thể, cần context rộng hơn |
| `multi_query` | Tạo N biến thể query → merge kết quả (RRF) | Query phức tạp, nhiều khía cạnh |
| `decompose` | Chia thành sub-query độc lập | Multi-hop reasoning |
| `self_query` | Parse metadata filter từ ngôn ngữ tự nhiên | Corpus có metadata phong phú |
| `route` | Phân loại query → chọn strategy phù hợp | Corpus đa dạng nhiều chủ đề |

---

### Bước 8 — Retrieval

Mở expander **"⚙️ Retrieval"** trong sidebar.

| Strategy | Cơ chế | Khi nào dùng |
|----------|--------|-------------|
| `hybrid` ⭐ | Dense + Sparse → RRF fusion | **Tốt nhất khi đã bật sparse embedding** |
| `dense` | Cosine similarity thuần | Câu hỏi ngữ nghĩa, không có từ khoá đặc biệt |
| `sparse` | BM25 / SPLADE keyword search | Tên riêng, số liệu, thuật ngữ chính xác |
| `multi_query` | N biến thể query → merge (RRF) | Query mơ hồ, nhiều cách hiểu |
| `parent_document` | Retrieve child chunk → trả parent | Cần context đầy đủ hơn chunk nhỏ |
| `sentence_window` | Mở rộng ±N câu xung quanh câu khớp | Corpus văn xuôi liên tục |
| `multi_hop` | Retrieve → LLM → follow-up → retrieve | Câu hỏi đòi hỏi lý luận nhiều bước |

Tham số:
- **Top-K**: Số chunk trả về trước khi rerank. Thường **10–20**.
- **Fusion method** (hybrid): `rrf` *(Reciprocal Rank Fusion — khuyến nghị)* / `weighted` / `dbsf`.

---

### Bước 9 — Post-retrieval *(tuỳ chọn)*

Mở expander **"⚙️ Post-retrieval"** trong sidebar.

**Reranker:**

| Reranker | Khi nào dùng |
|----------|-------------|
| `none` | Retrieval đã đủ tốt, muốn latency thấp |
| `cross_encoder` ⭐ | **Khuyến nghị VI** — `BAAI/bge-reranker-v2-m3` |
| `cohere` | Best API quality, 100+ ngôn ngữ |
| `llm` | Listwise reranking, không cần model riêng |

**Filters & Processors** *(chạy theo thứ tự)*:

| Tính năng | Tác dụng |
|-----------|---------|
| **Redundancy filter** | Loại chunk gần trùng lặp (cosine > ngưỡng) |
| **MMR filter** | Cân bằng relevance và diversity |
| **LLM filter** | LLM phân loại YES/NO từng chunk |
| **Compression** | LLM trích đoạn liên quan từ mỗi chunk |

**Context ordering:**
- `sandwich` ⭐ — Chunk tốt nhất ở đầu và cuối, giảm *"lost-in-the-middle"*.
- `relevance` — Giảm dần theo score.
- `original` — Giữ nguyên thứ tự từ retriever.

- **Top-N**: Số chunk giữ lại sau reranking. Thường **3–8**.

---

### Bước 10 — Prompt

Mở expander **"📝 Prompt"** trong sidebar.

| Template | Output | Khi nào dùng |
|----------|--------|-------------|
| `citation` ⭐ | Text + `[NGUỒN N]` inline | **Production** — verify được từng fact |
| `basic` | Plain text | Prototype nhanh |
| `conversational` | Text + lịch sử hội thoại | Chatbot, follow-up questions |
| `structured` | JSON: `answer + claims + sources + confidence` | Downstream code cần parse |

Tuỳ chọn:
- **Ngôn ngữ prompt**: `vi` / `en` / `both` *(tự động phát hiện)*.
- **Max context chars**: Giới hạn tổng ký tự context đưa vào prompt (0 = không giới hạn).
- **Rules**: Thêm instruction vào system prompt (ví dụ: "Trả lời súc tích trong 3 gạch đầu dòng").

---

### Bước 11 — Generation

Mở expander **"⚙️ Generation"** trong sidebar.

| Tình huống | Provider | Model | Ghi chú |
|-----------|---------|-------|---------|
| 🏆 Cân bằng nhất | OpenAI | `gpt-4.1-mini` | $0.40/$1.60 per 1M |
| 💰 Rẻ nhất (API) | OpenAI | `gpt-4o-mini` | $0.15/$0.60 per 1M |
| 🇻🇳 Tiếng Việt tốt | Anthropic | `claude-sonnet-4-6` | $3/$15 per 1M |
| 🆓 Miễn phí | Google | `gemini-2.0-flash` | Free tier có giới hạn |
| 🔒 Offline / Privacy | Ollama | `qwen2.5:7b` | ~4.7 GB RAM |
| 🪶 Máy yếu, offline | Ollama | `llama3.2:3b` | ~2 GB RAM |

Tham số:
- **Temperature**: `0.0` cho RAG *(deterministic, ít hallucinate nhất)*.
- **Max tokens**: Độ dài tối đa câu trả lời.
- **Streaming**: Bật để xem câu trả lời xuất hiện dần từng token.

---

### Chạy Query

Nhập câu hỏi vào ô text ở vùng chính và nhấn **▶️ Chạy**.

App hiển thị kết quả từng bước có thể expand/collapse:

- **Pre-retrieval**: Query gốc vs. query sau biến đổi.
- **Retrieval**: Top-K chunk với similarity score và nguồn file.
- **Post-retrieval**: Thứ tự chunk sau reranking, score thay đổi.
- **Prompt**: System message và user message đầy đủ gửi lên LLM.
- **Generation**: Câu trả lời streaming, inline citations, token usage, đoạn nguồn tương ứng.

---

## 9. Demo End-to-End

**Kịch bản:** Upload PDF phức tạp (có text, ảnh, bảng, công thức) → Hỏi đáp với trích dẫn nguồn chính xác.

### 9.1 Cấu hình Indexing khuyến nghị

| Bước | Setting | Lý do |
|------|---------|-------|
| **Upload** | 1–3 file PDF (học thuật / kỹ thuật) | Tài liệu có bảng, công thức, hình |
| **Loader** | `marker` | Chuyển đổi PDF → Markdown chất lượng cao, giữ cấu trúc bảng và công thức LaTeX |
| **Chunking** | `format_aware` | Cắt theo Markdown heading từ output của Marker — giữ nguyên ngữ nghĩa section |
| **Chunk size** | `1000` ký tự | Cân bằng giữa context đủ rộng và precision |
| **Chunk overlap** | `150` ký tự | ~15% overlap, tránh mất thông tin ở ranh giới |
| **Embedding** | OpenAI `text-embedding-3-small` | Nhanh, rẻ, chất lượng tốt |
| **Sparse** | Bật BM25 | Cải thiện recall cho từ khoá, tên riêng, số liệu |
| **Vector DB** | `chroma` hoặc `faiss` | Không cần config server, dùng được ngay |

### 9.2 Cấu hình Generation khuyến nghị

| Bước | Setting | Lý do |
|------|---------|-------|
| **Pre-retrieval** | `rewrite` | Sửa chính tả, làm rõ đại từ — latency thấp |
| **Retrieval** | `hybrid`, Top-K = 15 | Kết hợp dense + BM25 → recall tốt hơn |
| **Reranker** | `cross_encoder` — `BAAI/bge-reranker-v2-m3` | Multilingual, VI tốt, không cần GPU |
| **Top-N** | `5` | Giữ 5 chunk liên quan nhất sau rerank |
| **Redundancy filter** | Bật (threshold 0.92) | Loại chunk gần trùng trước khi rerank |
| **Context ordering** | `sandwich` | Giảm lost-in-the-middle |
| **Prompt template** | `citation` | Câu trả lời có `[NGUỒN N]` để verify |
| **Generation** | OpenAI `gpt-4.1-mini`, Temperature = 0.0 | Deterministic, ít hallucinate |
| **Streaming** | Bật | Xem câu trả lời xuất hiện dần |

### 9.3 Ví dụ câu hỏi thử nghiệm

Sau khi upload PDF học thuật/kỹ thuật, thử các câu hỏi:

```
# Câu hỏi ngữ nghĩa
"Phương pháp đề xuất trong bài có điểm gì khác biệt so với các phương pháp trước đó?"

# Câu hỏi cụ thể (tốt cho hybrid retrieval)
"Bảng 3 so sánh kết quả trên dataset nào?"

# Câu hỏi yêu cầu tổng hợp nhiều phần
"Tóm tắt các đóng góp chính của bài báo này?"

# Câu hỏi về công thức / số liệu
"Loss function được định nghĩa như thế nào trong bài?"
```

---

## 10. Pipeline Cache

App có hệ thống **step-level cache** để tránh chạy lại các bước tốn kém:

```
processed_data/
  <input_hash>/
    loader/<loader_fingerprint>/      # Documents đã parse
    chunking/<chunk_fingerprint>/     # Chunks đã cắt
    embedding/<embed_fingerprint>/    # Embedding vectors
    vector_db/<vdb_fingerprint>/      # Vector DB metadata + pipeline_chain.json
```

**Cơ chế Fingerprint Chain:**

```
input_hash    = SHA256(file_contents)
loader_key    = SHA256(input_hash + loader_config)
chunk_key     = SHA256(loader_key + chunking_config)
embed_key     = SHA256(chunk_key + embedding_config)
vdb_key       = SHA256(embed_key + vdb_config)
```

Điều này đảm bảo:
- Chỉ đổi **retrieval strategy** → không cần chạy lại bất kỳ bước nào ở Stage 1.
- Đổi **embedding model** → chạy lại từ bước Embedding trở đi.
- Thêm **file mới** → chạy lại tất cả.
- Cùng config → reuse 100%, không tốn API call.

**Quản lý cache:** Mở expander **"🗄️ Pipeline Cache"** ở cuối sidebar → xem danh sách cache, xoá cache cụ thể hoặc xoá tất cả.

---

## 11. config.yaml

File `config.yaml` chứa cấu hình mặc định. Thay đổi tại đây sẽ được áp dụng khi app khởi động lần đầu (nếu chưa có session state), hoặc khi gọi pipeline từ code (không qua UI).

Ví dụ cấu hình cho demo scenario (Section 9):

```yaml
data:
  input_dir: ./data
  language: both

indexing:
  loader:
    pdf_strategy: marker
    ocr_engine: none
    extract_tables: true
    extract_images: false

  chunking:
    strategy: format_aware
    chunk_size: 1000
    chunk_overlap: 150
    format_type: auto

  embedding:
    provider: openai
    model_name: text-embedding-3-small
    enable_sparse: true
    sparse_method: bm25

  vector_db:
    provider: chroma
    collection_name: rag_docs
    persist_dir: ./storage

query_pipeline:
  pre_retrieval:
    transformations: [rewrite]
    transformation_llm: gpt-4.1-mini

  retrieval:
    strategy: hybrid
    top_k: 15
    fusion_method: rrf

  post_retrieval:
    reranker: cross_encoder
    cross_encoder_model: BAAI/bge-reranker-v2-m3
    top_n: 5
    apply_redundancy: true
    redundancy_threshold: 0.92
    context_ordering: sandwich

  prompt:
    template: citation
    validate_citations: true

  generation:
    provider: openai
    model_name: gpt-4.1-mini
    temperature: 0.0
    max_tokens: 2048
    streaming: true
```

---

## 12. Troubleshooting

### `ModuleNotFoundError: No module named 'marker'`

```bash
pip install marker-pdf==1.6.2
```

Marker cần tải model (~1.5 GB) lần đầu chạy — đảm bảo có kết nối internet.

---

### `OPENAI_API_KEY not found` hoặc `AuthenticationError`

1. Kiểm tra file `.env` tồn tại ở thư mục gốc project.
2. Đảm bảo format đúng: `OPENAI_API_KEY=sk-proj-...` *(không có dấu nháy)*.
3. Đảm bảo chạy `streamlit run app.py` từ đúng thư mục gốc.

---

### Ollama: `Connection refused` hoặc `ConnectError`

```bash
# Khởi động Ollama server
ollama serve

# Kiểm tra các model đã có
ollama list

# Pull model nếu chưa có
ollama pull qwen2.5:7b
```

---

### `faiss` import error trên macOS (Apple Silicon)

```bash
pip uninstall faiss-cpu
pip install faiss-cpu==1.9.0.post1 --no-cache-dir
```

Nếu vẫn lỗi, thử cài qua conda:

```bash
conda install -c conda-forge faiss-cpu
```

---

### Chunking tạo ra quá nhiều chunk ngắn (< 50 ký tự)

- Tăng `chunk_size` lên 1000–1500.
- Kiểm tra loader: PDF có nhiều header/footer thừa không? Thử dùng `marker` hoặc `docling` để output Markdown sạch hơn.
- Bật option **"Bỏ qua chunk quá ngắn"** trong cài đặt chunking.

---

### Cosine similarity heatmap toàn màu nhạt (similarity thấp)

Embedding model không phù hợp với ngôn ngữ trong tài liệu. Thử đổi sang:

- `BAAI/bge-m3` (HuggingFace, multilingual)
- `embed-multilingual-v3.0` (Cohere)
- `text-embedding-3-large` (OpenAI, chất lượng cao hơn 3-small)

---

### Câu trả lời bị hallucinate (bịa thông tin không có trong tài liệu)

1. Mở tab **Retrieval** — kiểm tra xem các chunk retrieved có thực sự liên quan không.
2. Tăng `top_k` và bật **cross-encoder reranker**.
3. Đổi prompt template sang `citation` — LLM phải trích dẫn `[NGUỒN N]` cho mỗi claim.
4. Đặt `temperature = 0.0`.
5. Thêm rule vào system prompt: `"Chỉ trả lời dựa trên các đoạn context được cung cấp. Nếu không tìm thấy thông tin, hãy nói rõ."`

---

### Lỗi `torch` khi cài marker-pdf trên Windows

Marker cần `torch`. Cài PyTorch trước theo hướng dẫn chính thức:

```bash
# CPU only (không có GPU NVIDIA)
pip install torch==2.5.1+cpu -f https://download.pytorch.org/whl/torch_stable.html

# Sau đó cài marker
pip install marker-pdf==1.6.2
```

---

### Streamlit báo lỗi upload file quá lớn

```bash
streamlit run app.py --server.maxUploadSize 500
```

Hoặc tạo file `.streamlit/config.toml`:

```toml
[server]
maxUploadSize = 500
```

---

## 📄 License

MIT License — xem file [LICENSE](LICENSE) để biết chi tiết.

---

## 🤝 Contributing

Pull requests welcome! Vui lòng mở Issue trước khi thực hiện thay đổi lớn để thảo luận hướng đi.

---

*Built with ❤️ using Streamlit, LangChain, and the RAG community's best practices.*
