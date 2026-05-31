# 🔍 RAG Pipeline Visualizer

**RAG Pipeline Visualizer** là một ứng dụng Streamlit giúp bạn xây dựng, cấu hình và trực quan hóa từng bước trong pipeline RAG (Retrieval-Augmented Generation) — từ bước load tài liệu PDF đến bước sinh câu trả lời từ LLM.

> **Mục tiêu:** Giúp người dùng hiểu rõ tác động của từng cấu hình (chunking strategy, embedding model, vector DB, retrieval strategy, ...) đến chất lượng câu trả lời cuối cùng.

---

## 📋 Mục lục

1. [Tổng quan Pipeline](#1-tổng-quan-pipeline)
2. [Cấu trúc Project](#2-cấu-trúc-project)
3. [Yêu cầu hệ thống](#3-yêu-cầu-hệ-thống)
4. [Tạo môi trường & Cài đặt](#4-tạo-môi-trường--cài-đặt)
5. [Cấu hình API Keys](#5-cấu-hình-api-keys)
6. [Khởi động ứng dụng](#6-khởi-động-ứng-dụng)
7. [Hướng dẫn sử dụng chi tiết](#7-hướng-dẫn-sử-dụng-chi-tiết)
   - [Bước 0 — Chọn tài liệu nguồn](#bước-0--chọn-tài-liệu-nguồn)
   - [Bước 1 — Loader](#bước-1--loader-pdf)
   - [Bước 2 — Chunking](#bước-2--chunking)
   - [Bước 3 — Embedding](#bước-3--embedding)
   - [Bước 4 — Vector DB](#bước-4--vector-db)
   - [Bước 5 — Process](#bước-5--process)
   - [Bước 6 — Xem kết quả Indexing](#bước-6--xem-kết-quả-indexing)
   - [Bước 7 — Pre-retrieval](#bước-7--pre-retrieval-tuỳ-chọn)
   - [Bước 8 — Retrieval](#bước-8--retrieval)
   - [Bước 9 — Post-retrieval](#bước-9--post-retrieval-tuỳ-chọn)
   - [Bước 10 — Prompt](#bước-10--prompt)
   - [Bước 11 — Generation](#bước-11--generation)
   - [Bước 12 — Chạy Query](#bước-12--chạy-query)
8. [Demo End-to-End: PDF phức tạp → Câu trả lời có trích dẫn](#8-demo-end-to-end)
9. [Pipeline Cache](#9-pipeline-cache)
10. [config.yaml](#10-configyaml)
11. [Troubleshooting](#11-troubleshooting)

---

## 1. Tổng quan Pipeline

```
[File PDF] ──► [1. Loader] ──► [2. Chunking] ──► [3. Embedding] ──► [4. Vector DB]
                                                                            │
                                                                            ▼
[Câu trả lời] ◄── [11. Generation] ◄── [10. Prompt] ◄── [9. Post-retrieval] ◄── [8. Retrieval] ◄── [7. Pre-retrieval] ◄── [Query]
```

| Bước | Tên | Mô tả | Bắt buộc? |
|------|-----|--------|-----------|
| 1 | **Loader** | Parse PDF → `list[Document]` | ✅ |
| 2 | **Chunking** | Cắt Document thành chunk nhỏ | ✅ |
| 3 | **Embedding** | Embed chunk → dense/sparse vectors | ✅ |
| 4 | **Vector DB** | Lưu vectors + index | ✅ |
| 5–6 | **Process / Results** | Chạy indexing và xem kết quả | ✅ |
| 7 | **Pre-retrieval** | Biến đổi query trước khi tìm kiếm | ❌ Tuỳ chọn |
| 8 | **Retrieval** | Tìm kiếm chunk liên quan | ✅ |
| 9 | **Post-retrieval** | Rerank, filter, compress | ❌ Tuỳ chọn |
| 10 | **Prompt** | Xây dựng prompt đưa vào LLM | ✅ |
| 11 | **Generation** | LLM sinh câu trả lời cuối cùng | ✅ |

---

## 2. Cấu trúc Project

```
RefinedRAG/
├── app_visualizer.py          # Script chính — chạy bằng streamlit run
├── pipeline_cache.py          # Step-level disk cache
├── config.yaml                # Cấu hình mặc định cho toàn bộ pipeline
├── requirements.txt           # Dependencies
├── .env                       # API keys (tạo thủ công, KHÔNG commit)
│
├── loader/                    # Bước 1: Load PDF
│   ├── __init__.py
│   ├── base.py
│   ├── directory_loader.py    # PDFDocumentLoader (điểm vào chính)
│   ├── pdf_loader.py          # 7 PDF loader class
│   └── utils.py
│
├── chunking/                  # Bước 2: Chunking
│   ├── __init__.py
│   ├── base.py
│   ├── factory.py
│   ├── recursive.py
│   ├── token_based.py
│   ├── format_aware.py
│   ├── sentence_aware.py
│   ├── semantic.py
│   ├── hierarchical.py
│   ├── contextual.py
│   ├── deduplication.py
│   └── utils.py
│
├── embedding/                 # Bước 3: Embedding
│   ├── __init__.py
│   ├── base.py
│   ├── factory.py
│   └── utils.py
│
├── vector_db/                 # Bước 4: Vector Store
│   ├── __init__.py
│   ├── base.py
│   ├── factory.py
│   └── utils.py
│
├── pre_retrieval/             # Bước 7: Query Transformation
│   ├── __init__.py
│   ├── base.py
│   ├── pipeline.py
│   └── factory.py
│
├── retrieval/                 # Bước 8: Retrieval
│   ├── __init__.py
│   ├── base.py
│   └── factory.py
│
├── post_retrieval/            # Bước 9: Post-processing
│   ├── __init__.py
│   ├── base.py
│   ├── pipeline.py
│   └── factory.py
│
├── prompt/                    # Bước 10: Prompt Builder
│   ├── __init__.py
│   ├── base.py
│   ├── factory.py
│   ├── basic.py
│   ├── citation.py
│   ├── conversational.py
│   └── structured_output.py
│
└── generation/                # Bước 11: LLM Generation
    ├── __init__.py
    ├── base.py
    ├── factory.py
    ├── openai_generator.py
    ├── anthropic_generator.py
    ├── google_generator.py
    ├── ollama_generator.py
    └── cohere_generator.py
```

---

## 3. Yêu cầu hệ thống

| Yêu cầu | Tối thiểu | Khuyến nghị |
|---------|-----------|-------------|
| **Python** | 3.10 | 3.11 / 3.12 |
| **RAM** | 4 GB | 8 GB+ |
| **Disk** | 2 GB | 5 GB+ (models cache) |
| **GPU** | Không bắt buộc | NVIDIA CUDA (tăng tốc embedding/reranker) |
| **OS** | Windows 10 / macOS 12 / Ubuntu 20.04 | Bất kỳ |
| **Internet** | Cần để gọi API (OpenAI, Anthropic, ...) | — |

> 💡 Nếu dùng **Ollama** để chạy LLM local, khuyến nghị RAM ≥ 8 GB và có SSD.

---

## 4. Tạo môi trường & Cài đặt

### Bước 4.1 — Clone repository

```bash
git clone https://github.com/<your-username>/RefinedRAG.git
cd RefinedRAG
```

### Bước 4.2 — Tạo virtual environment

**Dùng `venv` (built-in):**

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

**Hoặc dùng `conda`:**

```bash
conda create -n rag_visualizer python=3.11 -y
conda activate rag_visualizer
```

### Bước 4.3 — Cài đặt dependencies bắt buộc

```bash
pip install -r requirements.txt
```

> ⏱️ Lần đầu mất 3–5 phút tuỳ tốc độ mạng.

### Bước 4.4 — Cài đặt thêm tuỳ strategy (tuỳ chọn)

**Nếu muốn dùng Marker PDF** (khuyến nghị cho PDF phức tạp có công thức, bảng):

```bash
pip install marker-pdf==1.6.2
```

**Nếu muốn dùng Docling:**

```bash
pip install docling==2.28.0
```

**Nếu muốn dùng Anthropic Claude:**

```bash
pip install anthropic==0.43.0
```

**Nếu muốn dùng Google Gemini:**

```bash
pip install google-generativeai==0.8.3 langchain-google-genai==2.0.11
```

**Nếu muốn dùng Ollama (LLM local):**

```bash
# 1. Cài Ollama server: https://ollama.com/download
# 2. Cài Python client:
pip install ollama==0.4.7 langchain-ollama==0.2.3

# 3. Pull model (ví dụ):
ollama pull qwen2.5:7b      # Khuyến nghị — tiếng Việt OK, ~4.7 GB
ollama pull llama3.2:3b     # Nhẹ nhất — ~2 GB
```

**Nếu muốn dùng Qdrant:**

```bash
pip install qdrant-client==1.13.3 langchain-qdrant==0.2.0
```

---

## 5. Cấu hình API Keys

Tạo file `.env` ở thư mục gốc của project:

```bash
touch .env      # macOS/Linux
# hoặc: New-Item .env -ItemType File  # Windows PowerShell
```

Mở `.env` và điền API keys cần thiết:

```dotenv
# OpenAI (dùng cho embedding + generation)
OPENAI_API_KEY=sk-proj-...

# Anthropic Claude
# ANTHROPIC_API_KEY=sk-ant-...

# Google Gemini
# GOOGLE_API_KEY=AIza...

# Cohere
# COHERE_API_KEY=...

# Qdrant Cloud (tuỳ chọn — để trống nếu dùng local)
# QDRANT_URL=https://...
# QDRANT_API_KEY=...

# Pinecone
# PINECONE_API_KEY=...
```

> ⚠️ **Quan trọng:** Thêm `.env` vào `.gitignore` để tránh lộ API key khi push lên GitHub.

```bash
echo ".env" >> .gitignore
```

---

## 6. Khởi động ứng dụng

```bash
# Đảm bảo đang ở thư mục gốc project và đã activate venv
streamlit run app_visualizer.py
```

Mặc định app chạy tại: **http://localhost:8501**

Nếu port 8501 đã bị chiếm:

```bash
streamlit run app_visualizer.py --server.port 8502
```

![Màn hình khởi động](docs/screenshots/00_launch.png)

---

## 7. Hướng dẫn sử dụng chi tiết

Giao diện app gồm hai phần:
- **Sidebar trái:** Tất cả cài đặt cấu hình (Bước 1–11)
- **Vùng chính phải:** Kết quả trực quan hóa từng bước

---

### Bước 0 — Chọn tài liệu nguồn

Ở đầu sidebar, bạn có hai cách nhập tài liệu:

**Cách 1 — Upload file trực tiếp:** Nhấn vào ô **"Kéo thả hoặc browse file PDF"** → chọn 1 hoặc nhiều file PDF.

**Cách 2 — Nhập đường dẫn thư mục:** Điền đường dẫn tuyệt đối đến thư mục chứa PDF. App sẽ tự động tìm tất cả file `.pdf` trong thư mục đó (bao gồm cả subfolder).

![Chọn tài liệu](docs/screenshots/01_upload.png)

> 💡 **Upload** phù hợp khi cần xử lý 1–5 file. **Thư mục** khi có nhiều file hoặc muốn chạy lại với dataset cố định.

---

### Bước 1 — Loader (PDF)

Mở expander **"1️⃣ Loader"** trong sidebar.

![Loader settings](docs/screenshots/02_loader_settings.png)

#### Các strategy và khi nào chọn:

| Strategy | Tốc độ | Bảng | Công thức | Khi nào dùng |
|----------|--------|------|-----------|--------------|
| `pypdf` | ⚡⚡⚡ Rất nhanh | ❌ | ❌ | Prototype nhanh, PDF text thuần |
| `pymupdf` | ⚡⚡⚡ Rất nhanh | ⚠️ Cơ bản | ❌ | PDF layout đơn giản |
| `pdfplumber` | ⚡⚡ Nhanh | ✅ Tốt nhất | ❌ | PDF có nhiều bảng dạng text |
| `marker` ⭐ | ⚡ Chậm | ✅✅ Rất tốt | ✅ LaTeX | **PDF phức tạp: bảng, công thức, hình** |
| `docling` | ⚡ Chậm | ✅✅ Rất tốt | ✅ | Tài liệu học thuật, báo cáo phức tạp |
| `unstructured` | ⚡ Chậm | ✅✅ Rất tốt | ✅ OCR | PDF scan, hình ảnh cần OCR |

**Kết quả hiển thị (tab "📄 Loader"):**

![Loader results](docs/screenshots/03_loader_results.png)

---

### Bước 2 — Chunking

Mở expander **"2️⃣ Chunking"** trong sidebar.

![Chunking settings](docs/screenshots/04_chunking_settings.png)

#### Các strategy và khi nào chọn:

| Strategy | Cơ chế | Best for |
|----------|--------|---------|
| `recursive` ⭐ | Đệ quy: đoạn→dòng→câu→ký tự | Mặc định tốt nhất |
| `token_based` | Đếm BPE token | Khi embedding model có token limit chặt |
| `format_aware` | Nhận diện Markdown heading, code block | **PDF qua Marker/Docling** |
| `sentence_aware` | Ranh giới câu (NLTK) | Q&A, FAQ, văn bản văn học |
| `semantic` | Cosine similarity | PDF nhiều chủ đề khác nhau |
| `hierarchical` | Parent (lớn) + Child (nhỏ) | Corpus lớn, cần độ chính xác cao |
| `contextual` | Recursive + LLM prefix | Production, có LLM budget |

- **Chunk size:** Số ký tự tối đa mỗi chunk. Thường 512–1500.
- **Chunk overlap:** Số ký tự chồng lắp giữa 2 chunk liên tiếp. Thường 10–15% của chunk size.

**Kết quả hiển thị (tab "✂️ Chunking"):**

![Chunking results](docs/screenshots/05_chunking_results.png)

---

### Bước 3 — Embedding

Mở expander **"3️⃣ Embedding"** trong sidebar.

![Embedding settings](docs/screenshots/06_embedding_settings.png)

#### Các provider và model:

| Provider | Model | Tiếng Việt | Chi phí | Best for |
|----------|-------|------------|---------|---------|
| **OpenAI** | `text-embedding-3-small` ⭐ | ⭐⭐⭐ | $0.02/1M | Mặc định tốt nhất |
| **OpenAI** | `text-embedding-3-large` | ⭐⭐⭐ | $0.13/1M | Độ chính xác cao nhất |
| **Cohere** | `embed-multilingual-v3.0` | ⭐⭐⭐⭐ | $0.10/1M | Corpus đa ngôn ngữ |
| **HuggingFace** | `BAAI/bge-m3` | ⭐⭐⭐⭐ | Miễn phí | Local, VI tốt |
| **HuggingFace** | `Qwen/Qwen3-Embedding` | ⭐⭐⭐⭐⭐ | Miễn phí | MTEB #1, VI tốt nhất |
| **FastEmbed** | `multilingual-e5-small` | ⭐⭐ | Miễn phí | CPU, không GPU |
| **Ollama** | `bge-m3` | ⭐⭐⭐⭐ | Miễn phí | Local server, privacy |

**Sparse Embedding (Hybrid Retrieval):**
- `BM25` — Lexical matching, nhanh, không cần GPU. Khuyến nghị bật cho corpus tiếng Việt.
- `SPLADE` — Sparse neural, hiệu quả hơn BM25, cần torch.

**Kết quả hiển thị (tab "🧮 Embedding"):**

![Embedding results](docs/screenshots/07_embedding_results.png)

---

### Bước 4 — Vector DB

Mở expander **"4️⃣ Vector DB"** trong sidebar.

![Vector DB settings](docs/screenshots/08_vdb_settings.png)

#### Các Vector DB và khi nào chọn:

| Store | Scale | Hybrid | Best for |
|-------|-------|--------|---------|
| **Chroma** ⭐ | < 1M | ✅ | Demo, dev, small corpus |
| **FAISS** | < 100M | ❌ | Prototype cực nhanh |
| **Qdrant** | 1B+ | ✅ ACORN | Production, filtering phức tạp |
| **LanceDB** | < 1B | ✅ | Embedded, không cần server |
| **Weaviate** | ~1B | ✅ | Hybrid out-of-the-box |
| **PGVector** | < 100M | ✅ | Đã có PostgreSQL infrastructure |
| **Pinecone** | ~1B | ✅ | Zero ops, startup |

---

### Bước 5 — Process

Sau khi cấu hình xong, nhấn nút **▶️ Process** ở cuối sidebar.

![Process button](docs/screenshots/09_process_button.png)

App chạy tuần tự 4 bước indexing với progress bar. Nút **▶️ Process** tự động bị **disabled** trong khi chạy. Nhấn **🛑 Stop** để dừng bất cứ lúc nào. Kết quả mỗi bước được **cache trên disk** — chạy lại với cùng config sẽ reuse kết quả cũ.

---

### Bước 6 — Xem kết quả Indexing

#### Tab "📄 Loader"

![Tab Loader](docs/screenshots/10_tab_loader.png)

Số document, tổng ký tự, bảng thống kê theo file, nội dung từng document.

#### Tab "✂️ Chunking"

![Tab Chunking](docs/screenshots/11_tab_chunking.png)

Histogram phân phối chunk size, metrics min/max/mean, xem từng chunk với metadata.

#### Tab "🧮 Embedding"

![Tab Embedding](docs/screenshots/12_tab_embedding.png)

Embedding matrix shape, cosine similarity heatmap, nearest neighbors của từng chunk.

#### Tab "🗃️ Vector DB"

![Tab VectorDB](docs/screenshots/13_tab_vdb.png)

Thông tin collection, persist path, sparse index info.

---

### Bước 7 — Pre-retrieval *(tuỳ chọn)*

Mở expander **"7️⃣ Pre-retrieval"** trong sidebar.

![Pre-retrieval settings](docs/screenshots/14_preretrieval_settings.png)

| Strategy | Cơ chế | Khi nào dùng |
|----------|--------|-------------|
| `none` ⭐ | Giữ nguyên query | Query đã rõ ràng |
| `rewrite` | LLM viết lại chuẩn hơn | Query ngắn, thiếu context |
| `expand` | Thêm từ đồng nghĩa | Corpus có nhiều thuật ngữ tương đương |
| `step_back` | Tổng quát hóa query | Query quá cụ thể |
| `multi_query` | Tạo N query con → merge (RRF) | Query phức tạp, nhiều khía cạnh |
| `decompose` | Chia sub-query độc lập | Multi-hop reasoning |
| `self_query` | Parse → metadata filter + semantic | Corpus có metadata phong phú |
| `route` | Phân loại → chọn strategy phù hợp | Corpus đa dạng |

---

### Bước 8 — Retrieval

Mở expander **"8️⃣ Retrieval"** trong sidebar.

![Retrieval settings](docs/screenshots/15_retrieval_settings.png)

| Strategy | Cơ chế | Khi nào dùng |
|----------|--------|-------------|
| `hybrid` ⭐ | Dense + Sparse RRF | Mặc định tốt nhất khi bật sparse |
| `dense` | Cosine similarity | Câu hỏi ngữ nghĩa |
| `sparse` | BM25 / SPLADE | Từ khoá, tên riêng, số liệu |
| `multi_query` | N phiên bản query → merge | Query mơ hồ |
| `parent_document` | Retrieve child, return parent | Cần nhiều context hơn |
| `sentence_window` | N câu xung quanh câu hit | Corpus văn xuôi |
| `multi_hop` | Multi-step reasoning | Multi-hop Q&A |

**Top-K:** Số chunk trả về. Thường 10–20 trước reranking.

---

### Bước 9 — Post-retrieval *(tuỳ chọn)*

Mở expander **"9️⃣ Post-retrieval"** trong sidebar.

![Post-retrieval settings](docs/screenshots/16_postretrieval_settings.png)

| Reranker | Khi nào dùng |
|----------|-------------|
| `none` | Retrieval đã đủ tốt, muốn latency thấp |
| `cross_encoder` ⭐ | Tốt nhất VI, không cần GPU (BAAI/bge-reranker-v2-m3) |
| `cohere` | Best API quality, 100+ ngôn ngữ |
| `llm` | Không có reranker model |

**Pipeline order:** MetadataFilter → RedundancyFilter → Reranker → LLMFilter → MMRFilter → Compressor → Orderer

**Context ordering `sandwich` ⭐:** Most relevant ở đầu và cuối, ít relevant ở giữa — giảm lost-in-the-middle hiệu quả nhất.

---

### Bước 10 — Prompt

Mở expander **"🔟 Prompt"** trong sidebar.

![Prompt settings](docs/screenshots/17_prompt_settings.png)

| Template | Output | Khi nào dùng |
|----------|--------|-------------|
| `citation` ⭐ | Text + [NGUỒN N] | Production, cần verify fact |
| `basic` | Plain text | Prototype nhanh |
| `conversational` | Text + lịch sử | Chatbot, follow-up questions |
| `structured` | JSON (claims+sources+confidence) | Downstream code cần parse |

---

### Bước 11 — Generation

Mở expander **"1️⃣1️⃣ Generation"** trong sidebar.

![Generation settings](docs/screenshots/18_generation_settings.png)

| Tình huống | Provider | Model |
|-----------|---------|-------|
| 🏆 Chất lượng + Tiết kiệm | OpenAI | `gpt-4.1-mini` ($0.40/$1.60 per 1M) |
| 💰 Rẻ nhất (API) | OpenAI | `gpt-4o-mini` ($0.15/$0.60 per 1M) |
| 🇻🇳 Tiếng Việt tốt | Anthropic | `claude-haiku-4-5` ($0.80/$4 per 1M) |
| 🆓 Miễn phí | Google | `gemini-2.0-flash` (free tier) |
| 🔒 Offline / Privacy | Ollama | `qwen2.5:7b` (~4.7 GB RAM) |
| 🪶 Máy yếu, offline | Ollama | `llama3.2:3b` (~2 GB RAM) |

**Temperature:** `0.0` cho RAG (deterministic, ít hallucinate). **Streaming:** Bật để xem câu trả lời xuất hiện dần.

---

### Bước 12 — Chạy Query

Chuyển sang tab **"🔎 Query Pipeline"** ở vùng chính.

![Query tab](docs/screenshots/19_query_tab.png)

Nhập câu hỏi và nhấn **▶️ Chạy**. App hiển thị kết quả từng bước:

**Pre-retrieval** — query gốc vs. query đã biến đổi:

![Pre-retrieval results](docs/screenshots/20_preretrieval_results.png)

**Retrieval** — top-K chunk với similarity score:

![Retrieval results](docs/screenshots/21_retrieval_results.png)

**Post-retrieval** — thứ tự sau reranking so với trước:

![Post-retrieval results](docs/screenshots/22_postretrieval_results.png)

**Prompt** — xem đầy đủ system message và user message gửi lên LLM:

![Prompt display](docs/screenshots/23_prompt_display.png)

**Generation** — câu trả lời streaming với inline citations, token usage, và đoạn nguồn tương ứng:

![Generation output](docs/screenshots/24_generation_output.png)

---

## 8. Demo End-to-End

**Kịch bản:** Upload PDF phức tạp (có text, bảng, công thức, hình ảnh) → hỏi đáp với trích dẫn nguồn.

### Cấu hình khuyến nghị:

| Bước | Setting |
|------|---------|
| **File** | Upload 1–3 file PDF phức tạp |
| **Loader** | `marker` — Markdown chất lượng cao |
| **Chunking** | `format_aware` — nhận diện Markdown heading từ Marker output |
| **Embedding** | OpenAI `text-embedding-3-small` + Bật BM25 sparse |
| **Vector DB** | `chroma` — không cần config, persist tự động |
| **Retrieval** | `hybrid` + Top-K = 15 |
| **Post-retrieval** | `cross_encoder` + Top-N = 5 + Ordering = `sandwich` |
| **Prompt** | `citation` |
| **Generation** | OpenAI `gpt-4.1-mini` + Temperature = 0 + Streaming = On |

### Kết quả sau khi Process:

![Demo process done](docs/screenshots/25_demo_process.png)

### Câu trả lời với trích dẫn nguồn:

![Demo answer](docs/screenshots/26_demo_answer.png)

---

## 9. Pipeline Cache

App có hệ thống **step-level cache** để tránh chạy lại các bước tốn kém:

```
processed_data/
  <input_hash>/
    loader/<loader_key>/    # Cache documents đã load
    chunking/<chunk_key>/   # Cache chunks
    embedding/<embed_key>/  # Cache embedding vectors
    vector_db/<vdb_key>/    # Cache metadata Vector DB
```

**Cơ chế fingerprint chain:** `step_key = SHA256(prev_key + config)`. Chỉ đổi retrieval strategy → không phải chạy lại loading/chunking/embedding. Đổi embedding model → chạy lại từ bước embedding. Thêm file mới → chạy lại tất cả.

**Quản lý cache:** Mở expander **"🗄️ Pipeline Cache"** ở cuối sidebar.

---

## 10. config.yaml

File `config.yaml` chứa cấu hình mặc định cho toàn bộ pipeline khi gọi từ code (không qua UI):

```yaml
data:
  language: both

loader:
  pdf_strategy: marker

chunking:
  strategy: format_aware
  chunk_size: 1000
  chunk_overlap: 150

embedding:
  provider: openai
  model_name: text-embedding-3-small
  sparse_method: bm25

vector_db:
  provider: chroma
  collection_name: rag_docs
  persist_directory: ./storage/chroma

query_pipeline:
  pre_retrieval:
    transformations: ["none"]
  retrieval:
    strategy: hybrid
    top_k: 15
  post_retrieval:
    reranker: cross_encoder
    top_n: 5
    context_ordering: sandwich
  prompt:
    template: citation
    language: both
  generation:
    provider: openai
    model_name: gpt-4.1-mini
    temperature: 0.0
    max_tokens: 2048
    streaming: true
```

---

## 11. Troubleshooting

**`ModuleNotFoundError: No module named 'marker'`**
```bash
pip install marker-pdf==1.6.2
```

**`OPENAI_API_KEY not found`**

Kiểm tra file `.env` tồn tại và có nội dung đúng. Đảm bảo app chạy từ đúng thư mục chứa `.env`.

**Ollama: `Connection refused`**
```bash
ollama serve
ollama list
ollama pull qwen2.5:7b
```

**Chunking quá nhiều chunk ngắn (<50 chars)**

Tăng `chunk_size` hoặc bật **"Bỏ qua chunk quá ngắn"**. Kiểm tra xem PDF có nhiều header/footer bị trích xuất không.

**Cosine heatmap toàn màu nhạt (similarity thấp)**

Đổi sang model đa ngôn ngữ (`BAAI/bge-m3`, `embed-multilingual-v3.0`).

**Câu trả lời bị hallucinate**

1. Kiểm tra tab Retrieval — chunk retrieved có thực sự liên quan không?
2. Tăng Top-K và bật cross-encoder reranker
3. Đổi prompt template sang `citation`
4. Đặt temperature = 0.0

---

## 📄 License

MIT License — xem file [LICENSE](LICENSE) để biết chi tiết.

---

## 🤝 Contributing

Pull requests welcome! Vui lòng mở Issue trước khi làm thay đổi lớn.

---

*Built with ❤️ using Streamlit, LangChain, and the RAG community's best practices.*
