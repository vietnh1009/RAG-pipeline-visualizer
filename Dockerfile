# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  RAG-pipeline-visualizer — Dockerfile                                    ║
# ║                                                                          ║
# ║  Vì sao Docker là phương án tốt nhất cho repo này (không chỉ vì pin lib): ║
# ║    · Khoá luôn Python 3.11 — requirements.txt KHÔNG làm được điều này    ║
# ║    · Cài sẵn Java 17  → opendataloader chạy được ngay                    ║
# ║    · Cài sẵn poppler + tesseract (kèm tiếng Việt) → unstructured OCR OK  ║
# ║    · Cài sẵn NLTK punkt → sentence_aware không cần tải lúc runtime       ║
# ║                                                                          ║
# ║  Build mặc định (nhẹ, ~2 GB):                                            ║
# ║      docker build -t rag-viz .                                           ║
# ║  Build kèm loader nâng cao (nặng, ~8 GB, có model AI):                   ║
# ║      docker build -t rag-viz --build-arg PROFILE=full .                  ║
# ╚══════════════════════════════════════════════════════════════════════════╝

FROM python:3.11-slim-bookworm AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive \
    # model HuggingFace/Marker cache vào volume, không nằm trong image
    HF_HOME=/cache/huggingface \
    TORCH_HOME=/cache/torch \
    NLTK_DATA=/usr/share/nltk_data \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# ── Gói hệ thống ─────────────────────────────────────────────────────────────
#  openjdk-17-jre-headless : BẮT BUỘC cho loader `opendataloader`
#  poppler-utils           : pdf2image, dùng bởi unstructured
#  tesseract-ocr(-vie/eng) : OCR cho PDF scan
#  libgl1 / libglib2.0-0   : opencv (kéo theo bởi unstructured, docling)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        openjdk-17-jre-headless \
        poppler-utils \
        tesseract-ocr tesseract-ocr-vie tesseract-ocr-eng \
        libgl1 libglib2.0-0 \
        curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Python deps: copy riêng để tận dụng layer cache ──────────────────────────
COPY requirements.txt requirements-extra.txt ./

# Nếu có requirements.lock.txt (sinh bằng scripts/freeze_lock.py) thì ưu tiên
# dùng nó — tái lập 100% kể cả dependency gián tiếp.
COPY requirements.lock.tx[t] ./
RUN python -m pip install --upgrade pip setuptools wheel \
 && if [ -f requirements.lock.txt ]; then \
        echo ">> Cài từ requirements.lock.txt (tái lập chính xác)"; \
        pip install -r requirements.lock.txt; \
    else \
        echo ">> Cài từ requirements.txt"; \
        pip install -r requirements.txt; \
    fi

# ── Tải sẵn NLTK data — tránh gọi nltk.download() lúc runtime ────────────────
# Quan trọng: chunking/sentence_aware.py gọi nltk.download("punkt_tab") khi
# thiếu data. Trong container không có mạng, lệnh đó sẽ ném LookupError.
RUN python -c "import nltk; nltk.download('punkt', download_dir='/usr/share/nltk_data', quiet=True); nltk.download('punkt_tab', download_dir='/usr/share/nltk_data', quiet=True); nltk.download('wordnet', download_dir='/usr/share/nltk_data', quiet=True)"

# ── Profile: base (mặc định) | full (loader nâng cao + provider khác) ────────
ARG PROFILE=base
RUN if [ "$PROFILE" = "full" ]; then \
      pip install \
        "docling==2.28.0" \
        "unstructured[pdf]==0.16.12" \
        "opendataloader-pdf" \
        "cohere==5.13.12" "langchain-cohere==0.4.3" \
        "ollama==0.4.7" "langchain-ollama==0.2.3" \
        "anthropic==0.43.0" \
        "google-generativeai==0.8.3" "langchain-google-genai==2.0.11" \
        "qdrant-client==1.13.3" "langchain-qdrant==0.2.0" \
        "lancedb==0.17.0" \
        "underthesea==6.8.4" ; \
    fi

# ── PyTorch bản GPU (tuỳ chọn) ──────────────────────────────────────────────
# KHÔNG cần base image CUDA: wheel PyTorch đã đóng gói sẵn CUDA runtime bên
# trong. Thứ duy nhất cần từ host là driver NVIDIA + nvidia-container-toolkit,
# do runtime inject vào container.
#
#   docker build -t rag-viz --build-arg TORCH_CUDA=cu124 .
#   docker run --gpus all ...
#
# Chọn cuXXX nào? Chạy `nvidia-smi` trên HOST, xem dòng "CUDA Version",
# rồi lấy bản PyTorch <= con số đó. Hoặc: python scripts/install_torch.py
ARG TORCH_CUDA=""
RUN if [ -n "$TORCH_CUDA" ]; then \
      echo ">> Cài PyTorch GPU ($TORCH_CUDA) — ~2.5 GB"; \
      pip uninstall -y torch torchvision torchaudio 2>/dev/null || true; \
      pip install torch torchvision \
          --index-url "https://download.pytorch.org/whl/${TORCH_CUDA}"; \
    fi

# ── Source code (copy sau cùng: sửa code không phải cài lại lib) ─────────────
COPY . .

RUN mkdir -p /app/processed_data /app/storage /app/data /cache \
 && useradd -m -u 1000 raguser \
 && chown -R raguser:raguser /app /cache
USER raguser

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
  CMD curl -fsS http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.maxUploadSize=500"]
