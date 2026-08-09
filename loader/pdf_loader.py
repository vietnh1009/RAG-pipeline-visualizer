"""
loader/pdf_loader.py
====================
PDF loaders — một class cho mỗi chiến lược parsing.

| Class                   | Strategy        | Điểm mạnh                                    |
|-------------------------|-----------------|----------------------------------------------|
| PyPDFLoader             | pypdf           | Nhanh, text layer, không cần dep thêm        |
| PyMuPDFLoader           | pymupdf         | Nhanh nhất, layout tốt hơn, text layer       |
| PDFPlumberLoader        | pdfplumber      | Trích bảng tốt nhất (text layer)             |
| UnstructuredPDFLoader   | unstructured    | Tốt nhất: OCR + bảng + hình ảnh             |
| DoclingPDFLoader        | docling         | IBM parser, Markdown output xuất sắc         |
| MarkerPDFLoader         | marker          | Markdown chất lượng cao, bảng & công thức   |
| OpenDataLoaderPDFLoader | opendataloader  | #1 benchmark (0.90), bounding box, no GPU   |
"""

from __future__ import annotations

import contextlib
import logging
import os
import re
from pathlib import Path

from langchain_core.documents import Document

from loader.base import BaseLoader, Language
from loader.utils import clean_text, table_html_to_markdown

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Vision API helper — dùng chung cho OpenAI và Ollama
# ---------------------------------------------------------------------------
_VISION_PROMPT = (
    "This image is a figure from an academic/technical PDF. "
    "Describe its content concisely in 1-2 sentences, "
    "focusing on what is shown (chart, diagram, photo, table, etc.) "
    "and the key information it conveys."
)

def _ollama_ensure_model(model: str, base_url: str) -> None:
    """Kiểm tra model có trong Ollama chưa; tự pull về nếu chưa có (blocking)."""
    from openai import OpenAI
    client  = OpenAI(base_url=base_url, api_key="ollama")
    models  = [m.id for m in client.models.list().data]

    # Ollama trả model id dạng "qwen2-vl:7b" hoặc "qwen2-vl"
    available = {m.split(":")[0] for m in models} | set(models)
    tag       = model.split(":")[0]

    if model in available or tag in available:
        logger.info("Ollama model '%s' đã có sẵn.", model)
        return

    print(f"[Ollama] Model '{model}' chưa được pull. Đang tải về...")
    logger.info("Ollama: pulling model '%s' ...", model)

    # Dùng ollama Python SDK nếu có, fallback sang subprocess
    try:
        import ollama as _ollama
        # ollama.pull() stream progress
        for progress in _ollama.pull(model, stream=True):
            status   = getattr(progress, "status", "")
            completed = getattr(progress, "completed", None)
            total     = getattr(progress, "total", None)
            if total and completed:
                pct = completed / total * 100
                print(f"\r[Ollama] {status} {pct:.1f}%", end="", flush=True)
            elif status:
                print(f"[Ollama] {status}", flush=True)
        print()   # newline sau progress
        logger.info("Ollama: pull '%s' hoàn tất.", model)

    except ImportError:
        # Fallback: subprocess
        import subprocess
        logger.info("ollama SDK không có, dùng subprocess.")
        result = subprocess.run(
            ["ollama", "pull", model],
            capture_output=False,   # cho phép print trực tiếp ra terminal
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"ollama pull '{model}' thất bại (exit {result.returncode})")
        logger.info("Ollama: pull '%s' hoàn tất (subprocess).", model)


def _call_vision_api(
    image_b64:  str,
    mime:       str,
    model:      str,
    provider:   str = "openai",
    base_url:   str = "http://localhost:11434/v1",
    prompt:     str = _VISION_PROMPT,
    max_tokens: int = 200,
) -> str:
    """
    Gọi Vision API để mô tả ảnh. Hỗ trợ OpenAI và Ollama.

    provider = "openai" → dùng OPENAI_API_KEY từ env
    provider = "ollama" → gọi Ollama local (OpenAI-compatible endpoint),
                          tự pull model nếu chưa có
    """
    from openai import OpenAI

    if provider == "ollama":
        _ollama_ensure_model(model, base_url)   # auto-pull nếu cần
        client = OpenAI(base_url=base_url, api_key="ollama")
    else:
        client = OpenAI()   # đọc OPENAI_API_KEY từ env / .env

    response = client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url",
             "image_url": {"url": f"data:{mime};base64,{image_b64}"}},
        ]}],
    )
    return response.choices[0].message.content.strip()

# ---------------------------------------------------------------------------
# Thư mục cache ảnh của Marker — nằm trong project, không bị xóa khi app tắt.
# Đường dẫn tính từ vị trí file này (loader/pdf_loader.py) → lên 1 cấp = project root.
# Kết quả: <project_root>/.marker_cache/images/
# ---------------------------------------------------------------------------
MARKER_CACHE_DIR: Path = Path(__file__).resolve().parent.parent / ".marker_cache" / "images"

# Thư mục cache ảnh của Docling (tách riêng khỏi Marker)
# Kết quả: <project_root>/.docling_cache/images/
DOCLING_CACHE_DIR: Path = Path(__file__).resolve().parent.parent / ".docling_cache" / "images"

# ===========================================================================
# 1. PyPDF — đọc text layer, không cần thư viện thêm
# ===========================================================================

class PyPDFLoader(BaseLoader):
    """
    Text-layer PDF extraction via LangChain built-in PyPDFLoader.

    Không cần cài thêm gì ngoài pypdf (đã bao gồm trong langchain-community).
    Trả về 1 Document / trang. Không có OCR, không trích bảng.
    Dùng cho: PDF digital text thuần, khi ưu tiên tốc độ.
    """

    def load(self, file_path: str) -> list[Document]:
        from langchain_community.document_loaders import PyPDFLoader as _PyPDF

        docs = _PyPDF(file_path).load()
        for doc in docs:
            doc.page_content = clean_text(doc.page_content)
            doc.metadata.update({"file_type": "pdf", "pdf_strategy": "pypdf"})
        return self._stamp(docs)


# ===========================================================================
# 2. PyMuPDF — đọc text layer, nhanh nhất
# ===========================================================================

class PyMuPDFLoader(BaseLoader):
    """
    PDF extraction via PyMuPDF (fitz).

    Nhanh hơn PyPDF, giữ thứ tự đọc tốt hơn.
    Trả về 1 Document / trang. Không có OCR.
    Dùng cho: PDF text layer lớn, khi tốc độ là ưu tiên số 1.

    Cần cài: pip install pymupdf
    """

    def load(self, file_path: str) -> list[Document]:
        import fitz  # PyMuPDF

        pdf   = fitz.open(file_path)
        docs  = []
        total = len(pdf)

        for page_num, page in enumerate(pdf, start=1):
            text = clean_text(page.get_text("text"))
            if text:
                docs.append(Document(
                    page_content=text,
                    metadata={
                        "source": file_path, "page": page_num,
                        "total_pages": total, "file_type": "pdf",
                        "pdf_strategy": "pymupdf",
                    },
                ))
        pdf.close()
        return self._stamp(docs)


# ===========================================================================
# 3. pdfplumber — trích bảng tốt nhất trong nhóm đọc text layer
# ===========================================================================

class PDFPlumberLoader(BaseLoader):
    """
    PDF extraction via pdfplumber — best table extraction for text-layer PDFs.

    Chuyển đổi bảng phát hiện được thành Markdown inline với văn bản xung quanh.
    Không có OCR. Chậm hơn PyMuPDF.
    Dùng cho: báo cáo tài chính, data sheet, PDF nhiều bảng (text layer).

    Tham số:
        extract_tables : Bật/tắt trích xuất bảng → Markdown.
    """

    def __init__(self, language: Language = "both", extract_tables: bool = True):
        super().__init__(language)
        self.extract_tables = extract_tables

    def load(self, file_path: str) -> list[Document]:
        import pdfplumber

        docs = []
        with pdfplumber.open(file_path) as pdf:
            total = len(pdf.pages)
            for page_num, page in enumerate(pdf.pages, start=1):
                parts: list[str] = []

                raw = page.extract_text(x_tolerance=2, y_tolerance=2) or ""
                if raw:
                    parts.append(raw)

                if self.extract_tables:
                    for table in page.extract_tables():
                        if not table:
                            continue
                        header, *rows = table
                        header = [str(h or "") for h in header]
                        md = [
                            "| " + " | ".join(header) + " |",
                            "| " + " | ".join(["---"] * len(header)) + " |",
                        ]
                        for row in rows:
                            cells = ([str(c or "") for c in row] + [""] * len(header))[:len(header)]
                            md.append("| " + " | ".join(cells) + " |")
                        parts.append("\n".join(md))

                content = clean_text("\n\n".join(parts))
                if content:
                    docs.append(Document(
                        page_content=content,
                        metadata={
                            "source": file_path, "page": page_num,
                            "total_pages": total, "file_type": "pdf",
                            "pdf_strategy": "pdfplumber",
                            "has_table": self.extract_tables and bool(page.extract_tables()),
                        },
                    ))
        return self._stamp(docs)


# ===========================================================================
# 4. Unstructured — toàn diện nhất: OCR + bảng + hình
# ===========================================================================

class UnstructuredPDFLoader(BaseLoader):
    """
    Trích PDF bằng Unstructured.io — chất lượng toàn diện nhất.

    Xử lý: text layer, PDF scan (qua OCR), bảng (→ Markdown), hình ảnh.
    Trả về 1 Document / trang với nội dung đã được phân loại và clean.

    Cần cài:
        pip install "unstructured[pdf]" unstructured-inference
        # Thêm OCR:
        pip install pytesseract Pillow         (Tesseract)
        pip install paddlepaddle paddleocr     (PaddleOCR)

    Tham số:
        ocr_engine     : "tesseract" | "paddleocr" | "none"
        extract_tables : Chuyển bảng thành Markdown.
        extract_images : Dùng VLM mô tả hình ảnh trong PDF.
        vision_model   : OpenAI model để mô tả hình (cần API key).
    """

    _LANG_MAP = {"vi": "vie", "en": "eng", "both": "vie+eng"}

    def __init__(
        self,
        language:        Language = "both",
        ocr_engine:      str  = "none",
        extract_tables:  bool = True,
        extract_images:  bool = False,
        vision_model:    str  = "gpt-4o-mini",
        vision_provider: str  = "openai",
        ollama_base_url: str  = "http://localhost:11434/v1",
    ):
        super().__init__(language)
        self.ocr_engine      = ocr_engine
        self.extract_tables  = extract_tables
        self.extract_images  = extract_images
        self.vision_model    = vision_model
        self.vision_provider = vision_provider
        self.ollama_base_url = ollama_base_url

    def load(self, file_path: str) -> list[Document]:
        from unstructured.partition.pdf import partition_pdf

        # hi_res: cần unstructured-inference, render từng trang thành ảnh
        # → chỉ dùng khi OCR thực sự cần thiết (PDF scan)
        # fast: đủ tốt cho PDF text layer, không cần dep thêm
        strategy = "hi_res" if self.ocr_engine != "none" else "fast"

        partition_kwargs: dict = {
            "filename":              file_path,
            "strategy":              strategy,
            "infer_table_structure": self.extract_tables,
        }
        if self.ocr_engine != "none":
            partition_kwargs["languages"] = [self._LANG_MAP.get(self.language, "vie+eng")]

        elements = partition_pdf(**partition_kwargs)

        pages: dict[int, list[str]] = {}
        for el in elements:
            page_num = getattr(el.metadata, "page_number", 1) or 1
            pages.setdefault(page_num, [])
            el_type = type(el).__name__

            if el_type == "Table" and self.extract_tables:
                html = getattr(el.metadata, "text_as_html", None) or str(el)
                pages[page_num].append(table_html_to_markdown(html))

            elif el_type == "Image":
                if self.extract_images:
                    # Dùng VLM mô tả ảnh (cần OPENAI_API_KEY)
                    desc = self._describe_image(el)
                    pages[page_num].append(f"[Figure: {desc}]" if desc else "[Figure]")
                else:
                    # Không dùng VLM: giữ placeholder để biết có ảnh ở đây
                    alt = str(el).strip()
                    pages[page_num].append(f"[Figure: {alt}]" if alt else "[Figure]")

            elif el_type not in ("Header", "Footer", "PageBreak"):
                text = str(el).strip()
                if text:
                    pages[page_num].append(text)

        docs = []
        for page_num in sorted(pages):
            content = clean_text("\n\n".join(pages[page_num]))
            if content:
                docs.append(Document(
                    page_content=content,
                    metadata={
                        "source": file_path, "page": page_num,
                        "file_type": "pdf", "pdf_strategy": "unstructured",
                        "ocr_engine": self.ocr_engine,
                    },
                ))
        return self._stamp(docs)

    def _describe_image(self, element) -> str:
        image_b64 = getattr(element.metadata, "image_base64", None)
        if not image_b64:
            return ""
        try:
            return _call_vision_api(
                image_b64=image_b64,
                mime="image/png",
                model=self.vision_model,
                provider=self.vision_provider,
                base_url=self.ollama_base_url,
                max_tokens=150,
            )
        except Exception as e:
            logger.warning("VLM description failed: %s", e)
            return ""


# ===========================================================================
# 5. Docling — IBM Research, Markdown đầu ra rất sạch
# ===========================================================================

@contextlib.contextmanager
def _force_eager_attention():
    """
    Ép transformers nạp model bằng eager attention, chỉ trong phạm vi with-block.

    Vì sao cần: Docling nạp model CodeFormula (kiến trúc SmolVLM/Idefics3) ở bên
    trong, không cho truyền tham số xuống. Transformers mặc định dùng SDPA khi
    torch >= 2.1.1, mà Idefics3VisionTransformer chưa khai báo hỗ trợ SDPA →
        ValueError: Idefics3VisionTransformer does not support an attention
        implementation through torch.nn.functional.scaled_dot_product_attention
    Chính model card của SmolDocling cũng dùng eager khi chạy CPU.

    Patch cả hai điểm vì Auto* class TỰ override from_pretrained, nên vá lớp cha
    PreTrainedModel là chưa đủ. Luôn khôi phục ở finally để không rò rỉ sang các
    model khác trong app (embedding, reranker...).
    """
    saved: list = []
    try:
        from transformers import modeling_utils
        from transformers.models.auto import auto_factory
    except Exception:
        yield                      # không có transformers → chạy nguyên trạng
        return

    targets = [modeling_utils.PreTrainedModel, auto_factory._BaseAutoModelClass]

    def _make(f):
        def wrapper(cls, *a, **kw):
            kw.setdefault("attn_implementation", "eager")
            try:
                return f(cls, *a, **kw)
            except TypeError:
                # Bản transformers cũ chưa nhận kwarg này
                kw.pop("attn_implementation", None)
                return f(cls, *a, **kw)
        return classmethod(wrapper)

    for owner in targets:
        orig = owner.__dict__.get("from_pretrained")
        if orig is None:
            continue
        func = orig.__func__ if isinstance(orig, classmethod) else orig
        setattr(owner, "from_pretrained", _make(func))
        saved.append((owner, orig))

    logger.info("Đã ép eager attention cho %d điểm nạp model", len(saved))
    try:
        yield
    finally:
        for owner, orig in saved:
            setattr(owner, "from_pretrained", orig)


class DoclingPDFLoader(BaseLoader):
    """
    PDF extraction via Docling (IBM Research, 2024).

    Tạo Markdown chất lượng cao với bảng và layout được giữ nguyên cấu trúc.
    Trả về toàn bộ tài liệu như 1 Document.
    Dùng cho: research papers, tài liệu kỹ thuật, báo cáo có layout phức tạp.

    Cần cài: pip install docling

    Tham số:
        extract_tables : Hint cho Docling (tự xử lý bảng, luôn bật mặc định).
    """

    def __init__(self, language: Language = "both", extract_tables: bool = True,
                 extract_images: bool = True):
        super().__init__(language)
        self.extract_tables = extract_tables
        self.extract_images = extract_images

    def load(self, file_path: str) -> list[Document]:
        import re, base64 as _b64, io
        from docling.document_converter import DocumentConverter, PdfFormatOption
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.datamodel.base_models import InputFormat

        # ── Pipeline options ─────────────────────────────────────────────────
        # Formula enrichment nạp model CodeFormula (kiến trúc SmolVLM/Idefics3).
        # Bản `transformers` mới mặc định dùng SDPA attention và đòi kiến trúc
        # phải khai báo hỗ trợ — Idefics3VisionTransformer thì chưa, nên ném:
        #     ValueError: Idefics3VisionTransformer does not support an attention
        #     implementation through torch.nn.functional.scaled_dot_product_attention
        #
        # Phần parse PDF chính KHÔNG cần model này. Trước đây enrichment được bật
        # vô điều kiện → chỉ cần transformers mới là docling hỏng hoàn toàn.
        # Giờ: thử có enrichment, lỗi kiểu VLM/attention thì tự lùi về không có.
        #
        # Tắt hẳn từ đầu (bỏ qua lần thử đầu):  DOCLING_FORMULA_ENRICHMENT=0
        _formula_env = os.environ.get("DOCLING_FORMULA_ENRICHMENT", "1").strip().lower()
        _want_formula = _formula_env not in ("0", "false", "no", "off")

        def _make_opts(with_formula: bool):
            o = PdfPipelineOptions()
            o.generate_picture_images = True
            o.images_scale = 2.0   # tăng resolution ảnh (mặc định 1.0 → nhỏ, mờ)
            if with_formula and hasattr(o, "do_formula_enrichment"):
                o.do_formula_enrichment = True
            return o

        def _convert(with_formula: bool):
            return DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(
                        pipeline_options=_make_opts(with_formula))
                }
            ).convert(file_path)

        def _is_vlm_attention_error(exc: BaseException) -> bool:
            """Lỗi do model enrichment, không phải do PDF hay do Docling core."""
            msg = str(exc)
            return any(tok in msg for tok in (
                "attn_implementation",
                "scaled_dot_product_attention",
                "Idefics3",
                "does not support an attention implementation",
            ))

        if _want_formula:
            logger.info("DoclingPDFLoader: thử với formula enrichment (eager attention)")
            try:
                # Ép eager attention — nếu không, model CodeFormula sẽ ném
                # ValueError về scaled_dot_product_attention và ta mất LaTeX.
                with _force_eager_attention():
                    result = _convert(True)
            except Exception as exc:
                if not _is_vlm_attention_error(exc):
                    raise
                logger.warning(
                    "DoclingPDFLoader: formula enrichment không chạy được trên môi "
                    "trường này (%s: %s). Đang parse lại KHÔNG có enrichment — "
                    "công thức sẽ ra dạng [formula_N] thay vì LaTeX. "
                    "Muốn bỏ hẳn lần thử này, đặt DOCLING_FORMULA_ENRICHMENT=0.",
                    type(exc).__name__, str(exc)[:160],
                )
                result = _convert(False)
        else:
            logger.info("DoclingPDFLoader: formula enrichment đã tắt qua env var")
            result = _convert(False)

        doc           = result.document
        raw_markdown  = doc.export_to_markdown()

        # ── Xử lý <!-- formula-not-decoded --> ───────────────────────────────
        formula_counter = [0]
        def _replace_formula(m: re.Match) -> str:
            formula_counter[0] += 1
            return f"$[formula_{formula_counter[0]}]$"
        raw_markdown = re.sub(r'<!--\s*formula-not-decoded\s*-->', _replace_formula, raw_markdown)

        # ── Xử lý <!-- image --> ─────────────────────────────────────────────
        docling_images_dir = None
        pics = getattr(doc, "pictures", []) or []
        logger.debug("DoclingPDFLoader: doc.pictures: %d ảnh", len(pics))

        if pics:
            pdf_stem  = Path(file_path).stem
            safe_name = re.sub(r'[^\w\-.]', '_', pdf_stem)[:80]
            img_dir   = DOCLING_CACHE_DIR / safe_name
            img_dir.mkdir(parents=True, exist_ok=True)

            img_counter = [0]
            image_refs: list[dict] = []

            def _replace_img_placeholder(m: re.Match) -> str:
                idx = img_counter[0]
                img_counter[0] += 1

                if idx >= len(pics):
                    logger.debug("DoclingPDFLoader: placeholder #%d không có ảnh", idx)
                    return ""

                pic     = pics[idx]
                img_ref = getattr(pic, "image", None)
                pil_img = getattr(img_ref, "pil_image", None) if img_ref else None

                if pil_img is None:
                    logger.debug("DoclingPDFLoader: ảnh #%d: pil_image = None", idx)
                    return ""

                logger.debug("DoclingPDFLoader: ảnh #%d: size=%s", idx, pil_img.size)

                dest = img_dir / f"picture_{idx:03d}.png"
                try:
                    pil_img.save(str(dest))
                except Exception as e:
                    logger.warning("DoclingPDFLoader: không lưu %s: %s", dest, e)

                fig_idx = idx + 1
                # Dùng đường dẫn file thay vì base64 inline.
                # Lý do: chuỗi base64 dài hàng nghìn ký tự → RecursiveCharacterTextSplitter
                # cắt thẳng vào giữa chuỗi → chunk vô nghĩa toàn base64.
                # File đã được lưu ở `dest`, markdown reference đủ để render trong app.
                token = f"![Figure {fig_idx}]({dest.as_posix()})"

                image_refs.append({
                    "figure_index": fig_idx,
                    "path":         dest.as_posix(),
                    "description":  f"Figure {fig_idx}",
                    "placeholder":  f"![Figure {fig_idx}]",
                })

                return token

            raw_markdown = re.sub(
                r'<!--\s*image\s*-->',
                _replace_img_placeholder,
                raw_markdown,
            )
            docling_images_dir = img_dir.as_posix()
            logger.debug("DoclingPDFLoader: replaced %d placeholders", img_counter[0])
        else:
            image_refs = []

        # ── Clean text: chỉ clean đoạn text, KHÔNG đụng vào base64 ──────────
        IMG_RE = re.compile(
            r'(!\[[^\]]*\]\(data:image/[^)]+\))',
            re.DOTALL,
        )
        segments = IMG_RE.split(raw_markdown)
        cleaned_segments = []
        for i, seg in enumerate(segments):
            if IMG_RE.fullmatch(seg):
                cleaned_segments.append(seg)
            else:
                cleaned_segments.append(clean_text(seg))
        final_markdown = "".join(cleaned_segments)

        return self._stamp([Document(
            page_content=final_markdown,
            metadata={
                "source":            file_path,
                "file_type":         "pdf",
                "pdf_strategy":      "docling",
                "marker_images_dir": docling_images_dir,
                "image_refs":        image_refs,
            },
        )])


# ===========================================================================
# 6. Marker — datalab.to, Markdown chất lượng cao từ mọi PDF
# ===========================================================================

# ─────────────────────────────────────────────────────────────────────────────
# Sửa công thức toán bị Marker escape hỏng
#
# Layout model của Marker phân loại một số dòng là Text thay vì Equation.
# Block Equation → LaTeX chuẩn trong $$...$$ (render đẹp).
# Block Text     → chạy qua bộ escape Markdown, biến `_` thành `\_` để chống bị
#                  hiểu là in nghiêng. Kết quả: LaTeX vừa mất delimiter, vừa bị
#                  escape hỏng → KaTeX parse fail → hiện nguyên văn:
#                      P\_a = P1 + u\_a(P2 - P1) \tag{l}
#
# Không sửa được model của Marker từ ngoài, nhưng sửa được output:
#   · trong $...$ / $$...$$ đã có sẵn → chỉ gỡ escape
#   · dòng trần có LỆNH LaTeX thật    → gỡ escape rồi bọc $$...$$
#
# Điều kiện cố tình siết chặt: CHỈ tin khi thấy lệnh LaTeX (\frac, \tag, \sqrt…).
# Đoán theo dấu `\_` đơn thuần sẽ bọc nhầm văn xuôi kiểu "my\_file.txt" —
# bọc nhầm văn xuôi tệ hơn nhiều so với bỏ sót một công thức.
_MD_ESCAPE = re.compile(r'\\([_*\[\]()#+\-.!`~])')
_MATH_SPAN = re.compile(r'(\$\$.+?\$\$|\$[^$\n]+?\$)', re.S)
_LATEX_CMD = re.compile(
    r'\\(frac|dfrac|tfrac|tag|sqrt|sum|prod|int|lim|cdot|times|div'
    r'|leq|geq|neq|approx|equiv|partial|nabla|forall|exists'
    r'|alpha|beta|gamma|delta|epsilon|theta|lambda|sigma|mu|rho|phi|psi|omega|pi|infty'
    r'|left|right|begin|end|mathbf|mathrm|mathcal|mathbb|boldsymbol|text|hat|bar|vec|tilde)\b')


# KaTeX (bộ render của Streamlit) KHÔNG nuốt được các lệnh này:
#   \tag      — chỉ hợp lệ ở display mode cấp cao nhất, ngữ cảnh khác là ném lỗi
#   \label \ref \eqref \nonumber \notag  — của amsmath, KaTeX không có
# Gặp một trong số đó, KaTeX bỏ NGUYÊN biểu thức và in ra dạng thô. Đây là lý do
# công thức đã gỡ escape đúng rồi mà vẫn không render được.
# Xử lý: tách \tag{3} thành nhãn '(3)' đặt NGOÀI phần toán — giữ số hiệu công
# thức mà không làm KaTeX chết; các lệnh còn lại bỏ hẳn.
_TAG_CMD  = re.compile(r'\\(?:tag)\*?\s*\{([^}]*)\}')
_DROP_CMD = re.compile(r'\\(?:label|ref|eqref|nonumber|notag)\s*(?:\{[^}]*\})?')


def _unescape_md(s: str) -> str:
    return _MD_ESCAPE.sub(r'\1', s)


def _strip_katex_unsupported(body: str) -> tuple[str, str | None]:
    """Trả về (thân công thức đã sạch, nhãn tag hoặc None)."""
    label = None
    m = _TAG_CMD.search(body)
    if m:
        label = m.group(1).strip()
        body = _TAG_CMD.sub('', body)
    body = _DROP_CMD.sub('', body)
    body = _strip_dangling_backslash(body)
    return body.strip(), label


def _strip_dangling_backslash(body: str) -> str:
    # Marker đôi khi để lại một backslash đơn ở cuối thân công thức:
    #     $$...\frac{...}{2}|\$$
    # Backslash đó nuốt mất dấu $ đóng (thành escape \$), biểu thức
    # mất cân → KaTeX ném ParseError và in nguyên văn màu ĐỎ.
    #
    # Số backslash CHẴN thì hợp lệ (\\ là xuống dòng trong LaTeX) → giữ.
    # Số LẺ thì luôn sai → bỏ đúng một cái.
    body = body.rstrip()
    n = len(body) - len(body.rstrip(chr(92)))
    if n % 2 == 1:
        body = body[:-1].rstrip()
    return body


def _line_is_math(s: str) -> bool:
    s = s.strip()
    if not s or s.startswith(('#', '>', '|', '-', '*', '!', '[', '```')):
        return False
    return bool(_LATEX_CMD.search(s))


def repair_marker_math(markdown: str) -> tuple[str, int]:
    """Trả về (markdown_đã_sửa, số_dòng_được_bọc_thêm)."""
    out: list[str] = []
    in_code = False
    n_wrapped = 0
    for line in markdown.split("\n"):
        if line.lstrip().startswith("```"):
            in_code = not in_code
            out.append(line)
            continue
        if in_code:
            out.append(line)
            continue
        if "$" in line:
            def _fix_span(m):
                span = _unescape_md(m.group(0))
                if span.startswith("$$"):
                    inner, lab = _strip_katex_unsupported(span[2:-2])
                    return f"$${inner}$$" + (f" ({lab})" if lab else "")
                inner, lab = _strip_katex_unsupported(span[1:-1])
                return f"${inner}$" + (f" ({lab})" if lab else "")
            out.append(_MATH_SPAN.sub(_fix_span, line))
        elif _line_is_math(line):
            body, lab = _strip_katex_unsupported(_unescape_md(line.strip()))
            out.append(f"$${body}$$" + (f" ({lab})" if lab else ""))
            n_wrapped += 1
        else:
            out.append(line)
    return "\n".join(out), n_wrapped


class MarkerPDFLoader(BaseLoader):
    """
    Trích PDF bằng Marker (datalab.to / Vik Paruchuri, 2024).

    Marker chuyển PDF, kể cả bản scan, thành Markdown chất lượng cao bằng cách
    ghép nhiều model: Surya OCR (đa ngôn ngữ), model layout (cấu trúc trang),
    model bảng (→ Markdown), model công thức (→ LaTeX).

    Điểm mạnh: chạy hoàn toàn cục bộ, không cần API; xử lý tốt cả PDF có text
    layer lẫn PDF scan; giữ được bảng, công thức và code block; Markdown đầu ra
    sạch, heading rõ ràng; hỗ trợ 90+ ngôn ngữ nhờ Surya.

    Cần cài ``pip install marker-pdf``; lần chạy đầu tải model khoảng 2–4 GB.

    Tham số
    -------
    langs   : Danh sách mã ngôn ngữ ISO 639-1 cho OCR; None để tự nhận diện.
    device  : "cpu" | "cuda" | "mps" (Apple Silicon)
    workers : Số tiến trình worker, mặc định 1.

    Hiệu năng: CPU khoảng 30–60 giây/trang, GPU khoảng 2–5 giây/trang. Hợp với
    xử lý theo lô ngoại tuyến, không hợp thời gian thực.
    """

    _LANG_MAP = {"vi": ["vi"], "en": ["en"], "both": ["vi", "en"]}

    @staticmethod
    def _auto_device() -> str:
        """Tự động chọn device tốt nhất có sẵn: cuda > mps > cpu."""
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            if torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
        return "cpu"

    def __init__(
        self,
        language:        Language        = "both",
        langs:           list[str] | None = None,
        device:          str | None      = None,
        workers:         int             = 1,
        describe_images: bool            = False,
        vision_model:    str             = "gpt-4o-mini",
        vision_provider: str             = "openai",
        ollama_base_url: str             = "http://localhost:11434/v1",
    ):
        super().__init__(language)
        self._langs          = langs or self._LANG_MAP.get(language, ["vi", "en"])
        self.device          = device or self._auto_device()
        self.workers         = workers
        self.describe_images = describe_images
        self.vision_model    = vision_model
        self.vision_provider = vision_provider
        self.ollama_base_url = ollama_base_url
        logger.info("MarkerPDFLoader: device='%s', describe_images=%s", self.device, describe_images)

    # ------------------------------------------------------------------
    # Hàm phụ trợ
    # ------------------------------------------------------------------

    def _describe_image_file(self, img_path: Path) -> str:
        """Gửi file ảnh lên VLM và trả về mô tả 1-2 câu."""
        import base64
        import mimetypes
        mime, _ = mimetypes.guess_type(str(img_path))
        mime = mime or "image/jpeg"
        b64  = base64.b64encode(img_path.read_bytes()).decode()
        return _call_vision_api(
            image_b64=b64,
            mime=mime,
            model=self.vision_model,
            provider=self.vision_provider,
            base_url=self.ollama_base_url,
        )

    def _replace_image_links(
        self, markdown: str, output_dir: Path
    ) -> tuple[str, list[dict]]:
        """
        Thay thế mọi `![alt](path)` trong Markdown bằng mô tả VLM.
        Trả về (markdown_mới, image_refs).

        image_refs là list các dict dạng {
            "figure_index": int,   # thứ tự ảnh trong document (bắt đầu từ 1)
            "path":         str,   # đường dẫn tuyệt đối đến file ảnh gốc
            "description":  str,   # mô tả VLM
            "placeholder":  str,   # chuỗi xuất hiện trong page_content
        }
        Giúp retriever downstream có thể map text chunk → ảnh gốc.
        """
        import re

        pattern     = re.compile(r'!\[([^\]]*)\]\(([^)]+)\)')
        image_refs: list[dict] = []
        fig_counter = [0]

        def replacer(m: re.Match) -> str:
            alt     = m.group(1)
            img_ref = m.group(2)

            img_path = Path(img_ref)
            if not img_path.is_absolute():
                img_path = output_dir / img_path

            if not img_path.exists():
                logger.warning("Marker image not found: %s", img_path)
                return f"[Figure: {alt}]" if alt else "[Figure]"

            fig_counter[0] += 1
            fig_idx = fig_counter[0]

            try:
                desc = self._describe_image_file(img_path)
                logger.info("  Described image '%s': %s", img_path.name, desc[:60])
            except Exception as exc:
                print(f"\n[VLM ERROR] {img_path.name}: {exc}\n")
                logger.warning("VLM description failed for %s: %s", img_path.name, exc)
                if not hasattr(self, "_vlm_errors"):
                    self._vlm_errors: list[str] = []
                self._vlm_errors.append(f"{img_path.name}: {exc}")
                desc = alt or img_path.name

            placeholder = f"[Figure {fig_idx}: {desc}]"

            # Lưu ref để gắn vào metadata
            image_refs.append({
                "figure_index": fig_idx,
                "path":         img_path.as_posix(),
                "description":  desc,
                "placeholder":  placeholder,
            })

            return placeholder

        new_markdown = pattern.sub(replacer, markdown)
        return new_markdown, image_refs

    # ------------------------------------------------------------------
    # Hàm load chính
    # ------------------------------------------------------------------

    def load(self, file_path: str) -> list[Document]:
        from marker.converters.pdf import PdfConverter
        from marker.models import create_model_dict
        from marker.config.parser import ConfigParser

        # ── Tạo thư mục cache riêng cho từng PDF ─────────────────────────────
        pdf_stem  = Path(file_path).stem
        safe_name = re.sub(r'[^\w\-.]', '_', pdf_stem)[:80]
        img_dir   = MARKER_CACHE_DIR / safe_name
        img_dir.mkdir(parents=True, exist_ok=True)
        logger.info("MarkerPDFLoader: image cache dir = %s", img_dir)

        # Marker ConfigParser mong đợi `languages` là CHUỖI phân tách bởi dấu
        # phẩy — bên trong nó gọi v.split(","). Truyền list vào sẽ ném:
        #     AttributeError: 'list' object has no attribute 'split'
        # (marker/config/parser.py :: generate_config_dict)
        # Lưu ý: self._langs vẫn giữ nguyên dạng list cho metadata ở cuối hàm.
        langs_cfg = (",".join(self._langs)
                     if isinstance(self._langs, (list, tuple))
                     else str(self._langs))

        config = {
            "languages":     langs_cfg,
            "device":        self.device,
            "workers":       self.workers,
            "output_format": "markdown",
        }
        config_parser = ConfigParser(config)
        converter     = PdfConverter(
            config=config_parser.generate_config_dict(),
            artifact_dict=create_model_dict(),
            processor_list=config_parser.get_processors(),
            renderer=config_parser.get_renderer(),
        )

        rendered = converter(file_path)

        # ── Lưu ảnh từ rendered.images xuống cache dir ───────────────────────
        saved_images: dict[str, str] = {}
        images_dict = getattr(rendered, "images", {}) or {}
        for img_name, img_obj in images_dict.items():
            dest = img_dir / Path(img_name).name
            try:
                if hasattr(img_obj, "save"):              # PIL.Image
                    img_obj.save(str(dest))
                elif isinstance(img_obj, (bytes, bytearray)):
                    dest.write_bytes(img_obj)
                else:
                    logger.warning("Marker image '%s': kiểu không xác định %s", img_name, type(img_obj))
                    continue
                saved_images[img_name] = dest.as_posix()
                logger.info("  Cached image: %s → %s", img_name, dest)
            except Exception as exc:
                logger.warning("Không lưu được ảnh '%s': %s", img_name, exc)

        markdown = clean_text(rendered.markdown)

        # Sửa công thức bị Marker escape hỏng — xem repair_marker_math ở trên
        markdown, _n_math = repair_marker_math(markdown)
        if _n_math:
            logger.info("MarkerPDFLoader: đã bọc lại %d dòng công thức bị escape hỏng",
                        _n_math)

        # ── Xử lý link ảnh trong Markdown ────────────────────────────────────
        image_refs: list[dict] = []

        if self.describe_images:
            logger.info("MarkerPDFLoader: mô tả ảnh bằng VLM '%s'…", self.vision_model)
            markdown, image_refs = self._replace_image_links(markdown, img_dir)
            marker_images_dir = None
        else:
            def _abs_path(m: re.Match) -> str:
                alt      = m.group(1)
                img_ref  = m.group(2)
                img_name = Path(img_ref).name
                for key, abs_path in saved_images.items():
                    if Path(key).name == img_name:
                        return f"![{alt}]({abs_path})"
                cached = img_dir / img_name
                if cached.exists():
                    return f"![{alt}]({cached.as_posix()})"
                logger.warning("Không tìm thấy ảnh '%s'", img_ref)
                return f"![{alt}]({img_ref})"

            markdown = re.sub(r'!\[([^\]]*)\]\(([^)]+)\)', _abs_path, markdown)
            marker_images_dir = img_dir.as_posix()

            # Khi không dùng VLM: vẫn ghi image_refs với path để retriever dùng
            img_re = re.compile(r'!\[([^\]]*)\]\(([^)]+)\)')
            for idx, m in enumerate(img_re.finditer(markdown), start=1):
                img_path = Path(m.group(2))
                if img_path.exists():
                    image_refs.append({
                        "figure_index": idx,
                        "path":         img_path.as_posix(),
                        "description":  m.group(1) or f"Figure {idx}",
                        "placeholder":  m.group(0),
                    })

        return self._stamp([Document(
            page_content=markdown,
            metadata={
                "source":             file_path,
                "file_type":          "pdf",
                "pdf_strategy":       "marker",
                "languages":          self._langs,
                "describe_images":    self.describe_images,
                "marker_images_dir":  marker_images_dir,
                # image_refs: list[dict] — map text placeholder → ảnh gốc
                "image_refs":         image_refs,
            },
        )])


# ---------------------------------------------------------------------------
# OpenDataLoaderPDFLoader
# ---------------------------------------------------------------------------

import subprocess
import tempfile


def _check_java() -> None:
    """
    Kiểm tra Java 11+ có sẵn trong PATH không.
    Raise RuntimeError với hướng dẫn cài đặt nếu không tìm thấy.
    """
    try:
        result = subprocess.run(
            ["java", "-version"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            raise RuntimeError(
                "Java không hoạt động (exit code != 0).\n"
                "opendataloader-pdf yêu cầu Java 11+.\n"
                "Tải tại: https://adoptium.net/"
            )
    except FileNotFoundError:
        raise RuntimeError(
            "Java chưa được cài đặt.\n"
            "opendataloader-pdf yêu cầu Java 11+.\n"
            "Tải tại: https://adoptium.net/\n"
            "Sau khi cài: java -version"
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError("Timeout khi kiểm tra Java — đảm bảo Java không bị treo.")


class OpenDataLoaderPDFLoader(BaseLoader):
    """
    Trích PDF bằng opendataloader-pdf, trả về list[Document].

    Theo đúng hợp đồng BaseLoader như PyPDFLoader, MarkerPDFLoader:
        loader = OpenDataLoaderPDFLoader(language="both", hybrid="docling-fast")
        docs   = loader.load("report.pdf")

    Hai chế độ
    ----------
      fast (mặc định) : Java cục bộ, tất định, ~0.05 giây/trang, độ chính xác 0.72
      hybrid          : có backend AI cho trang phức tạp, ~0.43 giây/trang, 0.90

    Benchmark trên 200 PDF thực tế (tổng thể · bảng · heading):
      opendataloader [hybrid]  0.90 · 0.93 · 0.83  ← hạng nhất
      opendataloader [fast]    0.72 · 0.49 · 0.76
      docling                  0.86 · 0.89 · 0.80
      marker                   0.83 · 0.81 · 0.80

    Cài đặt
    -------
      pip install opendataloader-pdf             (cần Java 11+)
      pip install "opendataloader-pdf[hybrid]"   (cho chế độ hybrid)

    Chế độ hybrid cần chạy server nền trước ở một terminal riêng:
      opendataloader-pdf-hybrid --port 5002
      # với PDF scan / tiếng Việt:
      opendataloader-pdf-hybrid --port 5002 --force-ocr --ocr-lang "vi,en"

    Tham số
    -------
    language        : Ngôn ngữ corpus ("vi" | "en" | "both").
    hybrid          : None → chế độ fast cục bộ, không cần GPU hay server.
                      "docling-fast" → chế độ hybrid, chính xác cao nhất.
    hybrid_port     : Cổng của server hybrid, mặc định 5002.
    use_struct_tree : Dùng thẻ cấu trúc sẵn có của Tagged PDF, mặc định False.
    """

    def __init__(
        self,
        language:        Language   = "both",
        hybrid:          str | None = None,
        hybrid_port:     int        = 5002,
        use_struct_tree: bool       = False,
    ) -> None:
        super().__init__(language)
        self.hybrid          = hybrid
        self.hybrid_port     = hybrid_port
        self.use_struct_tree = use_struct_tree

    def load(self, file_path: str) -> list[Document]:
        _check_java()
        self._validate(file_path)
        try:
            docs = self._load_native(file_path)
        except ImportError as exc:
            if "opendataloader_pdf" in str(exc) or "langchain-opendataloader-pdf" in str(exc):
                docs = self._load_langchain_fallback(file_path)
            else:
                raise
        return self._stamp(docs)

    def _load_native(self, file_path: str) -> list[Document]:
        try:
            import opendataloader_pdf
        except ImportError:
            raise ImportError(
                "opendataloader_pdf chưa được cài.\n"
                "  pip install opendataloader-pdf\n"
                "(Java 11+ required)"
            )
        except Exception as exc:
            # Bắt lỗi lúc import (JVM không khởi động được, thiếu native lib trên
            # Windows) để báo đúng nguyên nhân, thay vì lặng lẽ rơi về loader dự phòng.
            raise RuntimeError(
                f"opendataloader_pdf đã được cài nhưng không import được.\n"
                f"Nguyên nhân: {exc}\n\n"
                f"Kiểm tra: python -c \"import opendataloader_pdf\" để xem lỗi đầy đủ."
            ) from exc
        convert_kwargs: dict = {
            "input_path":   [file_path],
            "format":       "markdown",
            "image_output": "embedded",
        }
        if self.hybrid:
            convert_kwargs["hybrid"] = self.hybrid
        if self.use_struct_tree:
            convert_kwargs["use_struct_tree"] = True

        with tempfile.TemporaryDirectory() as tmp_dir:
            convert_kwargs["output_dir"] = tmp_dir
            try:
                opendataloader_pdf.convert(**convert_kwargs)
            except Exception as exc:
                self._raise_helpful_error(file_path, exc)

            md_files = sorted(Path(tmp_dir).rglob("*.md"))
            if not md_files:
                md_files = sorted(Path(tmp_dir).rglob("*.txt"))

            docs: list[Document] = []
            for md_file in md_files:
                text = md_file.read_text(encoding="utf-8", errors="replace")
                text = clean_text(text)
                if not text:
                    continue
                docs.append(Document(
                    page_content=text,
                    metadata=self._make_metadata(file_path),
                ))

        if not docs:
            docs = [Document(
                page_content="",
                metadata={**self._make_metadata(file_path), "warning": "empty_output"},
            )]
        return docs

    def _load_langchain_fallback(self, file_path: str) -> list[Document]:
        try:
            from langchain_opendataloader_pdf import OpenDataLoaderPDFLoader as _ODL
        except ImportError:
            raise ImportError(
                "Không tìm thấy opendataloader_pdf hoặc langchain-opendataloader-pdf.\n\n"
                "Cài một trong hai:\n"
                "  pip install opendataloader-pdf\n"
                "  pip install langchain-opendataloader-pdf\n"
                "(Java 11+ required)"
            )
        # Đường fallback KHÔNG hỗ trợ hybrid mode và use_struct_tree — API của
        # wrapper LangChain không nhận hai tham số này. Nếu người dùng đã chọn
        # chúng, phải báo cho họ biết, nếu không họ sẽ so sánh chất lượng loader
        # dựa trên metadata sai.
        if self.hybrid or self.use_struct_tree:
            logger.warning(
                "opendataloader: đang chạy qua fallback langchain-opendataloader-pdf, "
                "vốn KHÔNG hỗ trợ hybrid mode / use_struct_tree. "
                "Đã chọn hybrid=%s, use_struct_tree=%s nhưng thực tế chạy fast mode. "
                "Cài `pip install \"opendataloader-pdf[hybrid]\"` để dùng hybrid thật.",
                self.hybrid, self.use_struct_tree,
            )

        loader = _ODL(file_path=[file_path], format="text")
        docs   = loader.load()
        for doc in docs:
            doc.metadata.update(self._make_metadata(file_path))
            # Ghi đè theo những gì THỰC SỰ đã chạy, không phải theo ý định của
            # người dùng — _make_metadata() đọc self.hybrid nên sẽ nói dối ở đây.
            doc.metadata["via"]             = "langchain-opendataloader-pdf"
            doc.metadata["mode"]            = "fast"
            doc.metadata["use_struct_tree"] = False
            doc.metadata["format"]          = "text"   # native dùng markdown
        return docs

    def _make_metadata(self, file_path: str) -> dict:
        return {
            "source":          file_path,
            "file_type":       "pdf",
            "parser":          "opendataloader",
            "mode":            f"hybrid:{self.hybrid}" if self.hybrid else "fast",
            "use_struct_tree": self.use_struct_tree,
        }

    def _validate(self, file_path: str) -> None:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File không tồn tại: {file_path}")
        if path.suffix.lower() != ".pdf":
            raise ValueError(
                f"OpenDataLoaderPDFLoader chỉ hỗ trợ .pdf, nhận: {path.suffix}"
            )

    def _raise_helpful_error(self, file_path: str, exc: Exception) -> None:
        msg = str(exc).lower()
        if self.hybrid and ("connection refused" in msg or "connect" in msg):
            raise RuntimeError(
                f"Không kết nối được hybrid server tại port {self.hybrid_port}.\n\n"
                f"Khởi động server trước:\n"
                f"  opendataloader-pdf-hybrid --port {self.hybrid_port}\n\n"
                f"Với PDF scan / tiếng Việt:\n"
                f"  opendataloader-pdf-hybrid --port {self.hybrid_port} "
                f"--force-ocr --ocr-lang \"vi,en\"\n\n"
                f"Sau đó chạy lại."
            ) from exc
        if "java" in msg:
            raise RuntimeError(
                "Lỗi liên quan đến Java. Đảm bảo Java 11+ đã cài:\n"
                "  java -version\n"
                "Tải tại: https://adoptium.net/"
            ) from exc
        raise RuntimeError(
            f"opendataloader-pdf lỗi khi xử lý '{file_path}':\n{exc}"
        ) from exc
