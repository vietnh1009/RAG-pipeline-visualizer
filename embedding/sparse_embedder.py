"""
embedding/sparse_embedder.py
=============================
Embedder sparse cho hybrid retrieval — BM25 và SPLADE.

Vector dense nắm ngữ nghĩa, vector sparse nắm từ khoá chính xác. Kết hợp cả hai
gần như luôn tốt hơn từng cách riêng, nhất là với tiếng Việt khi danh từ riêng,
mã sản phẩm và thuật ngữ cần khớp đúng chữ.

BM25
----
Chấm điểm từ khoá theo xác suất cổ điển. Nhanh, không cần GPU, không cần dữ
liệu huấn luyện. Sinh ra {token: điểm_bm25} cho mỗi document.
Bắt buộc gọi ``.fit(corpus_texts)`` trước khi embed.

SPLADE
------
Sparse Lexical and Expansion Model (Formal et al., 2021). Học biểu diễn sparse
qua masked-language model, tự mở rộng "đái tháo đường" →
{"tiểu_đường", "glucose", "insulin", …}. Cần model transformer, nên có GPU.
"""

from __future__ import annotations

from typing import Literal


class BM25Embedder:
    """
    Embedding sparse BM25 bằng thư viện rank-bm25.

    Ví dụ
    -----
    >>> bm25 = BM25Embedder()
    >>> bm25.fit(corpus_texts)                    # dựng thống kê IDF
    >>> bm25.embed_documents(texts)               # list[dict[str, float]]
    >>> bm25.embed_query("câu hỏi của tôi")       # dict[str, float]
    """

    def __init__(self):
        self._bm25 = None

    def fit(self, corpus: list[str]) -> "BM25Embedder":
        """Dựng thống kê IDF từ corpus. Bắt buộc gọi trước khi embed."""
        from rank_bm25 import BM25Okapi

        self._bm25 = BM25Okapi([self._tokenize(t) for t in corpus])
        return self

    def embed_documents(self, texts: list[str]) -> list[dict[str, float]]:
        """Trả về một vector sparse cho mỗi document."""
        return [self._vector(t) for t in texts]

    def embed_query(self, query: str) -> dict[str, float]:
        """Trả về vector sparse cho một chuỗi query."""
        return self._vector(query)

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return text.lower().split()

    def _vector(self, text: str) -> dict[str, float]:
        if self._bm25 is None:
            raise RuntimeError("Call .fit(corpus_texts) before embedding.")
        tokens = list(set(self._tokenize(text)))
        scores: dict[str, float] = {}
        for token in tokens:
            score = float(self._bm25.get_scores([token]).max())
            if score > 0:
                scores[token] = score
        return scores


class SPLADEEmbedder:
    """
    Embedding sparse SPLADE qua masked-language model.

    Sinh vector sparse số chiều lớn trải trên toàn bộ từ vựng; các vị trí khác 0
    thể hiện độ quan trọng của từng token, kể cả những từ liên quan được mở rộng
    ngầm.

    Tham số
    -------
    model_name : Model SPLADE trên HuggingFace.
                 Mặc định "naver/splade-cocondenser-ensembledistil".
    device     : "cpu" | "cuda"
    """

    def __init__(
        self,
        model_name: str = "naver/splade-cocondenser-ensembledistil",
        device:     str = "cpu",
    ):
        self.model_name = model_name
        self.device     = device
        self._model     = None
        self._tokenizer = None

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForMaskedLM, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model     = AutoModelForMaskedLM.from_pretrained(self.model_name)
        self._model.to(self.device)
        self._model.eval()

    def embed_documents(self, texts: list[str]) -> list[dict[str, float]]:
        """Trả về một vector sparse cho mỗi document."""
        return [self._vector(t) for t in texts]

    def embed_query(self, query: str) -> dict[str, float]:
        """Trả về vector sparse cho một chuỗi query."""
        return self._vector(query)

    # SPLADE dựa vào model chứ không phụ thuộc corpus, nên fit() là rỗng
    def fit(self, corpus: list[str]) -> "SPLADEEmbedder":
        return self

    def _vector(self, text: str) -> dict[str, float]:
        import torch

        self._load()
        tokens = self._tokenizer(
            text, return_tensors="pt", truncation=True,
            max_length=512, padding=True,
        )
        tokens = {k: v.to(self.device) for k, v in tokens.items()}

        with torch.no_grad():
            logits = self._model(**tokens).logits           # (1, seq, vocab)

        # Gộp theo cách của SPLADE: ReLU → log1p → lấy max theo chiều chuỗi
        weights = torch.log1p(torch.relu(logits))
        agg     = weights.max(dim=1).values.squeeze(0)     # (vocab,)

        nonzero    = agg.nonzero(as_tuple=True)[0].tolist()
        vocab      = self._tokenizer.get_vocab()
        id_to_tok  = {v: k for k, v in vocab.items()}

        return {
            id_to_tok[idx]: float(agg[idx])
            for idx in nonzero
            if idx in id_to_tok
        }


# ---------------------------------------------------------------------------
# Factory tạo embedder sparse
# ---------------------------------------------------------------------------

SparseMethod = Literal["bm25", "splade"]


def get_sparse_embedder(
    method:     SparseMethod = "bm25",
    model_name: str = "naver/splade-cocondenser-ensembledistil",
    device:     str = "cpu",
) -> BM25Embedder | SPLADEEmbedder:
    """
    Trả về một embedder sparse.

    Tham số
    -------
    method     : "bm25" | "splade"
    model_name : Tên model SPLADE; BM25 bỏ qua.
    device     : "cpu" | "cuda", chỉ áp cho SPLADE.
    """
    if method == "bm25":
        return BM25Embedder()
    if method == "splade":
        return SPLADEEmbedder(model_name=model_name, device=device)
    raise ValueError(f"Unknown sparse method: '{method}'")
