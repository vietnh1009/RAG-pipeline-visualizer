"""
embedding/sparse_embedder.py
=============================
Sparse embedders for hybrid retrieval — BM25 and SPLADE.

Dense vectors capture semantic meaning; sparse vectors capture exact
keywords.  Hybrid retrieval (dense + sparse) consistently outperforms
either approach alone — especially for Vietnamese where proper nouns,
product codes, and technical terms need exact matching.

BM25
----
Classic probabilistic keyword scoring.  Fast, no GPU, no training data.
Produces {token: bm25_score} per document.
Must call .fit(corpus_texts) before embedding documents.

SPLADE
------
Sparse Lexical and Expansion Model (Formal et al., 2021).
Learns sparse representations via a masked-language model, automatically
expanding "đái tháo đường" → {"tiểu_đường", "glucose", "insulin", ...}.
Requires a transformer model; GPU recommended for speed.
"""

from __future__ import annotations

from typing import Literal


class BM25Embedder:
    """
    BM25 sparse embedding using rank-bm25.

    Usage
    -----
    >>> bm25 = BM25Embedder()
    >>> bm25.fit(corpus_texts)                    # build IDF statistics
    >>> doc_vecs  = bm25.embed_documents(texts)   # list[dict[str, float]]
    >>> query_vec = bm25.embed_query("my query")  # dict[str, float]
    """

    def __init__(self):
        self._bm25 = None

    def fit(self, corpus: list[str]) -> "BM25Embedder":
        """Build BM25 IDF statistics from the corpus. Must be called before embedding."""
        from rank_bm25 import BM25Okapi

        self._bm25 = BM25Okapi([self._tokenize(t) for t in corpus])
        return self

    def embed_documents(self, texts: list[str]) -> list[dict[str, float]]:
        """Return a sparse vector per document."""
        return [self._vector(t) for t in texts]

    def embed_query(self, query: str) -> dict[str, float]:
        """Return a sparse vector for one query string."""
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
    SPLADE sparse embedding via a masked-language model.

    Produces high-dimensional sparse vectors over the full vocabulary.
    Non-zero entries represent importance of each token including
    implicitly expanded related terms.

    Parameters
    ----------
    model_name : HuggingFace SPLADE model.
                 Default: "naver/splade-cocondenser-ensembledistil"
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
        """Return a sparse vector per document."""
        return [self._vector(t) for t in texts]

    def embed_query(self, query: str) -> dict[str, float]:
        """Return a sparse vector for one query string."""
        return self._vector(query)

    # No fit() needed — SPLADE is model-based, not corpus-dependent.
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

        # SPLADE aggregation: max over sequence, ReLU, log1p
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
# Factory
# ---------------------------------------------------------------------------

SparseMethod = Literal["bm25", "splade"]


def get_sparse_embedder(
    method:     SparseMethod = "bm25",
    model_name: str = "naver/splade-cocondenser-ensembledistil",
    device:     str = "cpu",
) -> BM25Embedder | SPLADEEmbedder:
    """
    Return a sparse embedder instance.

    Parameters
    ----------
    method     : "bm25" | "splade"
    model_name : SPLADE model name (ignored for BM25).
    device     : "cpu" | "cuda" (SPLADE only).
    """
    if method == "bm25":
        return BM25Embedder()
    if method == "splade":
        return SPLADEEmbedder(model_name=model_name, device=device)
    raise ValueError(f"Unknown sparse method: '{method}'")
