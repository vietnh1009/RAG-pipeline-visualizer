"""
chunking/deduplication.py
=========================
Post-split deduplication — remove duplicate or near-duplicate chunks.

Run after any chunker to clean the chunk list before embedding.

| Method   | Mechanism                       | Speed      | Catches                  |
|----------|---------------------------------|------------|--------------------------|
| exact    | MD5 hash comparison             | Very fast  | Identical text only      |
| minhash  | MinHash + LSH shingle similarity| Fast       | Near-duplicates (~typos) |
| semantic | Cosine similarity of embeddings | Slower     | Paraphrases, translations|
"""

from __future__ import annotations

import hashlib
from typing import Literal

from langchain_core.documents import Document


def deduplicate_chunks(
    chunks:    list[Document],
    method:    Literal["exact", "minhash", "semantic"] = "exact",
    threshold: float = 0.95,
    model:     str = "sentence-transformers/all-MiniLM-L6-v2",
) -> list[Document]:
    """
    Remove duplicate or near-duplicate chunks.

    Parameters
    ----------
    chunks    : Output of any chunker's ``.split()`` method.
    method    : Deduplication strategy (see module docstring).
    threshold : Similarity above which two chunks are considered duplicates
                (used by minhash and semantic methods).
    model     : Sentence-transformers model for semantic dedup.

    Returns
    -------
    Deduplicated list of Documents (first occurrence kept; order preserved).
    """
    if method == "exact":
        return _exact(chunks)
    if method == "minhash":
        return _minhash(chunks, threshold)
    if method == "semantic":
        return _semantic(chunks, threshold, model)
    raise ValueError(f"Unknown dedup method: '{method}'")


# ---------------------------------------------------------------------------
# Exact hash
# ---------------------------------------------------------------------------

def _exact(chunks: list[Document]) -> list[Document]:
    seen:   set[str]       = set()
    unique: list[Document] = []
    for chunk in chunks:
        h = hashlib.md5(chunk.page_content.encode()).hexdigest()
        if h not in seen:
            seen.add(h)
            unique.append(chunk)
    return unique


# ---------------------------------------------------------------------------
# MinHash LSH
# ---------------------------------------------------------------------------

def _minhash(chunks: list[Document], threshold: float) -> list[Document]:
    from datasketch import MinHash, MinHashLSH

    lsh    = MinHashLSH(threshold=threshold, num_perm=128)
    unique: list[Document] = []

    for i, chunk in enumerate(chunks):
        mh = MinHash(num_perm=128)
        for word in chunk.page_content.lower().split():
            mh.update(word.encode("utf-8"))
        key = f"chunk_{i}"
        if not lsh.query(mh):
            lsh.insert(key, mh)
            unique.append(chunk)

    return unique


# ---------------------------------------------------------------------------
# Semantic (cosine similarity)
# ---------------------------------------------------------------------------

def _semantic(chunks: list[Document], threshold: float, model_name: str) -> list[Document]:
    import numpy as np
    from sentence_transformers import SentenceTransformer

    model      = SentenceTransformer(model_name)
    texts      = [c.page_content for c in chunks]
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)

    unique_indices: list[int] = []
    for i in range(len(chunks)):
        is_dup = any(
            float(np.dot(embeddings[i], embeddings[j])) >= threshold
            for j in unique_indices
        )
        if not is_dup:
            unique_indices.append(i)

    return [chunks[i] for i in unique_indices]
