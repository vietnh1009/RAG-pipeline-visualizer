"""
embedding/utils.py
==================
Shared utility functions used across embedding modules.
"""

from __future__ import annotations


def truncate_embeddings(
    vectors: list[list[float]],
    dimensions: int,
) -> list[list[float]]:
    """
    Truncate dense vectors to fewer dimensions (Matryoshka / MRL).

    Works with models trained with Matryoshka Representation Learning where
    the first N dimensions capture the most important information.

    Compatible models: OpenAI text-embedding-3-*, Nomic Embed v2,
                       Qwen3-Embedding, E5-mistral-7b.

    Parameters
    ----------
    vectors    : Full-dimension embedding vectors.
    dimensions : Target number of dimensions (must be ≤ original dim).
    """
    return [v[:dimensions] for v in vectors]


def binarise_embeddings(vectors: list[list[float]]) -> list[list[int]]:
    """
    Convert float32 vectors to binary (0/1) — 32× storage reduction.

    Binary embeddings use Hamming distance (XOR + popcount) which is
    30–40× faster than float cosine similarity.
    Typical recall@10 vs float32: ~90–96 %.

    Best used as a first-pass filter: search binary, re-rank with float32.
    """
    return [[1 if v > 0 else 0 for v in vec] for vec in vectors]


def quantise_to_int8(vectors: list[list[float]]) -> list[list[int]]:
    """
    Scalar Quantisation (SQ8): float32 → int8 — 4× RAM reduction.

    Speed gain: 2–4× faster ANN search.
    Quality loss: recall@10 typically drops ~1 % only.

    This is the first optimisation to apply in production before exploring
    more complex approaches (PQ, binary).
    """
    import numpy as np

    result: list[list[int]] = []
    for vec in vectors:
        arr    = np.array(vec, dtype=np.float32)
        scaled = np.clip(arr * 127, -128, 127).astype(np.int8)
        result.append(scaled.tolist())
    return result
