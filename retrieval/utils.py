"""
retrieval/utils.py
==================
Các hàm gộp kết quả xếp hạng dùng chung cho các module retrieval.
"""

from __future__ import annotations

from collections import defaultdict

from langchain_core.documents import Document


def reciprocal_rank_fusion(
    ranked_lists: list[list[Document]],
    k: int = 60,
) -> list[Document]:
    """
    Reciprocal Rank Fusion (RRF) — gộp nhiều danh sách đã xếp hạng.

    Điểm RRF:  score(d) = Σ  1 / (k + rank_i(d))

    k=60 là hằng số chuẩn theo Cormack et al. (2009). k lớn làm chênh lệch mượt
    hơn, k nhỏ thiên vị mạnh cho các hạng đầu.

    Trả về document sắp theo điểm RRF giảm dần.
    """
    scores:  dict[str, float]    = defaultdict(float)
    doc_map: dict[str, Document] = {}

    for ranked in ranked_lists:
        for rank, doc in enumerate(ranked, start=1):
            key           = doc.page_content.strip()
            scores[key]  += 1.0 / (k + rank)
            doc_map[key]  = doc

    for doc in doc_map.values():
        doc.metadata["rrf_score"] = scores[doc.page_content.strip()]

    return [doc_map[k] for k in sorted(scores, key=scores.__getitem__, reverse=True)]


def weighted_fusion(
    dense_docs:  list[Document],
    sparse_docs: list[Document],
    alpha: float = 0.5,
) -> list[Document]:
    """
    Kết hợp tuyến tính có trọng số giữa điểm dense và sparse.

    Điểm cuối = alpha * dense_norm + (1 - alpha) * sparse_norm, mỗi danh sách
    được chuẩn hoá min-max trước khi cộng.

    alpha = 1.0 → thuần dense; alpha = 0.0 → thuần sparse.
    """
    def normalise(docs: list[Document], key: str) -> dict[str, float]:
        raw = [doc.metadata.get(key, 0.0) for doc in docs]
        # Không có điểm tường minh thì chấm theo thứ hạng
        # (phần lớn VectorStore dense không tự đặt relevance_score)
        if not raw or all(s == 0.0 for s in raw):
            raw = [1.0 / (i + 1) for i in range(len(docs))]
        lo, hi = min(raw), max(raw)
        rng    = hi - lo or 1.0
        return {doc.page_content.strip(): (raw[i] - lo) / rng for i, doc in enumerate(docs)}

    d_scores = normalise(dense_docs,  "relevance_score")
    s_scores = normalise(sparse_docs, "bm25_score")

    all_docs: dict[str, Document] = {
        doc.page_content.strip(): doc
        for doc in dense_docs + sparse_docs
    }
    combined: dict[str, float] = {
        key: alpha * d_scores.get(key, 0.0) + (1 - alpha) * s_scores.get(key, 0.0)
        for key in all_docs
    }
    for key, doc in all_docs.items():
        doc.metadata["hybrid_score"] = combined[key]

    return [all_docs[k] for k in sorted(combined, key=combined.__getitem__, reverse=True)]


def distribution_based_fusion(ranked_lists: list[list[Document]]) -> list[Document]:
    """
    Distribution-Based Score Fusion (DBSF).

    Chuẩn hoá z-score điểm của từng danh sách rồi lấy trung bình. Ít nhạy với
    điểm ngoại lai hơn min-max.
    """
    import math

    key_scores: dict[str, list[float]] = defaultdict(list)
    doc_map:    dict[str, Document]    = {}

    for ranked in ranked_lists:
        raw = [doc.metadata.get("relevance_score", 0.0) for doc in ranked]
        if not raw:
            continue
        # Không có điểm tường minh thì chấm theo thứ hạng
        if all(s == 0.0 for s in raw):
            raw = [1.0 / (i + 1) for i in range(len(ranked))]
        mean = sum(raw) / len(raw)
        std  = math.sqrt(sum((x - mean) ** 2 for x in raw) / len(raw)) or 1.0
        for doc, score in zip(ranked, raw):
            key = doc.page_content.strip()
            key_scores[key].append((score - mean) / std)
            doc_map[key] = doc

    final = {k: sum(v) / len(v) for k, v in key_scores.items()}
    for doc in doc_map.values():
        doc.metadata["dbsf_score"] = final.get(doc.page_content.strip(), 0.0)

    return [doc_map[k] for k in sorted(final, key=final.__getitem__, reverse=True)]
