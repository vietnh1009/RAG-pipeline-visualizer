"""
retrieval/
==========
Truy hồi tài liệu — Stage 6 của pipeline RAG.

API công khai
-------------
    from retrieval import get_retriever, build_retriever_from_config

    retriever = get_retriever("hybrid", vector_store=store,
                              documents=chunks, top_k=10)
    docs = retriever.retrieve(transform_result)

Các chiến lược
--------------
  dense           Tìm ANN theo độ tương đồng vector
  sparse          Tìm từ khoá bằng BM25
  hybrid          Dense + BM25, gộp bằng RRF / weighted / DBSF ← khuyến nghị
  multi_query     Chạy N query rồi gộp bằng RRF
  parent_document Tìm chunk con, trả về chunk cha làm ngữ cảnh
  sentence_window Tìm theo câu, mở rộng ±window câu lân cận
  multi_hop       Lặp truy hồi rồi suy luận, cho câu hỏi nhiều bước
  contextual      Dense + ngưỡng điểm + đa dạng hoá MMR
"""

from retrieval.factory import get_retriever, build_retriever_from_config

__all__ = ["get_retriever", "build_retriever_from_config"]
