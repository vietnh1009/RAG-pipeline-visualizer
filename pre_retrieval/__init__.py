"""
pre_retrieval/
==============
Biến đổi query — Stage 5 của pipeline RAG.

API công khai
-------------
    from pre_retrieval import build_pipeline, build_pipeline_from_config

    pipeline = build_pipeline(
        transformations=["rewrite", "multi_query"],
        llm_model="claude-haiku-4-5-20251001",
        multi_query_count=3,
    )
    result = pipeline.transform("rag hoạt động thế nào?")
    result.queries          # danh sách query để đưa sang retrieval
    result.metadata_filter  # self_query điền
    result.retrieval_path   # route điền

Các phép biến đổi
-----------------
  none         Không biến đổi
  rewrite      Sửa chính tả/ngữ pháp, khử đại từ, chuẩn hoá
  expand       Thêm từ đồng nghĩa / từ liên quan
  hyde         Sinh tài liệu giả định rồi embed tài liệu đó
  step_back    Lùi về câu hỏi nền tảng rộng hơn
  multi_query  Tách thành N góc nhìn truy hồi
  decompose    Chẻ câu hỏi ghép thành các câu hỏi con
  self_query   Trích metadata filter từ ngôn ngữ tự nhiên
  route        Định tuyến query tới nhánh retrieval phù hợp
"""

from pre_retrieval.factory  import build_pipeline, build_pipeline_from_config
from pre_retrieval.pipeline import PreRetrievalPipeline
from pre_retrieval.base     import TransformResult

__all__ = [
    "build_pipeline",
    "build_pipeline_from_config",
    "PreRetrievalPipeline",
    "TransformResult",
]
