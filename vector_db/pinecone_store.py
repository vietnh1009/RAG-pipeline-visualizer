"""
vector_db/pinecone_store.py
============================
Pinecone — vector DB serverless có quản lý: không phải vận hành, tự co giãn,
API ổn định.

Tránh index lại: xem ``index.describe_index_stats().total_vector_count``; lớn
hơn 0 và force_reindex=False thì trả về index sẵn có.

Quy mô  : ~ 1 tỷ vector (bản Serverless)
Dùng khi: muốn cloud có quản lý, không tự vận hành hạ tầng.
Hạn chế : phụ thuộc nhà cung cấp, chi phí tăng theo quy mô.

Biến môi trường: PINECONE_API_KEY
"""

from __future__ import annotations

import logging
import os

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore

from vector_db.base import BaseVectorStore

logger = logging.getLogger(__name__)


class PineconeVectorStore(BaseVectorStore):
    """
    Vector store Pinecone.

    Tham số
    -------
    collection_name : Tên index trong Pinecone.
    dimension       : Số chiều vector, phải khớp model embedding.
    metric          : "cosine" | "euclidean" | "dotproduct"
    cloud           : Nhà cung cấp cho bản serverless ("aws" | "gcp" | "azure").
    region          : Vùng, ví dụ "us-east-1".
    force_reindex   : Xoá index và tạo lại.
    """

    def __init__(
        self,
        collection_name: str  = "rag",
        dimension:       int  = 1536,
        metric:          str  = "cosine",
        cloud:           str  = "aws",
        region:          str  = "us-east-1",
        force_reindex:   bool = False,
    ):
        super().__init__(collection_name, force_reindex)
        self.dimension = dimension
        self.metric    = metric
        self.cloud     = cloud
        self.region    = region

    def get_or_create(self, chunks: list[Document], embedder) -> VectorStore:
        from pinecone import Pinecone, ServerlessSpec
        from langchain_pinecone import PineconeVectorStore as _PVS

        api_key     = os.environ["PINECONE_API_KEY"]
        pc          = Pinecone(api_key=api_key)
        lc_embedder = self._langchain_embedder(embedder)
        existing    = [idx.name for idx in pc.list_indexes()]

        # Index đã có dữ liệu thì dùng lại
        if self.collection_name in existing and not self.force_reindex:
            index = pc.Index(self.collection_name)
            count = index.describe_index_stats().total_vector_count
            if count > 0:
                logger.info("Pinecone: index '%s' has %d vectors — skipping.", self.collection_name, count)
                return _PVS(index=index, embedding=lc_embedder)

        # force_reindex: xoá index cũ
        if self.collection_name in existing and self.force_reindex:
            pc.delete_index(self.collection_name)
            existing = []

        # Tạo index serverless
        if self.collection_name not in existing:
            logger.info("Pinecone: creating index '%s'.", self.collection_name)
            pc.create_index(
                name=self.collection_name,
                dimension=self.dimension,
                metric=self.metric,
                spec=ServerlessSpec(cloud=self.cloud, region=self.region),
            )

        logger.info("Pinecone: upserting %d chunks.", len(chunks))
        return _PVS.from_documents(
            documents=self.sanitize_metadata(chunks),
            embedding=lc_embedder,
            index_name=self.collection_name,
        )
