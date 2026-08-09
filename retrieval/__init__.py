"""
retrieval/
==========
Document retrieval package — Stage 6 of the full RAG pipeline.

Public API
----------
    from retrieval import get_retriever, build_retriever_from_config

    retriever = get_retriever("hybrid", vector_store=store,
                              documents=chunks, top_k=10)
    docs = retriever.retrieve(transform_result)

Available strategies
--------------------
  dense           Standard ANN vector similarity search
  sparse          BM25 keyword search
  hybrid          Dense + BM25 fused via RRF / weighted / DBSF  ← recommended
  multi_query     Run N queries, merge results via RRF
  parent_document Search child chunks, return parent context
  sentence_window Search sentences, expand to ±window neighbours
  multi_hop       Iterative retrieve-then-reason for complex questions
  contextual      Dense + score threshold + MMR diversity
"""

from retrieval.factory import get_retriever, build_retriever_from_config

__all__ = ["get_retriever", "build_retriever_from_config"]
