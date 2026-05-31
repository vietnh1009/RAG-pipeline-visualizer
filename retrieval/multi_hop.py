"""
retrieval/multi_hop.py
=======================
Multi-Hop Retrieval — iterative retrieve-then-reason for complex questions.

Multi-hop questions require chaining multiple retrieval steps:
  Q: "Who founded the company that acquired DeepMind?"
  Hop 1: retrieve → "Google acquired DeepMind"
  Hop 2: retrieve → "Larry Page and Sergey Brin founded Google"
  Answer: Larry Page and Sergey Brin

Algorithm
---------
1. Retrieve k docs for the current query.
2. Feed accumulated context to the LLM: should it continue or stop?
3. If continue → generate a follow-up query and repeat.
4. Stop when LLM says DONE or max_hops is reached.
5. Return all retrieved docs deduplicated and capped to top_k.

Use when: multi-step reasoning, relationship chaining, comparison across
          multiple documents that each contain only partial information.
"""

from __future__ import annotations

import logging

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore

from retrieval.base import BaseRetriever
from retrieval.utils import deduplicate

logger = logging.getLogger(__name__)

_HOP_PROMPT = (
    "You are answering a question by iteratively retrieving documents.\n\n"
    "Original question: {question}\n\n"
    "Retrieved context so far:\n{context}\n\n"
    "Do you have enough information to fully answer the original question?\n"
    "If YES → respond with: DONE\n"
    "If NO  → respond with a single follow-up search query to find the "
    "missing information (output ONLY the query, nothing else):"
)


class MultiHopRetriever(BaseRetriever):
    """
    Iteratively retrieve and reason until the question can be answered.

    Parameters
    ----------
    vector_store : Populated LangChain VectorStore.
    top_k        : Final documents returned (across all hops).
    max_hops     : Maximum retrieval iterations.
    candidate_k  : Documents to retrieve per hop.
    llm_model    : LLM used to decide whether to continue.
    llm_provider : "openai" | "anthropic" | "google"
    """

    def __init__(
        self,
        vector_store: VectorStore,
        top_k:        int = 5,
        max_hops:     int = 3,
        candidate_k:  int = 5,
        llm_model:    str = "gpt-4.1-mini",
        llm_provider: str = "openai",
    ):
        super().__init__(vector_store, top_k)
        self.max_hops     = max_hops
        self.candidate_k  = candidate_k
        self.llm_model    = llm_model
        self.llm_provider = llm_provider

    def retrieve(self, result) -> list[Document]:
        original_query = result.original_query
        current_query  = result.queries[0] if result.queries else original_query
        all_docs: list[Document] = []

        for hop in range(self.max_hops):
            hop_docs = self._search(
                query=current_query, k=self.candidate_k,
                filter=result.metadata_filter,
            )
            all_docs.extend(hop_docs)

            context  = "\n\n".join(d.page_content for d in all_docs[:10])
            followup = self._ask_followup(original_query, context)

            if not followup or followup.upper().startswith("DONE"):
                logger.debug("MultiHop: stopping at hop %d.", hop + 1)
                break

            logger.debug("MultiHop hop %d: follow-up='%s'.", hop + 1, followup[:60])
            current_query = followup

        return deduplicate(all_docs)[:self.top_k]

    def _ask_followup(self, question: str, context: str) -> str:
        prompt = _HOP_PROMPT.format(question=question, context=context[:3000])
        try:
            if self.llm_provider == "openai":
                from openai import OpenAI
                r = OpenAI().chat.completions.create(
                    model=self.llm_model, temperature=0, max_tokens=128,
                    messages=[{"role": "user", "content": prompt}],
                )
                return r.choices[0].message.content.strip()
            if self.llm_provider == "anthropic":
                import anthropic
                r = anthropic.Anthropic().messages.create(
                    model=self.llm_model, max_tokens=128,
                    messages=[{"role": "user", "content": prompt}],
                )
                return r.content[0].text.strip()
        except Exception as exc:
            logger.warning("MultiHop LLM call failed: %s", exc)
        return "DONE"
