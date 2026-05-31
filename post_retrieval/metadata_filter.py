"""
post_retrieval/metadata_filter.py
===================================
Hard Metadata Filter — keep only documents matching explicit conditions.

Unlike vector DB metadata filtering (which runs before/during ANN search),
this runs AFTER retrieval as a safety net for:
  - Conditions not supported by the vector DB's filter API.
  - Dynamic conditions only known at post-retrieval time.
  - Debugging / validating retrieval quality.

Supported operators
-------------------
  eq       : exact equality       field_value == value
  ne       : not equal            field_value != value
  in       : membership           field_value in [value, ...]
  gt       : greater than         field_value > value
  lt       : less than            field_value < value
  gte      : greater or equal     field_value >= value
  lte      : less or equal        field_value <= value
  contains : substring match      value in str(field_value)

All conditions are combined with AND logic.
"""

from __future__ import annotations

from langchain_core.documents import Document

from post_retrieval.base import BasePostProcessor


class MetadataFilter(BasePostProcessor):
    """
    Keep only documents that match all specified metadata conditions.

    Parameters
    ----------
    conditions : List of condition dicts, each with keys:
                   field    : metadata key to filter on
                   operator : eq | ne | in | gt | lt | gte | lte | contains
                   value    : comparison value

    Example
    -------
    >>> f = MetadataFilter(conditions=[
    ...     {"field": "language", "operator": "eq",       "value": "vi"},
    ...     {"field": "year",     "operator": "gte",      "value": 2022},
    ...     {"field": "source",   "operator": "contains", "value": "policy"},
    ... ])
    """

    def __init__(self, conditions: list[dict]):
        self.conditions = conditions

    def process(self, query: str, docs: list[Document]) -> list[Document]:
        return [doc for doc in docs if self._matches(doc)]

    def _matches(self, doc: Document) -> bool:
        for cond in self.conditions:
            field    = cond["field"]
            operator = cond["operator"]
            value    = cond["value"]
            fval     = doc.metadata.get(field)

            if operator == "eq"       and fval != value:                    return False
            if operator == "ne"       and fval == value:                    return False
            if operator == "in"       and fval not in value:                return False
            if operator == "gt"       and not (fval is not None and fval > value):  return False
            if operator == "lt"       and not (fval is not None and fval < value):  return False
            if operator == "gte"      and not (fval is not None and fval >= value): return False
            if operator == "lte"      and not (fval is not None and fval <= value): return False
            if operator == "contains" and value not in str(fval or ""):     return False
        return True
