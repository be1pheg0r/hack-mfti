from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GraphNodeNames:
    """Идентификаторы узлов графа should-split."""

    clean: str = "clean"
    pre_filter: str = "pre_filter"
    pre_filter_exit: str = "pre_filter_exit"
    rag_split: str = "rag_split"
    categorize: str = "categorize"
    llm_exit: str = "llm_exit"
    draft_generate: str = "draft_generate"
    drafts_exit: str = "drafts_exit"


@dataclass(frozen=True)
class GraphRouteNames:
    """Идентификаторы роутов для conditional edges."""

    to_rag: str = "to_rag"
    to_pre_filter_exit: str = "to_pre_filter_exit"
    to_categorize: str = "to_categorize"
    to_draft_generate: str = "to_draft_generate"
    to_llm_exit: str = "to_llm_exit"


@dataclass(frozen=True)
class StageNames:
    """Ключи стадий для замера времени."""

    clean: str = "clean"
    pre_filter: str = "pre_filter"
    rag_split: str = "rag_split"
    categorize: str = "categorize"
    draft_generate: str = "draft_generate"
    pre_filter_exit: str = "pre_filter_exit"
    llm_exit: str = "llm_exit"
    drafts_exit: str = "drafts_exit"

