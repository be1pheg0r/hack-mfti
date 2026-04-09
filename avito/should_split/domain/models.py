from __future__ import annotations

from typing import *

from pydantic import BaseModel, Field


class DraftCandidate(BaseModel):
    """Черновик объявления для микрокатегории.

    Attributes:
        mc_id: Идентификатор микрокатегории.
        mc_title: Название микрокатегории.
        text: Сгенерированный черновик текста (в заглушке пустой).
    """

    mc_id: int
    mc_title: str
    text: str = ""


class RetrievedExample(BaseModel):
    """Кандидат-объявление из retrieval слоя.

    Attributes:
        text: Текст объявления-кандидата.
        shouldSplit: Эталонный признак split для кандидата.
        categories: Названия целевых микрокатегорий кандидата.
        score: Релевантность кандидата.
    """

    text: str
    shouldSplit: bool
    categories: list[str] = Field(default_factory=list)
    score: float = 0.0


class RunState(BaseModel):
    """Состояние выполнения графа should-split.

    Attributes:
        description: Входной текст объявления.
        shouldSplit: Итоговое решение о необходимости split.
        pre_filter_verdict: Вердикт префильтра.
        rag_split_verdict: Вердикт стадии rag_split.
        multiCats: Флаг наличия нескольких тематических категорий в одном объявлении.
        categorized_mc_ids: Категории, найденные на стадии categorize.
        top_k_examples: Примеры, использованные в RAG-промпте.
        drafts: Список draft-кандидатов (заглушка).
        stage_durations_sec: Время выполнения стадий в секундах.
        metadata: Служебные метаданные этапов.
    """

    description: str
    shouldSplit: bool = False
    multiCats: bool = False
    pre_filter_verdict: bool = False
    rag_split_verdict: bool = False
    categorized_mc_ids: list[int] = Field(default_factory=list)
    top_k_examples: list[RetrievedExample] = Field(default_factory=list)
    drafts: list[DraftCandidate] = Field(default_factory=list)
    stage_durations_sec: dict[str, float] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

