from __future__ import annotations
from typing import *

from avito.prompts import (
    shouldSplitInstructionPrompt,
    shouldSplitPromptWithoutRAG,
    shouldSplitPromptWithRAG,
)


def test_should_split_instruction_contains_bullet_rule_false() -> None:
    normalized = shouldSplitInstructionPrompt.lower()
    assert "буллет" in normalized
    assert "нумерац" in normalized
    assert "в таком случае ставь false" in normalized
    assert "ыполняю ремонтные и отделочные работы" in normalized
    assert "стрoительствo и ремонт" in normalized
    assert "укладка плитки в доме квартире быстро качественно и не дорого" in normalized
    assert "только одна микрокатегория" in normalized
    assert "только если мастер явно подает услугу как отдельную" in normalized
    assert "устанавливаем натяжные потолки любой сложности" in normalized
    assert "профессиональную услугу по малярным работам" in normalized
    assert "специализированный оффер по малярным работам" in normalized
    assert "самостоятельная отдельная услуга" in normalized
    assert "даже без слов-маркеров" in normalized
    assert "качественные ремонты ванных комнат и сан.узлов под ключ" in normalized
    assert "105 + 102 + 103" in normalized
    assert '"multicats": true/false' in normalized


def test_should_split_user_prompt_without_rag_contains_false_hint() -> None:
    prompt = shouldSplitPromptWithoutRAG("- Сантехника\n- Электрика\nДелаю отдельно")
    normalized = prompt.lower()
    assert "буллетами/нумерацией" in normalized
    assert "выбирай shouldsplit=false" in normalized
    assert "только при явной отдельной подаче услуги" in normalized
    assert "может быть true даже без слов" in normalized
    assert "верни json" in normalized


def test_should_split_user_prompt_with_rag_contains_false_hint() -> None:
    prompt = shouldSplitPromptWithRAG(
        desc="1) Сантехника\n2) Электрика\nОтдельно",
        top_k=[("тест", False, [])],
    )
    normalized = prompt.lower()
    assert "буллетами/нумерацией" in normalized
    assert "выбирай shouldsplit=false" in normalized
    assert "только при явной отдельной подаче услуги" in normalized
    assert "может быть true даже без слов" in normalized
    assert "верни json" in normalized


