from __future__ import annotations

from avito.prompts import (
    draftGenerationPrompt,
    shouldSplitInstructionPrompt,
    shouldSplitPromptWithoutRAG,
    shouldSplitPromptWithRAG,
)


def test_should_split_instruction_contains_neutral_bullets_rule() -> None:
    normalized = shouldSplitInstructionPrompt.lower()
    assert "буллет" in normalized
    assert "нумерац" in normalized
    assert "нейтральны" in normalized
    assert "ыполняю ремонтные и отделочные работы" in normalized
    assert "стрoительствo и ремонт" in normalized
    assert "укладка плитки в доме квартире быстро качественно и не дорого" in normalized
    assert "даже если все написано в одном объявлении и в виде списка" in normalized
    assert "даже если выделяется только одна дополнительная микрокатегория" in normalized
    assert "устанавливаем натяжные потолки любой сложности" in normalized
    assert "профессиональную услугу по малярным работам" in normalized
    assert "качественные ремонты ванных комнат и сан.узлов под ключ" in normalized
    assert "105 + 102 + 103" in normalized
    assert '"shouldsplit": true/false' in normalized


def test_should_split_user_prompt_without_rag_contains_neutral_hint() -> None:
    prompt = shouldSplitPromptWithoutRAG("- Сантехника\n- Электрика\nДелаю отдельно")
    normalized = prompt.lower()
    assert "буллеты/нумерация" in normalized
    assert "нейтральны" in normalized
    assert "по смыслу текста" in normalized
    assert "хотя бы одну дополнительную самостоятельную микрокатегорию" in normalized
    assert "верни json" in normalized


def test_should_split_user_prompt_with_rag_contains_neutral_hint() -> None:
    prompt = shouldSplitPromptWithRAG(
        desc="1) Сантехника\n2) Электрика\nОтдельно",
        top_k=[("тест", False, [])],
    )
    normalized = prompt.lower()
    assert "буллеты/нумерация" in normalized
    assert "нейтральны" in normalized
    assert "по смыслу текста" in normalized
    assert "хотя бы одну дополнительную самостоятельную микрокатегорию" in normalized
    assert "основывай решение" in normalized
    assert "не делай выводы из домыслов" in normalized
    assert "верни json" in normalized


def test_categorization_prompt_with_rag_requires_no_guessing() -> None:
    from avito.prompts import categorizationPromptWithRAG

    prompt = categorizationPromptWithRAG(
        desc="делаю ремонт ванной",
        top_k=[("пример", True, ["102"])],
    )
    normalized = prompt.lower()

    assert "основывайся на категориях из rag-примеров" in normalized
    assert "не добавляй категории из предположений" in normalized


def test_draft_generation_prompt_contains_target_category() -> None:
    prompt = draftGenerationPrompt(
        description="делаю сантехнику",
        mc_title="Сантехника",
    )
    normalized = prompt.lower()

    assert "исходное объявление" in normalized
    assert "целевая микрокатегория" in normalized
    assert "сантехника" in normalized


