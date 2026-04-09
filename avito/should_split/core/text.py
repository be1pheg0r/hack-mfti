from __future__ import annotations

from avito.constants import SHOULD_SPLIT_CLEAN_TEXT_PATTERN, SHOULD_SPLIT_MULTI_SPACE_PATTERN


def clean_description(text: str) -> str:
    """Нормализует описание объявления до кириллицы и пробелов.

    Args:
        text: Исходный текст объявления.

    Returns:
        Очищенный текст.
    """
    cleaned = SHOULD_SPLIT_CLEAN_TEXT_PATTERN.sub(" ", text)
    cleaned = SHOULD_SPLIT_MULTI_SPACE_PATTERN.sub(" ", cleaned)
    return cleaned.strip().lower()

