from __future__ import annotations

import argparse
import logging
import json
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import *

from pydantic import BaseModel, field_validator

from avito.constants import (
    RELABEL_DEFAULT_INPUT_FILENAME,
    RELABEL_DEFAULT_MODEL_NAME,
    RELABEL_DEFAULT_SAVE_EVERY,
    RELABEL_OUTPUT_TIMESTAMP_FORMAT,
)
from avito.prompts import (
    categorizationInstructionPrompt,
    categorizationPromptWithoutRAG,
    shouldSplitInstructionPrompt,
    shouldSplitPromptWithoutRAG,
)
from common.files import read_json
from common.logger import AVITO_SHOULD_SPLIT_LOGGER as logger
from common.logger import MISTRAL_LOGGER
from common.mistral import MistralCallConfig, call_mistral
from common.paths import get_avito_data_dpath, get_avito_gitignore_data_dpath
from tqdm.auto import tqdm


class RelabelCliConfig(BaseModel):
    """Параметры запуска переразметки датасета.

    Attributes:
        input_path: Путь к входному JSON-датасету.
        output_path: Путь к выходному JSON-файлу.
        model_name: Имя модели Mistral.
        save_every: Период промежуточного сохранения.
    """

    input_path: Path
    output_path: Path
    model_name: str = RELABEL_DEFAULT_MODEL_NAME
    save_every: int = RELABEL_DEFAULT_SAVE_EVERY

    @field_validator("save_every")
    @classmethod
    def validate_save_every(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("save_every должен быть положительным")
        return value


class OutputSample(BaseModel):
    """Целевая запись переразмеченного датасета.

    Attributes:
        itemId: Идентификатор объявления.
        sourceMcId: Исходный id микрокатегории.
        sourceMcTitle: Исходное название микрокатегории.
        description: Текст объявления.
        targetDetectedMcIds: Идентификаторы найденных микрокатегорий в виде строки списка.
        targetSplitMcIds: Дубликат targetDetectedMcIds в виде строки списка.
        shouldSplit: Признак необходимости сплита.
    """

    itemId: str
    sourceMcId: int
    sourceMcTitle: str
    description: str
    targetDetectedMcIds: str
    targetSplitMcIds: str
    shouldSplit: bool


def _default_output_path() -> Path:
    timestamp = datetime.now().strftime(RELABEL_OUTPUT_TIMESTAMP_FORMAT)
    return Path(get_avito_gitignore_data_dpath()) / f"{timestamp}.json"


def _build_cli_config() -> RelabelCliConfig:
    parser = argparse.ArgumentParser(description="Relabel Avito dataset with Mistral without RAG")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(get_avito_data_dpath()) / RELABEL_DEFAULT_INPUT_FILENAME,
        help="Путь к входному датасету",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_default_output_path(),
        help="Путь к выходному json",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=RELABEL_DEFAULT_MODEL_NAME,
        help="Имя модели Mistral",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=RELABEL_DEFAULT_SAVE_EVERY,
        help="Сохранять промежуточный результат каждые N записей",
    )
    args = parser.parse_args()
    return RelabelCliConfig(
        input_path=args.input,
        output_path=args.output,
        model_name=args.model,
        save_every=args.save_every,
    )


def _write_output(samples: list[OutputSample], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    serialized = [sample.model_dump() for sample in samples]
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(serialized, file, ensure_ascii=False, indent=2)


def _parse_should_split(response_text: str) -> bool:
    text = response_text.strip()
    if not text:
        return False

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            value = parsed.get("shouldSplit", False)
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                return value.strip().lower() == "true"
            return bool(value)
    except json.JSONDecodeError:
        pass

    normalized = text.lower()
    if normalized in {"true", "false"}:
        return normalized == "true"
    if "shouldsplit" in normalized and "false" in normalized:
        return False
    if "shouldsplit" in normalized and "true" in normalized:
        return True
    return "true" in normalized and "false" not in normalized


def _parse_category_ids(response_text: str) -> list[int]:
    ids: list[int] = []
    for chunk in response_text.replace("\n", ",").split(","):
        token = chunk.strip()
        if token.isdigit():
            ids.append(int(token))
    # Удаляем дубликаты и сохраняем порядок ответа модели.
    return list(dict.fromkeys(ids))


def _as_list_string(values: list[int]) -> str:
    return str(values)


@contextmanager
def _mute_mistral_logger() -> Iterator[None]:
    """Временно отключает логгер mistral-call только в рамках этого скрипта."""
    original_disabled = MISTRAL_LOGGER.disabled
    original_level = MISTRAL_LOGGER.level
    try:
        MISTRAL_LOGGER.disabled = True
        MISTRAL_LOGGER.setLevel(logging.CRITICAL)
        yield
    finally:
        MISTRAL_LOGGER.disabled = original_disabled
        MISTRAL_LOGGER.setLevel(original_level)


def _predict_sample(
    sample: dict[str, Any],
    model_name: str,
    mistral_caller: Callable[..., str],
) -> OutputSample:
    description = str(sample.get("description", ""))
    should_split_prompt = shouldSplitPromptWithoutRAG(desc=description)

    should_split_response = mistral_caller(
        MistralCallConfig(models_list=[model_name]),
        messages=[
            {"role": "system", "content": shouldSplitInstructionPrompt},
            {"role": "user", "content": should_split_prompt},
        ],
        temperature=0.0,
    )
    should_split = _parse_should_split(str(should_split_response))

    category_ids: list[int] = []
    if should_split:
        categorization_prompt = categorizationPromptWithoutRAG(desc=description)
        categorization_response = mistral_caller(
            MistralCallConfig(models_list=[model_name]),
            messages=[
                {"role": "system", "content": categorizationInstructionPrompt},
                {"role": "user", "content": categorization_prompt},
            ],
            temperature=0.0,
        )
        category_ids = _parse_category_ids(str(categorization_response))

    categories_string = _as_list_string(category_ids)
    return OutputSample(
        itemId=str(sample.get("itemId", "")),
        sourceMcId=int(sample.get("sourceMcId", 0)),
        sourceMcTitle=str(sample.get("sourceMcTitle", "")),
        description=description,
        targetDetectedMcIds=categories_string,
        targetSplitMcIds=categories_string,
        shouldSplit=should_split,
    )


def relabel_dataset(
    config: RelabelCliConfig,
    mistral_caller: Callable[..., str] = call_mistral,
) -> Path:
    """Переразмечает датасет через Mistral без RAG.

    Args:
        config: Параметры переразметки.
        mistral_caller: Функция вызова Mistral (для тестов можно подменить).

    Returns:
        Путь к итоговому выходному файлу.
    """
    raw_dataset = read_json(config.input_path)
    if not isinstance(raw_dataset, list):
        raise ValueError("Ожидается JSON-массив объектов во входном датасете")

    processed: list[OutputSample] = []
    total = len(raw_dataset)

    logger.info("[relabel] start | total=%s | model=%s", total, config.model_name)

    with _mute_mistral_logger():
        progress = tqdm(raw_dataset, total=total, desc="Relabeling", unit="sample")
        for index, sample in enumerate(progress, start=1):
            if not isinstance(sample, dict):
                logger.warning("[relabel] skip non-dict sample at index=%s", index - 1)
                continue

            predicted = _predict_sample(
                sample=sample,
                model_name=config.model_name,
                mistral_caller=mistral_caller,
            )
            processed.append(predicted)

            if index % config.save_every == 0:
                _write_output(samples=processed, output_path=config.output_path)
                logger.info("[relabel] checkpoint saved | processed=%s/%s", index, total)

    _write_output(samples=processed, output_path=config.output_path)
    logger.info("[relabel] done | saved=%s | output=%s", len(processed), config.output_path)
    return config.output_path


def main() -> None:
    config = _build_cli_config()
    relabel_dataset(config=config)


if __name__ == "__main__":
    main()


