from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import AliasChoices, BaseModel, Field

from common.logger import AVITO_MICROCATS_LOGGER as logger
from common.mistral import MistralCallConfig, call_mistral


class _MistralDetectedResponse(BaseModel):
    detected_mc_ids: list[Any] = Field(
        default_factory=list,
        validation_alias=AliasChoices("detectedMcIds", "detected_mc_ids"),
    )


class _ParsedMistralResult(BaseModel):
    detected_mc_ids: list[int]
    is_valid_json: bool


def _preview_text(text: str, *, limit: int = 240) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return f"{compact[:limit]}..."


def _default_catalog_path() -> Path:
    return Path(__file__).resolve().parent.parent / "data" / "rnc_dataset.csv"


@lru_cache(maxsize=1)
def _load_global_category_catalog() -> dict[int, str]:
    catalog_path = _default_catalog_path()
    if not catalog_path.exists():
        return {}

    try:
        df = pd.read_csv(catalog_path, usecols=["sourceMcId", "sourceMcTitle"])
    except Exception:  # noqa: BLE001
        return {}

    mapping: dict[int, str] = {}
    for _, row in df.drop_duplicates().iterrows():
        raw_id = row.get("sourceMcId")
        title = str(row.get("sourceMcTitle", "")).strip()
        if raw_id is None or not title:
            continue
        try:
            mc_id = int(raw_id)
        except (TypeError, ValueError):
            continue
        mapping.setdefault(mc_id, title)
    return mapping


def build_mc_id_title_mapping(df: pd.DataFrame, *, allowed_ids: set[int]) -> dict[int, str]:
    mapping: dict[int, str] = {}
    for _, row in df.iterrows():
        raw_id = row.get("sourceMcId")
        title = str(row.get("sourceMcTitle", "")).strip()
        if raw_id is None:
            continue
        try:
            mc_id = int(raw_id)
        except (TypeError, ValueError):
            continue
        if mc_id not in allowed_ids:
            continue
        if mc_id not in mapping and title:
            mapping[mc_id] = title

    global_catalog = _load_global_category_catalog()
    for mc_id, title in global_catalog.items():
        if mc_id in allowed_ids and mc_id not in mapping and title:
            mapping[mc_id] = title

    for mc_id in sorted(allowed_ids):
        mapping.setdefault(mc_id, "unknown")

    return mapping


def _coerce_int_list(value: Any) -> list[int]:
    if isinstance(value, list):
        result: list[int] = []
        for item in value:
            try:
                result.append(int(item))
            except (TypeError, ValueError):
                continue
        return result
    return []


def _sanitize_detected_mc_ids(
    raw_ids: Any,
    *,
    allowed_ids: set[int],
    source_mc_id: int | None,
) -> list[int]:
    seen: set[int] = set()
    sanitized: list[int] = []
    for mc_id in _coerce_int_list(raw_ids):
        if source_mc_id is not None and mc_id == source_mc_id:
            continue
        if mc_id not in allowed_ids or mc_id in seen:
            continue
        seen.add(mc_id)
        sanitized.append(mc_id)
    return sanitized


def _parse_mistral_detected_ids_with_status(
    payload_text: str,
    *,
    allowed_ids: set[int],
    source_mc_id: int | None,
) -> _ParsedMistralResult:
    try:
        parsed = _MistralDetectedResponse.model_validate_json(payload_text)
    except Exception as error:  # noqa: BLE001
        logger.warning(
            "Mistral вернул JSON не по контракту микрокатегорий: %s | payload=%s",
            error,
            _preview_text(payload_text),
        )
        return _ParsedMistralResult(detected_mc_ids=[], is_valid_json=False)

    return _ParsedMistralResult(
        detected_mc_ids=_sanitize_detected_mc_ids(
            parsed.detected_mc_ids,
            allowed_ids=allowed_ids,
            source_mc_id=source_mc_id,
        ),
        is_valid_json=True,
    )


def parse_mistral_detected_ids(
    payload_text: str,
    *,
    allowed_ids: set[int],
    source_mc_id: int | None,
) -> list[int]:
    parsed = _parse_mistral_detected_ids_with_status(
        payload_text,
        allowed_ids=allowed_ids,
        source_mc_id=source_mc_id,
    )
    return parsed.detected_mc_ids


def _parse_target_ids(value: Any) -> list[int]:
    if isinstance(value, list):
        return _coerce_int_list(value)
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except Exception:  # noqa: BLE001
            return []
        return _coerce_int_list(parsed)
    return []


def _build_train_few_shot_examples(
    df: pd.DataFrame,
    *,
    category_map: dict[int, str],
    max_examples: int = 2,
) -> str:
    required_columns = {"description", "sourceMcId", "sourceMcTitle", "targetDetectedMcIds"}
    if not required_columns.issubset(df.columns):
        return ""

    examples: list[str] = []
    train_df = df[df["split"].astype(str).eq("train")] if "split" in df.columns else df

    if train_df.empty:
        return ""

    for _, row in train_df.iterrows():
        description = str(row.get("description", "")).strip()
        source_title = str(row.get("sourceMcTitle", "")).strip()
        raw_source_id = row.get("sourceMcId")
        if not description or raw_source_id is None:
            continue
        try:
            source_mc_id = int(raw_source_id)
        except (TypeError, ValueError):
            continue

        target_ids = [
            mc_id
            for mc_id in _parse_target_ids(row.get("targetDetectedMcIds"))
            if mc_id in category_map and mc_id != source_mc_id
        ]
        dedup_target = list(dict.fromkeys(target_ids))

        examples.append(
            "\n".join(
                [
                    f"sourceMcId: {source_mc_id}",
                    f"sourceMcTitle: {source_title}",
                    f"description: {description}",
                    f"response: {{\"detectedMcIds\": {dedup_target}}}",
                ]
            )
        )
        if len(examples) >= max_examples:
            break

    return "\n\n".join(f"Пример {idx + 1}:\n{example}" for idx, example in enumerate(examples))


def _build_few_shot_examples(category_map: dict[int, str]) -> str:
    ids = sorted(category_map)
    if len(ids) < 3:
        return ""

    source_id = ids[0]
    other_1 = ids[1]
    other_2 = ids[2]

    source_title = category_map[source_id]
    other_title_1 = category_map[other_1]
    other_title_2 = category_map[other_2]

    return (
        "Пример 1:\n"
        f"sourceMcId: {source_id}, sourceMcTitle: {source_title}\n"
        "description: Делаем комплексную услугу под ключ, включая дополнительные работы в составе проекта.\n"
        "response: {\"detectedMcIds\": []}\n\n"
        "Пример 2:\n"
        f"sourceMcId: {source_id}, sourceMcTitle: {source_title}\n"
        f"description: Выполняем {other_title_1} отдельно и {other_title_2} отдельно, помимо основной услуги.\n"
        f"response: {{\"detectedMcIds\": [{other_1}, {other_2}]}}"
    )


def build_mistral_messages(
    *,
    description: str,
    source_mc_id: int | None,
    source_title: str,
    category_map: dict[int, str],
) -> list[dict[str, str]]:
    few_shot = _build_few_shot_examples(category_map)
    source_id_str = "unknown" if source_mc_id is None else str(source_mc_id)

    return [
        {
            "role": "system",
            "content": (
                "Ты определяешь дополнительные микрокатегории услуг по тексту объявления. "
                "Верни ТОЛЬКО JSON-объект формата {\"detectedMcIds\": [int, ...]}. "
                "Правила: выбирай только id из словаря категорий; не возвращай sourceMcId; "
                "если услуга упоминается как часть комплексной услуги без признака самостоятельности, не добавляй ее."
            ),
        },
        {
            "role": "user",
            "content": (
                f"sourceMcId: {source_id_str}\n"
                f"sourceMcTitle: {source_title}\n"
                f"description: {description}\n"
                f"categories: {category_map}\n"
                f"fewShot:\n{few_shot}\n"
                "Ответ верни только JSON без пояснений."
            ),
        },
    ]


def predict_detected_mc_ids_with_mistral(
    df: pd.DataFrame,
    *,
    allowed_ids: set[int],
    mistral_config: MistralCallConfig,
) -> list[list[int]]:
    category_map = build_mc_id_title_mapping(df, allowed_ids=allowed_ids)
    unknown_count = sum(1 for title in category_map.values() if title == "unknown")
    logger.info(
        "Mistral microcats: allowed_ids=%d, mapping_size=%d, unknown_titles=%d",
        len(allowed_ids),
        len(category_map),
        unknown_count,
    )

    train_few_shot = _build_train_few_shot_examples(df, category_map=category_map)
    fallback_few_shot = _build_few_shot_examples(category_map)
    combined_few_shot = train_few_shot or fallback_few_shot
    logger.info(
        "Mistral microcats: few-shot source=%s, length=%d",
        "train" if train_few_shot else "fallback",
        len(combined_few_shot),
    )

    detected_rows: list[list[int]] = []

    for row_index, row in df.iterrows():
        description = str(row.get("description", "")).strip()
        source_title = str(row.get("sourceMcTitle", "")).strip()
        source_id_raw = row.get("sourceMcId")
        if source_id_raw is None:
            source_mc_id = None
        else:
            try:
                source_mc_id = int(source_id_raw)
            except (TypeError, ValueError):
                source_mc_id = None

        messages = build_mistral_messages(
            description=description,
            source_mc_id=source_mc_id,
            source_title=source_title,
            category_map=category_map,
        )
        if combined_few_shot:
            messages[1]["content"] = messages[1]["content"].replace(
                f"fewShot:\n{fallback_few_shot}\n",
                f"fewShot:\n{combined_few_shot}\n",
            )

        try:
            response = call_mistral(
                mistral_config,
                messages=messages,
                temperature=0.0,
                top_p=1.0,
                response_format={"type": "json_object"},
            )
            logger.debug(
                "Mistral microcats row=%s source_mc_id=%s response=%s",
                row_index,
                source_mc_id,
                _preview_text(response),
            )
        except Exception as error:  # noqa: BLE001
            logger.warning(f"Ошибка вызова Mistral для микрокатегорий: {error}")
            detected_rows.append([])
            continue

        parsed = _parse_mistral_detected_ids_with_status(
            response,
            allowed_ids=allowed_ids,
            source_mc_id=source_mc_id,
        )

        if not parsed.is_valid_json:
            logger.info(
                "Mistral microcats row=%s invalid JSON on first pass; retrying",
                row_index,
            )
            retry_messages = [
                {
                    "role": "system",
                    "content": (
                        "Верни строго JSON формата {\"detectedMcIds\": [int, ...]} "
                        "без любого дополнительного текста."
                    ),
                },
                messages[1],
            ]
            try:
                retry_response = call_mistral(
                    mistral_config,
                    messages=retry_messages,
                    temperature=0.0,
                    top_p=1.0,
                    response_format={"type": "json_object"},
                )
                logger.debug(
                    "Mistral microcats row=%s retry response=%s",
                    row_index,
                    _preview_text(retry_response),
                )
                parsed = _parse_mistral_detected_ids_with_status(
                    retry_response,
                    allowed_ids=allowed_ids,
                    source_mc_id=source_mc_id,
                )
            except Exception as error:  # noqa: BLE001
                logger.warning(f"Повторный вызов Mistral для JSON-контракта завершился ошибкой: {error}")

        logger.info(
            "Mistral microcats row=%s source_mc_id=%s predicted=%s desc=%s",
            row_index,
            source_mc_id,
            parsed.detected_mc_ids,
            _preview_text(description, limit=120),
        )
        detected_rows.append(parsed.detected_mc_ids)

    non_empty = sum(1 for row in detected_rows if row)
    logger.info(
        "Mistral microcats: completed rows=%d, non_empty_predictions=%d",
        len(detected_rows),
        non_empty,
    )

    return detected_rows
