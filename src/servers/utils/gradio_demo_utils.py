from __future__ import annotations

import json
import random
from functools import lru_cache
from pathlib import Path
from typing import *
from urllib.parse import urlsplit, urlunsplit
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd
from pydantic import BaseModel, ConfigDict, field_validator

from common.logger import GRADIO_DEMO_LOGGER as logger
from common.paths import PathLike, get_project_root
from src.sber.constants import (
    DEFAULT_GRADIO_DEMO_CLASSIFICATION_THRESHOLD,
    DEFAULT_GRADIO_DEMO_HOST,
    DEFAULT_GRADIO_DEMO_PIPELINE_BASE_URL,
    DEFAULT_GRADIO_DEMO_PORT,
    DEFAULT_GRADIO_DEMO_SHARE,
    DEFAULT_GRADIO_DEMO_TIMEOUT_SEC,
    DEFAULT_GRADIO_DEMO_TITLE,
)

DEFAULT_PIPELINE_BASE_URL: str = DEFAULT_GRADIO_DEMO_PIPELINE_BASE_URL
DEFAULT_DUMMY_EXAMPLES_CSV: PathLike = "data/raw/merged_features_with_judge_scores.csv"
QUICK_QUERY_EXAMPLES: list[str] = [
    "Кто написал роман 'Война и мир'?",
    "Москва находится в Германии.",
    "В каком году началась Первая мировая война?",
]


class GradioDemoConfig(BaseModel):
    """Конфигурация Gradio demo для tabular pipeline backend.

    Attributes:
        host: Хост, на котором поднимается Gradio UI.
        port: Порт Gradio UI.
        share: Флаг внешнего публичного share-ссылки.
        title: Заголовок UI.
        pipeline_base_url: Базовый URL tabular pipeline сервера.
        timeout_sec: Таймаут HTTP-запросов к backend-серверам.
        classification_threshold: Порог интерпретации `hallucination_score` для label.
        dummy_mode: Включает UX dummy-режима для быстрого smoke-теста.
        dummy_examples_csv: CSV с train-примерами для случайного dummy-запуска.
    """

    model_config = ConfigDict(frozen=True)

    host: str = DEFAULT_GRADIO_DEMO_HOST
    port: int = DEFAULT_GRADIO_DEMO_PORT
    share: bool = DEFAULT_GRADIO_DEMO_SHARE
    title: str = DEFAULT_GRADIO_DEMO_TITLE
    pipeline_base_url: str = DEFAULT_PIPELINE_BASE_URL
    timeout_sec: float = DEFAULT_GRADIO_DEMO_TIMEOUT_SEC
    classification_threshold: float = DEFAULT_GRADIO_DEMO_CLASSIFICATION_THRESHOLD
    dummy_mode: bool = False
    dummy_examples_csv: PathLike = DEFAULT_DUMMY_EXAMPLES_CSV

    @field_validator("host", "title", "pipeline_base_url")
    @classmethod
    def validate_non_empty_string(cls, value: str) -> str:
        normalized: str = value.strip()
        if not normalized:
            raise ValueError("Строковое поле конфигурации не может быть пустым")
        return normalized

    @field_validator("port")
    @classmethod
    def validate_positive_port(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("port должен быть положительным")
        return value

    @field_validator("timeout_sec")
    @classmethod
    def validate_timeout(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("timeout_sec должен быть положительным")
        return value

    @field_validator("classification_threshold")
    @classmethod
    def validate_classification_threshold(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError("classification_threshold должен быть в диапазоне [0, 1]")
        return value


def _normalize_base_url(url: str) -> str:
    normalized: str = url.rstrip("/")
    parsed = urlsplit(normalized)
    if parsed.hostname != "0.0.0.0":
        return normalized

    auth_prefix: str = ""
    if parsed.username is not None:
        auth_prefix = parsed.username
        if parsed.password is not None:
            auth_prefix = f"{auth_prefix}:{parsed.password}"
        auth_prefix = f"{auth_prefix}@"

    port_suffix: str = f":{parsed.port}" if parsed.port is not None else ""
    patched_netloc: str = f"{auth_prefix}127.0.0.1{port_suffix}"
    return urlunsplit((parsed.scheme, patched_netloc, parsed.path, parsed.query, parsed.fragment)).rstrip("/")


class DummyTrainSample(BaseModel):
    """Структура одного train-примера для dummy-режима.

    Attributes:
        query: Текст пользовательского запроса.
        model_answer: Ответ модели для классификатора.
        correct_answer: Эталонный ответ из train-данных.
        features: Полный набор признаков строки train-датасета.
    """

    query: str
    model_answer: str
    correct_answer: str
    features: dict[str, Any]


def _resolve_dummy_examples_csv(csv_path: PathLike) -> Path:
    path: Path = Path(csv_path).expanduser()
    if path.is_absolute():
        return path
    return Path(get_project_root()) / path


@lru_cache(maxsize=4)
def _load_dummy_samples(csv_path: str) -> tuple[DummyTrainSample, ...]:
    dataset_path: Path = _resolve_dummy_examples_csv(csv_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"CSV с dummy-примерами не найден: {dataset_path}")

    frame: pd.DataFrame = pd.read_csv(dataset_path)
    required_columns: set[str] = {"query", "model_answer", "correct_answer"}
    missing_columns: set[str] = required_columns - set(frame.columns)
    if missing_columns:
        raise ValueError(f"В dummy CSV отсутствуют колонки: {sorted(missing_columns)}")

    samples: list[DummyTrainSample] = []

    def sanitize_value(value: Any) -> Any:
        if pd.isna(value):
            return None
        if hasattr(value, "item"):
            try:
                return value.item()
            except Exception:
                return value
        return value

    for _, row in frame.iterrows():
        query: str = str(row["query"]).strip()
        model_answer: str = str(row["model_answer"]).strip()
        correct_answer: str = str(row["correct_answer"]).strip()
        if not query or not model_answer:
            continue
        features: dict[str, Any] = {
            str(column_name): sanitize_value(row[column_name])
            for column_name in frame.columns
            if column_name not in {"is_hallucination", "comment"}
        }
        samples.append(
            DummyTrainSample(
                query=query,
                model_answer=model_answer,
                correct_answer=correct_answer,
                features=features,
            )
        )

    if not samples:
        raise ValueError(f"Не удалось найти валидные строки в dummy CSV: {dataset_path}")
    return tuple(samples)


def pick_dummy_train_sample(config: GradioDemoConfig) -> DummyTrainSample:
    """Возвращает случайный train-пример для dummy UX в Gradio."""
    samples: tuple[DummyTrainSample, ...] = _load_dummy_samples(str(config.dummy_examples_csv))
    return random.choice(samples)


def preload_dummy_samples(config: GradioDemoConfig) -> None:
    """Прогревает кэш train-сэмплов при старте Gradio-сервера."""
    _ = _load_dummy_samples(str(config.dummy_examples_csv))


def _post_json(url: str, payload: Mapping[str, Any], timeout_sec: float) -> dict[str, Any]:
    body: bytes = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request: Request = Request(
        url=url,
        data=body,
        method="POST",
        headers={"Content-Type": "application/json; charset=utf-8"},
    )
    try:
        with urlopen(request, timeout=timeout_sec) as response:
            raw: str = response.read().decode("utf-8")
            data: Any = json.loads(raw) if raw else {}
            if not isinstance(data, dict):
                raise ValueError("HTTP ответ должен быть JSON-объектом")
            return data
    except HTTPError as error:
        raw_error: str = error.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"HTTP {error.code} при запросе {url}: {raw_error}") from error
    except URLError as error:
        raise RuntimeError(f"Не удалось подключиться к {url}: {error}") from error


def _extract_stage_timings_text(response: Mapping[str, Any]) -> str:
    def first_float(key: str) -> float | None:
        value: Any = response.get(key)
        if isinstance(value, list) and value:
            return float(value[0])
        return None

    t_feature: float | None = first_float("t_feature_extraction_sec")
    t_classification: float | None = first_float("t_classification_sec")
    t_total: float | None = first_float("t_total_sec")
    if t_total is None:
        t_total = first_float("t_sample_sec")

    parts: list[str] = []
    if t_feature is not None:
        parts.append(f"feature_extraction={t_feature:.4f}s")
    if t_classification is not None:
        parts.append(f"classification={t_classification:.4f}s")
    if t_total is not None:
        parts.append(f"total={t_total:.4f}s")
    return "\n".join(parts)


def query_tabular_pipeline(
    *,
    query: str,
    model_answer: str,
    config: GradioDemoConfig,
    precomputed_features: Mapping[str, Any] | None = None,
) -> tuple[str, float | None, str]:
    """Запрашивает классификацию у tabular pipeline сервера."""
    endpoint: str = f"{_normalize_base_url(config.pipeline_base_url)}/v1/tabular/predict"
    payload: dict[str, Any] = {
        "query": query,
        "model_answer": model_answer,
        "threshold": config.classification_threshold,
    }
    if precomputed_features is not None:
        payload["features"] = dict(precomputed_features)
    response: dict[str, Any] = _post_json(endpoint, payload, timeout_sec=config.timeout_sec)

    preds_raw: Any = response.get("pred_is_hallucination", [])
    if not isinstance(preds_raw, list) or not preds_raw:
        raise ValueError("Некорректный ответ pipeline: нет pred_is_hallucination")

    score_raw: Any = response.get("hallucination_score", [])
    score: float | None = None
    if isinstance(score_raw, list) and score_raw:
        score = float(score_raw[0])

    if score is None:
        prediction: int = int(preds_raw[0])
    else:
        prediction = int(score >= config.classification_threshold)
    label: str = "галлюцинация" if prediction == 1 else "не галлюцинация"

    timings_text: str = _extract_stage_timings_text(response)

    return label, score, timings_text


def run_pipeline(query: str, model_answer: str, config: GradioDemoConfig) -> tuple[str, str, str]:
    """Прогоняет пару query/answer через tabular pipeline backend по HTTP."""
    normalized_query: str = query.strip()
    normalized_answer: str = model_answer.strip()
    if not normalized_query:
        raise ValueError("Запрос не должен быть пустым")
    if not normalized_answer:
        raise ValueError("Ответ модели не должен быть пустым")

    logger.info("Запускаю пайплайн демо: featureextractor -> tabular_classifier")
    label, score, timings_text = query_tabular_pipeline(query=normalized_query, model_answer=normalized_answer, config=config)

    result_text: str = label if score is None else f"{label} (score={score:.4f})"
    details: str = f"query={normalized_query}\nmodel_answer={normalized_answer}"
    return result_text, details, timings_text


def run_dummy_pipeline(config: GradioDemoConfig) -> tuple[str, str, str, str, str, str]:
    """Запускает dummy-сценарий Gradio на случайном train-примере."""
    sample: DummyTrainSample = pick_dummy_train_sample(config)
    label, score, timings_text = query_tabular_pipeline(
        query=sample.query,
        model_answer=sample.model_answer,
        config=config,
        precomputed_features=sample.features,
    )
    classifier_result: str = label if score is None else f"{label} (score={score:.4f})"
    details: str = f"dummy_mode=true\nquery={sample.query}\nmodel_answer={sample.model_answer}"
    return sample.query, sample.model_answer, sample.correct_answer, classifier_result, details, timings_text


def build_gradio_predict_fn(config: GradioDemoConfig) -> Callable[[str, str], tuple[str, str, str]]:
    """Возвращает callable для привязки к Gradio UI."""

    def predict(query: str, model_answer: str) -> tuple[str, str, str]:
        try:
            return run_pipeline(query=query, model_answer=model_answer, config=config)
        except Exception as error:
            logger.error("Ошибка в Gradio демо пайплайне: %s", error)
            return "", f"Ошибка: {error}", ""

    return predict


def build_gradio_dummy_predict_fn(config: GradioDemoConfig) -> Callable[[], tuple[str, str, str, str, str, str]]:
    """Возвращает callable для запуска dummy-сценария без ручного ввода."""

    def predict() -> tuple[str, str, str, str, str, str]:
        try:
            return run_dummy_pipeline(config=config)
        except Exception as error:
            logger.error("Ошибка в Gradio dummy пайплайне: %s", error)
            return "", "", "", "", f"Ошибка: {error}", ""

    return predict


__all__ = [
    "DummyTrainSample",
    "GradioDemoConfig",
    "QUICK_QUERY_EXAMPLES",
    "build_gradio_dummy_predict_fn",
    "build_gradio_predict_fn",
    "pick_dummy_train_sample",
    "preload_dummy_samples",
    "query_tabular_pipeline",
    "run_dummy_pipeline",
    "run_pipeline",
]



