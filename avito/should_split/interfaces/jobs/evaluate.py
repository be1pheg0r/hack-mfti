from __future__ import annotations

import argparse
import ast
import csv
import json
import time
from pathlib import Path
from typing import *

from pydantic import BaseModel, field_validator
from tqdm.auto import tqdm

from avito.constants import GRADIO_DEFAULT_API_URL, PIPELINE_API_CLIENT_DEFAULT_TIMEOUT_SEC
from avito.should_split.api.client import PipelineApiClient
from avito.should_split.core.config import ShouldSplitGraphConfig
from common.files import read_json
from common.logger import AVITO_SHOULD_SPLIT_LOGGER as logger
from common.paths import get_avito_data_dpath, get_avito_gitignore_data_dpath


class EvaluateCliConfig(BaseModel):
    """Параметры запуска evaluate-скрипта.

    Attributes:
        dataset_path: Путь к JSON-датасету для оценки.
        config_path: Опциональный путь к YAML-конфигу (используется для fallback timeout).
        limit: Опциональный лимит числа объявлений для оценки.
        output_path: Путь к JSON-отчету с полем prediction.
        api_url: Базовый URL уже запущенного API сервера.
        api_timeout_sec: Таймаут одного HTTP-запроса к API.
            Если не задан, берется из should_split_graph.yaml.
        fail_fast: Если True, остановить evaluate на первой ошибке строки.
    """

    dataset_path: Path = Path(get_avito_data_dpath()) / "rnc_dataset_markup.json"
    config_path: Path | None = None
    limit: int | None = None
    output_path: Path = Path(get_avito_gitignore_data_dpath()) / "rnc_dataset_markup_with_predictions.json"
    api_url: str = GRADIO_DEFAULT_API_URL
    api_timeout_sec: int | None = None
    fail_fast: bool = False

    @field_validator("limit")
    @classmethod
    def validate_limit(cls, value: int | None) -> int | None:
        if value is not None and value <= 0:
            raise ValueError("limit должен быть положительным")
        return value

    @field_validator("api_timeout_sec")
    @classmethod
    def validate_api_timeout_sec(cls, value: int | None) -> int | None:
        if value is None:
            return None
        if value <= 0:
            raise ValueError("api_timeout_sec должен быть положительным")
        return value


class EvaluationSample(BaseModel):
    """Нормализованная запись для подсчета метрик.

    Attributes:
        gt_ids: Эталонные дополнительные микрокатегории.
        pred_ids: Предсказанные дополнительные микрокатегории.
        gt_should_split: Эталонный shouldSplit.
        pred_should_split: Предсказанный shouldSplit.
    """

    gt_ids: set[int]
    pred_ids: set[int]
    gt_should_split: bool
    pred_should_split: bool


class EvaluationMetrics(BaseModel):
    """Итоговые метрики оценки should-split пайплайна.

    Attributes:
        samples_count: Число обработанных объявлений.
        true_positive: Число корректно предложенных дополнительных mcId.
        false_positive: Число лишних предложенных дополнительных mcId.
        false_negative: Число пропущенных эталонных дополнительных mcId.
        precision_micro: Precision (micro) по дополнительным микрокатегориям.
        recall_micro: Recall (micro) по дополнительным микрокатегориям.
        f1_micro: F1-score (micro) по дополнительным микрокатегориям.
        should_split_accuracy: Accuracy по shouldSplit.
        categories_accuracy: Accuracy по полному совпадению множества извлеченных категорий.
        inference_time_total_sec: Суммарное время инференса по обработанным строкам.
        inference_time_avg_sec: Среднее время инференса на строку.
        inference_time_p95_sec: 95-й перцентиль времени инференса.
    """

    samples_count: int
    true_positive: int
    false_positive: int
    false_negative: int
    precision_micro: float
    recall_micro: float
    f1_micro: float
    should_split_accuracy: float
    categories_accuracy: float
    inference_time_total_sec: float
    inference_time_avg_sec: float
    inference_time_p95_sec: float


def _percentile_95(values: list[float]) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    index = int(round(0.95 * (len(sorted_values) - 1)))
    return sorted_values[index]


def _safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _parse_should_split(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y"}:
            return True
        if normalized in {"false", "0", "no", "n"}:
            return False
    return bool(value)


def parse_mc_ids(value: Any) -> list[int]:
    """Парсит поле микрокатегорий из JSON-записи в список int."""
    if isinstance(value, list):
        parsed = value
    elif isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        try:
            parsed = ast.literal_eval(stripped)
        except (ValueError, SyntaxError):
            return []
    else:
        return []

    if not isinstance(parsed, list):
        return []

    ids: list[int] = []
    for item in parsed:
        if isinstance(item, bool):
            continue
        if isinstance(item, int):
            ids.append(item)
            continue
        if isinstance(item, str) and item.strip().isdigit():
            ids.append(int(item.strip()))

    # Сохраняем порядок и убираем дубликаты.
    return list(dict.fromkeys(ids))


def aggregate_metrics(
    samples: list[EvaluationSample],
    inference_durations_sec: list[float] | None = None,
) -> EvaluationMetrics:
    """Агрегирует micro-метрики и shouldSplit accuracy по набору сэмплов."""
    tp = 0
    fp = 0
    fn = 0
    correct_should_split = 0
    exact_categories_matches = 0

    for sample in samples:
        tp += len(sample.pred_ids & sample.gt_ids)
        fp += len(sample.pred_ids - sample.gt_ids)
        fn += len(sample.gt_ids - sample.pred_ids)
        if sample.pred_should_split == sample.gt_should_split:
            correct_should_split += 1
        if sample.pred_ids == sample.gt_ids:
            exact_categories_matches += 1

    precision_micro = _safe_divide(tp, tp + fp)
    recall_micro = _safe_divide(tp, tp + fn)
    f1_micro = _safe_divide(2 * precision_micro * recall_micro, precision_micro + recall_micro)
    should_split_accuracy = _safe_divide(correct_should_split, len(samples))
    categories_accuracy = _safe_divide(exact_categories_matches, len(samples))
    durations = inference_durations_sec or []
    inference_time_total_sec = sum(durations)
    inference_time_avg_sec = _safe_divide(inference_time_total_sec, len(durations))
    inference_time_p95_sec = _percentile_95(durations)

    return EvaluationMetrics(
        samples_count=len(samples),
        true_positive=tp,
        false_positive=fp,
        false_negative=fn,
        precision_micro=precision_micro,
        recall_micro=recall_micro,
        f1_micro=f1_micro,
        should_split_accuracy=should_split_accuracy,
        categories_accuracy=categories_accuracy,
        inference_time_total_sec=inference_time_total_sec,
        inference_time_avg_sec=inference_time_avg_sec,
        inference_time_p95_sec=inference_time_p95_sec,
    )


def _parse_json_object(value: Any, field_name: str) -> dict[str, Any]:
    if isinstance(value, dict):
        return value

    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as error:
            raise ValueError(f"Поле {field_name} содержит невалидный JSON") from error

        if isinstance(parsed, dict):
            return parsed

    raise ValueError(f"Поле {field_name} должно быть JSON-объектом или JSON-строкой объекта")


def _try_parse_json_object(value: Any, field_name: str) -> dict[str, Any] | None:
    """Пытается распарсить JSON-объект; пустое значение трактуется как отсутствие эталона."""
    if value is None:
        return None

    if isinstance(value, str) and not value.strip():
        return None

    return _parse_json_object(value=value, field_name=field_name)


def _normalize_row_for_api_evaluate(row: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Нормализует строку датасета к request/response формату для API-only evaluate.

    Поддерживает:
    - официальный формат с полями request/response;
    - legacy markup-формат (itemId/sourceMcId/sourceMcTitle/description/targetSplitMcIds/shouldSplit).
    """
    if "request" in row:
        request_data = _parse_json_object(row.get("request"), field_name="request")
        response_data = _try_parse_json_object(row.get("response"), field_name="response")
        return request_data, response_data

    if "description" not in row:
        raise ValueError("Строка должна содержать request/response или legacy-поля с description")

    request_data = {
        "itemId": row.get("itemId", 0),
        "mcId": int(row.get("sourceMcId", 0)),
        "mcTitle": str(row.get("sourceMcTitle", "")),
        "description": str(row.get("description", "")),
    }

    # Для валидации используем только targetSplitMcIds.
    gt_ids = parse_mc_ids(row.get("targetSplitMcIds", []))

    response_data = {
        "detectedMcIds": gt_ids,
        "shouldSplit": _parse_should_split(row.get("shouldSplit", False)),
        "drafts": [],
    }
    return request_data, response_data


def _build_official_artifacts_for_row(
    row: dict[str, Any],
    client: PipelineApiClient,
) -> tuple[EvaluationSample | None, dict[str, Any]]:
    request_data, response_data = _normalize_row_for_api_evaluate(row)

    model_response = client.infer(
        item_id=request_data.get("itemId", 0),
        mc_id=int(request_data.get("mcId", 0)),
        mc_title=str(request_data.get("mcTitle", "")),
        description=str(request_data.get("description", "")),
    )
    prediction = model_response.model_dump()

    if response_data is None:
        return None, prediction

    source_mc_id = int(request_data.get("mcId", 0))
    gt_ids = set(parse_mc_ids(response_data.get("detectedMcIds", [])))
    pred_ids = set(int(mc_id) for mc_id in model_response.detectedMcIds)
    gt_ids.discard(source_mc_id)
    pred_ids.discard(source_mc_id)

    pred_should_split = bool(model_response.shouldSplit)
    if not pred_should_split:
        pred_ids = set()

    evaluation_sample = EvaluationSample(
        gt_ids=gt_ids,
        pred_ids=pred_ids,
        gt_should_split=_parse_should_split(response_data.get("shouldSplit", False)),
        pred_should_split=pred_should_split,
    )
    return evaluation_sample, prediction


def _write_predictions(output_path: Path, rows_with_predictions: list[Any]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() == ".csv":
        dict_rows = [row for row in rows_with_predictions if isinstance(row, dict)]
        fieldnames: list[str] = []
        for row in dict_rows:
            for key in row.keys():
                if key not in fieldnames:
                    fieldnames.append(key)

        with open(output_path, "w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            for row in dict_rows:
                serialized_row: dict[str, Any] = {}
                for key in fieldnames:
                    value = row.get(key, "")
                    if isinstance(value, (dict, list)):
                        serialized_row[key] = json.dumps(value, ensure_ascii=False)
                    else:
                        serialized_row[key] = value
                writer.writerow(serialized_row)
        return

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(rows_with_predictions, file, ensure_ascii=False, indent=2)


def _read_csv_rows(dataset_path: Path) -> list[dict[str, str]]:
    with open(dataset_path, "r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None:
            raise ValueError("CSV-датасет должен содержать заголовок колонок")
        return [dict(row) for row in reader]


def _load_dataset_rows(dataset_path: Path) -> list[dict[str, Any]]:
    suffix = dataset_path.suffix.lower()
    if suffix == ".json":
        dataset = read_json(dataset_path)
        if not isinstance(dataset, list):
            raise ValueError("Ожидается JSON-массив объектов во входном датасете")
        return dataset

    if suffix == ".csv":
        return _read_csv_rows(dataset_path)

    raise ValueError("Поддерживаются только .json и .csv датасеты")


def evaluate_official_dataset(
    dataset_path: Path,
    api_url: str,
    api_timeout_sec: int = PIPELINE_API_CLIENT_DEFAULT_TIMEOUT_SEC,
    fail_fast: bool = False,
    limit: int | None = None,
    prediction_rows: list[Any] | None = None,
) -> EvaluationMetrics:
    """Считает метрики по JSON/CSV-датасету request/response через API."""
    dataset = _load_dataset_rows(dataset_path)
    rows = dataset[:limit] if limit is not None else dataset
    samples: list[EvaluationSample] = []
    inference_durations_sec: list[float] = []
    client = PipelineApiClient(base_url=api_url, timeout_sec=api_timeout_sec)
    failed_rows = 0

    logger.info(
        "[evaluate] start | dataset=%s | rows=%s | api_url=%s | timeout=%ss | fail_fast=%s",
        dataset_path,
        len(rows),
        api_url,
        api_timeout_sec,
        fail_fast,
    )

    progress = tqdm(rows, total=len(rows), desc="Evaluating API", unit="sample")
    for index, row in enumerate(progress):
        if not isinstance(row, dict):
            if prediction_rows is not None:
                prediction_rows.append(row)
            continue

        started_at = time.perf_counter()
        try:
            eval_sample, prediction = _build_official_artifacts_for_row(row=row, client=client)
            inference_durations_sec.append(time.perf_counter() - started_at)
        except Exception as error:
            inference_durations_sec.append(time.perf_counter() - started_at)
            failed_rows += 1
            if prediction_rows is not None:
                prediction_rows.append(
                    {
                        **dict(row),
                        "prediction_error": str(error),
                    }
                )
            if fail_fast:
                raise ValueError(f"Ошибка обработки строки {index} в official датасете: {error}") from error
            logger.warning("[evaluate] row=%s skipped due to error: %s", index, error)
            continue

        if eval_sample is not None:
            samples.append(eval_sample)

        if prediction_rows is not None:
            prediction_row = dict(row)
            if _try_parse_json_object(prediction_row.get("response"), field_name="response") is None:
                prediction_row["response"] = prediction
            prediction_row["prediction"] = prediction
            prediction_rows.append(prediction_row)

    metrics = aggregate_metrics(samples, inference_durations_sec=inference_durations_sec)
    logger.info(
        "[evaluate] done | rows=%s | failed_rows=%s | precision_micro=%.6f | recall_micro=%.6f | f1_micro=%.6f | should_split_accuracy=%.6f | categories_accuracy=%.6f | infer_total=%.3fs | infer_avg=%.3fs | infer_p95=%.3fs",
        metrics.samples_count,
        failed_rows,
        metrics.precision_micro,
        metrics.recall_micro,
        metrics.f1_micro,
        metrics.should_split_accuracy,
        metrics.categories_accuracy,
        metrics.inference_time_total_sec,
        metrics.inference_time_avg_sec,
        metrics.inference_time_p95_sec,
    )
    return metrics


def _resolve_api_timeout_sec(config_path: Path | None, cli_timeout_sec: int | None) -> int:
    """Возвращает timeout official API: CLI override или значение из pipeline-конфига."""
    if cli_timeout_sec is not None:
        return cli_timeout_sec

    runtime_config = ShouldSplitGraphConfig.model_validate({"config_path": config_path}) if config_path is not None else ShouldSplitGraphConfig.from_default_yaml()
    return runtime_config.graph.pipeline_request_timeout_sec


def _build_cli_config() -> EvaluateCliConfig:
    parser = argparse.ArgumentParser(description="Evaluate Avito should-split via running API server")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(get_avito_data_dpath()) / "rnc_dataset_markup.json",
        help="Путь к датасету разметки (.json или .csv)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Опциональный путь к YAML-конфигу (только для fallback timeout)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Опциональный лимит числа объявлений",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(get_avito_gitignore_data_dpath()) / "rnc_dataset_markup_with_predictions.json",
        help="Путь к JSON-датасету с добавленным полем prediction",
    )
    parser.add_argument(
        "--api-url",
        type=str,
        default=GRADIO_DEFAULT_API_URL,
        help="Базовый URL уже запущенного API сервера",
    )
    parser.add_argument(
        "--api-timeout-sec",
        type=int,
        default=None,
        help="Таймаут одного HTTP-запроса к API в official режиме (сек). По умолчанию берется из конфига.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Остановить evaluate на первой ошибке строки в official режиме",
    )
    args = parser.parse_args()
    return EvaluateCliConfig(
        dataset_path=args.dataset,
        config_path=args.config,
        limit=args.limit,
        output_path=args.output,
        api_url=args.api_url,
        api_timeout_sec=args.api_timeout_sec,
        fail_fast=args.fail_fast,
    )


def main() -> None:
    config = _build_cli_config()
    prediction_rows: list[Any] = []

    api_timeout_sec = _resolve_api_timeout_sec(
        config_path=config.config_path,
        cli_timeout_sec=config.api_timeout_sec,
    )
    metrics = evaluate_official_dataset(
        dataset_path=config.dataset_path,
        api_url=config.api_url,
        api_timeout_sec=api_timeout_sec,
        fail_fast=config.fail_fast,
        limit=config.limit,
        prediction_rows=prediction_rows,
    )

    _write_predictions(output_path=config.output_path, rows_with_predictions=prediction_rows)
    logger.info("[evaluate] predictions saved | output=%s", config.output_path)
    print(json.dumps(metrics.model_dump(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

