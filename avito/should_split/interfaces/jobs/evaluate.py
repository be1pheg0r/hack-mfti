from __future__ import annotations

import argparse
import ast
import csv
import json
import logging
from contextlib import contextmanager
from pathlib import Path
from typing import *

from pydantic import BaseModel, field_validator
from tqdm.auto import tqdm

from avito.constants import GRADIO_DEFAULT_API_URL
from avito.should_split.api.client import PipelineApiClient
from avito.should_split.core.config import ShouldSplitGraphConfig
from avito.should_split.core.pipeline import ShouldSplitPipeline, build_default_pipeline
from common.files import read_json
from common.logger import AVITO_SHOULD_SPLIT_LOGGER as logger
from common.logger import MISTRAL_LOGGER
from common.paths import get_avito_data_dpath, get_avito_gitignore_data_dpath


class EvaluateCliConfig(BaseModel):
    """Параметры запуска evaluate-скрипта.

    Attributes:
        dataset_path: Путь к JSON-датасету для оценки.
        config_path: Опциональный путь к YAML-конфигу пайплайна.
        limit: Опциональный лимит числа объявлений для оценки.
        output_path: Путь к JSON-отчету с полем prediction.
        official_test: Если True, ожидается датасет формата request/response
            и инференс выполняется через уже запущенный API сервер.
        api_url: Базовый URL API сервера для режима official_test.
    """

    dataset_path: Path = Path(get_avito_data_dpath()) / "rnc_dataset_markup.json"
    config_path: Path | None = None
    limit: int | None = None
    output_path: Path = Path(get_avito_gitignore_data_dpath()) / "rnc_dataset_markup_with_predictions.json"
    official_test: bool = False
    api_url: str = GRADIO_DEFAULT_API_URL

    @field_validator("limit")
    @classmethod
    def validate_limit(cls, value: int | None) -> int | None:
        if value is not None and value <= 0:
            raise ValueError("limit должен быть положительным")
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
    """

    samples_count: int
    true_positive: int
    false_positive: int
    false_negative: int
    precision_micro: float
    recall_micro: float
    f1_micro: float
    should_split_accuracy: float


@contextmanager
def _mute_mistral_logger() -> Iterator[None]:
    """Временно отключает логгер mistral-call на время evaluate-прохода."""
    original_disabled = MISTRAL_LOGGER.disabled
    original_level = MISTRAL_LOGGER.level
    try:
        MISTRAL_LOGGER.disabled = True
        MISTRAL_LOGGER.setLevel(logging.CRITICAL)
        yield
    finally:
        MISTRAL_LOGGER.disabled = original_disabled
        MISTRAL_LOGGER.setLevel(original_level)


@contextmanager
def _mute_pipeline_logger() -> Iterator[None]:
    """Временно отключает логи should-split пайплайна в evaluate-цикле."""
    original_disabled = logger.disabled
    original_level = logger.level
    try:
        logger.disabled = True
        logger.setLevel(logging.CRITICAL)
        yield
    finally:
        logger.disabled = original_disabled
        logger.setLevel(original_level)


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


def aggregate_metrics(samples: list[EvaluationSample]) -> EvaluationMetrics:
    """Агрегирует micro-метрики и shouldSplit accuracy по набору сэмплов."""
    tp = 0
    fp = 0
    fn = 0
    correct_should_split = 0

    for sample in samples:
        tp += len(sample.pred_ids & sample.gt_ids)
        fp += len(sample.pred_ids - sample.gt_ids)
        fn += len(sample.gt_ids - sample.pred_ids)
        if sample.pred_should_split == sample.gt_should_split:
            correct_should_split += 1

    precision_micro = _safe_divide(tp, tp + fp)
    recall_micro = _safe_divide(tp, tp + fn)
    f1_micro = _safe_divide(2 * precision_micro * recall_micro, precision_micro + recall_micro)
    should_split_accuracy = _safe_divide(correct_should_split, len(samples))

    return EvaluationMetrics(
        samples_count=len(samples),
        true_positive=tp,
        false_positive=fp,
        false_negative=fn,
        precision_micro=precision_micro,
        recall_micro=recall_micro,
        f1_micro=f1_micro,
        should_split_accuracy=should_split_accuracy,
    )


def _build_evaluation_sample(sample: dict[str, Any], pipeline: ShouldSplitPipeline) -> EvaluationSample:
    description = str(sample.get("description", ""))
    source_mc_id = int(sample.get("sourceMcId", 0))

    state = pipeline.invoke(description)

    gt_ids = set(parse_mc_ids(sample.get("targetSplitMcIds", [])))
    pred_ids = set(int(mc_id) for mc_id in state.categorized_mc_ids)

    gt_ids.discard(source_mc_id)
    pred_ids.discard(source_mc_id)

    pred_should_split = bool(state.shouldSplit)
    # Для оценки дополнительных категорий учитываем только случаи split=True.
    if not pred_should_split:
        pred_ids = set()

    return EvaluationSample(
        gt_ids=gt_ids,
        pred_ids=pred_ids,
        gt_should_split=_parse_should_split(sample.get("shouldSplit", False)),
        pred_should_split=pred_should_split,
    )


def _build_prediction_payload(sample: dict[str, Any], pipeline: ShouldSplitPipeline) -> dict[str, Any]:
    state = pipeline.invoke(str(sample.get("description", "")))

    drafts = [
        {
            "mcId": draft.mc_id,
            "mcTitle": draft.mc_title,
            "text": draft.text,
        }
        for draft in state.drafts
    ]
    return {
        "detectedMcIds": [int(mc_id) for mc_id in state.categorized_mc_ids],
        "shouldSplit": bool(state.shouldSplit),
        "drafts": drafts,
    }


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


def _build_official_artifacts_for_row(
    row: dict[str, Any],
    client: PipelineApiClient,
) -> tuple[EvaluationSample | None, dict[str, Any]]:
    request_data = _parse_json_object(row.get("request"), field_name="request")
    response_data = _try_parse_json_object(row.get("response"), field_name="response")

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


def _build_artifacts_for_row(
    sample: dict[str, Any],
    pipeline: ShouldSplitPipeline,
) -> tuple[EvaluationSample, dict[str, Any]]:
    prediction = _build_prediction_payload(sample=sample, pipeline=pipeline)

    source_mc_id = int(sample.get("sourceMcId", 0))
    gt_ids = set(parse_mc_ids(sample.get("targetSplitMcIds", [])))
    pred_ids = set(int(mc_id) for mc_id in prediction["detectedMcIds"])
    gt_ids.discard(source_mc_id)
    pred_ids.discard(source_mc_id)

    pred_should_split = bool(prediction["shouldSplit"])
    if not pred_should_split:
        pred_ids = set()

    evaluation_sample = EvaluationSample(
        gt_ids=gt_ids,
        pred_ids=pred_ids,
        gt_should_split=_parse_should_split(sample.get("shouldSplit", False)),
        pred_should_split=pred_should_split,
    )
    return evaluation_sample, prediction


def _write_predictions(output_path: Path, rows_with_predictions: list[Any]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
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


def evaluate_markup_dataset(
    dataset_path: Path,
    pipeline: ShouldSplitPipeline,
    limit: int | None = None,
    prediction_rows: list[Any] | None = None,
) -> EvaluationMetrics:
    """Считает метрики качества should-split пайплайна по JSON/CSV-датасету."""
    dataset = _load_dataset_rows(dataset_path)
    rows = dataset[:limit] if limit is not None else dataset
    samples: list[EvaluationSample] = []

    logger.info("[evaluate] start | dataset=%s | rows=%s", dataset_path, len(rows))

    with _mute_mistral_logger(), _mute_pipeline_logger():
        progress = tqdm(rows, total=len(rows), desc="Evaluating", unit="sample")
        for row in progress:
            if not isinstance(row, dict):
                if prediction_rows is not None:
                    prediction_rows.append(row)
                continue

            eval_sample, prediction = _build_artifacts_for_row(sample=row, pipeline=pipeline)
            samples.append(eval_sample)

            if prediction_rows is not None:
                prediction_row = dict(row)
                prediction_row["prediction"] = prediction
                prediction_rows.append(prediction_row)

    metrics = aggregate_metrics(samples)
    logger.info(
        "[evaluate] done | rows=%s | precision_micro=%.6f | recall_micro=%.6f | f1_micro=%.6f | should_split_accuracy=%.6f",
        metrics.samples_count,
        metrics.precision_micro,
        metrics.recall_micro,
        metrics.f1_micro,
        metrics.should_split_accuracy,
    )
    return metrics


def evaluate_official_dataset(
    dataset_path: Path,
    api_url: str,
    limit: int | None = None,
    prediction_rows: list[Any] | None = None,
) -> EvaluationMetrics:
    """Считает метрики по official JSON/CSV-датасету request/response через API."""
    dataset = _load_dataset_rows(dataset_path)
    rows = dataset[:limit] if limit is not None else dataset
    samples: list[EvaluationSample] = []
    client = PipelineApiClient(base_url=api_url)

    logger.info("[evaluate official] start | dataset=%s | rows=%s | api_url=%s", dataset_path, len(rows), api_url)

    progress = tqdm(rows, total=len(rows), desc="Evaluating official", unit="sample")
    for index, row in enumerate(progress):
        if not isinstance(row, dict):
            if prediction_rows is not None:
                prediction_rows.append(row)
            continue

        try:
            eval_sample, prediction = _build_official_artifacts_for_row(row=row, client=client)
        except Exception as error:
            raise ValueError(f"Ошибка обработки строки {index} в official датасете: {error}") from error

        if eval_sample is not None:
            samples.append(eval_sample)

        if prediction_rows is not None:
            prediction_row = dict(row)
            if _try_parse_json_object(prediction_row.get("response"), field_name="response") is None:
                prediction_row["response"] = prediction
            prediction_row["prediction"] = prediction
            prediction_rows.append(prediction_row)

    metrics = aggregate_metrics(samples)
    logger.info(
        "[evaluate official] done | rows=%s | precision_micro=%.6f | recall_micro=%.6f | f1_micro=%.6f | should_split_accuracy=%.6f",
        metrics.samples_count,
        metrics.precision_micro,
        metrics.recall_micro,
        metrics.f1_micro,
        metrics.should_split_accuracy,
    )
    return metrics


def _build_cli_config() -> EvaluateCliConfig:
    parser = argparse.ArgumentParser(description="Evaluate Avito should-split pipeline on markup dataset")
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
        help="Опциональный путь к YAML-конфигу пайплайна",
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
        "--official-test",
        action="store_true",
        help="Режим official: датасет должен иметь поля request/response, инференс идет через API",
    )
    parser.add_argument(
        "--api-url",
        type=str,
        default=GRADIO_DEFAULT_API_URL,
        help="Базовый URL уже запущенного API сервера для режима official",
    )
    args = parser.parse_args()
    return EvaluateCliConfig(
        dataset_path=args.dataset,
        config_path=args.config,
        limit=args.limit,
        output_path=args.output,
        official_test=args.official_test,
        api_url=args.api_url,
    )


def _disable_drafts(config: ShouldSplitGraphConfig) -> ShouldSplitGraphConfig:
    config_data = config.model_dump()
    graph_data = dict(config_data.get("graph", {}))
    graph_data["enable_drafts"] = False
    config_data["graph"] = graph_data
    return ShouldSplitGraphConfig.model_validate(config_data)


def _build_pipeline(config_path: Path | None) -> ShouldSplitPipeline:
    if config_path is not None:
        base_config = ShouldSplitGraphConfig.model_validate({"config_path": config_path})
    else:
        base_config = ShouldSplitGraphConfig.from_default_yaml()

    runtime_config = _disable_drafts(base_config)
    return build_default_pipeline(config=runtime_config)


def main() -> None:
    config = _build_cli_config()
    prediction_rows: list[Any] = []

    if config.official_test:
        metrics = evaluate_official_dataset(
            dataset_path=config.dataset_path,
            api_url=config.api_url,
            limit=config.limit,
            prediction_rows=prediction_rows,
        )
    else:
        pipeline = _build_pipeline(config.config_path)
        metrics = evaluate_markup_dataset(
            dataset_path=config.dataset_path,
            pipeline=pipeline,
            limit=config.limit,
            prediction_rows=prediction_rows,
        )

    _write_predictions(output_path=config.output_path, rows_with_predictions=prediction_rows)
    logger.info("[evaluate] predictions saved | output=%s", config.output_path)
    print(json.dumps(metrics.model_dump(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

