from __future__ import annotations

"""Оценивает модель как evaluate.py, но фичи извлекаются on-the-fly."""

import argparse
import math
from pathlib import Path
from typing import *

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator

from common.logger import SBER_EVALUATION_LOGGER as logger
from common.paths import PathLike, get_data_bench_dpath
from src.servers.utils.tabular_pipeline_utils import (
    TabularPipelineRequest,
    TabularPipelineServerConfig,
    TabularPipelineService,
)
from src.sber.constants import DEFAULT_FEATURE_EXTRACTION_CONFIGS_FPATH, DEFAULT_SBER_SCRIPT_MODEL_NAME
from src.sber.utils.evaluate_utils import evaluate_scoring_results


class FullEvaluateConfig(BaseModel):
    """Конфигурация полного evaluate-пайплайна.

    Attributes:
        input_csv: Входной CSV с колонками query/prompt и model_answer.
        output_csv: Выходной CSV со score-колонками.
        checkpoint_dir: Директория tabular-чекпоинта.
        feature_model_name: Имя/путь LLM модели для извлечения фичей.
        feature_config_path: YAML-конфиг feature extractor.
        feature_batch_size: Размер батча feature extractor.
        input_query_column: Явная query-колонка (если не задана, auto: query -> prompt).
        threshold: Опциональный override порога классификации.
        report_dir: Каталог для текстового отчета и графиков.
        save_plots: Сохранять ли графики оценки.
    """

    model_config = ConfigDict(frozen=True)

    input_csv: PathLike = Field(default_factory=lambda: Path(get_data_bench_dpath()) / "bench_processed_judged.csv")
    output_csv: PathLike = Field(default_factory=lambda: Path(get_data_bench_dpath()) / "bench_processed_judged_tabular_scores_full_eval.csv")
    checkpoint_dir: PathLike = "sber_tabular/latest"
    feature_model_name: str = DEFAULT_SBER_SCRIPT_MODEL_NAME
    feature_config_path: PathLike = DEFAULT_FEATURE_EXTRACTION_CONFIGS_FPATH
    feature_batch_size: int = 2
    input_query_column: str | None = None
    threshold: float | None = None
    report_dir: PathLike | None = None
    save_plots: bool = True

    @field_validator("feature_batch_size")
    @classmethod
    def validate_feature_batch_size(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("feature_batch_size должен быть положительным")
        return value

    @field_validator("input_query_column")
    @classmethod
    def validate_input_query_column(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized: str = value.strip()
        if not normalized:
            raise ValueError("input_query_column не может быть пустым")
        return normalized

    @field_validator("threshold")
    @classmethod
    def validate_threshold(cls, value: float | None) -> float | None:
        if value is None:
            return value
        if not 0.0 <= value <= 1.0:
            raise ValueError("threshold должен быть в диапазоне [0, 1]")
        return value


def parse_args() -> FullEvaluateConfig:
    """Парсит CLI-аргументы full evaluate скрипта."""
    parser = argparse.ArgumentParser(description="Полный evaluate: feature extractor + tabular classifier")
    parser.add_argument("--input-csv", type=str, default=str(Path(get_data_bench_dpath()) / "bench_processed_judged.csv"))
    parser.add_argument(
        "--output-csv",
        type=str,
        default=str(Path(get_data_bench_dpath()) / "bench_processed_judged_tabular_scores_full_eval.csv"),
    )
    parser.add_argument("--checkpoint-dir", type=str, default="sber_tabular/latest")
    parser.add_argument("--feature-model-name", type=str, default=DEFAULT_SBER_SCRIPT_MODEL_NAME)
    parser.add_argument("--feature-config-path", type=str, default=str(DEFAULT_FEATURE_EXTRACTION_CONFIGS_FPATH))
    parser.add_argument("--feature-batch-size", type=int, default=2)
    parser.add_argument("--input-query-column", type=str, default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--report-dir", type=str, default=None)
    parser.add_argument("--save-plots", action=argparse.BooleanOptionalAction, default=True)
    namespace: argparse.Namespace = parser.parse_args()
    return FullEvaluateConfig.model_validate(vars(namespace))


def _resolve_query_column_name(columns: Sequence[str], preferred_column: str | None = None) -> str:
    """Определяет колонку запроса во входном CSV."""
    available: set[str] = {str(column) for column in columns}
    if preferred_column is not None:
        if preferred_column in available:
            return preferred_column
        raise ValueError(f"Колонка {preferred_column} не найдена во входном CSV")

    for candidate in ("query", "prompt"):
        if candidate in available:
            return candidate
    raise ValueError("Во входном CSV должна быть колонка query или prompt")


def run(config: FullEvaluateConfig) -> Path:
    """Скорит датасет с online feature extraction и сохраняет CSV/отчет."""
    input_path: Path = Path(config.input_csv)
    output_path: Path = Path(config.output_csv)

    dataframe: pd.DataFrame = pd.read_csv(input_path)
    query_column: str = _resolve_query_column_name(
        columns=[str(column) for column in dataframe.columns],
        preferred_column=config.input_query_column,
    )
    if "model_answer" not in dataframe.columns:
        raise ValueError("Во входном CSV должна быть колонка model_answer")

    queries: list[str] = dataframe[query_column].astype("string").fillna("").str.strip().tolist()
    model_answers: list[str] = dataframe["model_answer"].astype("string").fillna("").str.strip().tolist()
    if any(not query for query in queries):
        raise ValueError(f"Во входном CSV найдены пустые значения в колонке {query_column}")
    if any(not answer for answer in model_answers):
        raise ValueError("Во входном CSV найдены пустые значения в колонке model_answer")

    service_config: TabularPipelineServerConfig = TabularPipelineServerConfig(
        serve_mode="tabular_pipeline",
        feature_extractor_mode="real",
        feature_model_name=config.feature_model_name,
        checkpoint_dir=config.checkpoint_dir,
        feature_config_path=config.feature_config_path,
        feature_batch_size=config.feature_batch_size,
    )
    service: TabularPipelineService = TabularPipelineService(config=service_config)

    response: dict[str, list[float] | list[int]] = service.predict(
        TabularPipelineRequest(
            queries=queries,
            model_answers=model_answers,
            threshold=config.threshold,
        )
    )

    result: pd.DataFrame = dataframe.copy()
    if "query" not in result.columns:
        result["query"] = queries
    result["hallucination_score"] = [float(value) for value in response["hallucination_score"]]
    result["predict_proba"] = [float(value) for value in response["hallucination_score"]]
    result["pred_is_hallucination"] = [int(value) for value in response["pred_is_hallucination"]]
    result["entailment_score"] = [float(value) for value in response["entailment_score"]]
    result["t_feature_extraction_sec"] = [float(value) for value in response["t_feature_extraction_sec"]]
    result["t_classification_sec"] = [float(value) for value in response["t_classification_sec"]]
    result["t_total_sec"] = [float(value) for value in response["t_total_sec"]]
    result["t_sample_sec"] = [float(value) for value in response["t_sample_sec"]]

    total_batches: int = max(int(math.ceil(len(result) / max(config.feature_batch_size, 1))), 1)
    export_dataframe: pd.DataFrame = result.assign(t_sample_ms=result["t_sample_sec"] * 1000.0)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    export_dataframe.to_csv(output_path, index=False)

    report_dir: Path = Path(config.report_dir) if config.report_dir is not None else output_path.parent / "reports"
    summary = evaluate_scoring_results(
        dataframe=result,
        output_csv_fpath=output_path,
        report_dpath=report_dir,
        save_plots_enabled=config.save_plots,
        total_batches=total_batches,
    )

    logger.info("\n%s", summary.report_text)
    logger.info("Сохранил отчет: %s", summary.report_fpath)
    logger.info("Сохранил файл с предсказаниями: %s", output_path)
    return output_path


def main() -> None:
    """CLI entrypoint."""
    run(parse_args())


if __name__ == "__main__":
    main()

