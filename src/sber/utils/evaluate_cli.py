from __future__ import annotations

"""Оценивает tabular-модель на CSV-датасете и сохраняет отчеты."""

import argparse
import time
from pathlib import Path

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator

from common.logger import SBER_EVALUATION_LOGGER as logger
from common.paths import PathLike, get_data_bench_dpath
from src.sber.models.tabular_hallucination import TabularHallucinationPredictor
from src.sber.utils.evaluate_utils import evaluate_scoring_results


class ScoreConfig(BaseModel):
    """Конфигурация скрипта оценки tabular-модели.

    Attributes:
        input_csv: Входной CSV для скоринга.
        output_csv: Выходной CSV со score-колонками.
        checkpoint_dir: Директория tabular-чекпоинта.
        threshold: Опциональный override порога классификации.
        report_dir: Каталог для текстового отчета и графиков.
        save_plots: Сохранять ли графики оценки.
    """

    model_config = ConfigDict(frozen=True)

    input_csv: PathLike = Field(default_factory=lambda: Path(get_data_bench_dpath()) / "bench_processed_judged_mapped.csv")
    output_csv: PathLike = Field(default_factory=lambda: Path(get_data_bench_dpath()) / "bench_processed_judged_tabular_scores.csv")
    checkpoint_dir: PathLike = "sber_tabular/latest"
    threshold: float | None = None
    report_dir: PathLike | None = None
    save_plots: bool = True

    @field_validator("threshold")
    @classmethod
    def validate_probability_threshold(cls, value: float | None) -> float | None:
        if value is None:
            return value
        if not 0.0 <= value <= 1.0:
            raise ValueError("Порог должен быть в диапазоне [0, 1]")
        return value


def parse_args() -> ScoreConfig:
    """Парсит CLI-аргументы."""
    parser = argparse.ArgumentParser(description="Оценка tabular-модели на benchmark CSV")
    parser.add_argument(
        "--input-csv",
        "--input_csv",
        dest="input_csv",
        type=str,
        default=str(Path(get_data_bench_dpath()) / "bench_processed_judged_mapped.csv"),
    )
    parser.add_argument(
        "--output-csv",
        "--output_csv",
        dest="output_csv",
        type=str,
        default=str(Path(get_data_bench_dpath()) / "bench_processed_judged_tabular_scores.csv"),
    )
    parser.add_argument("--checkpoint-dir", "--checkpoint_dir", dest="checkpoint_dir", type=str, default="sber_tabular/latest")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--report-dir", "--report_dir", dest="report_dir", type=str, default=None)
    parser.add_argument("--save-plots", "--save_plots", dest="save_plots", action=argparse.BooleanOptionalAction, default=True)
    namespace: argparse.Namespace = parser.parse_args()
    return ScoreConfig.model_validate(vars(namespace))


def score_dataframe(
    df: pd.DataFrame,
    checkpoint_dir: PathLike,
    *,
    threshold: float | None = None,
) -> tuple[pd.DataFrame, int]:
    """Скорит DataFrame через сохраненный tabular-чекпоинт."""
    predictor = TabularHallucinationPredictor(checkpoint_dir=checkpoint_dir)
    infer_start: float = time.perf_counter()
    result: pd.DataFrame = predictor.predict_dataframe(df, threshold=threshold)
    infer_end: float = time.perf_counter()

    per_sample_sec: float = (infer_end - infer_start) / max(len(result), 1)
    result["t_sample_sec"] = per_sample_sec
    return result, 1


def run(config: ScoreConfig) -> Path:
    """Скорит датасет tabular-моделью и сохраняет CSV/отчет."""
    input_path = Path(config.input_csv)
    output_path = Path(config.output_csv)
    logger.info("Читаю датасет для оценки: %s", input_path)
    dataframe = pd.read_csv(input_path)
    scored, total_batches = score_dataframe(
        df=dataframe,
        checkpoint_dir=config.checkpoint_dir,
        threshold=config.threshold,
    )
    logger.info("Скоринг завершен, строк=%s", len(scored))

    export_dataframe: pd.DataFrame = scored.assign(t_sample_ms=scored["t_sample_sec"] * 1000.0)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    export_dataframe.to_csv(output_path, index=False)

    report_dir: Path = Path(config.report_dir) if config.report_dir is not None else output_path.parent / "reports"
    summary = evaluate_scoring_results(
        dataframe=scored,
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
    """Точка входа CLI."""
    run(parse_args())


if __name__ == "__main__":
    main()



