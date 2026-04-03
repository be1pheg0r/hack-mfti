from __future__ import annotations

"""Scores `knowledge_bench_private.csv` with the HF hallucination classifier."""

import argparse
import time
from pathlib import Path
from typing import *

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator
from tqdm.auto import tqdm

from common.logger import SBER_HOOKS_LOGGER as logger
from common.paths import PathLike, get_data_bench_dpath
from src.sber.constants import (
    DEFAULT_SBER_HF_MODEL_REPO_ID,
    DEFAULT_SBER_HF_NLI_COMPUTE_DTYPE,
    DEFAULT_SBER_HF_NLI_CONFIG_FPATH,
)
from src.sber.models.hf_nli_clf import HFNLIClf, HFNLIClfConfig
from src.sber.utils.evaluate_utils import evaluate_scoring_results


class ScoreConfig(BaseModel):
    """Конфигурация скрипта скоринга benchmark-датасета.

    Attributes:
        input_csv: Входной CSV с колонками `correct_answer` и `model_answer`.
        output_csv: Выходной CSV со score-колонками.
        repo_id: Hugging Face repo ID модели-классификатора.
        batch_size: Размер батча для инференса.
        nli_config_path: Путь к YAML-конфигу `HFNLIClf`.
        compute_dtype: Тип вычислений (float32/float16/bfloat16).
        report_dir: Каталог для текстового отчета и графиков.
        save_plots: Сохранять ли графики оценки.
    """

    model_config = ConfigDict(frozen=True)

    input_csv: PathLike = Field(default_factory=lambda: Path(get_data_bench_dpath()) / "knowledge_bench_private.csv")
    output_csv: PathLike = Field(default_factory=lambda: Path(get_data_bench_dpath()) / "knowledge_bench_private_scores.csv")
    repo_id: str = DEFAULT_SBER_HF_MODEL_REPO_ID
    batch_size: int = 8
    nli_config_path: PathLike = DEFAULT_SBER_HF_NLI_CONFIG_FPATH
    compute_dtype: str = DEFAULT_SBER_HF_NLI_COMPUTE_DTYPE
    report_dir: PathLike | None = None
    save_plots: bool = True

    @field_validator("repo_id")
    @classmethod
    def validate_repo_id(cls, value: str) -> str:
        normalized: str = value.strip()
        if not normalized:
            raise ValueError("repo_id не может быть пустым")
        return normalized

    @field_validator("batch_size")
    @classmethod
    def validate_batch_size(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("batch_size должен быть положительным")
        return value

    @field_validator("compute_dtype")
    @classmethod
    def validate_compute_dtype(cls, value: str) -> str:
        normalized: str = value.strip().lower()
        allowed: set[str] = {"float32", "float16", "bfloat16"}
        if normalized not in allowed:
            raise ValueError("compute_dtype должен быть одним из: float32, float16, bfloat16")
        return normalized


def parse_args() -> ScoreConfig:
    """Парсит CLI-аргументы."""
    parser = argparse.ArgumentParser(description="Скоринг knowledge_bench_private.csv через HF классификатор")
    parser.add_argument("--input-csv", type=str, default=str(Path(get_data_bench_dpath()) / "knowledge_bench_private.csv"))
    parser.add_argument("--output-csv", type=str, default=str(Path(get_data_bench_dpath()) / "knowledge_bench_private_scores.csv"))
    parser.add_argument("--repo-id", type=str, default=DEFAULT_SBER_HF_MODEL_REPO_ID)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--nli-config-path", type=str, default=str(DEFAULT_SBER_HF_NLI_CONFIG_FPATH))
    parser.add_argument("--compute-dtype", type=str, default=DEFAULT_SBER_HF_NLI_COMPUTE_DTYPE)
    parser.add_argument("--report-dir", type=str, default=None)
    parser.add_argument("--save-plots", action=argparse.BooleanOptionalAction, default=True)
    namespace: argparse.Namespace = parser.parse_args()
    return ScoreConfig.model_validate(vars(namespace))


def score_dataframe(
    df: pd.DataFrame,
    repo_id: str,
    batch_size: int,
    nli_config_path: PathLike,
    compute_dtype: str,
) -> tuple[pd.DataFrame, int]:
    """Скорит DataFrame с колонками `correct_answer` и `model_answer`."""
    nli_config: HFNLIClfConfig = HFNLIClfConfig.from_yaml(nli_config_path).model_copy(
        update={"repo_id": repo_id, "batch_size": batch_size, "compute_dtype": compute_dtype}
    )
    clf: HFNLIClf = HFNLIClf(config=nli_config)

    if "correct_answer" not in df.columns or "model_answer" not in df.columns:
        raise ValueError("Ожидаются колонки correct_answer и model_answer")

    entailment_scores: list[float] = []
    predicted_hallucinations: list[int] = []
    per_sample_times: list[float] = []
    total_batches: int = (len(df) + batch_size - 1) // batch_size
    progress_bar = tqdm(total=total_batches, desc="Scoring benchmark")

    for start in range(0, len(df), batch_size):
        batch = df.iloc[start : start + batch_size]
        batch_premises: list[str] = batch["correct_answer"].astype(str).tolist()
        batch_hypotheses: list[str] = batch["model_answer"].astype(str).tolist()

        infer_start: float = time.perf_counter()
        scores: list[float] = clf.predict_entailment_proba(
            premises=batch_premises,
            hypotheses=batch_hypotheses,
            batch_size=batch_size,
        )
        preds: list[int] = clf.predict_is_hallucination(
            premises=batch_premises,
            hypotheses=batch_hypotheses,
            batch_size=batch_size,
        )
        infer_end: float = time.perf_counter()

        batch_time_sec: float = infer_end - infer_start
        batch_sample_time_sec: float = batch_time_sec / max(len(batch), 1)
        entailment_scores.extend(float(score) for score in scores)
        predicted_hallucinations.extend(int(value) for value in preds)
        per_sample_times.extend([batch_sample_time_sec] * len(batch))
        progress_bar.update(1)

    progress_bar.close()
    result = df.copy()
    result["entailment_score"] = entailment_scores
    result["hallucination_score"] = [1.0 - score for score in entailment_scores]
    result["pred_is_hallucination"] = predicted_hallucinations
    result["t_sample_sec"] = per_sample_times
    return result, total_batches


def run(config: ScoreConfig) -> Path:
    """Скорит benchmark и сохраняет CSV."""
    input_path = Path(config.input_csv)
    output_path = Path(config.output_csv)
    logger.info("Читаю benchmark: %s", input_path)
    dataframe = pd.read_csv(input_path)
    scored, total_batches = score_dataframe(
        df=dataframe,
        repo_id=config.repo_id,
        batch_size=config.batch_size,
        nli_config_path=config.nli_config_path,
        compute_dtype=config.compute_dtype,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(output_path, index=False)

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
    logger.info("Сохранил score-файл: %s", output_path)
    return output_path


def main() -> None:
    """Точка входа CLI."""
    run(parse_args())


if __name__ == "__main__":
    main()


