from __future__ import annotations

"""Scores `knowledge_bench_private.csv` with the HF hallucination classifier."""

import argparse
import re
import time
from pathlib import Path
from typing import *

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator
from sklearn.metrics import fbeta_score
from tqdm.auto import tqdm

from common.logger import SBER_EVALUATION_LOGGER as logger
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
        hallucination_threshold: Порог бинаризации hallucination_score.
        confidence_floor: Минимальная confidence, ниже которой positive-предсказание гасится.
        threshold_search_mode: Режим подбора threshold: none/grid/optuna.
        threshold_search_beta: Бета для F-beta при подборе threshold (beta>1 штрафует FN).
        threshold_trials: Количество trial для Optuna-оптимизации threshold.
        numeric_overlap_weight: Вес штрафа за несовпадение числовых фактов.
        entity_overlap_weight: Вес штрафа за несовпадение named entities.
        token_precision_weight: Вес штрафа за низкую token precision.
        ambiguity_discount_weight: Вес снижения score в ambiguous-зоне.
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
    hallucination_threshold: float = 0.5
    confidence_floor: float = 0.0
    threshold_search_mode: str = "none"
    threshold_search_beta: float = 2.0
    threshold_trials: int = 50
    numeric_overlap_weight: float = 0.0
    entity_overlap_weight: float = 0.0
    token_precision_weight: float = 0.0
    ambiguity_discount_weight: float = 0.0

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

    @field_validator("hallucination_threshold", "confidence_floor")
    @classmethod
    def validate_probability_threshold(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError("Порог должен быть в диапазоне [0, 1]")
        return value

    @field_validator("threshold_search_mode")
    @classmethod
    def validate_threshold_search_mode(cls, value: str) -> str:
        normalized: str = value.strip().lower()
        allowed: set[str] = {"none", "grid", "optuna"}
        if normalized not in allowed:
            raise ValueError("threshold_search_mode должен быть одним из: none, grid, optuna")
        return normalized

    @field_validator("threshold_search_beta")
    @classmethod
    def validate_threshold_search_beta(cls, value: float) -> float:
        if value <= 0.0:
            raise ValueError("threshold_search_beta должен быть больше нуля")
        return value

    @field_validator("threshold_trials")
    @classmethod
    def validate_threshold_trials(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("threshold_trials должен быть положительным")
        return value

    @field_validator(
        "numeric_overlap_weight",
        "entity_overlap_weight",
        "token_precision_weight",
        "ambiguity_discount_weight",
    )
    @classmethod
    def validate_non_negative_weight(cls, value: float) -> float:
        if value < 0.0:
            raise ValueError("Вес не может быть отрицательным")
        return value


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
    parser.add_argument("--hallucination-threshold", type=float, default=0.5)
    parser.add_argument("--confidence-floor", type=float, default=0.0)
    parser.add_argument("--threshold-search-mode", type=str, default="none")
    parser.add_argument("--threshold-search-beta", type=float, default=2.0)
    parser.add_argument("--threshold-trials", type=int, default=50)
    parser.add_argument("--numeric-overlap-weight", type=float, default=0.0)
    parser.add_argument("--entity-overlap-weight", type=float, default=0.0)
    parser.add_argument("--token-precision-weight", type=float, default=0.0)
    parser.add_argument("--ambiguity-discount-weight", type=float, default=0.0)
    namespace: argparse.Namespace = parser.parse_args()
    return ScoreConfig.model_validate(vars(namespace))


def _extract_numbers(text: str) -> set[str]:
    return set(re.findall(r"\d+(?:[.,]\d+)?", text))


def _extract_named_entities(text: str) -> set[str]:
    title_case_entities: set[str] = set(
        re.findall(r"\b[A-ZА-ЯЁ][a-zа-яё]+(?:[-\s][A-ZА-ЯЁ][a-zа-яё]+)*\b", text)
    )
    acronyms: set[str] = set(re.findall(r"\b[A-ZА-ЯЁ]{2,}\b", text))
    entities: set[str] = {value.strip() for value in title_case_entities.union(acronyms) if value.strip()}
    return {value.lower() for value in entities}


def _tokenize_text(text: str) -> list[str]:
    return re.findall(r"[\w\-]+", text.lower(), flags=re.UNICODE)


def _overlap_ratio(source_items: set[str], reference_items: set[str], empty_default: float = 1.0) -> float:
    if not source_items:
        return empty_default
    intersection_count: int = len(source_items.intersection(reference_items))
    return float(intersection_count / max(len(source_items), 1))


def _token_precision(predicted_text: str, reference_text: str) -> float:
    predicted_tokens: list[str] = _tokenize_text(predicted_text)
    reference_tokens: set[str] = set(_tokenize_text(reference_text))
    if not predicted_tokens:
        return 1.0
    matched: int = sum(1 for token in predicted_tokens if token in reference_tokens)
    return float(matched / len(predicted_tokens))


def _build_fact_features(correct_answer: str, model_answer: str) -> tuple[float, float, float]:
    numeric_overlap: float = _overlap_ratio(_extract_numbers(model_answer), _extract_numbers(correct_answer))
    entity_overlap: float = _overlap_ratio(_extract_named_entities(model_answer), _extract_named_entities(correct_answer))
    token_precision: float = _token_precision(predicted_text=model_answer, reference_text=correct_answer)
    return numeric_overlap, entity_overlap, token_precision


def _compute_adjusted_hallucination_score(
    *,
    base_hallucination_score: float,
    entailment_score: float,
    numeric_overlap: float,
    entity_overlap: float,
    token_precision: float,
    numeric_overlap_weight: float,
    entity_overlap_weight: float,
    token_precision_weight: float,
    ambiguity_discount_weight: float,
) -> tuple[float, float]:
    confidence: float = abs(entailment_score - base_hallucination_score)
    fact_mismatch_boost: float = (
        numeric_overlap_weight * (1.0 - numeric_overlap)
        + entity_overlap_weight * (1.0 - entity_overlap)
        + token_precision_weight * (1.0 - token_precision)
    )
    ambiguity_discount: float = ambiguity_discount_weight * (1.0 - confidence)
    adjusted_score: float = max(0.0, min(1.0, base_hallucination_score + fact_mismatch_boost - ambiguity_discount))
    return adjusted_score, confidence


def _find_best_threshold_grid(scores: Sequence[float], labels: Sequence[int], beta: float) -> float:
    y_true: list[int] = [int(value) for value in labels]
    y_score: list[float] = [float(value) for value in scores]
    best_threshold: float = 0.5
    best_metric: float = -1.0
    for step in range(1, 100):
        threshold: float = step / 100.0
        preds: list[int] = [int(score >= threshold) for score in y_score]
        metric_value: float = float(fbeta_score(y_true, preds, beta=beta, pos_label=1, zero_division=0))
        if metric_value > best_metric:
            best_metric = metric_value
            best_threshold = threshold
    return best_threshold


def _find_best_threshold_optuna(
    scores: Sequence[float],
    labels: Sequence[int],
    *,
    beta: float,
    trials: int,
) -> float:
    try:
        import optuna
    except ImportError:
        logger.warning("Optuna не установлена, fallback на grid-search threshold")
        return _find_best_threshold_grid(scores=scores, labels=labels, beta=beta)

    y_true: list[int] = [int(value) for value in labels]
    y_score: list[float] = [float(value) for value in scores]

    def objective(trial: Any) -> float:
        threshold: float = float(trial.suggest_float("threshold", 0.01, 0.99))
        preds: list[int] = [int(score >= threshold) for score in y_score]
        return float(fbeta_score(y_true, preds, beta=beta, pos_label=1, zero_division=0))

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=trials, show_progress_bar=False)
    return float(study.best_params["threshold"])


def score_dataframe(
    df: pd.DataFrame,
    repo_id: str,
    batch_size: int,
    nli_config_path: PathLike,
    compute_dtype: str,
    *,
    hallucination_threshold: float = 0.5,
    confidence_floor: float = 0.0,
    numeric_overlap_weight: float = 0.0,
    entity_overlap_weight: float = 0.0,
    token_precision_weight: float = 0.0,
    ambiguity_discount_weight: float = 0.0,
) -> tuple[pd.DataFrame, int]:
    """Скорит DataFrame с колонками `correct_answer` и `model_answer`."""
    nli_config: HFNLIClfConfig = HFNLIClfConfig.from_yaml(nli_config_path).model_copy(
        update={"repo_id": repo_id, "batch_size": batch_size, "compute_dtype": compute_dtype}
    )
    clf: HFNLIClf = HFNLIClf(config=nli_config)

    if "correct_answer" not in df.columns or "model_answer" not in df.columns:
        raise ValueError("Ожидаются колонки correct_answer и model_answer")

    entailment_scores: list[float] = []
    hallucination_scores: list[float] = []
    predicted_hallucinations: list[int] = []
    confidence_scores: list[float] = []
    numeric_overlaps: list[float] = []
    entity_overlaps: list[float] = []
    token_precisions: list[float] = []
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
        infer_end: float = time.perf_counter()

        batch_time_sec: float = infer_end - infer_start
        batch_sample_time_sec: float = batch_time_sec / max(len(batch), 1)
        entailment_scores.extend(float(score) for score in scores)
        batch_hallucination_scores: list[float] = [1.0 - float(score) for score in scores]

        for index in range(len(batch)):
            numeric_overlap, entity_overlap, token_precision = _build_fact_features(
                correct_answer=batch_premises[index],
                model_answer=batch_hypotheses[index],
            )
            adjusted_score, confidence = _compute_adjusted_hallucination_score(
                base_hallucination_score=batch_hallucination_scores[index],
                entailment_score=float(scores[index]),
                numeric_overlap=numeric_overlap,
                entity_overlap=entity_overlap,
                token_precision=token_precision,
                numeric_overlap_weight=numeric_overlap_weight,
                entity_overlap_weight=entity_overlap_weight,
                token_precision_weight=token_precision_weight,
                ambiguity_discount_weight=ambiguity_discount_weight,
            )
            hallucination_scores.append(adjusted_score)
            confidence_scores.append(confidence)
            numeric_overlaps.append(numeric_overlap)
            entity_overlaps.append(entity_overlap)
            token_precisions.append(token_precision)

            predicted_value: int = int(adjusted_score >= hallucination_threshold)
            if predicted_value == 1 and confidence < confidence_floor:
                predicted_value = 0
            predicted_hallucinations.append(predicted_value)

        per_sample_times.extend([batch_sample_time_sec] * len(batch))
        progress_bar.update(1)

    progress_bar.close()
    result = df.copy()
    result["entailment_score"] = entailment_scores
    result["hallucination_score"] = hallucination_scores
    result["confidence"] = confidence_scores
    result["numeric_overlap"] = numeric_overlaps
    result["entity_overlap"] = entity_overlaps
    result["token_precision"] = token_precisions
    result["pred_is_hallucination"] = predicted_hallucinations
    result["t_sample_sec"] = per_sample_times
    return result, total_batches


def _resolve_threshold(
    dataframe: pd.DataFrame,
    *,
    threshold_search_mode: str,
    threshold_search_beta: float,
    threshold_trials: int,
    fallback_threshold: float,
    confidence_floor: float,
) -> float:
    if threshold_search_mode == "none" or "is_hallucination" not in dataframe.columns:
        return fallback_threshold

    scores: list[float] = [float(value) for value in dataframe["hallucination_score"].tolist()]
    labels: list[int] = [int(value) for value in dataframe["is_hallucination"].tolist()]
    if threshold_search_mode == "optuna":
        best_threshold: float = _find_best_threshold_optuna(
            scores=scores,
            labels=labels,
            beta=threshold_search_beta,
            trials=threshold_trials,
        )
    else:
        best_threshold = _find_best_threshold_grid(scores=scores, labels=labels, beta=threshold_search_beta)

    recalculated_preds: list[int] = []
    for score, confidence in zip(scores, dataframe["confidence"].tolist()):
        predicted_value: int = int(float(score) >= best_threshold)
        if predicted_value == 1 and float(confidence) < confidence_floor:
            predicted_value = 0
        recalculated_preds.append(predicted_value)
    dataframe["pred_is_hallucination"] = recalculated_preds
    logger.info("Подобран threshold=%.4f (%s)", best_threshold, threshold_search_mode)
    return best_threshold


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
        hallucination_threshold=config.hallucination_threshold,
        confidence_floor=config.confidence_floor,
        numeric_overlap_weight=config.numeric_overlap_weight,
        entity_overlap_weight=config.entity_overlap_weight,
        token_precision_weight=config.token_precision_weight,
        ambiguity_discount_weight=config.ambiguity_discount_weight,
    )

    selected_threshold: float = _resolve_threshold(
        dataframe=scored,
        threshold_search_mode=config.threshold_search_mode,
        threshold_search_beta=config.threshold_search_beta,
        threshold_trials=config.threshold_trials,
        fallback_threshold=config.hallucination_threshold,
        confidence_floor=config.confidence_floor,
    )
    logger.info("Используем threshold=%.4f, confidence_floor=%.4f", selected_threshold, config.confidence_floor)

    output_columns: list[str] = [*list(dataframe.columns), "pred_is_hallucination", "t_sample_ms"]
    export_dataframe: pd.DataFrame = scored.assign(t_sample_ms=scored["t_sample_sec"] * 1000.0)[output_columns]

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
    logger.info("Сохранил score-файл: %s", output_path)
    return output_path


def main() -> None:
    """Точка входа CLI."""
    run(parse_args())


if __name__ == "__main__":
    main()


