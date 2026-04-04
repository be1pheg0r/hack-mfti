from __future__ import annotations

from pathlib import Path
from typing import *

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pydantic import BaseModel, ConfigDict, Field, field_validator
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from common.logger import SBER_EVALUATION_LOGGER as logger
from common.paths import PathLike


class TimingMetrics(BaseModel):
    """Агрегированные метрики времени инференса.

    Attributes:
        total_samples: Количество обработанных сэмплов.
        total_batches: Количество батчей инференса.
        total_inference_sec: Суммарное время инференса в секундах.
        mean_sample_sec: Среднее время на один сэмпл.
        p50_sample_sec: Медиана времени на сэмпл.
        p95_sample_sec: 95-й перцентиль времени на сэмпл.
        throughput_samples_per_sec: Пропускная способность по сэмплам/сек.
    """

    model_config = ConfigDict(frozen=True)

    total_samples: int
    total_batches: int
    total_inference_sec: float
    mean_sample_sec: float
    p50_sample_sec: float
    p95_sample_sec: float
    throughput_samples_per_sec: float

    @field_validator("total_samples", "total_batches")
    @classmethod
    def validate_non_negative_int(cls, value: int) -> int:
        if value < 0:
            raise ValueError("Числовой параметр не может быть отрицательным")
        return value


class ClassificationMetrics(BaseModel):
    """Качество классификации по бинарным меткам галлюцинации.

    Attributes:
        has_labels: Есть ли в данных истинные метки.
        accuracy: Accuracy предсказаний.
        precision: Precision для класса галлюцинации.
        recall: Recall для класса галлюцинации.
        f1: F1-score для класса галлюцинации.
        average_precision: PR-AUC (Average Precision) по score.
        weighted_average_precision: Weighted PR-AUC как в notebook-валидации.
    """

    model_config = ConfigDict(frozen=True)

    has_labels: bool
    accuracy: float | None = None
    precision: float | None = None
    recall: float | None = None
    f1: float | None = None
    average_precision: float | None = None
    weighted_average_precision: float | None = None


class EvaluationSummary(BaseModel):
    """Сводка оценки скоринга.

    Attributes:
        timing: Метрики скорости инференса.
        classification: Метрики качества классификации.
        report_text: Форматированный текстовый отчет.
        plots: Пути к сохраненным графикам.
        misclassified_fpath: Путь к CSV с ошибочными сэмплами для hard negative mining.
        report_fpath: Путь к txt-отчету.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    timing: TimingMetrics
    classification: ClassificationMetrics
    report_text: str
    plots: list[Path] = Field(default_factory=list)
    misclassified_fpath: Path | None = None
    report_fpath: Path | None = None


def _quantile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    index: float = (len(sorted_values) - 1) * q
    lower: int = int(index)
    upper: int = min(lower + 1, len(sorted_values) - 1)
    weight: float = index - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def compute_timing_metrics(sample_times: list[float], total_batches: int) -> TimingMetrics:
    """Вычисляет агрегированные метрики скорости инференса."""
    total_samples: int = len(sample_times)
    total_inference_sec: float = float(sum(sample_times))
    mean_sample_sec: float = total_inference_sec / max(total_samples, 1)
    sorted_times: list[float] = sorted(sample_times)
    p50_sample_sec: float = _quantile(sorted_times, 0.5)
    p95_sample_sec: float = _quantile(sorted_times, 0.95)
    throughput: float = total_samples / max(total_inference_sec, 1e-12)
    return TimingMetrics(
        total_samples=total_samples,
        total_batches=total_batches,
        total_inference_sec=total_inference_sec,
        mean_sample_sec=mean_sample_sec,
        p50_sample_sec=p50_sample_sec,
        p95_sample_sec=p95_sample_sec,
        throughput_samples_per_sec=throughput,
    )


def compute_classification_metrics(
    dataframe: pd.DataFrame,
    *,
    label_col: str,
    pred_col: str,
    score_col: str,
) -> ClassificationMetrics:
    """Вычисляет метрики качества, если присутствует столбец меток."""
    if label_col not in dataframe.columns:
        return ClassificationMetrics(has_labels=False)

    y_true: np.ndarray = np.array([int(value) for value in dataframe[label_col].tolist()], dtype=np.int64)
    y_pred: np.ndarray = np.array([int(value) for value in dataframe[pred_col].tolist()], dtype=np.int64)
    y_score: np.ndarray = np.array([float(value) for value in dataframe[score_col].tolist()], dtype=np.float64)

    # Повторяем notebook-логику: невалидные score заменяем на 0.5.
    if np.any(~np.isfinite(y_score)):
        y_score = np.nan_to_num(y_score, nan=0.5, posinf=1.0, neginf=0.0)

    pos_rate: float = float(y_true.mean()) if y_true.size else 0.0
    sample_weight: np.ndarray | None = None
    if 0.0 < pos_rate < 1.0:
        pos_weight: float = (1.0 - pos_rate) / pos_rate
        sample_weight = np.where(y_true == 1, pos_weight, 1.0).astype(np.float64)

    average_precision: float | None = None
    weighted_average_precision: float | None = None
    try:
        average_precision = float(average_precision_score(y_true, y_score))
    except ValueError:
        average_precision = None

    try:
        weighted_average_precision = float(
            average_precision_score(y_true, y_score, sample_weight=sample_weight)
        )
    except ValueError:
        weighted_average_precision = None

    return ClassificationMetrics(
        has_labels=True,
        accuracy=float(accuracy_score(y_true, y_pred)),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        average_precision=average_precision,
        weighted_average_precision=weighted_average_precision,
    )


def _save_histogram(dataframe: pd.DataFrame, score_col: str, output_dpath: Path) -> Path:
    fpath: Path = output_dpath / "hallucination_score_hist.png"
    plt.figure(figsize=(10, 6))
    sns.histplot(dataframe[score_col], bins=30, kde=True, color="#3366cc")
    plt.title("Hallucination Score Distribution")
    plt.xlabel("hallucination_score")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(fpath, dpi=140)
    plt.close()
    return fpath


def _save_timing_boxplot(dataframe: pd.DataFrame, time_col: str, output_dpath: Path) -> Path:
    fpath: Path = output_dpath / "sample_time_boxplot.png"
    sample_time_ms: pd.Series = dataframe[time_col].astype(float) * 1000.0
    plt.figure(figsize=(10, 4))
    sns.boxplot(x=sample_time_ms, color="#66aa00")
    plt.title("Per-sample Inference Time")
    plt.xlabel("milliseconds")
    plt.tight_layout()
    plt.savefig(fpath, dpi=140)
    plt.close()
    return fpath


def _save_confusion_matrix(
    dataframe: pd.DataFrame,
    *,
    label_col: str,
    pred_col: str,
    output_dpath: Path,
) -> Path:
    fpath: Path = output_dpath / "confusion_matrix.png"
    y_true: list[int] = [int(value) for value in dataframe[label_col].tolist()]
    y_pred: list[int] = [int(value) for value in dataframe[pred_col].tolist()]
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["pred_0", "pred_1"],
        yticklabels=["true_0", "true_1"],
    )
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(fpath, dpi=140)
    plt.close()
    return fpath


def save_score_plots(
    dataframe: pd.DataFrame,
    *,
    output_dpath: PathLike,
    score_col: str,
    time_col: str,
    label_col: str,
    pred_col: str,
) -> list[Path]:
    """Сохраняет графики оценки и возвращает пути к ним."""
    logger.info("Сохраняю графики оценки в %s", output_dpath)
    out_dir: Path = Path(output_dpath)
    out_dir.mkdir(parents=True, exist_ok=True)

    plots: list[Path] = [
        _save_histogram(dataframe=dataframe, score_col=score_col, output_dpath=out_dir),
        _save_timing_boxplot(dataframe=dataframe, time_col=time_col, output_dpath=out_dir),
    ]

    if label_col in dataframe.columns:
        plots.append(
            _save_confusion_matrix(
                dataframe=dataframe,
                label_col=label_col,
                pred_col=pred_col,
                output_dpath=out_dir,
            )
        )

    return plots


def save_misclassified_samples(
    dataframe: pd.DataFrame,
    *,
    output_dpath: PathLike,
    label_col: str,
    pred_col: str,
    score_col: str,
) -> Path | None:
    """Сохраняет ошибки классификации (FP/FN) для hard negative mining."""
    if label_col not in dataframe.columns:
        return None

    out_dir: Path = Path(output_dpath)
    out_dir.mkdir(parents=True, exist_ok=True)

    labeled: pd.DataFrame = dataframe.copy()
    labeled[label_col] = labeled[label_col].astype(int)
    labeled[pred_col] = labeled[pred_col].astype(int)
    misclassified: pd.DataFrame = labeled[labeled[label_col] != labeled[pred_col]].copy()
    if misclassified.empty:
        fpath_empty: Path = out_dir / "misclassified_samples.csv"
        misclassified.to_csv(fpath_empty, index=False)
        return fpath_empty

    misclassified["error_type"] = np.where(
        (misclassified[label_col] == 0) & (misclassified[pred_col] == 1),
        "false_positive",
        "false_negative",
    )
    if score_col in misclassified.columns:
        misclassified = misclassified.sort_values(by=score_col, ascending=False)
    fpath: Path = out_dir / "misclassified_samples.csv"
    misclassified.to_csv(fpath, index=False)
    logger.info("Сохранил ошибочные сэмплы: %s", fpath)
    return fpath


def build_text_report(
    *,
    timing: TimingMetrics,
    classification: ClassificationMetrics,
    output_csv_fpath: PathLike,
    plots: Sequence[Path],
    misclassified_fpath: Path | None,
) -> str:
    """Собирает человекочитаемый отчёт по метрикам."""
    total_inference_ms: float = timing.total_inference_sec * 1000.0
    mean_sample_ms: float = timing.mean_sample_sec * 1000.0
    p50_sample_ms: float = timing.p50_sample_sec * 1000.0
    p95_sample_ms: float = timing.p95_sample_sec * 1000.0

    lines: list[str] = []
    lines.append("=" * 88)
    lines.append("SBER NLI SCORING REPORT")
    lines.append("=" * 88)
    lines.append(f"output_csv: {Path(output_csv_fpath)}")
    lines.append("-" * 88)
    lines.append("TIMING")
    lines.append(f"  total_samples            : {timing.total_samples}")
    lines.append(f"  total_batches            : {timing.total_batches}")
    lines.append(f"  total_inference_ms       : {total_inference_ms:.3f}")
    lines.append(f"  mean_sample_ms           : {mean_sample_ms:.3f}")
    lines.append(f"  p50_sample_ms            : {p50_sample_ms:.3f}")
    lines.append(f"  p95_sample_ms            : {p95_sample_ms:.3f}")
    lines.append(f"  throughput_samples_per_s : {timing.throughput_samples_per_sec:.2f}")
    lines.append("-" * 88)
    lines.append("QUALITY")
    if classification.has_labels:
        lines.append(f"  accuracy                 : {classification.accuracy:.6f}")
        lines.append(f"  precision                : {classification.precision:.6f}")
        lines.append(f"  recall                   : {classification.recall:.6f}")
        lines.append(f"  f1                       : {classification.f1:.6f}")
        if classification.weighted_average_precision is None:
            lines.append("  weighted_average_precision: n/a")
        else:
            lines.append(f"  weighted_average_precision: {classification.weighted_average_precision:.6f}")
        if classification.average_precision is None:
            lines.append("  average_precision        : n/a")
        else:
            lines.append(f"  average_precision        : {classification.average_precision:.6f}")
    else:
        lines.append("  labels not found: quality metrics are skipped")
    lines.append("-" * 88)
    lines.append("HARD NEGATIVE MINING")
    if misclassified_fpath is None:
        lines.append("  misclassified_samples: labels not found")
    else:
        lines.append(f"  misclassified_samples: {misclassified_fpath}")
    lines.append("-" * 88)
    lines.append("PLOTS")
    if plots:
        for plot in plots:
            lines.append(f"  {plot}")
    else:
        lines.append("  disabled")
    lines.append("=" * 88)
    return "\n".join(lines)


def evaluate_scoring_results(
    dataframe: pd.DataFrame,
    *,
    output_csv_fpath: PathLike,
    report_dpath: PathLike,
    save_plots_enabled: bool,
    time_col: str = "t_sample_sec",
    score_col: str = "hallucination_score",
    label_col: str = "is_hallucination",
    pred_col: str = "pred_is_hallucination",
    total_batches: int,
) -> EvaluationSummary:
    """Вычисляет метрики/графики и возвращает сводный отчёт."""
    logger.info("Считаю сводные метрики скоринга")
    sample_times: list[float] = [float(value) for value in dataframe[time_col].tolist()]
    timing: TimingMetrics = compute_timing_metrics(sample_times=sample_times, total_batches=total_batches)
    classification: ClassificationMetrics = compute_classification_metrics(
        dataframe,
        label_col=label_col,
        pred_col=pred_col,
        score_col=score_col,
    )

    report_dir: Path = Path(report_dpath)
    report_dir.mkdir(parents=True, exist_ok=True)

    plots: list[Path] = []
    if save_plots_enabled:
        plots = save_score_plots(
            dataframe=dataframe,
            output_dpath=report_dir,
            score_col=score_col,
            time_col=time_col,
            label_col=label_col,
            pred_col=pred_col,
        )

    misclassified_fpath: Path | None = save_misclassified_samples(
        dataframe=dataframe,
        output_dpath=report_dir,
        label_col=label_col,
        pred_col=pred_col,
        score_col=score_col,
    )

    report_text: str = build_text_report(
        timing=timing,
        classification=classification,
        output_csv_fpath=output_csv_fpath,
        plots=plots,
        misclassified_fpath=misclassified_fpath,
    )
    report_fpath: Path = report_dir / "score_report.txt"
    report_fpath.write_text(report_text, encoding="utf-8")
    logger.info("Отчет сохранен: %s", report_fpath)

    return EvaluationSummary(
        timing=timing,
        classification=classification,
        report_text=report_text,
        plots=plots,
        misclassified_fpath=misclassified_fpath,
        report_fpath=report_fpath,
    )


__all__ = [
    "ClassificationMetrics",
    "EvaluationSummary",
    "TimingMetrics",
    "build_text_report",
    "compute_classification_metrics",
    "compute_timing_metrics",
    "evaluate_scoring_results",
    "save_misclassified_samples",
    "save_score_plots",
]

