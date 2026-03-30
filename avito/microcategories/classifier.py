from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer, OrdinalEncoder, StandardScaler
from xgboost import XGBClassifier

from avito.features import (
    ShouldSplitFeatureConfig,
    TextEncoderLike,
    append_embedding_features,
    extract_should_split_features,
    resolve_keyphrases,
)
from common.files import read_yaml
from common.logger import AVITO_MICROCATS_LOGGER as logger
from common.logger import fit_model_with_progress, log_block_separator


TARGET_COLUMN = "targetDetectedMcIds"
SPLIT_COLUMN = "split"


class MicrocategoryTrainingConfig(BaseModel):
    """Гиперпараметры обучения выделения микрокатегорий."""

    model_config = ConfigDict(frozen=True)

    random_state: int = 42
    merge_train_test_for_fit: bool = True
    threshold_grid: list[float] = Field(default_factory=lambda: [0.5])
    log_reg_c: float = 1.0
    max_iter: int = 1500
    categorical_features: list[str] = Field(default_factory=lambda: ["source_mc_id", "case_type"])

    @field_validator("threshold_grid")
    @classmethod
    def validate_thresholds(cls, values: list[float]) -> list[float]:
        if not values:
            return [0.5]
        normalized: list[float] = []
        for value in values:
            if not 0.0 < value <= 1.0:
                raise ValueError("Пороговые значения должны быть в (0, 1].")
            normalized.append(float(value))
        return normalized

    @field_validator("log_reg_c")
    @classmethod
    def validate_log_reg_c(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("log_reg_c должен быть положительным")
        return float(value)

    @field_validator("max_iter")
    @classmethod
    def validate_max_iter(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("max_iter должен быть положительным")
        return int(value)


class TrainedMicrocategoryModel(BaseModel):
    """Контейнер с обученной моделью выделения микрокатегорий."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    model_name: str
    pipeline: Pipeline
    threshold: float
    mlb_classes: list[int]
    metrics: dict[str, float]
    model_comparison_records: list[dict[str, float | str]] = Field(default_factory=list)


class ModelArchitectureConfig(BaseModel):
    """Конфиг одной кандидатной архитектуры микрокатегорий."""

    enabled: bool = True
    params: dict[str, Any] = Field(default_factory=dict)


@dataclass(frozen=True)
class MicrocategoryModelConfigPaths:
    one_vs_rest_logreg: Path
    one_vs_rest_random_forest: Path
    one_vs_rest_xgboost: Path


def get_microcategory_model_config_paths() -> MicrocategoryModelConfigPaths:
    """Возвращает абсолютные пути к конфигам архитектур microcategories."""

    config_dir = Path(__file__).resolve().parent.parent / "configs" / "microcategories"
    return MicrocategoryModelConfigPaths(
        one_vs_rest_logreg=config_dir / "one_vs_rest_logreg.yaml",
        one_vs_rest_random_forest=config_dir / "one_vs_rest_random_forest.yaml",
        one_vs_rest_xgboost=config_dir / "one_vs_rest_xgboost.yaml",
    )


def _build_architecture_map(paths: MicrocategoryModelConfigPaths) -> dict[str, Path]:
    return {
        "one_vs_rest_logreg": paths.one_vs_rest_logreg,
        "one_vs_rest_random_forest": paths.one_vs_rest_random_forest,
        "one_vs_rest_xgboost": paths.one_vs_rest_xgboost,
    }


def _load_architecture_configs() -> dict[str, ModelArchitectureConfig]:
    paths = get_microcategory_model_config_paths()
    architecture_files = _build_architecture_map(paths)

    configs: dict[str, ModelArchitectureConfig] = {}
    for architecture_name, config_path in architecture_files.items():
        raw_data = read_yaml(config_path)
        if raw_data is None:
            raise ValueError(f"Пустой конфиг архитектуры: {config_path}")
        if not isinstance(raw_data, dict):
            raise ValueError(f"Конфиг архитектуры должен быть словарем: {config_path}")
        configs[architecture_name] = ModelArchitectureConfig.model_validate(raw_data)

    if not any(config.enabled for config in configs.values()):
        raise ValueError("Хотя бы одна архитектура должна быть включена в model-конфигах микрокатегорий")

    return configs


def _parse_mc_list(value: Any) -> list[int]:
    if isinstance(value, (list, tuple, set)):
        return [int(v) for v in value]
    if isinstance(value, str):
        try:
            loaded = json.loads(value)
            if isinstance(loaded, (list, tuple, set)):
                return [int(v) for v in loaded]
        except json.JSONDecodeError:
            pass
    return []


def _prepare_targets(df: pd.DataFrame) -> tuple[np.ndarray, MultiLabelBinarizer]:
    targets = df[TARGET_COLUMN].apply(_parse_mc_list)
    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(targets)
    if y.ndim != 2 or y.shape[1] == 0:
        raise ValueError("Не удалось сформировать многометочные таргеты: пустой результат.")
    return y.astype(np.int8, copy=False), mlb


def _split_frame(
    X: pd.DataFrame,
    y: np.ndarray,
    split: pd.Series,
    *,
    merge_train_test_for_fit: bool,
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray]:
    train_mask = split.eq("train")
    test_mask = split.eq("test")
    val_mask = split.eq("val")

    if not train_mask.any() or not val_mask.any():
        raise ValueError("Ожидаются непустые группы split: train/val (+ optional test)")

    fit_mask = (train_mask | test_mask) if merge_train_test_for_fit else train_mask
    return (
        X.loc[fit_mask],
        y[fit_mask],
        X.loc[val_mask],
        y[val_mask],
    )


def _build_preprocessor(feature_frame: pd.DataFrame, config: MicrocategoryTrainingConfig) -> ColumnTransformer:
    categorical_features = config.categorical_features
    numeric_features = [column for column in feature_frame.columns if column not in categorical_features]

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="constant", fill_value=0.0),
            ),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "ordinal",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )


def _predict_proba(pipeline: Pipeline, X_data: pd.DataFrame) -> np.ndarray:
    if hasattr(pipeline, "predict_proba"):
        proba = pipeline.predict_proba(X_data)
        return np.asarray(proba, dtype=np.float64)
    if hasattr(pipeline, "decision_function"):
        decision = np.asarray(pipeline.decision_function(X_data), dtype=np.float64)
        # Сигмоида для двоичных решений
        return 1.0 / (1.0 + np.exp(-decision))
    raise ValueError("Модель не поддерживает predict_proba или decision_function")


def _evaluate_thresholds(
    y_true: np.ndarray,
    proba: np.ndarray,
    thresholds: Iterable[float],
) -> tuple[float, dict[str, float]]:
    best_threshold: float | None = None
    best_metrics: dict[str, float] = {}
    best_f1: float = -1.0

    for threshold in thresholds:
        y_pred = (proba >= threshold).astype(np.int8)
        micro_f1 = f1_score(y_true, y_pred, average="micro", zero_division=0)
        micro_precision = precision_score(y_true, y_pred, average="micro", zero_division=0)
        micro_recall = recall_score(y_true, y_pred, average="micro", zero_division=0)
        if micro_f1 > best_f1 or (np.isclose(micro_f1, best_f1) and (best_threshold is None or threshold < best_threshold)):
            best_f1 = micro_f1
            best_threshold = threshold
            best_metrics = {
                "micro_f1": float(micro_f1),
                "micro_precision": float(micro_precision),
                "micro_recall": float(micro_recall),
            }

    if best_threshold is None:
        raise RuntimeError("Не удалось подобрать порог для микрокатегорий")

    best_metrics["threshold"] = float(best_threshold)
    return best_threshold, best_metrics


def _make_estimator(
    architecture_name: str,
    *,
    params: dict[str, Any],
    cfg: MicrocategoryTrainingConfig,
) -> OneVsRestClassifier:
    effective_params = dict(params)

    if architecture_name == "one_vs_rest_logreg":
        effective_params.setdefault("C", cfg.log_reg_c)
        effective_params.setdefault("class_weight", "balanced")
        effective_params.setdefault("max_iter", cfg.max_iter)
        effective_params.setdefault("random_state", cfg.random_state)
        effective_params.setdefault("solver", "liblinear")
        base_estimator = LogisticRegression(**effective_params)
        return OneVsRestClassifier(base_estimator)

    if architecture_name == "one_vs_rest_random_forest":
        effective_params.setdefault("n_estimators", 400)
        effective_params.setdefault("max_depth", None)
        effective_params.setdefault("n_jobs", -1)
        effective_params.setdefault("random_state", cfg.random_state)
        base_estimator = RandomForestClassifier(**effective_params)
        return OneVsRestClassifier(base_estimator)

    if architecture_name == "one_vs_rest_xgboost":
        effective_params.setdefault("n_estimators", 400)
        effective_params.setdefault("learning_rate", 0.05)
        effective_params.setdefault("max_depth", 6)
        effective_params.setdefault("subsample", 0.9)
        effective_params.setdefault("colsample_bytree", 0.9)
        effective_params.setdefault("eval_metric", "logloss")
        effective_params.setdefault("n_jobs", -1)
        effective_params.setdefault("random_state", cfg.random_state)
        effective_params.setdefault("tree_method", "hist")
        effective_params.setdefault("objective", "binary:logistic")
        base_estimator = XGBClassifier(**effective_params)
        return OneVsRestClassifier(base_estimator)

    raise ValueError(f"Неизвестная архитектура: {architecture_name}")


def train_microcategory_model(
    df: pd.DataFrame,
    *,
    include_embeddings: bool = True,
    encoder: TextEncoderLike | None = None,
    feature_config: ShouldSplitFeatureConfig | None = None,
    training_config: MicrocategoryTrainingConfig | None = None,
) -> TrainedMicrocategoryModel:
    """Обучает мульти-лейбл модель выделения микрокатегорий.

    Args:
        df: Датасет с колонками description, targetDetectedMcIds и split (train/val/test).
        include_embeddings: Добавлять ли эмбеддинги FRIDA.
        encoder: Энкодер для генерации эмбеддингов при include_embeddings=True.
        feature_config: Конфиг извлечения признаков.
        training_config: Гиперпараметры обучения/порогов.

    Returns:
        TrainedMicrocategoryModel с пайплайном, лучшим порогом и метриками.
    """

    cfg = training_config or MicrocategoryTrainingConfig()
    feat_cfg = feature_config or ShouldSplitFeatureConfig()

    required_columns = {"description", "sourceMcId", "sourceMcTitle", TARGET_COLUMN, SPLIT_COLUMN}
    missing = required_columns.difference(df.columns)
    if missing:
        raise ValueError(f"В DataFrame отсутствуют обязательные колонки: {sorted(missing)}")

    keyphrases = resolve_keyphrases(df, feat_cfg)
    features = extract_should_split_features(df=df, config=feat_cfg, keyphrases=keyphrases)
    if include_embeddings:
        if encoder is None:
            raise ValueError("Для include_embeddings=True нужно передать encoder")
        features = append_embedding_features(
            features=features,
            descriptions=df["description"].tolist(),
            encoder=encoder,
            keyphrases=keyphrases,
            config=feat_cfg,
        )

    y, mlb = _prepare_targets(df)
    split = df[SPLIT_COLUMN].astype(str)

    X_fit, y_fit, X_val, y_val = _split_frame(
        X=features,
        y=y,
        split=split,
        merge_train_test_for_fit=cfg.merge_train_test_for_fit,
    )

    preprocessor = _build_preprocessor(features, cfg)
    architecture_configs = _load_architecture_configs()
    enabled_items = [(name, arch_cfg) for name, arch_cfg in architecture_configs.items() if arch_cfg.enabled]

    if not enabled_items:
        raise ValueError("Хотя бы одна архитектура должна быть включена для обучения микрокатегорий")

    log_block_separator(logger)
    logger.info("Старт сравнения архитектур microcategories")
    logger.info(f"Кандидатов: {len(enabled_items)}")
    logger.info(f"Порогов в сетке: {len(cfg.threshold_grid)}")
    logger.info(f"Признаков в матрице: {features.shape[1]}")
    log_block_separator(logger)

    model_rows: list[dict[str, float | str]] = []
    best_model_name: str | None = None
    best_pipeline: Pipeline | None = None
    best_threshold: float | None = None
    best_metrics: dict[str, float] | None = None
    best_micro_f1 = -1.0

    for architecture_index, (architecture_name, architecture_config) in enumerate(enabled_items, start=1):
        estimator = _make_estimator(
            architecture_name,
            params=architecture_config.params,
            cfg=cfg,
        )
        current_preprocessor = clone(preprocessor)
        pipeline = Pipeline(steps=[("preprocessor", current_preprocessor), ("model", estimator)])

        log_block_separator(logger)
        logger.info(f"[{architecture_index}/{len(enabled_items)}] Архитектура: {architecture_name}")
        logger.info(f"Базовые параметры: {architecture_config.params}")
        fit_model_with_progress(pipeline, X_fit, y_fit, X_val)

        val_proba = _predict_proba(pipeline, X_val)
        candidate_threshold, candidate_metrics = _evaluate_thresholds(
            y_true=y_val,
            proba=val_proba,
            thresholds=cfg.threshold_grid,
        )

        candidate_record: dict[str, float | str] = {"model": architecture_name, **candidate_metrics}
        model_rows.append(candidate_record)

        logger.info("Результаты на валидации:")
        logger.info(f"  micro_f1        = {candidate_metrics['micro_f1']:.6f}")
        logger.info(f"  micro_precision = {candidate_metrics['micro_precision']:.6f}")
        logger.info(f"  micro_recall    = {candidate_metrics['micro_recall']:.6f}")
        logger.info(f"  threshold       = {candidate_threshold:.3f}")

        if candidate_metrics["micro_f1"] > best_micro_f1 or (
            np.isclose(candidate_metrics["micro_f1"], best_micro_f1) and best_model_name is None
        ):
            best_micro_f1 = candidate_metrics["micro_f1"]
            best_model_name = architecture_name
            best_pipeline = pipeline
            best_threshold = candidate_threshold
            best_metrics = dict(candidate_metrics)

    if best_model_name is None or best_pipeline is None or best_threshold is None or best_metrics is None:
        raise RuntimeError("Не удалось выбрать лучшую архитектуру микрокатегорий")

    comparison_records = sorted(
        model_rows,
        key=lambda row: float(row.get("micro_f1", -1)),
        reverse=True,
    )

    logger.info("Рейтинг архитектур по micro_f1:")
    for rank, row in enumerate(comparison_records, start=1):
        logger.info(
            f"  #{rank} {row['model']}: micro_f1={float(row['micro_f1']):.6f}, "
            f"precision={float(row['micro_precision']):.6f}, recall={float(row['micro_recall']):.6f}, "
            f"threshold={float(row['threshold']):.3f}"
        )
    log_block_separator(logger)

    return TrainedMicrocategoryModel(
        model_name=best_model_name,
        pipeline=best_pipeline,
        threshold=best_threshold,
        mlb_classes=[int(cls) for cls in mlb.classes_.tolist()],
        metrics=best_metrics,
        model_comparison_records=comparison_records,
    )
