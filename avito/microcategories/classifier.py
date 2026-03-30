from __future__ import annotations

import json
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer, OrdinalEncoder, StandardScaler

from avito.should_split.features import (
    ShouldSplitFeatureConfig,
    TextEncoderLike,
    append_embedding_features,
    extract_should_split_features,
)
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

    features = extract_should_split_features(df=df, config=feat_cfg)
    if include_embeddings:
        if encoder is None:
            raise ValueError("Для include_embeddings=True нужно передать encoder")
        features = append_embedding_features(
            features=features,
            descriptions=df["description"].tolist(),
            encoder=encoder,
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
    estimator = OneVsRestClassifier(
        LogisticRegression(
            C=cfg.log_reg_c,
            class_weight="balanced",
            max_iter=cfg.max_iter,
            random_state=cfg.random_state,
            solver="liblinear",
        )
    )
    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", estimator)])

    log_block_separator(logger)
    logger.info("Старт обучения microcategories")
    logger.info(f"Порогов в сетке: {len(cfg.threshold_grid)}")
    logger.info(f"Признаков в матрице: {features.shape[1]}")
    log_block_separator(logger)

    fit_model_with_progress(pipeline, X_fit, y_fit, X_val)
    logger.info("Обучение завершено.")

    val_proba = _predict_proba(pipeline, X_val)
    best_threshold, best_metrics = _evaluate_thresholds(
        y_true=y_val,
        proba=val_proba,
        thresholds=cfg.threshold_grid,
    )

    logger.info("Результаты на валидации:")
    logger.info(f"  micro_f1        = {best_metrics['micro_f1']:.6f}")
    logger.info(f"  micro_precision = {best_metrics['micro_precision']:.6f}")
    logger.info(f"  micro_recall    = {best_metrics['micro_recall']:.6f}")
    logger.info(f"  threshold       = {best_threshold:.3f}")
    log_block_separator(logger)

    return TrainedMicrocategoryModel(
        model_name="one_vs_rest_logreg",
        pipeline=pipeline,
        threshold=best_threshold,
        mlb_classes=[int(cls) for cls in mlb.classes_.tolist()],
        metrics=best_metrics,
    )
