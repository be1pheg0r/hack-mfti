from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from pydantic import BaseModel, ConfigDict, Field
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from avito.config import ShouldSplitTrainingConfig
from avito.features import ShouldSplitFeatureConfig, TextEncoderLike, build_training_matrix


class TrainedShouldSplitModel(BaseModel):
    """Training result for shouldSplit model selection.

    Attributes:
        model_name: Name of the selected best model.
        pipeline: Fitted sklearn pipeline.
        val_metrics: Validation metrics for selected model.
        test_metrics: Test metrics for selected model.
        model_comparison_records: Validation metrics for all candidate models.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    model_name: str
    pipeline: Pipeline
    val_metrics: dict[str, float]
    test_metrics: dict[str, float]
    model_comparison_records: list[dict[str, float | str]] = Field(default_factory=list)


def _build_preprocessor(feature_frame: pd.DataFrame, config: ShouldSplitTrainingConfig) -> ColumnTransformer:
    categorical_features = config.categorical_features
    numeric_features = [column for column in feature_frame.columns if column not in categorical_features]

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="constant", fill_value=config.numeric_imputer_fill_value),
            ),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
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


def _candidate_models(config: ShouldSplitTrainingConfig) -> dict[str, Any]:
    candidates: dict[str, Any] = {}

    if config.logistic_regression.enabled:
        candidates["logistic_regression"] = LogisticRegression(
            random_state=config.random_state,
            max_iter=config.logistic_regression.max_iter,
            class_weight=config.logistic_regression.class_weight,
        )

    if config.random_forest.enabled:
        candidates["random_forest"] = RandomForestClassifier(
            random_state=config.random_state,
            n_estimators=config.random_forest.n_estimators,
            class_weight=config.random_forest.class_weight,
            n_jobs=-1,
        )

    if config.hist_gradient_boosting.enabled:
        candidates["hist_gradient_boosting"] = HistGradientBoostingClassifier(
            random_state=config.random_state,
            max_depth=config.hist_gradient_boosting.max_depth,
            max_iter=config.hist_gradient_boosting.max_iter,
            learning_rate=config.hist_gradient_boosting.learning_rate,
        )

    if config.catboost.enabled:
        candidates["catboost"] = CatBoostClassifier(
            random_seed=config.random_state,
            iterations=config.catboost.iterations,
            learning_rate=config.catboost.learning_rate,
            depth=config.catboost.depth,
            auto_class_weights=config.catboost.auto_class_weights,
            verbose=False,
            loss_function="Logloss",
        )

    return candidates


def _compute_metrics(y_true: pd.Series, y_pred: np.ndarray, y_prob: np.ndarray | None) -> dict[str, float]:
    y_pred_array = np.asarray(y_pred)

    metrics: dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred_array)),
        "precision_pos": float(precision_score(y_true, y_pred_array, pos_label=True, zero_division=0)),
        "recall_pos": float(recall_score(y_true, y_pred_array, pos_label=True, zero_division=0)),
        "f1_pos": float(f1_score(y_true, y_pred_array, pos_label=True, zero_division=0)),
    }
    if y_prob is not None:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        except ValueError:
            metrics["roc_auc"] = float("nan")
    else:
        metrics["roc_auc"] = float("nan")
    return metrics


def _split_frame(
    X: pd.DataFrame,
    y: pd.Series,
    split: pd.Series,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    train_mask = split.eq("train")
    val_mask = split.eq("val")
    test_mask = split.eq("test")

    if not train_mask.any() or not val_mask.any() or not test_mask.any():
        raise ValueError("Ожидаются непустые split-группы train/val/test.")

    return (
        X.loc[train_mask],
        y.loc[train_mask],
        X.loc[val_mask],
        y.loc[val_mask],
        X.loc[test_mask],
        y.loc[test_mask],
    )


def train_should_split_models(
    df: pd.DataFrame,
    *,
    include_embeddings: bool = False,
    encoder: TextEncoderLike | None = None,
    feature_config: ShouldSplitFeatureConfig | None = None,
    training_config: ShouldSplitTrainingConfig | None = None,
) -> TrainedShouldSplitModel:
    """Train and compare shouldSplit candidate models.

    Args:
        df: Input dataset with split labels and target.
        include_embeddings: Whether to append embedding features.
        encoder: Encoder for embedding extraction.
        feature_config: Feature extraction configuration.
        training_config: Model and preprocessing configuration.

    Returns:
        Pydantic training result with best model and metrics.
    """
    effective_training_config = training_config or ShouldSplitTrainingConfig()

    X, y, split = build_training_matrix(
        df=df,
        include_embeddings=include_embeddings,
        encoder=encoder,
        config=feature_config,
    )
    X_train, y_train, X_val, y_val, X_test, y_test = _split_frame(X=X, y=y, split=split)

    model_rows: list[dict[str, float | str]] = []
    best_name: str | None = None
    best_pipeline: Pipeline | None = None
    best_val_metrics: dict[str, float] | None = None
    best_val_accuracy = -1.0

    preprocessor = _build_preprocessor(X, config=effective_training_config)
    for model_name, estimator in _candidate_models(config=effective_training_config).items():
        pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", estimator)])
        pipeline.fit(X_train, y_train)

        val_pred = np.asarray(pipeline.predict(X_val))
        val_prob = (
            np.asarray(pipeline.predict_proba(X_val)[:, 1])
            if hasattr(pipeline, "predict_proba")
            else None
        )
        val_metrics = _compute_metrics(y_true=y_val, y_pred=val_pred, y_prob=val_prob)

        model_rows.append({"model": model_name, **val_metrics})
        if val_metrics["accuracy"] > best_val_accuracy:
            best_val_accuracy = val_metrics["accuracy"]
            best_name = model_name
            best_pipeline = pipeline
            best_val_metrics = val_metrics

    if best_pipeline is None or best_name is None or best_val_metrics is None:
        raise RuntimeError("Не удалось выбрать лучшую модель shouldSplit.")

    test_pred = np.asarray(best_pipeline.predict(X_test))
    test_prob = (
        np.asarray(best_pipeline.predict_proba(X_test)[:, 1])
        if hasattr(best_pipeline, "predict_proba")
        else None
    )
    test_metrics = _compute_metrics(y_true=y_test, y_pred=test_pred, y_prob=test_prob)

    comparison_records_raw = (
        pd.DataFrame(model_rows)
        .sort_values(by=effective_training_config.primary_metric, ascending=False)
        .reset_index(drop=True)
        .to_dict(orient="records")
    )
    comparison_records: list[dict[str, float | str]] = [
        {str(key): value for key, value in row.items()}
        for row in comparison_records_raw
    ]

    return TrainedShouldSplitModel(
        model_name=best_name,
        pipeline=best_pipeline,
        val_metrics=best_val_metrics,
        test_metrics=test_metrics,
        model_comparison_records=comparison_records,
    )