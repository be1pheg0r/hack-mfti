from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field
from sklearn.pipeline import Pipeline

from avito.features import (
    ShouldSplitFeatureConfig,
    TextEncoderLike,
    append_embedding_features,
    extract_should_split_features,
    resolve_keyphrases,
)
from common.logger import AVITO_MICROCATS_LOGGER as logger


class MicrocategoryArtifact(BaseModel):
    """Контракт сериализованного артефакта модели микрокатегорий."""

    model_config = ConfigDict(arbitrary_types_allowed=True, protected_namespaces=())

    model_name: str
    pipeline: Pipeline
    threshold: float
    mlb_classes: list[int]
    with_embeddings: bool = False
    feature_config: dict[str, Any] = Field(default_factory=dict)
    training_config: dict[str, Any] = Field(default_factory=dict)
    model_comparison_records: list[dict[str, Any]] = Field(default_factory=list)
    tuned_params: dict[str, Any] = Field(default_factory=dict)


class MicrocategoryInferenceResult(BaseModel):
    """Результат инференса микрокатегорий."""

    detected_mc_ids: list[list[int]]
    probabilities: list[list[float]] | None = None


def load_microcategory_artifact(artifact_path: str | Path) -> MicrocategoryArtifact:
    payload = joblib.load(artifact_path)
    if not isinstance(payload, dict):
        raise ValueError("Ожидается словарь в артефакте микрокатегорий.")

    pipeline = payload.get("pipeline")
    threshold = payload.get("threshold")
    mlb_classes = payload.get("mlb_classes")
    model_name = payload.get("model_name", "unknown")
    with_embeddings = bool(payload.get("with_embeddings", False))
    feature_config_payload = payload.get("feature_config") or {}
    training_config_payload = payload.get("training_config") or {}
    comparison_records = payload.get("model_comparison_records") or []
    tuned_params = payload.get("tuned_params") or {}

    if not isinstance(pipeline, Pipeline):
        raise ValueError("Поле 'pipeline' должно быть sklearn Pipeline")
    if not isinstance(threshold, (int, float)):
        raise ValueError("Поле 'threshold' должно быть числом")
    if not isinstance(mlb_classes, list):
        raise ValueError("Поле 'mlb_classes' должно быть списком")

    return MicrocategoryArtifact(
        model_name=str(model_name),
        pipeline=pipeline,
        threshold=float(threshold),
        mlb_classes=[int(cls) for cls in mlb_classes],
        with_embeddings=with_embeddings,
        feature_config=feature_config_payload if isinstance(feature_config_payload, dict) else {},
        training_config=training_config_payload if isinstance(training_config_payload, dict) else {},
        model_comparison_records=comparison_records if isinstance(comparison_records, list) else [],
        tuned_params=tuned_params if isinstance(tuned_params, dict) else {},
    )


def _resolve_feature_config(
    artifact: MicrocategoryArtifact,
    feature_config: ShouldSplitFeatureConfig | None,
) -> ShouldSplitFeatureConfig:
    if feature_config is not None:
        return feature_config
    if artifact.feature_config:
        return ShouldSplitFeatureConfig.model_validate(artifact.feature_config)
    return ShouldSplitFeatureConfig()


def predict_microcategories(
    df: pd.DataFrame,
    *,
    artifact: MicrocategoryArtifact,
    encoder: TextEncoderLike | None = None,
    feature_config: ShouldSplitFeatureConfig | None = None,
) -> MicrocategoryInferenceResult:
    """Предсказывает список микрокатегорий для объявлений."""
    resolved_feature_config = _resolve_feature_config(artifact=artifact, feature_config=feature_config)
    keyphrases = resolve_keyphrases(df, resolved_feature_config)
    features = extract_should_split_features(df=df, config=resolved_feature_config, keyphrases=keyphrases)

    if artifact.with_embeddings:
        if encoder is None:
            raise ValueError("Артефакт требует эмбеддинги, но encoder не передан")
        features = append_embedding_features(
            features=features,
            descriptions=df["description"].tolist(),
            encoder=encoder,
            keyphrases=keyphrases,
            config=resolved_feature_config,
        )

    if hasattr(artifact.pipeline, "predict_proba"):
        proba = np.asarray(artifact.pipeline.predict_proba(features), dtype=np.float64)
    elif hasattr(artifact.pipeline, "decision_function"):
        decision = np.asarray(artifact.pipeline.decision_function(features), dtype=np.float64)
        proba = 1.0 / (1.0 + np.exp(-decision))
    else:
        raise ValueError("Модель не поддерживает predict_proba/decision_function")

    y_pred = (proba >= artifact.threshold).astype(np.int8)
    classes = artifact.mlb_classes

    detected: list[list[int]] = []
    for row in y_pred:
        ids = [classes[idx] for idx, flag in enumerate(row.tolist()) if flag]
        detected.append(ids)

    probabilities: list[list[float]] | None = None
    if proba is not None:
        probabilities = [[float(p) for p in row.tolist()] for row in proba]

    return MicrocategoryInferenceResult(detected_mc_ids=detected, probabilities=probabilities)


def predict_microcategories_from_artifact(
    df: pd.DataFrame,
    *,
    artifact_path: str | Path,
    encoder: TextEncoderLike | None = None,
    feature_config: ShouldSplitFeatureConfig | None = None,
) -> MicrocategoryInferenceResult:
    artifact = load_microcategory_artifact(artifact_path)
    return predict_microcategories(df=df, artifact=artifact, encoder=encoder, feature_config=feature_config)
