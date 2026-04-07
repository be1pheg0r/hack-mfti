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


class ShouldSplitArtifact(BaseModel):
    """Контракт сериализованного артефакта модели shouldSplit.

    Attributes:
        best_model_name: Имя выбранной модели.
        pipeline: Обученный sklearn-пайплайн.
        with_embeddings: Использовались ли эмбеддинги при обучении.
        feature_config: Конфигурация извлечения признаков, примененная при обучении.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    best_model_name: str
    pipeline: Pipeline
    with_embeddings: bool = False
    feature_config: dict[str, Any] = Field(default_factory=dict)


class ShouldSplitInferenceResult(BaseModel):
    """Результат инференса для предсказаний shouldSplit.

    Attributes:
        predictions: Бинарные предсказания shouldSplit.
        probabilities: Опциональные вероятности положительного класса.
    """

    predictions: list[bool]
    probabilities: list[float] | None = None


def load_should_split_artifact(artifact_path: str | Path) -> ShouldSplitArtifact:
    """Загружает сериализованный артефакт shouldSplit с диска."""
    payload = joblib.load(artifact_path)
    if not isinstance(payload, dict):
        raise ValueError("Expected dict payload in shouldSplit artifact.")

    model_name = payload.get("best_model_name")
    pipeline = payload.get("pipeline")
    if not isinstance(model_name, str):
        raise ValueError("Artifact field 'best_model_name' must be a string.")
    if not isinstance(pipeline, Pipeline):
        raise ValueError("Artifact field 'pipeline' must be sklearn Pipeline.")

    with_embeddings = bool(payload.get("with_embeddings", False))
    feature_config_payload = payload.get("feature_config") or {}
    if not isinstance(feature_config_payload, dict):
        raise ValueError("Artifact field 'feature_config' must be a mapping.")

    return ShouldSplitArtifact(
        best_model_name=model_name,
        pipeline=pipeline,
        with_embeddings=with_embeddings,
        feature_config=feature_config_payload,
    )


def _resolve_feature_config(
    artifact: ShouldSplitArtifact,
    feature_config: ShouldSplitFeatureConfig | None,
) -> ShouldSplitFeatureConfig:
    if feature_config is not None:
        return feature_config

    if artifact.feature_config:
        return ShouldSplitFeatureConfig.model_validate(artifact.feature_config)

    return ShouldSplitFeatureConfig()


def predict_should_split(
    df: pd.DataFrame,
    *,
    artifact: ShouldSplitArtifact,
    encoder: TextEncoderLike | None = None,
    feature_config: ShouldSplitFeatureConfig | None = None,
) -> ShouldSplitInferenceResult:
    """Делает предсказания shouldSplit для новых строк.

    Args:
        df: Входной DataFrame с полями description/source.
        artifact: Загруженный артефакт модели shouldSplit.
        encoder: Энкодер для эмбеддинговых признаков при необходимости.
        feature_config: Явное переопределение конфигурации признаков.

    Returns:
        Предсказания и, при наличии, вероятности.
    """
    resolved_feature_config = _resolve_feature_config(artifact=artifact, feature_config=feature_config)
    keyphrases = resolve_keyphrases(df, resolved_feature_config)
    features = extract_should_split_features(df=df, config=resolved_feature_config, keyphrases=keyphrases)

    if artifact.with_embeddings:
        if encoder is None:
            raise ValueError("Artifact requires embeddings, but encoder is not provided.")
        features = append_embedding_features(
            features=features,
            descriptions=df["description"].tolist(),
            encoder=encoder,
            keyphrases=keyphrases,
            config=resolved_feature_config,
        )

    raw_pred = np.asarray(artifact.pipeline.predict(features))
    predictions = [bool(value) for value in raw_pred.tolist()]

    probabilities: list[float] | None = None
    if hasattr(artifact.pipeline, "predict_proba"):
        raw_prob = np.asarray(artifact.pipeline.predict_proba(features)[:, 1], dtype=np.float64)
        probabilities = [float(value) for value in raw_prob.tolist()]

    return ShouldSplitInferenceResult(predictions=predictions, probabilities=probabilities)


def predict_should_split_from_artifact(
    df: pd.DataFrame,
    *,
    artifact_path: str | Path,
    encoder: TextEncoderLike | None = None,
    feature_config: ShouldSplitFeatureConfig | None = None,
) -> ShouldSplitInferenceResult:
    """Упрощенная обертка для предсказаний по пути к сохраненному артефакту."""
    artifact = load_should_split_artifact(artifact_path)
    return predict_should_split(
        df=df,
        artifact=artifact,
        encoder=encoder,
        feature_config=feature_config,
    )
