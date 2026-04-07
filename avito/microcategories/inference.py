from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

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
from avito.microcategories.mistral_inference import predict_detected_mc_ids_with_mistral
from common.logger import AVITO_MICROCATS_LOGGER as logger
from common.mistral import MistralCallConfig


class MicrocategoryArtifact(BaseModel):
    """Контракт сериализованного артефакта модели микрокатегорий."""

    model_config = ConfigDict(arbitrary_types_allowed=True, protected_namespaces=())

    model_name: str
    pipeline: Pipeline | None = None
    threshold: float
    mlb_classes: list[int]
    with_embeddings: bool = False
    backend_type: Literal["sklearn", "mistral", "transformer"] = "sklearn"
    transformer_model_name: str | None = None
    transformer_model_dir: str | None = None
    feature_config: dict[str, Any] = Field(default_factory=dict)
    training_config: dict[str, Any] = Field(default_factory=dict)
    model_comparison_records: list[dict[str, Any]] = Field(default_factory=list)
    tuned_params: dict[str, Any] = Field(default_factory=dict)


class MicrocategoryInferenceResult(BaseModel):
    """Результат инференса микрокатегорий."""

    detected_mc_ids: list[list[int]]
    probabilities: list[list[float]] | None = None


def _validate_inference_frame(df: pd.DataFrame, *, backend: Literal["sklearn", "mistral", "transformer"]) -> None:
    required_columns = {"description", "sourceMcId", "sourceMcTitle"}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        raise ValueError(
            f"Для backend='{backend}' отсутствуют обязательные колонки: {sorted(missing_columns)}"
        )


def _predict_microcategories_mistral(
    df: pd.DataFrame,
    *,
    artifact: MicrocategoryArtifact,
    mistral_config: MistralCallConfig,
) -> MicrocategoryInferenceResult:
    _validate_inference_frame(df, backend="mistral")

    allowed_ids = set(artifact.mlb_classes)
    detected_rows = predict_detected_mc_ids_with_mistral(
        df,
        allowed_ids=allowed_ids,
        mistral_config=mistral_config,
    )

    return MicrocategoryInferenceResult(detected_mc_ids=detected_rows, probabilities=None)


def load_microcategory_artifact(artifact_path: str | Path) -> MicrocategoryArtifact:
    payload = joblib.load(artifact_path)
    if not isinstance(payload, dict):
        raise ValueError("Ожидается словарь в артефакте микрокатегорий.")

    pipeline = payload.get("pipeline")
    threshold = payload.get("threshold")
    mlb_classes = payload.get("mlb_classes")
    model_name = payload.get("model_name", "unknown")
    with_embeddings = bool(payload.get("with_embeddings", False))
    backend_type = str(payload.get("backend_type", "sklearn"))
    transformer_model_name = payload.get("transformer_model_name")
    transformer_model_dir = payload.get("transformer_model_dir")
    feature_config_payload = payload.get("feature_config") or {}
    training_config_payload = payload.get("training_config") or {}
    comparison_records = payload.get("model_comparison_records") or []
    tuned_params = payload.get("tuned_params") or {}

    if backend_type == "sklearn" and not isinstance(pipeline, Pipeline):
        raise ValueError("Для backend_type='sklearn' поле 'pipeline' должно быть sklearn Pipeline")
    if backend_type == "transformer" and not transformer_model_dir:
        raise ValueError("Для backend_type='transformer' требуется transformer_model_dir")
    if not isinstance(threshold, (int, float)):
        raise ValueError("Поле 'threshold' должно быть числом")
    if not isinstance(mlb_classes, list):
        raise ValueError("Поле 'mlb_classes' должно быть списком")

    resolved_backend: Literal["sklearn", "mistral", "transformer"]
    if backend_type in {"sklearn", "mistral", "transformer"}:
        resolved_backend = backend_type  # type: ignore[assignment]
    else:
        resolved_backend = "sklearn"

    return MicrocategoryArtifact(
        model_name=str(model_name),
        pipeline=pipeline,
        threshold=float(threshold),
        mlb_classes=[int(cls) for cls in mlb_classes],
        with_embeddings=with_embeddings,
        backend_type=resolved_backend,
        transformer_model_name=str(transformer_model_name) if transformer_model_name is not None else None,
        transformer_model_dir=str(transformer_model_dir) if transformer_model_dir is not None else None,
        feature_config=feature_config_payload if isinstance(feature_config_payload, dict) else {},
        training_config=training_config_payload if isinstance(training_config_payload, dict) else {},
        model_comparison_records=comparison_records if isinstance(comparison_records, list) else [],
        tuned_params=tuned_params if isinstance(tuned_params, dict) else {},
    )


def _predict_microcategories_transformer(
    df: pd.DataFrame,
    *,
    artifact: MicrocategoryArtifact,
) -> MicrocategoryInferenceResult:
    _validate_inference_frame(df, backend="transformer")
    if artifact.transformer_model_dir is None:
        raise ValueError("В артефакте отсутствует transformer_model_dir")

    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "Для transformer backend требуются torch и transformers. "
            "Установите зависимости из requirements-avito.txt"
        ) from exc

    model_dir = artifact.transformer_model_dir
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    texts = df["description"].astype(str).tolist()
    batch_size = int(artifact.training_config.get("transformer_eval_batch_size", 16))
    all_rows: list[np.ndarray] = []
    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start : start + batch_size]
        encoded = tokenizer(
            batch_texts,
            truncation=True,
            max_length=int(artifact.training_config.get("transformer_max_length", 256)),
            padding=True,
            return_tensors="pt",
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}
        with torch.no_grad():
            logits = model(**encoded).logits
        probs = torch.sigmoid(logits).detach().cpu().numpy().astype(np.float64)
        all_rows.append(probs)

    if not all_rows:
        return MicrocategoryInferenceResult(detected_mc_ids=[], probabilities=[])

    proba = np.vstack(all_rows)
    y_pred = (proba >= artifact.threshold).astype(np.int8)
    classes = artifact.mlb_classes

    detected: list[list[int]] = []
    for row in y_pred:
        ids = [classes[idx] for idx, flag in enumerate(row.tolist()) if flag]
        detected.append(ids)

    probabilities = [[float(p) for p in row.tolist()] for row in proba]
    return MicrocategoryInferenceResult(detected_mc_ids=detected, probabilities=probabilities)


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
    backend: Literal["sklearn", "mistral", "transformer"] = "sklearn",
    mistral_config: MistralCallConfig | None = None,
) -> MicrocategoryInferenceResult:
    """Предсказывает список микрокатегорий для объявлений."""
    if backend == "mistral":
        if mistral_config is None:
            raise ValueError("Для backend='mistral' нужно передать mistral_config")
        return _predict_microcategories_mistral(df, artifact=artifact, mistral_config=mistral_config)

    if backend == "transformer":
        return _predict_microcategories_transformer(df, artifact=artifact)

    _validate_inference_frame(df, backend="sklearn")
    if artifact.pipeline is None:
        raise ValueError("Для backend='sklearn' в артефакте должен быть pipeline")

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

    pipeline = artifact.pipeline
    if hasattr(pipeline, "predict_proba"):
        proba = np.asarray(pipeline.predict_proba(features), dtype=np.float64)
    elif hasattr(pipeline, "decision_function"):
        decision = np.asarray(pipeline.decision_function(features), dtype=np.float64)
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
    backend: Literal["sklearn", "mistral", "transformer"] = "sklearn",
    mistral_config: MistralCallConfig | None = None,
) -> MicrocategoryInferenceResult:
    artifact = load_microcategory_artifact(artifact_path)
    return predict_microcategories(
        df=df,
        artifact=artifact,
        encoder=encoder,
        feature_config=feature_config,
        backend=backend,
        mistral_config=mistral_config,
    )
