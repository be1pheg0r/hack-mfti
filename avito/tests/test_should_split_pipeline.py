from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import pytest

from avito.should_split.classifier import train_should_split_models
from avito.should_split.features import append_embedding_features, build_training_matrix, extract_should_split_features
from avito.should_split.inference import (
    ShouldSplitArtifact,
    load_should_split_artifact,
    predict_should_split,
    predict_should_split_from_artifact,
)


class _FakeEncoder:
    def encode(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(text)), float(idx)] for idx, text in enumerate(texts)]


def _sample_df() -> pd.DataFrame:
    rows = [
        {
            "description": "Ремонт под ключ, включая материалы",
            "sourceMcId": 1,
            "sourceMcTitle": "Ремонт квартир и домов под ключ",
            "shouldSplit": False,
            "split": "train",
        },
        {
            "description": "Электрика отдельно, также выполняем сантехнику",
            "sourceMcId": 2,
            "sourceMcTitle": "Электрика",
            "shouldSplit": True,
            "split": "train",
        },
        {
            "description": "- плитка\n- покраска\nпомимо штукатурки",
            "sourceMcId": 3,
            "sourceMcTitle": "Отделочные работы",
            "shouldSplit": True,
            "split": "train",
        },
        {
            "description": "Мелкий ремонт без доп. услуг",
            "sourceMcId": 2,
            "sourceMcTitle": "Электрика",
            "shouldSplit": False,
            "split": "train",
        },
        {
            "description": "Отдельно делаем демонтаж",
            "sourceMcId": 3,
            "sourceMcTitle": "Отделочные работы",
            "shouldSplit": True,
            "split": "val",
        },
        {
            "description": "Ремонт под ключ, в том числе доставка",
            "sourceMcId": 1,
            "sourceMcTitle": "Ремонт квартир и домов под ключ",
            "shouldSplit": False,
            "split": "val",
        },
        {
            "description": "Также выполняем электрику и сантехнику",
            "sourceMcId": 4,
            "sourceMcTitle": "Сантехника",
            "shouldSplit": True,
            "split": "test",
        },
        {
            "description": "Только сборка мебели",
            "sourceMcId": 5,
            "sourceMcTitle": "Сборка мебели",
            "shouldSplit": False,
            "split": "test",
        },
    ]
    return pd.DataFrame(rows)


def test_extract_should_split_features_counts_markers_and_bullets() -> None:
    df = _sample_df().iloc[[1, 2]].copy()

    features = extract_should_split_features(df)

    assert int(features.iloc[0]["split_marker_count"]) >= 1
    assert int(features.iloc[0]["complex_marker_count"]) == 0
    assert int(features.iloc[1]["has_bullets"]) == 1
    assert float(features.iloc[0]["marker_ratio"]) > 0


def test_append_embedding_features_adds_embedding_columns() -> None:
    df = _sample_df().iloc[:3].copy()
    base = extract_should_split_features(df)

    extended = append_embedding_features(base, descriptions=df["description"].tolist(), encoder=_FakeEncoder())

    assert "embedding_000" in extended.columns
    assert "embedding_001" in extended.columns
    assert len(extended) == len(base)


def test_build_training_matrix_with_embeddings() -> None:
    df = _sample_df().copy()

    X, y, split = build_training_matrix(df, include_embeddings=True, encoder=_FakeEncoder())

    assert len(X) == len(df)
    assert y.dtype == bool
    assert set(split.unique()) == {"train", "val", "test"}
    assert "source_mc_id" in X.columns
    assert "embedding_000" in X.columns


def test_train_should_split_models_smoke() -> None:
    df = _sample_df().copy()

    result = train_should_split_models(df=df, include_embeddings=False)

    assert result.model_name in {
        "logistic_regression",
        "random_forest",
        "hist_gradient_boosting",
        "catboost",
        "xgboost",
        "lightgbm",
    }
    assert 0.0 <= result.gt_should_split_ratio <= 1.0
    assert 0.0 <= result.model_should_split_ratio <= 1.0
    assert abs(result.ratio_delta) == pytest.approx(result.ratio_abs_delta)
    assert len(result.model_comparison_records) > 0


def test_predict_should_split_from_artifact_smoke(tmp_path: Path) -> None:
    df = _sample_df().copy()
    train_result = train_should_split_models(df=df, include_embeddings=False)

    artifact_path = tmp_path / "should_split_artifact.joblib"
    joblib.dump(
        {
            "best_model_name": train_result.model_name,
            "pipeline": train_result.pipeline,
            "with_embeddings": False,
            "feature_config": {"include_extra_text_features": True},
        },
        artifact_path,
    )

    loaded_artifact = load_should_split_artifact(artifact_path)
    assert loaded_artifact.best_model_name == train_result.model_name

    inference_result = predict_should_split_from_artifact(df=df.iloc[:3], artifact_path=artifact_path)

    assert len(inference_result.predictions) == 3
    assert inference_result.probabilities is None or len(inference_result.probabilities) == 3


def test_predict_should_split_requires_encoder_for_embedding_artifact() -> None:
    df = _sample_df().copy()
    train_result = train_should_split_models(df=df, include_embeddings=False)

    artifact = ShouldSplitArtifact(
        best_model_name=train_result.model_name,
        pipeline=train_result.pipeline,
        with_embeddings=True,
        feature_config={"include_extra_text_features": True},
    )

    with pytest.raises(ValueError):
        predict_should_split(df=df.iloc[:2], artifact=artifact)