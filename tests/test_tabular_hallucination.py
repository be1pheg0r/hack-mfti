from __future__ import annotations

from pathlib import Path
from typing import *

import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.sber.models.tabular_hallucination import (
    AutoFeatureSelectionConfig,
    FeatureGroupFlags,
    TabularHallucinationPredictor,
    TabularPreprocessor,
    TabularHallucinationTrainer,
    TabularTrainConfig,
)


def _build_dataset(size: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for index in range(size):
        rows.append(
            {
                "sample_id": index,
                "query": f"query {index}",
                "model_answer": "hallucination text" if index % 2 == 0 else "correct answer",
                "is_hallucination": int(index % 2 == 0),
                "mean_log_prob": float(index) / max(size, 1),
            }
        )
    return pd.DataFrame(rows)


def test_feature_flags_allow_tfidf_without_key_error() -> None:
    flags = FeatureGroupFlags(tfidf=True, text_features=False, uncertainty=False)
    assert flags.tfidf is True


def test_train_and_infer_roundtrip(tmp_path: Path) -> None:
    train_df = _build_dataset(20)
    val_df = _build_dataset(10)

    train_csv = tmp_path / "train.csv"
    val_csv = tmp_path / "val.csv"
    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)

    config = TabularTrainConfig(
        train_csv=train_csv,
        val_csv=val_csv,
        output_root=tmp_path / "checkpoints",
        run_name="test_run",
        feature_flags=FeatureGroupFlags(
            uncertainty=True,
            internal_scalars=False,
            probe_vec=False,
            attention_entropy=False,
            entropy_drops=False,
            moe_routing=False,
            text_features=True,
            tfidf=True,
        ),
        pca_n_components=None,
        tfidf_n_components=4,
        under_sampling=False,
        oversampling=False,
        plot_feature_distributions=False,
        iterations=20,
        early_stopping_rounds=5,
    )

    train_result = TabularHallucinationTrainer(config=config).train()
    checkpoint_dir = Path(train_result.checkpoint_dir)
    latest_dir = (tmp_path / "checkpoints") / "latest"

    predictor = TabularHallucinationPredictor(checkpoint_dir=checkpoint_dir)
    scored = predictor.predict_dataframe(val_df)

    assert "hallucination_score" in scored.columns
    assert "pred_is_hallucination" in scored.columns
    assert len(scored) == len(val_df)
    assert latest_dir.exists()
    assert train_result.best_architecture in {"catboost", "xgboost", "lightgbm", "logreg"}
    assert "average_precision" in train_result.validation_metrics


def test_preprocessor_accepts_prompt_and_feature_alias_columns() -> None:
    train_df = pd.DataFrame(
        {
            "prompt": ["q1", "q2", "q3", "q4"],
            "model_answer": ["a1", "a2", "a3", "a4"],
            "is_hallucination": [1, 0, 1, 0],
            "feature_uncertainty_token_logprob_mean": [0.1, 0.2, 0.3, 0.4],
        }
    )
    val_df = train_df.copy()

    config = TabularTrainConfig(
        train_csv="unused.csv",
        val_csv="unused.csv",
        feature_flags=FeatureGroupFlags(
            uncertainty=True,
            internal_scalars=False,
            probe_vec=False,
            attention_entropy=False,
            entropy_drops=False,
            moe_routing=False,
            text_features=False,
            tfidf=False,
        ),
        pca_n_components=None,
        tfidf_n_components=None,
        under_sampling=False,
        oversampling=False,
        plot_feature_distributions=False,
    )

    preprocessor = TabularPreprocessor(config=config)
    processed_train, processed_val, selected = preprocessor.fit_transform(train_df, val_df)

    assert "query" in processed_train.columns
    assert "mean_log_prob" in processed_train.columns
    assert "mean_log_prob" in processed_val.columns
    assert selected == ["mean_log_prob"]


def test_auto_feature_selection_drops_correlated_weak_features() -> None:
    train_df = pd.DataFrame(
        {
            "query": [f"q{i}" for i in range(40)],
            "model_answer": [f"a{i}" for i in range(40)],
            "is_hallucination": [i % 2 for i in range(40)],
            "mean_log_prob": [float(i % 10) for i in range(40)],
            "min_log_prob": [float(i % 10) + 1e-6 for i in range(40)],
            "max_log_prob": [float(i % 2) for i in range(40)],
        }
    )
    val_df = train_df.copy()

    config = TabularTrainConfig(
        train_csv="unused.csv",
        val_csv="unused.csv",
        feature_flags=FeatureGroupFlags(
            uncertainty=True,
            internal_scalars=False,
            probe_vec=False,
            attention_entropy=False,
            entropy_drops=False,
            moe_routing=False,
            text_features=False,
            tfidf=False,
        ),
        auto_feature_selection=AutoFeatureSelectionConfig(
            enabled=True,
            correlation_filter_enabled=True,
            feature_correlation_threshold=0.9,
            target_correlation_threshold=0.2,
            model_selection_enabled=False,
        ),
        pca_n_components=None,
        tfidf_n_components=None,
        under_sampling=False,
        oversampling=False,
        plot_feature_distributions=False,
    )

    preprocessor = TabularPreprocessor(config=config)
    _, _, selected = preprocessor.fit_transform(train_df, val_df)

    assert not ({"mean_log_prob", "min_log_prob"} <= set(selected))
    assert "max_log_prob" in selected


def test_auto_feature_selection_select_from_model_keeps_non_empty_features() -> None:
    train_df = _build_dataset(30)
    val_df = _build_dataset(10)

    config = TabularTrainConfig(
        train_csv="unused.csv",
        val_csv="unused.csv",
        feature_flags=FeatureGroupFlags(
            uncertainty=True,
            internal_scalars=False,
            probe_vec=False,
            attention_entropy=False,
            entropy_drops=False,
            moe_routing=False,
            text_features=False,
            tfidf=False,
        ),
        auto_feature_selection=AutoFeatureSelectionConfig(
            enabled=True,
            correlation_filter_enabled=False,
            model_selection_enabled=True,
            model_selection_method="select_from_model",
            selection_threshold="median",
        ),
        pca_n_components=None,
        tfidf_n_components=None,
        under_sampling=False,
        oversampling=False,
        plot_feature_distributions=False,
    )

    preprocessor = TabularPreprocessor(config=config)
    _, _, selected = preprocessor.fit_transform(train_df, val_df)

    assert selected
    assert set(selected).issubset({"mean_log_prob"})


def test_preprocessor_scaling_with_auto_feature_selection_has_consistent_scaler_features() -> None:
    train_df = pd.DataFrame(
        {
            "query": [f"q{i}" for i in range(30)],
            "model_answer": [f"a{i}" for i in range(30)],
            "is_hallucination": [i % 2 for i in range(30)],
            "mean_log_prob": [float(i) for i in range(30)],
            "min_log_prob": [float(i) + 1e-6 for i in range(30)],
        }
    )
    val_df = train_df.copy()

    config = TabularTrainConfig(
        train_csv="unused.csv",
        val_csv="unused.csv",
        feature_flags=FeatureGroupFlags(
            uncertainty=True,
            internal_scalars=False,
            probe_vec=False,
            attention_entropy=False,
            entropy_drops=False,
            moe_routing=False,
            text_features=False,
            tfidf=False,
        ),
        scaling=True,
        auto_feature_selection=AutoFeatureSelectionConfig(
            enabled=True,
            correlation_filter_enabled=True,
            feature_correlation_threshold=0.9,
            target_correlation_threshold=0.2,
            model_selection_enabled=False,
        ),
        pca_n_components=None,
        tfidf_n_components=None,
        under_sampling=False,
        oversampling=False,
        plot_feature_distributions=False,
    )

    preprocessor = TabularPreprocessor(config=config)
    preprocessor.fit_transform(train_df, val_df)
    transformed = preprocessor.transform(val_df)

    assert all(name in transformed.columns for name in preprocessor.selected_feature_names)


def test_preprocessor_transform_is_backward_compatible_when_scaler_has_extra_fit_features() -> None:
    train_df = pd.DataFrame(
        {
            "query": ["q1", "q2", "q3", "q4"],
            "model_answer": ["a1", "a2", "a3", "a4"],
            "is_hallucination": [1, 0, 1, 0],
            "mean_log_prob": [0.1, 0.2, 0.3, 0.4],
            "max_log_prob": [0.0, 1.0, 0.0, 1.0],
        }
    )

    config = TabularTrainConfig(
        train_csv="unused.csv",
        val_csv="unused.csv",
        feature_flags=FeatureGroupFlags(
            uncertainty=True,
            internal_scalars=False,
            probe_vec=False,
            attention_entropy=False,
            entropy_drops=False,
            moe_routing=False,
            text_features=False,
            tfidf=False,
        ),
        scaling=False,
        pca_n_components=None,
        tfidf_n_components=None,
        under_sampling=False,
        oversampling=False,
        plot_feature_distributions=False,
    )

    preprocessor = TabularPreprocessor(config=config)
    preprocessor.selected_feature_names = ["mean_log_prob"]
    preprocessor.scaler = StandardScaler().fit(train_df[["mean_log_prob", "max_log_prob"]])

    transformed = preprocessor.transform(train_df)

    assert "mean_log_prob" in transformed.columns


