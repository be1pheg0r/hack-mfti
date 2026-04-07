from __future__ import annotations

from pathlib import Path
from typing import *

import pandas as pd

from src.sber.models.tabular_hallucination import (
    FeatureGroupFlags,
    TabularHallucinationPredictor,
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

