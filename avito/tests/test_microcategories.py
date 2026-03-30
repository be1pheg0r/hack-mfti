from __future__ import annotations

import pandas as pd

from avito.microcategories.classifier import MicrocategoryTrainingConfig, train_microcategory_model
from avito.microcategories.inference import MicrocategoryArtifact, predict_microcategories
from avito.should_split.features import ShouldSplitFeatureConfig


def _make_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "description": [
                "Делаю сантехнику и разводку труб",
                "Электрика отдельно, щитки, разводка",
                "Делаю сантехника и электрика вместе",
                "Просто ремонт под ключ",
            ],
            "sourceMcId": [101, 102, 101, 101],
            "sourceMcTitle": [
                "Сантехника",
                "Электрика",
                "Сантехника",
                "Ремонт под ключ",
            ],
            "caseType": ["multi_service_split", "multi_service_split", "turnkey_split", "turnkey_no_split"],
            "targetDetectedMcIds": [
                [101],
                [102],
                [101, 102],
                [],
            ],
            "split": ["train", "train", "val", "test"],
        }
    )


def test_microcategories_training_and_inference_runs() -> None:
    df = _make_df()
    training_config = MicrocategoryTrainingConfig(
        threshold_grid=[0.5],
        merge_train_test_for_fit=False,
        log_reg_c=1.0,
        max_iter=200,
    )
    feature_config = ShouldSplitFeatureConfig(include_extra_text_features=False)

    result = train_microcategory_model(
        df=df,
        include_embeddings=False,
        encoder=None,
        feature_config=feature_config,
        training_config=training_config,
    )

    assert result.metrics["micro_f1"] >= 0.0
    assert result.threshold > 0

    artifact = MicrocategoryArtifact(
        model_name=result.model_name,
        pipeline=result.pipeline,
        threshold=result.threshold,
        mlb_classes=result.mlb_classes,
        with_embeddings=False,
        feature_config=feature_config.model_dump(mode="json"),
        training_config=training_config.model_dump(mode="json"),
    )

    inference_df = df.iloc[[0]].copy()
    output = predict_microcategories(
        inference_df,
        artifact=artifact,
        encoder=None,
        feature_config=feature_config,
    )

    assert len(output.detected_mc_ids) == 1
    assert isinstance(output.detected_mc_ids[0], list)
