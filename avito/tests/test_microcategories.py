from __future__ import annotations

import pandas as pd
import pytest

from avito.features import ShouldSplitFeatureConfig
from avito.microcategories.classifier import MicrocategoryTrainingConfig, train_microcategory_model
from avito.microcategories.inference import MicrocategoryArtifact, predict_microcategories
from avito.microcategories.mistral_inference import build_mc_id_title_mapping, build_mistral_messages
from common.mistral import MistralCallConfig


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


def test_microcategories_mistral_backend_filters_ids(monkeypatch: pytest.MonkeyPatch) -> None:
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

    artifact = MicrocategoryArtifact(
        model_name=result.model_name,
        pipeline=result.pipeline,
        threshold=result.threshold,
        mlb_classes=[101, 102],
        with_embeddings=False,
        feature_config=feature_config.model_dump(mode="json"),
        training_config=training_config.model_dump(mode="json"),
    )

    monkeypatch.setattr(
        "avito.microcategories.mistral_inference.call_mistral",
        lambda *_args, **_kwargs: '{"detectedMcIds": [102, 999, 102]}',
    )

    mistral_config = MistralCallConfig(models_list=["mistral-small-latest"])
    output = predict_microcategories(
        df.iloc[[0]],
        artifact=artifact,
        backend="mistral",
        mistral_config=mistral_config,
    )

    assert output.detected_mc_ids == [[102]]
    assert output.probabilities is None


def test_microcategories_mistral_backend_excludes_source_mcid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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

    artifact = MicrocategoryArtifact(
        model_name=result.model_name,
        pipeline=result.pipeline,
        threshold=result.threshold,
        mlb_classes=[101, 102],
        with_embeddings=False,
        feature_config=feature_config.model_dump(mode="json"),
        training_config=training_config.model_dump(mode="json"),
    )

    monkeypatch.setattr(
        "avito.microcategories.mistral_inference.call_mistral",
        lambda *_args, **_kwargs: '{"detectedMcIds": [101, 102]}',
    )

    mistral_config = MistralCallConfig(models_list=["mistral-small-latest"])
    output = predict_microcategories(
        df.iloc[[0]],
        artifact=artifact,
        backend="mistral",
        mistral_config=mistral_config,
    )

    assert output.detected_mc_ids == [[102]]


def test_microcategories_mistral_backend_invalid_json_returns_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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

    artifact = MicrocategoryArtifact(
        model_name=result.model_name,
        pipeline=result.pipeline,
        threshold=result.threshold,
        mlb_classes=[101, 102],
        with_embeddings=False,
        feature_config=feature_config.model_dump(mode="json"),
        training_config=training_config.model_dump(mode="json"),
    )

    monkeypatch.setattr(
        "avito.microcategories.mistral_inference.call_mistral",
        lambda *_args, **_kwargs: 'not-json',
    )

    mistral_config = MistralCallConfig(models_list=["mistral-small-latest"])
    output = predict_microcategories(
        df.iloc[[0]],
        artifact=artifact,
        backend="mistral",
        mistral_config=mistral_config,
    )

    assert output.detected_mc_ids == [[]]


def test_microcategories_mistral_backend_requires_config() -> None:
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

    artifact = MicrocategoryArtifact(
        model_name=result.model_name,
        pipeline=result.pipeline,
        threshold=result.threshold,
        mlb_classes=[101, 102],
        with_embeddings=False,
        feature_config=feature_config.model_dump(mode="json"),
        training_config=training_config.model_dump(mode="json"),
    )

    with pytest.raises(ValueError):
        predict_microcategories(
            df.iloc[[0]],
            artifact=artifact,
            backend="mistral",
            mistral_config=None,
        )


def test_microcategories_inference_requires_required_columns_for_backends() -> None:
    broken_df = pd.DataFrame(
        {
            "description": ["текст"],
            "sourceMcId": [101],
        }
    )
    fit_df = _make_df()
    training_config = MicrocategoryTrainingConfig(
        threshold_grid=[0.5],
        merge_train_test_for_fit=False,
        log_reg_c=1.0,
        max_iter=200,
    )
    feature_config = ShouldSplitFeatureConfig(include_extra_text_features=False)
    result = train_microcategory_model(
        df=fit_df,
        include_embeddings=False,
        encoder=None,
        feature_config=feature_config,
        training_config=training_config,
    )

    artifact = MicrocategoryArtifact(
        model_name=result.model_name,
        pipeline=result.pipeline,
        threshold=result.threshold,
        mlb_classes=[101, 102],
        with_embeddings=False,
        feature_config=feature_config.model_dump(mode="json"),
        training_config=training_config.model_dump(mode="json"),
    )

    with pytest.raises(ValueError):
        predict_microcategories(
            broken_df,
            artifact=artifact,
            backend="sklearn",
        )

    with pytest.raises(ValueError):
        predict_microcategories(
            broken_df,
            artifact=artifact,
            backend="mistral",
            mistral_config=MistralCallConfig(models_list=["mistral-small-latest"]),
        )


def test_build_mc_id_title_mapping_uses_allowed_ids_only() -> None:
    df = pd.DataFrame(
        {
            "sourceMcId": [101, 999],
            "sourceMcTitle": ["Сантехника", "Чужая категория"],
        }
    )

    mapping = build_mc_id_title_mapping(df, allowed_ids={101, 102})

    assert mapping[101] == "Сантехника"
    assert isinstance(mapping[102], str)
    assert mapping[102].strip() != ""
    assert 999 not in mapping


def test_build_mistral_messages_contains_categories_and_examples() -> None:
    messages = build_mistral_messages(
        description="Отдельно делаем электрику и сантехнику",
        source_mc_id=101,
        source_title="Ремонт под ключ",
        category_map={101: "Ремонт под ключ", 102: "Электрика", 103: "Сантехника"},
    )

    assert len(messages) == 2
    assert "categories" in messages[1]["content"]
    assert "fewShot" in messages[1]["content"]
