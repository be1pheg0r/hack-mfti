from __future__ import annotations

import pandas as pd
import pytest

from avito.features import ShouldSplitFeatureConfig, extract_should_split_features
import avito.features as features_module


def test_extract_should_split_features_builds_rapidfuzz_vector_by_category(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        features_module,
        "_load_catalog_keyphrases_by_category",
        lambda: (
            (101, ("ремонт квартиры", "под ключ")),
            (102, ("сантехника", "разводка труб")),
            (103, ("электрика", "монтаж проводки")),
        ),
    )

    df = pd.DataFrame(
        {
            "description": [
                "Сделаем сантехнику и разводку труб в санузле.",
                "",
            ]
        }
    )
    config = ShouldSplitFeatureConfig(
        include_extra_text_features=False,
        include_split_marker_window=False,
        include_rapidfuzz_score=True,
    )

    result = extract_should_split_features(df=df, config=config)

    rapidfuzz_columns = [
        "max_keyphrase_rapidfuzz_mc_101",
        "max_keyphrase_rapidfuzz_mc_102",
        "max_keyphrase_rapidfuzz_mc_103",
    ]
    assert all(column in result.columns for column in rapidfuzz_columns)
    assert "max_keyphrase_rapidfuzz" not in result.columns

    row_0_scores = result.loc[0, rapidfuzz_columns]
    assert row_0_scores.max() == result.loc[0, "max_keyphrase_rapidfuzz_mc_102"]

    row_1_scores = result.loc[1, rapidfuzz_columns]
    assert (row_1_scores == 0.0).all()


def test_extract_should_split_features_requires_description_column() -> None:
    config = ShouldSplitFeatureConfig(
        include_extra_text_features=False,
        include_split_marker_window=False,
        include_rapidfuzz_score=False,
    )

    with pytest.raises(ValueError, match="обязательные колонки"):
        extract_should_split_features(df=pd.DataFrame({"text": ["abc"]}), config=config)

