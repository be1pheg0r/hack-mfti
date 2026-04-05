from __future__ import annotations
from typing import *

import random
from pathlib import Path

import pytest
from pydantic import ValidationError

from common.paths import get_sber_configs_dpath
from src.sber.utils.extract_features_cli import (
    ScriptConfig,
    _build_feature_column_names,
    _flatten_feature_groups,
    _sample_balanced_queries_and_answers,
    _sample_temperature,
)
from src.sber.models.extract_features import FeatureGroups


def test_sample_temperature_is_deterministic_for_fixed_seed() -> None:
    rng_a: random.Random = random.Random(17)
    rng_b: random.Random = random.Random(17)

    sampled_a: list[float] = [
        _sample_temperature(rng=rng_a, mean=1.0, std=0.25, min_value=0.3, max_value=1.7)
        for _ in range(5)
    ]
    sampled_b: list[float] = [
        _sample_temperature(rng=rng_b, mean=1.0, std=0.25, min_value=0.3, max_value=1.7)
        for _ in range(5)
    ]

    assert sampled_a == sampled_b
    assert all(0.3 <= value <= 1.7 for value in sampled_a)


def test_sample_temperature_with_zero_std_returns_mean_with_clipping() -> None:
    rng: random.Random = random.Random(1)

    assert _sample_temperature(rng=rng, mean=1.1, std=0.0, min_value=0.3, max_value=1.7) == pytest.approx(1.1)
    assert _sample_temperature(rng=rng, mean=0.1, std=0.0, min_value=0.3, max_value=1.7) == pytest.approx(0.3)
    assert _sample_temperature(rng=rng, mean=2.5, std=0.0, min_value=0.3, max_value=1.7) == pytest.approx(1.7)


def test_script_config_rejects_temperature_bounds_in_wrong_order(tmp_path: Any) -> None:
    with pytest.raises(ValidationError):
        ScriptConfig(
            n=2,
            output_csv=tmp_path / "features.csv",
            temperature_min=1.5,
            temperature_max=1.0,
        )


def test_script_config_accepts_none_n_and_rejects_non_positive_n(tmp_path: Any) -> None:
    config: ScriptConfig = ScriptConfig(n=None, output_csv=tmp_path / "features.csv")
    assert config.n is None

    with pytest.raises(ValidationError, match="n должен быть положительным"):
        ScriptConfig(n=0, output_csv=tmp_path / "features.csv")


def test_script_config_uses_yaml_datasets_config_by_default() -> None:
    config: ScriptConfig = ScriptConfig(n=None)
    assert Path(config.datasets_config_path).name == "datasets_configs.yaml"
    assert Path(config.datasets_config_path) == Path(get_sber_configs_dpath()) / "datasets_configs.yaml"


def test_balanced_sampling_takes_equal_count_from_each_dataset_and_shuffles() -> None:
    datasets_map: dict[str, list[dict[str, Any]]] = {
        "rubq-20": [
            {"question_text": "rq1", "answer_text": "rubq_a1"},
            {"question_text": "rq2", "answer_text": "rubq_a2"},
            {"question_text": "rq3", "answer_text": "rubq_a3"},
            {"question_text": "rq4", "answer_text": "rubq_a4"},
        ],
        "tape-chegeka.raw": [
            {"question": "cq1", "answer": "chegeka_a1"},
            {"question": "cq2", "answer": "chegeka_a2"},
            {"question": "cq3", "answer": "chegeka_a3"},
            {"question": "cq4", "answer": "chegeka_a4"},
        ],
    }

    sampled_queries, sampled_answers = _sample_balanced_queries_and_answers(
        datasets_map=datasets_map,
        n=4,
        seed=11,
    )

    assert len(sampled_queries) == 4
    assert len(sampled_answers) == 4

    rubq_count: int = sum(answer.startswith("rubq_") for answer in sampled_answers)
    chegeka_count: int = sum(answer.startswith("chegeka_") for answer in sampled_answers)
    assert rubq_count == 2
    assert chegeka_count == 2

    # Проверяем, что после общего shuffle выдача не остается блочно по датасетам.
    assert sampled_answers not in [
        ["rubq_a1", "rubq_a2", "chegeka_a1", "chegeka_a2"],
        ["chegeka_a1", "chegeka_a2", "rubq_a1", "rubq_a2"],
    ]


def test_sampling_with_none_n_takes_all_samples_from_all_datasets() -> None:
    datasets_map: dict[str, list[dict[str, Any]]] = {
        "rubq-20": [
            {"question_text": "rq1", "answer_text": "rubq_a1"},
            {"question_text": "rq2", "answer_text": "rubq_a2"},
        ],
        "tape-chegeka.raw": [
            {"question": "cq1", "answer": "chegeka_a1"},
            {"question": "cq2", "answer": "chegeka_a2"},
            {"question": "cq3", "answer": "chegeka_a3"},
        ],
    }

    sampled_queries, sampled_answers = _sample_balanced_queries_and_answers(
        datasets_map=datasets_map,
        n=None,
        seed=11,
    )

    assert len(sampled_queries) == 5
    assert len(sampled_answers) == 5
    assert sorted(sampled_answers) == ["chegeka_a1", "chegeka_a2", "chegeka_a3", "rubq_a1", "rubq_a2"]


def test_balanced_sampling_rejects_n_not_divisible_by_dataset_count() -> None:
    datasets_map: dict[str, list[dict[str, Any]]] = {
        "rubq-20": [{"question_text": "rq1", "answer_text": "rubq_a1"}],
        "tape-chegeka.raw": [{"question": "cq1", "answer": "chegeka_a1"}],
    }

    with pytest.raises(ValueError, match="должен делиться"):
        _sample_balanced_queries_and_answers(datasets_map=datasets_map, n=3, seed=1)


def test_balanced_sampling_rejects_insufficient_dataset_size() -> None:
    datasets_map: dict[str, list[dict[str, Any]]] = {
        "rubq-20": [
            {"question_text": "rq1", "answer_text": "rubq_a1"},
        ],
        "tape-chegeka.raw": [
            {"question": "cq1", "answer": "chegeka_a1"},
            {"question": "cq2", "answer": "chegeka_a2"},
        ],
    }

    with pytest.raises(ValueError, match="недостаточно сэмплов"):
        _sample_balanced_queries_and_answers(datasets_map=datasets_map, n=4, seed=1)


def test_feature_columns_are_semantic_and_order_matches_flatten() -> None:
    features = FeatureGroups(
        uncertainty=[0.1] * 12,
        internal_scalars=[0.2] * 6,
        probe_vec=[0.3] * 4,
        attention_entropy=[0.4] * 6,
        entropy_drops=[0.5],
        moe_routing=[0.6] * 10,
    )

    probe_layers = [0, 2]
    names = _build_feature_column_names(features=features, probe_layers=probe_layers)
    flat = _flatten_feature_groups(features)

    assert len(names) == len(flat)
    assert names[0] == "feature_uncertainty_token_logprob_mean"
    assert "feature_internal_pre_answer_norm_layer_0" in names
    assert "feature_probe_vec_0" in names
    assert "feature_attention_entropy_mean_layer_2" in names
    assert "feature_entropy_drop_layer_0_to_2" in names
    assert "feature_moe_active_ratio_std" in names
    assert all(name.startswith("feature_") for name in names)
    assert all(not name.startswith("feature_") or name != f"feature_{idx}" for idx, name in enumerate(names))
