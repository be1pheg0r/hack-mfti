from __future__ import annotations
from typing import *

import json
from pathlib import Path

import pytest
import pandas as pd
from pydantic import ValidationError

from common.paths import get_sber_configs_dpath
from src.sber.utils.extract_features_cli import (
    ScriptConfig,
    _build_feature_column_names,
    _coerce_field_to_float_if_needed,
    _extract_expected_float_field_name,
    _flatten_feature_groups,
    _resolve_answer_column_name,
    _resolve_query_column_name,
    _sample_balanced_queries_and_answers,
    _sample_dataframe_rows,
    _try_patch_model_config_for_float_field,
)
from src.sber.models.extract_features import FeatureGroups




def test_script_config_accepts_none_n_and_rejects_non_positive_n(tmp_path: Any) -> None:
    config: ScriptConfig = ScriptConfig(n=None, output_csv=tmp_path / "features.csv")
    assert config.n is None

    with pytest.raises(ValidationError, match="n должен быть положительным"):
        ScriptConfig(n=0, output_csv=tmp_path / "features.csv")


def test_script_config_uses_yaml_datasets_config_by_default() -> None:
    config: ScriptConfig = ScriptConfig(n=None)
    assert Path(config.datasets_config_path).name == "datasets_configs.yaml"
    assert Path(config.datasets_config_path) == Path(get_sber_configs_dpath()) / "datasets_configs.yaml"



def test_resolve_query_column_name_prefers_query_then_prompt() -> None:
    assert _resolve_query_column_name(["id", "query"]) == "query"
    assert _resolve_query_column_name(["id", "prompt"]) == "prompt"


def test_resolve_query_column_name_raises_when_column_missing() -> None:
    with pytest.raises(ValueError, match="query или prompt"):
        _resolve_query_column_name(["id", "text"])


def test_resolve_answer_column_name_prefers_model_answer_then_fallbacks() -> None:
    assert _resolve_answer_column_name(["id", "model_answer"]) == "model_answer"
    assert _resolve_answer_column_name(["id", "answer"]) == "answer"
    assert _resolve_answer_column_name(["id", "generated_answer"]) == "generated_answer"
    assert _resolve_answer_column_name(["id", "correct_answer"]) == "correct_answer"


def test_resolve_answer_column_name_raises_when_column_missing() -> None:
    with pytest.raises(ValueError, match="колонка с ответом"):
        _resolve_answer_column_name(["id", "text"])


def test_sample_dataframe_rows_is_deterministic_for_n() -> None:
    dataframe: pd.DataFrame = pd.DataFrame({"query": [f"q{i}" for i in range(6)]})

    sampled_a: pd.DataFrame = _sample_dataframe_rows(dataframe, n=3, seed=10)
    sampled_b: pd.DataFrame = _sample_dataframe_rows(dataframe, n=3, seed=10)

    assert sampled_a["query"].tolist() == sampled_b["query"].tolist()
    assert len(sampled_a) == 3


def test_extract_expected_float_field_name_parses_error_text() -> None:
    text: str = "TypeError: Field 'routed_scaling_factor' expected float, got int (value: 1)"
    assert _extract_expected_float_field_name(text) == "routed_scaling_factor"


def test_coerce_field_to_float_if_needed_changes_nested_int_fields() -> None:
    payload: dict[str, Any] = {
        "routed_scaling_factor": 1,
        "nested": {"routed_scaling_factor": 2, "other": 3},
        "items": [{"routed_scaling_factor": 4}],
    }

    changed: bool = _coerce_field_to_float_if_needed(payload, field_name="routed_scaling_factor")

    assert changed is True
    assert isinstance(payload["routed_scaling_factor"], float)
    assert isinstance(payload["nested"]["routed_scaling_factor"], float)
    assert isinstance(payload["items"][0]["routed_scaling_factor"], float)


def test_try_patch_model_config_for_float_field_patches_local_config(tmp_path: Path) -> None:
    model_dir: Path = tmp_path / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    config_fpath: Path = model_dir / "config.json"
    config_fpath.write_text('{"routed_scaling_factor": 1}', encoding="utf-8")

    error: Exception = ValueError("Field 'routed_scaling_factor' expected float, got int (value: 1)")
    patched: bool = _try_patch_model_config_for_float_field(str(model_dir), error)

    assert patched is True
    patched_payload: dict[str, Any] = json.loads(config_fpath.read_text(encoding="utf-8"))
    assert isinstance(patched_payload["routed_scaling_factor"], float)


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
