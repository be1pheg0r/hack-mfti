from __future__ import annotations

from pathlib import Path
from typing import *

import pandas as pd
import pytest

from src.sber.utils.map_bench_columns_cli import (
    BenchMappingConfig,
    build_val_feature_to_train_mapping,
    run,
)


def test_mapping_size_for_default_config() -> None:
    mapping = build_val_feature_to_train_mapping()
    assert len(mapping) == 1613
    assert mapping["feature_uncertainty_token_logprob_mean"] == "mean_log_prob"
    assert mapping["feature_probe_vec_1535"] == "probe_vec_1535"
    assert mapping["feature_entropy_drop_layer_24_to_25"] == "entropy_drop_24_to_25"


def test_run_maps_columns_and_saves_output(tmp_path: Path) -> None:
    mapping = build_val_feature_to_train_mapping()

    val_columns = {
        "sample_id": [1],
        "query": ["q"],
        "is_hallucination": [0],
        "judge_score": [0.1],
    }
    for source_name in mapping.keys():
        val_columns[source_name] = [0.5]

    train_columns = {
        "sample_id": [1],
        "query": ["q"],
        "is_hallucination": [0],
        "judge_score": [0.1],
    }
    for target_name in mapping.values():
        train_columns[target_name] = [0.5]

    train_path = tmp_path / "train.csv"
    val_path = tmp_path / "val.csv"
    output_path = tmp_path / "mapped.csv"

    pd.DataFrame(train_columns).to_csv(train_path, index=False)
    pd.DataFrame(val_columns).to_csv(val_path, index=False)

    config = BenchMappingConfig(
        train_csv=train_path,
        val_csv=val_path,
        output_csv=output_path,
        strict=False,
        inplace=False,
    )
    run(config=config)

    mapped_df = pd.read_csv(output_path)
    assert "mean_log_prob" in mapped_df.columns
    assert "feature_uncertainty_token_logprob_mean" not in mapped_df.columns
    assert "probe_vec_1535" in mapped_df.columns


def test_run_fails_on_missing_mapping_sources(tmp_path: Path) -> None:
    mapping = build_val_feature_to_train_mapping()

    val_columns = {
        "sample_id": [1],
        "query": ["q"],
        "is_hallucination": [0],
        "judge_score": [0.1],
    }
    # Deliberately keep only one source column to trigger validation error.
    first_source = next(iter(mapping.keys()))
    val_columns[first_source] = [0.5]

    train_columns = {
        "sample_id": [1],
        "query": ["q"],
        "is_hallucination": [0],
        "judge_score": [0.1],
    }
    for target_name in mapping.values():
        train_columns[target_name] = [0.5]

    train_path = tmp_path / "train.csv"
    val_path = tmp_path / "val.csv"
    output_path = tmp_path / "mapped.csv"

    pd.DataFrame(train_columns).to_csv(train_path, index=False)
    pd.DataFrame(val_columns).to_csv(val_path, index=False)

    config = BenchMappingConfig(
        train_csv=train_path,
        val_csv=val_path,
        output_csv=output_path,
        strict=True,
        inplace=False,
    )

    with pytest.raises(ValueError, match="missing source feature columns"):
        run(config=config)
