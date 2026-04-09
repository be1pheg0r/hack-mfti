from __future__ import annotations

from pathlib import Path
from typing import *

import pandas as pd

from src.sber.utils.new_data_cli import NewDataConfig, run


def test_new_data_builds_combined_dataset_without_dedup_by_prompt(tmp_path: Path, monkeypatch: Any) -> None:
    source_csv: Path = tmp_path / "source.csv"
    output_csv: Path = tmp_path / "output.csv"

    pd.DataFrame(
        {
            "query": ["same_prompt", "positive_only", "same_prompt", "negative_only"],
            "model_answer": ["ans1", "ans2", "ans3", "ans4"],
            "is_hallucination": [1, 1, 0, 0],
            "feature_probe_vec_0": [0.1, 0.2, 0.3, 0.4],
        }
    ).to_csv(source_csv, index=False)

    def fake_extract_run(config: Any) -> Path:
        input_df: pd.DataFrame = pd.read_csv(config.input_csv_path)
        result_df: pd.DataFrame = input_df.copy()
        result_df["query"] = result_df["prompt"]
        result_df["feature_probe_vec_0"] = [10.0 + idx for idx in range(len(result_df))]
        result_df["hallucination_score"] = [100.0 for _ in range(len(result_df))]
        result_df["is_hallucination"] = [0 for _ in range(len(result_df))]
        output_path: Path = Path(config.output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result_df.to_csv(output_path, index=False)
        return output_path

    monkeypatch.setattr("src.sber.utils.new_data_cli.extract_features_run", fake_extract_run)

    output_path: Path = run(
        NewDataConfig(
            source_csv=source_csv,
            output_csv=output_csv,
            feature_config_path=tmp_path / "hooks.yaml",
        )
    )

    assert output_path == output_csv
    result: pd.DataFrame = pd.read_csv(output_csv)

    # same_prompt присутствует и у positives, и у negatives source CSV, дубликаты сохраняются.
    assert sorted(result["prompt"].tolist()) == ["negative_only", "positive_only", "same_prompt", "same_prompt"]
    assert len(result) == 4
    assert int(result["is_hallucination"].sum()) == 2


def test_new_data_supports_legacy_target_typo_column(tmp_path: Path, monkeypatch: Any) -> None:
    source_csv: Path = tmp_path / "source_typo.csv"
    output_csv: Path = tmp_path / "output_typo.csv"

    pd.DataFrame(
        {
            "prompt": ["p1", "p2"],
            "model_answer": ["a1", "a2"],
            "is_hallutination": ["True", "False"],
        }
    ).to_csv(source_csv, index=False)

    def fake_extract_run(config: Any) -> Path:
        input_df: pd.DataFrame = pd.read_csv(config.input_csv_path)
        result_df: pd.DataFrame = input_df.copy()
        result_df["query"] = result_df["prompt"]
        result_df.to_csv(config.output_csv, index=False)
        return Path(config.output_csv)

    monkeypatch.setattr("src.sber.utils.new_data_cli.extract_features_run", fake_extract_run)

    run(
        NewDataConfig(
            source_csv=source_csv,
            output_csv=output_csv,
            feature_config_path=tmp_path / "hooks.yaml",
        )
    )

    result: pd.DataFrame = pd.read_csv(output_csv)
    assert "is_hallucination" in result.columns
    assert int(result["is_hallucination"].sum()) == 1


