from __future__ import annotations

from pathlib import Path
from typing import *

import pandas as pd

from src.sber.utils.new_data_cli import NewDataConfig, run


def test_new_data_builds_combined_dataset_and_deduplicates_by_prompt(tmp_path: Path, monkeypatch: Any) -> None:
    source_csv: Path = tmp_path / "source.csv"
    output_csv: Path = tmp_path / "output.csv"

    pd.DataFrame(
        {
            "query": ["same_prompt", "positive_only", "negative_in_source"],
            "model_answer": ["ans1", "ans2", "ans3"],
            "is_hallucination": [1, 1, 0],
            "feature_probe_vec_0": [0.1, 0.2, 0.3],
        }
    ).to_csv(source_csv, index=False)

    def fake_retrieve_all_datasets(config: Any, force_download: bool = False) -> dict[str, Any]:
        _ = (config, force_download)
        return {
            "rubq-20": [{"question_text": "same_prompt", "answer_text": "good_answer"}],
            "tape-chegeka.raw": [{"question": "neg_prompt", "answer": "good_answer_2"}],
        }

    def fake_unpack_dataset(dataset: Iterable[dict[str, Any]], name: str) -> tuple[list[str], list[str]]:
        if name == "rubq-20":
            return [str(item["question_text"]) for item in dataset], [str(item["answer_text"]) for item in dataset]
        return [str(item["question"]) for item in dataset], [str(item["answer"]) for item in dataset]

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

    monkeypatch.setattr("src.sber.utils.new_data_cli.retrieve_all_datasets", fake_retrieve_all_datasets)
    monkeypatch.setattr("src.sber.utils.new_data_cli.unpack_dataset", fake_unpack_dataset)
    monkeypatch.setattr("src.sber.utils.new_data_cli.extract_features_run", fake_extract_run)

    output_path: Path = run(
        NewDataConfig(
            source_csv=source_csv,
            output_csv=output_csv,
            datasets_config_path=tmp_path / "datasets.yaml",
            feature_config_path=tmp_path / "hooks.yaml",
        )
    )

    assert output_path == output_csv
    result: pd.DataFrame = pd.read_csv(output_csv)

    # same_prompt присутствует и у positives, и у negatives, после dedup остается один.
    assert sorted(result["prompt"].tolist()) == ["neg_prompt", "positive_only", "same_prompt"]
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

    monkeypatch.setattr(
        "src.sber.utils.new_data_cli.retrieve_all_datasets",
        lambda config, force_download=False: {"rubq-20": [{"question_text": "neg", "answer_text": "ok"}]},
    )
    monkeypatch.setattr("src.sber.utils.new_data_cli.unpack_dataset", lambda dataset, name: (["neg"], ["ok"]))

    def fake_extract_run(config: Any) -> Path:
        pd.DataFrame(
            {
                "prompt": ["neg"],
                "query": ["neg"],
                "model_answer": ["ok"],
                "is_hallucination": [0],
            }
        ).to_csv(config.output_csv, index=False)
        return Path(config.output_csv)

    monkeypatch.setattr("src.sber.utils.new_data_cli.extract_features_run", fake_extract_run)

    run(
        NewDataConfig(
            source_csv=source_csv,
            output_csv=output_csv,
            datasets_config_path=tmp_path / "datasets.yaml",
            feature_config_path=tmp_path / "hooks.yaml",
        )
    )

    result: pd.DataFrame = pd.read_csv(output_csv)
    assert "is_hallucination" in result.columns
    assert int(result["is_hallucination"].sum()) == 1


