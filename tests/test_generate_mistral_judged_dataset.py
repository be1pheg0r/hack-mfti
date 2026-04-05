from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.sber.utils.generate_mistral_judged_dataset_cli import merge_with_existing, normalize_judge_score


def test_normalize_judge_score_json_and_plain_text() -> None:
    assert normalize_judge_score('{"judge_score": "галлюцинация"}') == "галлюцинация"
    assert normalize_judge_score("не галлюцинация") == "не галлюцинация"
    assert normalize_judge_score("unknown") == "неизвестно"


def test_merge_with_existing_prefills_processed_rows(tmp_path: Path) -> None:
    base_df = pd.DataFrame(
        {
            "dataset_name": ["rubq-20", "rubq-20"],
            "query": ["q1", "q2"],
            "correct_answer": ["a1", "a2"],
            "model_answer": ["", ""],
            "judge_score": [pd.NA, pd.NA],
            "is_hallucination": [pd.NA, pd.NA],
        }
    )
    existing_df = pd.DataFrame(
        {
            "query": ["q2"],
            "model_answer": ["m2"],
            "judge_score": ["галлюцинация"],
            "is_hallucination": [1],
        }
    )
    output_csv = tmp_path / "existing.csv"
    existing_df.to_csv(output_csv, index=False)

    merged = merge_with_existing(base_df=base_df, output_csv=output_csv, resume_enabled=True)

    second = merged[merged["query"] == "q2"].iloc[0]
    assert str(second["model_answer"]) == "m2"
    assert str(second["judge_score"]) == "галлюцинация"
    assert int(second["is_hallucination"]) == 1
