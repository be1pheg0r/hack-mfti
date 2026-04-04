from __future__ import annotations

import pandas as pd

from scripts.train_hf_nli import build_balanced_train_df, is_candidate_better


def test_build_balanced_train_df_downsamples_majority() -> None:
    dataframe = pd.DataFrame(
        {
            "correct_answer": [f"c{i}" for i in range(8)],
            "model_answer": [f"m{i}" for i in range(8)],
            "is_hallucination": [1, 1, 1, 1, 1, 0, 0, 0],
        }
    )

    balanced = build_balanced_train_df(dataframe, seed=42)

    counts: dict[int, int] = balanced["is_hallucination"].value_counts().to_dict()
    assert counts[0] == counts[1]
    assert len(balanced) == 6


def test_is_candidate_better_uses_primary_then_secondary() -> None:
    baseline = {"weighted_average_precision": 0.90, "f1": 0.80}

    better_primary = {"weighted_average_precision": 0.91, "f1": 0.70}
    assert (
        is_candidate_better(
            candidate=better_primary,
            baseline=baseline,
            primary_key="weighted_average_precision",
            secondary_key="f1",
            min_relative_improvement=0.0,
        )
        is True
    )

    tie_primary_better_secondary = {"weighted_average_precision": 0.90, "f1": 0.82}
    assert (
        is_candidate_better(
            candidate=tie_primary_better_secondary,
            baseline=baseline,
            primary_key="weighted_average_precision",
            secondary_key="f1",
            min_relative_improvement=0.0,
        )
        is True
    )

    worse = {"weighted_average_precision": 0.89, "f1": 0.99}
    assert (
        is_candidate_better(
            candidate=worse,
            baseline=baseline,
            primary_key="weighted_average_precision",
            secondary_key="f1",
            min_relative_improvement=0.0,
        )
        is False
    )


