from __future__ import annotations

import pandas as pd
import pytest

from src.sber.utils.train_hf_nli_cli import (
    apply_hallucination_threshold,
    build_balanced_train_df,
    calculate_warmup_steps,
    is_candidate_better,
    resolve_hallucination_threshold,
    resolve_question_column,
    select_class_weight_dataframe,
)


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


def test_resolve_hallucination_threshold_grid_prefers_recall_when_beta_high() -> None:
    dataframe = pd.DataFrame(
        {
            "hallucination_score": [0.49, 0.20, 0.80, 0.30],
            "is_hallucination": [1, 0, 1, 0],
            "pred_is_hallucination": [0, 0, 1, 0],
        }
    )

    threshold = resolve_hallucination_threshold(
        dataframe,
        threshold_search_mode="grid",
        threshold_search_beta=2.0,
        threshold_trials=20,
        fallback_threshold=0.5,
    )

    assert threshold <= 0.49


def test_apply_hallucination_threshold_rebuilds_predictions() -> None:
    dataframe = pd.DataFrame(
        {
            "hallucination_score": [0.2, 0.5, 0.9],
            "pred_is_hallucination": [1, 1, 0],
        }
    )

    result = apply_hallucination_threshold(dataframe, threshold=0.6)

    assert result["pred_is_hallucination"].tolist() == [0, 0, 1]


def test_calculate_warmup_steps_uses_single_epoch_ratio() -> None:
    assert calculate_warmup_steps(num_batches_per_epoch=120, warmup_ratio_per_epoch=0.1) == 12


def test_select_class_weight_dataframe_can_use_raw_or_balanced() -> None:
    raw_df = pd.DataFrame({"is_hallucination": [1, 1, 1, 0]})
    balanced_df = pd.DataFrame({"is_hallucination": [1, 0]})

    selected_without_undersampling = select_class_weight_dataframe(
        train_df_raw=raw_df,
        train_df_balanced=balanced_df,
        undersampling_enabled=False,
        class_weights_before_balancing=False,
    )
    selected_raw = select_class_weight_dataframe(
        train_df_raw=raw_df,
        train_df_balanced=balanced_df,
        undersampling_enabled=True,
        class_weights_before_balancing=True,
    )
    selected_balanced = select_class_weight_dataframe(
        train_df_raw=raw_df,
        train_df_balanced=balanced_df,
        undersampling_enabled=True,
        class_weights_before_balancing=False,
    )

    assert selected_without_undersampling is raw_df
    assert selected_raw is raw_df
    assert selected_balanced is balanced_df


def test_resolve_question_column_uses_known_candidates() -> None:
    dataframe = pd.DataFrame({"prompt": ["q1"], "model_answer": ["a1"], "is_hallucination": [0]})

    resolved = resolve_question_column(dataframe)

    assert resolved == "prompt"


def test_resolve_question_column_raises_without_question_column() -> None:
    dataframe = pd.DataFrame({"model_answer": ["a1"], "is_hallucination": [0], "correct_answer": ["gt"]})

    with pytest.raises(ValueError, match="Колонка вопроса не найдена"):
        resolve_question_column(dataframe)

