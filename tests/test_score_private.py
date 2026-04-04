from __future__ import annotations

from pathlib import Path
from typing import *

import pandas as pd
import pytest

from scripts.evaluate import score_dataframe
from src.sber.utils.evaluate_utils import evaluate_scoring_results


def test_score_dataframe_adds_prediction_and_timing_columns(monkeypatch: Any) -> None:
    class DummyClf:
        def __init__(self, config: Any) -> None:
            self.config = config

        def predict_entailment_proba(self, premises: Sequence[str], hypotheses: Sequence[str], batch_size: int) -> list[float]:
            assert len(premises) == len(hypotheses)
            return [0.8 for _ in premises]

        def predict_is_hallucination(self, premises: Sequence[str], hypotheses: Sequence[str], batch_size: int) -> list[int]:
            assert len(premises) == len(hypotheses)
            return [0 for _ in premises]

    monkeypatch.setattr("scripts.evaluate.HFNLIClf", DummyClf)

    dataframe = pd.DataFrame(
        {
            "correct_answer": ["a1", "a2", "a3"],
            "model_answer": ["m1", "m2", "m3"],
            "is_hallucination": [0, 1, 0],
        }
    )

    scored, total_batches = score_dataframe(
        df=dataframe,
        repo_id="be1pheg0r/hack-mfti-sbercase",
        batch_size=2,
        nli_config_path=Path("configs/sber/hf_nli_clf.yaml"),
        compute_dtype="float32",
    )

    assert total_batches == 2
    assert "entailment_score" in scored.columns
    assert "hallucination_score" in scored.columns
    assert "pred_is_hallucination" in scored.columns
    assert "t_sample_sec" in scored.columns
    assert scored["entailment_score"].tolist() == [0.8, 0.8, 0.8]
    assert scored["hallucination_score"].tolist() == pytest.approx([0.2, 0.2, 0.2])
    assert scored["pred_is_hallucination"].tolist() == [0, 0, 0]
    assert all(float(value) >= 0.0 for value in scored["t_sample_sec"].tolist())


def test_evaluate_scoring_results_writes_report_and_plots(tmp_path: Path) -> None:
    scored = pd.DataFrame(
        {
            "correct_answer": ["a1", "a2", "a3", "a4"],
            "model_answer": ["m1", "m2", "m3", "m4"],
            "is_hallucination": [0, 1, 1, 0],
            "pred_is_hallucination": [0, 1, 0, 0],
            "hallucination_score": [0.1, 0.9, 0.4, 0.2],
            "t_sample_sec": [0.01, 0.02, 0.015, 0.03],
        }
    )

    summary = evaluate_scoring_results(
        dataframe=scored,
        output_csv_fpath=tmp_path / "scored.csv",
        report_dpath=tmp_path / "reports",
        save_plots_enabled=True,
        total_batches=2,
    )

    assert summary.timing.total_samples == 4
    assert summary.classification.has_labels is True
    assert summary.report_fpath is not None
    assert summary.report_fpath.exists()
    assert summary.misclassified_fpath is not None
    assert summary.misclassified_fpath.exists()
    assert "SBER NLI SCORING REPORT" in summary.report_text
    assert len(summary.plots) >= 2
    assert all(path.exists() for path in summary.plots)


