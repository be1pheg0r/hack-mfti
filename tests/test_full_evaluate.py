from __future__ import annotations

from pathlib import Path
from typing import *

import pandas as pd
import pytest

from src.sber.utils.full_evaluate_cli import FullEvaluateConfig, run


class _DummyService:
    """Легковесный mock сервиса для тестов full evaluate."""

    captured_queries: list[str] = []
    captured_feature_batch_size: int | None = None

    def __init__(self, config: object) -> None:
        _DummyService.captured_feature_batch_size = int(getattr(config, "feature_batch_size", 0))

    def predict(self, request: Any) -> dict[str, list[float] | list[int]]:
        queries: list[str] = list(request.queries)
        _DummyService.captured_queries = queries
        size: int = len(queries)
        return {
            "pred_is_hallucination": [1 for _ in range(size)],
            "hallucination_score": [0.73 for _ in range(size)],
            "entailment_score": [0.27 for _ in range(size)],
            "t_feature_extraction_sec": [0.01 for _ in range(size)],
            "t_classification_sec": [0.002 for _ in range(size)],
            "t_total_sec": [0.012 for _ in range(size)],
            "t_sample_sec": [0.012 for _ in range(size)],
        }


def test_full_evaluate_writes_predict_proba_for_query_column(tmp_path: Path, monkeypatch: Any) -> None:
    input_csv: Path = tmp_path / "input.csv"
    output_csv: Path = tmp_path / "output.csv"
    pd.DataFrame(
        {
            "query": ["q1", "q2"],
            "model_answer": ["a1", "a2"],
        }
    ).to_csv(input_csv, index=False)

    monkeypatch.setattr("src.sber.utils.full_evaluate_cli.TabularPipelineService", _DummyService)

    output_path: Path = run(
        FullEvaluateConfig(
            input_csv=input_csv,
            output_csv=output_csv,
            feature_batch_size=7,
        )
    )

    assert output_path == output_csv
    result: pd.DataFrame = pd.read_csv(output_csv)
    assert "predict_proba" in result.columns
    assert result["predict_proba"].tolist() == pytest.approx([0.73, 0.73])
    assert result["pred_is_hallucination"].tolist() == [1, 1]
    assert _DummyService.captured_feature_batch_size == 7


def test_full_evaluate_supports_prompt_column(tmp_path: Path, monkeypatch: Any) -> None:
    input_csv: Path = tmp_path / "input_prompt.csv"
    output_csv: Path = tmp_path / "output_prompt.csv"
    pd.DataFrame(
        {
            "prompt": ["p1", "p2"],
            "model_answer": ["a1", "a2"],
        }
    ).to_csv(input_csv, index=False)

    monkeypatch.setattr("src.sber.utils.full_evaluate_cli.TabularPipelineService", _DummyService)

    run(
        FullEvaluateConfig(
            input_csv=input_csv,
            output_csv=output_csv,
        )
    )

    result: pd.DataFrame = pd.read_csv(output_csv)
    assert "query" in result.columns
    assert _DummyService.captured_queries == ["p1", "p2"]


def test_full_evaluate_requires_model_answer_column(tmp_path: Path) -> None:
    input_csv: Path = tmp_path / "bad.csv"
    output_csv: Path = tmp_path / "out.csv"
    pd.DataFrame({"query": ["q1"]}).to_csv(input_csv, index=False)

    with pytest.raises(ValueError, match="model_answer"):
        run(FullEvaluateConfig(input_csv=input_csv, output_csv=output_csv))

