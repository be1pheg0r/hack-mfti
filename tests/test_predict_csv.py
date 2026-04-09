from __future__ import annotations

from pathlib import Path
from typing import *

import pandas as pd
import pytest

from src.sber.utils.predict_csv_cli import PredictCsvConfig, run


class _DummyService:
    """Легковесный mock сервиса для тестов predict_csv."""

    captured_feature_batch_size: int | None = None

    def __init__(self, config: object) -> None:
        _DummyService.captured_feature_batch_size = int(getattr(config, "feature_batch_size", 0))

    def predict(self, request: Any) -> dict[str, list[float] | list[int]]:
        size: int = len(request.queries)
        return {
            "pred_is_hallucination": [0 for _ in range(size)],
            "hallucination_score": [0.42 for _ in range(size)],
            "entailment_score": [0.58 for _ in range(size)],
            "t_feature_extraction_sec": [0.01 for _ in range(size)],
            "t_classification_sec": [0.002 for _ in range(size)],
            "t_total_sec": [0.012 for _ in range(size)],
            "t_sample_sec": [0.012 for _ in range(size)],
        }


def test_predict_csv_writes_predict_proba_for_prompt_column(tmp_path: Path, monkeypatch: Any) -> None:
    input_csv: Path = tmp_path / "input.csv"
    output_csv: Path = tmp_path / "output.csv"
    pd.DataFrame(
        {
            "prompt": ["p1", "p2"],
            "model_answer": ["a1", "a2"],
        }
    ).to_csv(input_csv, index=False)

    monkeypatch.setattr("src.sber.utils.predict_csv_cli.TabularPipelineService", _DummyService)

    output_path: Path = run(
        PredictCsvConfig(
            input_csv=input_csv,
            output_csv=output_csv,
            feature_batch_size=5,
        )
    )

    assert output_path == output_csv
    result: pd.DataFrame = pd.read_csv(output_csv)
    assert "predict_proba" in result.columns
    assert result["predict_proba"].tolist() == pytest.approx([0.42, 0.42])
    assert _DummyService.captured_feature_batch_size == 5


def test_predict_csv_requires_model_answer_column(tmp_path: Path) -> None:
    input_csv: Path = tmp_path / "bad.csv"
    output_csv: Path = tmp_path / "out.csv"
    pd.DataFrame({"prompt": ["p1"]}).to_csv(input_csv, index=False)

    with pytest.raises(ValueError, match="model_answer"):
        run(PredictCsvConfig(input_csv=input_csv, output_csv=output_csv))

