from __future__ import annotations

import logging
from pathlib import Path
from typing import *

import pandas as pd
import pytest

import src.sber.utils.full_evaluate_cli as full_evaluate_cli
from src.sber.utils.full_evaluate_cli import FullEvaluateConfig, run


class _DummyService:
    """Легковесный mock сервиса для тестов full evaluate."""

    captured_queries: list[str] = []
    captured_query_batches: list[list[str]] = []
    captured_feature_batch_size: int | None = None

    def __init__(self, config: object) -> None:
        _DummyService.captured_feature_batch_size = int(getattr(config, "feature_batch_size", 0))

    def predict(self, request: Any) -> dict[str, list[float] | list[int]]:
        queries: list[str] = list(request.queries)
        _DummyService.captured_query_batches.append(queries)
        _DummyService.captured_queries.extend(queries)
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
    _DummyService.captured_queries = []
    _DummyService.captured_query_batches = []

    output_path: Path = run(
        FullEvaluateConfig(
            input_csv=input_csv,
            output_csv=output_csv,
            feature_batch_size=7,
            save_plots=False,
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
    _DummyService.captured_queries = []
    _DummyService.captured_query_batches = []

    run(
        FullEvaluateConfig(
            input_csv=input_csv,
            output_csv=output_csv,
            save_plots=False,
        )
    )

    result: pd.DataFrame = pd.read_csv(output_csv)
    assert "query" in result.columns
    assert _DummyService.captured_queries == ["p1", "p2"]


def test_full_evaluate_shows_progress_and_splits_batches(tmp_path: Path, monkeypatch: Any) -> None:
    input_csv: Path = tmp_path / "input_many.csv"
    output_csv: Path = tmp_path / "output_many.csv"
    pd.DataFrame(
        {
            "query": ["q1", "q2", "q3", "q4", "q5"],
            "model_answer": ["a1", "a2", "a3", "a4", "a5"],
        }
    ).to_csv(input_csv, index=False)

    progress_calls: list[dict[str, Any]] = []

    def _fake_tqdm(iterable: Iterable[int], **kwargs: Any) -> Iterable[int]:
        progress_calls.append(dict(kwargs))
        return iterable

    monkeypatch.setattr("src.sber.utils.full_evaluate_cli.TabularPipelineService", _DummyService)
    monkeypatch.setattr("src.sber.utils.full_evaluate_cli.tqdm", _fake_tqdm)
    _DummyService.captured_queries = []
    _DummyService.captured_query_batches = []

    run(
        FullEvaluateConfig(
            input_csv=input_csv,
            output_csv=output_csv,
            feature_batch_size=2,
            save_plots=False,
        )
    )

    assert _DummyService.captured_query_batches == [["q1", "q2"], ["q3", "q4"], ["q5"]]
    assert progress_calls and progress_calls[0]["desc"] == "Full evaluate"
    assert progress_calls[0]["total"] == 3


def test_full_evaluate_suppresses_hook_info_logs_during_predict(tmp_path: Path, monkeypatch: Any) -> None:
    input_csv: Path = tmp_path / "input_logs.csv"
    output_csv: Path = tmp_path / "output_logs.csv"
    pd.DataFrame(
        {
            "query": ["q1", "q2"],
            "model_answer": ["a1", "a2"],
        }
    ).to_csv(input_csv, index=False)

    observed_levels: list[int] = []

    class _ObserveHooksService(_DummyService):
        def predict(self, request: Any) -> dict[str, list[float] | list[int]]:
            observed_levels.append(full_evaluate_cli.SBER_HOOKS_LOGGER.level)
            return super().predict(request)

    monkeypatch.setattr("src.sber.utils.full_evaluate_cli.TabularPipelineService", _ObserveHooksService)
    _DummyService.captured_queries = []
    _DummyService.captured_query_batches = []
    initial_level: int = full_evaluate_cli.SBER_HOOKS_LOGGER.level

    run(
        FullEvaluateConfig(
            input_csv=input_csv,
            output_csv=output_csv,
            feature_batch_size=1,
            save_plots=False,
        )
    )

    assert observed_levels
    assert all(level >= logging.WARNING for level in observed_levels)
    assert full_evaluate_cli.SBER_HOOKS_LOGGER.level == initial_level


def test_full_evaluate_requires_model_answer_column(tmp_path: Path) -> None:
    input_csv: Path = tmp_path / "bad.csv"
    output_csv: Path = tmp_path / "out.csv"
    pd.DataFrame({"query": ["q1"]}).to_csv(input_csv, index=False)

    with pytest.raises(ValueError, match="model_answer"):
        run(FullEvaluateConfig(input_csv=input_csv, output_csv=output_csv))

