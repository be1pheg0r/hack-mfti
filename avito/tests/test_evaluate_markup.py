from __future__ import annotations

import csv
import json
from types import SimpleNamespace

from avito.should_split.interfaces.jobs.evaluate import (
    EvaluationSample,
    _disable_drafts,
    aggregate_metrics,
    evaluate_markup_dataset,
    evaluate_official_dataset,
    parse_mc_ids,
)
from avito.should_split.api.schemas import AvitoApiResponse
from avito.should_split.core.config import ShouldSplitGraphConfig


def test_parse_mc_ids_supports_stringified_list_and_deduplicates() -> None:
    assert parse_mc_ids("[101, '102', 101, 'foo']") == [101, 102]
    assert parse_mc_ids("[]") == []


def test_aggregate_metrics_computes_micro_scores() -> None:
    samples = [
        EvaluationSample(
            gt_ids={1, 2},
            pred_ids={2, 3},
            gt_should_split=True,
            pred_should_split=True,
        ),
        EvaluationSample(
            gt_ids=set(),
            pred_ids=set(),
            gt_should_split=False,
            pred_should_split=False,
        ),
    ]

    metrics = aggregate_metrics(samples)

    assert metrics.true_positive == 1
    assert metrics.false_positive == 1
    assert metrics.false_negative == 1
    assert metrics.precision_micro == 0.5
    assert metrics.recall_micro == 0.5
    assert metrics.f1_micro == 0.5
    assert metrics.should_split_accuracy == 1.0


def test_aggregate_metrics_handles_empty_sample_list() -> None:
    metrics = aggregate_metrics([])

    assert metrics.samples_count == 0
    assert metrics.precision_micro == 0.0
    assert metrics.recall_micro == 0.0
    assert metrics.f1_micro == 0.0
    assert metrics.should_split_accuracy == 0.0


def test_disable_drafts_forces_graph_flag() -> None:
    config = ShouldSplitGraphConfig.model_validate({"graph": {"enable_drafts": True}})
    updated = _disable_drafts(config)

    assert config.graph.enable_drafts is True
    assert updated.graph.enable_drafts is False


def test_evaluate_official_dataset_uses_api_client_and_computes_metrics(tmp_path, monkeypatch) -> None:
    dataset_path = tmp_path / "official_dataset.json"
    dataset = [
        {
            "request": json.dumps(
                {
                    "itemId": 5001,
                    "mcId": 101,
                    "mcTitle": "Ремонт квартир и домов под ключ",
                    "description": "Текст объявления",
                },
                ensure_ascii=False,
            ),
            "response": json.dumps(
                {
                    "detectedMcIds": [101, 102, 103],
                    "shouldSplit": True,
                    "drafts": [],
                },
                ensure_ascii=False,
            ),
        }
    ]
    dataset_path.write_text(json.dumps(dataset, ensure_ascii=False), encoding="utf-8")

    class FakePipelineApiClient:
        def __init__(self, base_url: str) -> None:
            self.base_url = base_url

        def infer(self, item_id: int | str, mc_id: int, mc_title: str, description: str) -> AvitoApiResponse:
            assert item_id == 5001
            assert mc_id == 101
            assert mc_title == "Ремонт квартир и домов под ключ"
            assert description == "Текст объявления"
            return AvitoApiResponse(
                detectedMcIds=[101, 102],
                shouldSplit=True,
                drafts=[],
            )

    monkeypatch.setattr(
        "avito.should_split.interfaces.jobs.evaluate.PipelineApiClient",
        FakePipelineApiClient,
    )

    prediction_rows: list[dict] = []
    metrics = evaluate_official_dataset(
        dataset_path=dataset_path,
        api_url="http://127.0.0.1:8080",
        prediction_rows=prediction_rows,
    )

    assert metrics.samples_count == 1
    assert metrics.true_positive == 1
    assert metrics.false_positive == 0
    assert metrics.false_negative == 1
    assert metrics.precision_micro == 1.0
    assert metrics.recall_micro == 0.5
    assert round(metrics.f1_micro, 6) == round(2 / 3, 6)
    assert metrics.should_split_accuracy == 1.0

    assert len(prediction_rows) == 1
    assert prediction_rows[0]["prediction"] == {
        "detectedMcIds": [101, 102],
        "shouldSplit": True,
        "drafts": [],
    }


def test_evaluate_markup_dataset_supports_csv(tmp_path) -> None:
    dataset_path = tmp_path / "markup_dataset.csv"
    with open(dataset_path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["description", "sourceMcId", "targetSplitMcIds", "shouldSplit"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "description": "Описание",
                "sourceMcId": "101",
                "targetSplitMcIds": "[101, 102]",
                "shouldSplit": "true",
            }
        )

    class FakePipeline:
        def invoke(self, description: str) -> SimpleNamespace:
            assert description == "Описание"
            return SimpleNamespace(categorized_mc_ids=[101, 102], shouldSplit=True, drafts=[])

    prediction_rows: list[dict[str, object]] = []
    metrics = evaluate_markup_dataset(
        dataset_path=dataset_path,
        pipeline=FakePipeline(),
        prediction_rows=prediction_rows,
    )

    assert metrics.samples_count == 1
    assert metrics.true_positive == 1
    assert metrics.false_positive == 0
    assert metrics.false_negative == 0
    assert metrics.precision_micro == 1.0
    assert metrics.recall_micro == 1.0
    assert metrics.f1_micro == 1.0
    assert metrics.should_split_accuracy == 1.0

    assert len(prediction_rows) == 1
    assert prediction_rows[0]["prediction"] == {
        "detectedMcIds": [101, 102],
        "shouldSplit": True,
        "drafts": [],
    }


def test_evaluate_official_dataset_supports_csv(tmp_path, monkeypatch) -> None:
    dataset_path = tmp_path / "official_dataset.csv"
    request_payload = {
        "itemId": 5001,
        "mcId": 101,
        "mcTitle": "Ремонт квартир и домов под ключ",
        "description": "Текст объявления",
    }
    response_payload = {
        "detectedMcIds": [101, 102, 103],
        "shouldSplit": True,
        "drafts": [],
    }

    with open(dataset_path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["request", "response"])
        writer.writeheader()
        writer.writerow(
            {
                "request": json.dumps(request_payload, ensure_ascii=False),
                "response": json.dumps(response_payload, ensure_ascii=False),
            }
        )

    class FakePipelineApiClient:
        def __init__(self, base_url: str) -> None:
            self.base_url = base_url

        def infer(self, item_id: int | str, mc_id: int, mc_title: str, description: str) -> AvitoApiResponse:
            assert item_id == 5001
            assert mc_id == 101
            assert mc_title == "Ремонт квартир и домов под ключ"
            assert description == "Текст объявления"
            return AvitoApiResponse(
                detectedMcIds=[101, 102],
                shouldSplit=True,
                drafts=[],
            )

    monkeypatch.setattr(
        "avito.should_split.interfaces.jobs.evaluate.PipelineApiClient",
        FakePipelineApiClient,
    )

    prediction_rows: list[dict[str, object]] = []
    metrics = evaluate_official_dataset(
        dataset_path=dataset_path,
        api_url="http://127.0.0.1:8080",
        prediction_rows=prediction_rows,
    )

    assert metrics.samples_count == 1
    assert metrics.true_positive == 1
    assert metrics.false_positive == 0
    assert metrics.false_negative == 1
    assert metrics.precision_micro == 1.0
    assert metrics.recall_micro == 0.5
    assert round(metrics.f1_micro, 6) == round(2 / 3, 6)
    assert metrics.should_split_accuracy == 1.0


def test_evaluate_official_dataset_handles_empty_response_and_fills_it(tmp_path, monkeypatch) -> None:
    dataset_path = tmp_path / "official_without_gt.csv"
    request_payload = {
        "itemId": 5001,
        "mcId": 101,
        "mcTitle": "Ремонт квартир и домов под ключ",
        "description": "Текст объявления",
    }

    with open(dataset_path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["request", "response"])
        writer.writeheader()
        writer.writerow(
            {
                "request": json.dumps(request_payload, ensure_ascii=False),
                "response": "",
            }
        )

    class FakePipelineApiClient:
        def __init__(self, base_url: str) -> None:
            self.base_url = base_url

        def infer(self, item_id: int | str, mc_id: int, mc_title: str, description: str) -> AvitoApiResponse:
            return AvitoApiResponse(
                detectedMcIds=[101, 102],
                shouldSplit=True,
                drafts=[],
            )

    monkeypatch.setattr(
        "avito.should_split.interfaces.jobs.evaluate.PipelineApiClient",
        FakePipelineApiClient,
    )

    prediction_rows: list[dict[str, object]] = []
    metrics = evaluate_official_dataset(
        dataset_path=dataset_path,
        api_url="http://127.0.0.1:8080",
        prediction_rows=prediction_rows,
    )

    assert metrics.samples_count == 0
    assert metrics.precision_micro == 0.0
    assert metrics.recall_micro == 0.0
    assert metrics.f1_micro == 0.0
    assert metrics.should_split_accuracy == 0.0

    assert len(prediction_rows) == 1
    assert prediction_rows[0]["response"] == {
        "detectedMcIds": [101, 102],
        "shouldSplit": True,
        "drafts": [],
    }
    assert prediction_rows[0]["prediction"] == {
        "detectedMcIds": [101, 102],
        "shouldSplit": True,
        "drafts": [],
    }


