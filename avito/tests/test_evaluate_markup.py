from __future__ import annotations

import csv
import json

import pytest

from avito.should_split.api.client import PipelineApiClientError
from avito.should_split.interfaces.jobs.evaluate import (
    EvaluationSample,
    _resolve_api_timeout_sec,
    _write_predictions,
    aggregate_metrics,
    evaluate_official_dataset,
    parse_mc_ids,
)
from avito.should_split.api.schemas import AvitoApiResponse


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
    assert metrics.categories_accuracy == 0.5
    assert metrics.inference_time_total_sec == 0.0


def test_aggregate_metrics_handles_empty_sample_list() -> None:
    metrics = aggregate_metrics([])

    assert metrics.samples_count == 0
    assert metrics.precision_micro == 0.0
    assert metrics.recall_micro == 0.0
    assert metrics.f1_micro == 0.0
    assert metrics.should_split_accuracy == 0.0
    assert metrics.categories_accuracy == 0.0
    assert metrics.inference_time_total_sec == 0.0
    assert metrics.inference_time_avg_sec == 0.0
    assert metrics.inference_time_p95_sec == 0.0


def test_aggregate_metrics_category_accuracy_ignores_order_and_counts_time() -> None:
    samples = [
        EvaluationSample(
            gt_ids={101, 102},
            pred_ids={102, 101},
            gt_should_split=True,
            pred_should_split=True,
        ),
        EvaluationSample(
            gt_ids={103},
            pred_ids={104},
            gt_should_split=True,
            pred_should_split=True,
        ),
    ]

    metrics = aggregate_metrics(samples, inference_durations_sec=[0.1, 0.2, 0.3])

    assert metrics.categories_accuracy == 0.5
    assert metrics.inference_time_total_sec == pytest.approx(0.6)
    assert metrics.inference_time_avg_sec == pytest.approx(0.2)
    assert metrics.inference_time_p95_sec == pytest.approx(0.3)


def test_resolve_api_timeout_sec_reads_from_config_when_cli_not_set(tmp_path) -> None:
    config_path = tmp_path / "should_split_graph.yaml"
    config_path.write_text(
        """
graph:
  pipeline_request_timeout_sec: 123
""".strip(),
        encoding="utf-8",
    )

    assert _resolve_api_timeout_sec(config_path=config_path, cli_timeout_sec=None) == 123


def test_resolve_api_timeout_sec_prefers_cli_override(tmp_path) -> None:
    config_path = tmp_path / "should_split_graph.yaml"
    config_path.write_text(
        """
graph:
  pipeline_request_timeout_sec: 123
""".strip(),
        encoding="utf-8",
    )

    assert _resolve_api_timeout_sec(config_path=config_path, cli_timeout_sec=77) == 77


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
        def __init__(self, base_url: str, timeout_sec: int = 60) -> None:
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


def test_evaluate_official_dataset_continues_after_timeout_and_writes_prediction_error(tmp_path, monkeypatch) -> None:
    dataset_path = tmp_path / "official_timeout_recover.json"
    dataset = [
        {
            "request": {
                "itemId": 1,
                "mcId": 101,
                "mcTitle": "Категория",
                "description": "Первый",
            },
            "response": {
                "detectedMcIds": [101, 102],
                "shouldSplit": True,
                "drafts": [],
            },
        },
        {
            "request": {
                "itemId": 2,
                "mcId": 101,
                "mcTitle": "Категория",
                "description": "Второй",
            },
            "response": {
                "detectedMcIds": [101, 102],
                "shouldSplit": True,
                "drafts": [],
            },
        },
    ]
    dataset_path.write_text(json.dumps(dataset, ensure_ascii=False), encoding="utf-8")

    class FakePipelineApiClient:
        calls = 0
        observed_timeout: int | None = None

        def __init__(self, base_url: str, timeout_sec: int = 60) -> None:
            self.base_url = base_url
            FakePipelineApiClient.observed_timeout = timeout_sec

        def infer(self, item_id: int | str, mc_id: int, mc_title: str, description: str) -> AvitoApiResponse:
            FakePipelineApiClient.calls += 1
            if FakePipelineApiClient.calls == 1:
                raise PipelineApiClientError("Request failed: timed out")
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
        api_timeout_sec=123,
        prediction_rows=prediction_rows,
    )

    assert FakePipelineApiClient.observed_timeout == 123
    assert metrics.samples_count == 1
    assert metrics.precision_micro == 1.0
    assert metrics.recall_micro == 1.0
    assert metrics.f1_micro == 1.0
    assert metrics.should_split_accuracy == 1.0

    assert len(prediction_rows) == 2
    assert prediction_rows[0]["prediction_error"] == "Request failed: timed out"
    assert prediction_rows[1]["prediction"] == {
        "detectedMcIds": [101, 102],
        "shouldSplit": True,
        "drafts": [],
    }


def test_evaluate_official_dataset_fail_fast_raises_on_timeout(tmp_path, monkeypatch) -> None:
    dataset_path = tmp_path / "official_fail_fast.json"
    dataset = [
        {
            "request": {
                "itemId": 1,
                "mcId": 101,
                "mcTitle": "Категория",
                "description": "Первый",
            },
            "response": {
                "detectedMcIds": [101, 102],
                "shouldSplit": True,
                "drafts": [],
            },
        }
    ]
    dataset_path.write_text(json.dumps(dataset, ensure_ascii=False), encoding="utf-8")

    class FakePipelineApiClient:
        def __init__(self, base_url: str, timeout_sec: int = 60) -> None:
            self.base_url = base_url

        def infer(self, item_id: int | str, mc_id: int, mc_title: str, description: str) -> AvitoApiResponse:
            raise PipelineApiClientError("Request failed: timed out")

    monkeypatch.setattr(
        "avito.should_split.interfaces.jobs.evaluate.PipelineApiClient",
        FakePipelineApiClient,
    )

    with pytest.raises(ValueError, match="Ошибка обработки строки 0"):
        evaluate_official_dataset(
            dataset_path=dataset_path,
            api_url="http://127.0.0.1:8080",
            fail_fast=True,
        )



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
        def __init__(self, base_url: str, timeout_sec: int = 60) -> None:
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
        def __init__(self, base_url: str, timeout_sec: int = 60) -> None:
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


def test_evaluate_official_dataset_supports_legacy_markup_json_rows(tmp_path, monkeypatch) -> None:
    dataset_path = tmp_path / "legacy_markup.json"
    dataset = [
        {
            "itemId": "1000001",
            "sourceMcId": 101,
            "sourceMcTitle": "Ремонт квартир и домов под ключ",
            "description": "Делаем также сантехнические работы",
            "targetDetectedMcIds": "[102]",
            "targetSplitMcIds": "[102]",
            "shouldSplit": True,
            "caseType": "legacy",
            "split": None,
        }
    ]
    dataset_path.write_text(json.dumps(dataset, ensure_ascii=False), encoding="utf-8")

    class FakePipelineApiClient:
        def __init__(self, base_url: str, timeout_sec: int = 60) -> None:
            self.base_url = base_url

        def infer(self, item_id: int | str, mc_id: int, mc_title: str, description: str) -> AvitoApiResponse:
            assert item_id == "1000001"
            assert mc_id == 101
            assert mc_title == "Ремонт квартир и домов под ключ"
            assert description == "Делаем также сантехнические работы"
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
    assert metrics.precision_micro == 1.0
    assert metrics.recall_micro == 1.0
    assert metrics.f1_micro == 1.0
    assert metrics.should_split_accuracy == 1.0
    assert len(prediction_rows) == 1


def test_evaluate_official_dataset_legacy_ignores_target_detected_mc_ids(tmp_path, monkeypatch) -> None:
    dataset_path = tmp_path / "legacy_target_detected_ignored.json"
    dataset = [
        {
            "itemId": "1000180",
            "sourceMcId": 101,
            "sourceMcTitle": "Ремонт квартир и домов под ключ",
            "description": "делаем сантехнику отдельно",
            "targetDetectedMcIds": "[102]",
            "targetSplitMcIds": "[]",
            "shouldSplit": False,
        }
    ]
    dataset_path.write_text(json.dumps(dataset, ensure_ascii=False), encoding="utf-8")

    class FakePipelineApiClient:
        def __init__(self, base_url: str, timeout_sec: int = 60) -> None:
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

    metrics = evaluate_official_dataset(
        dataset_path=dataset_path,
        api_url="http://127.0.0.1:8080",
    )

    # sourceMcId=101 исключается, поэтому остается pred={102}, gt=set().
    assert metrics.samples_count == 1
    assert metrics.false_positive == 1
    assert metrics.true_positive == 0
    assert metrics.false_negative == 0


def test_write_predictions_saves_csv_when_output_has_csv_suffix(tmp_path) -> None:
    output_path = tmp_path / "predictions.csv"
    rows_with_predictions = [
        {
            "itemId": "1",
            "description": "тест",
            "prediction": {
                "detectedMcIds": [101],
                "shouldSplit": True,
                "drafts": [],
            },
        }
    ]

    _write_predictions(output_path=output_path, rows_with_predictions=rows_with_predictions)

    with open(output_path, "r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        rows = list(reader)

    assert output_path.exists()
    assert len(rows) == 1
    assert rows[0]["itemId"] == "1"
    parsed_prediction = json.loads(rows[0]["prediction"])
    assert parsed_prediction["detectedMcIds"] == [101]


