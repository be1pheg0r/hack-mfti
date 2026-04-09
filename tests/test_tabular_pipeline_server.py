from __future__ import annotations

from typing import *

import pandas as pd

from src.servers.utils.tabular_pipeline_utils import (
    DummyFeatureExtractorBackend,
    LLMFeatureExtractorBackend,
    TabularPipelineRequest,
    TabularPipelineServerConfig,
    TabularPipelineService,
    parse_tabular_pipeline_request,
)


def test_parse_tabular_pipeline_request_supports_single_pair_payload() -> None:
    request = parse_tabular_pipeline_request({"query": "q", "model_answer": "a", "threshold": 0.7})

    assert request.queries == ["q"]
    assert request.model_answers == ["a"]
    assert request.threshold == 0.7


def test_parse_tabular_pipeline_request_supports_single_features_payload() -> None:
    request = parse_tabular_pipeline_request(
        {
            "query": "q",
            "model_answer": "a",
            "features": {"mean_log_prob": -0.2, "n_answer_tokens": 4.0},
        }
    )

    assert request.queries == ["q"]
    assert request.model_answers == ["a"]
    assert request.precomputed_features is not None
    assert request.precomputed_features[0]["mean_log_prob"] == -0.2


def test_tabular_pipeline_service_dummy_mode_returns_expected_contract() -> None:
    config = TabularPipelineServerConfig(serve_mode="dummy", feature_extractor_mode="dummy")
    service = TabularPipelineService(config=config)

    request = TabularPipelineRequest(queries=["q1", "q2"], model_answers=["a1", ""], threshold=0.5)
    response = service.predict(request)

    assert set(response.keys()) == {
        "pred_is_hallucination",
        "hallucination_score",
        "entailment_score",
        "t_feature_extraction_sec",
        "t_classification_sec",
        "t_total_sec",
        "t_sample_sec",
    }
    assert len(response["pred_is_hallucination"]) == 2
    assert len(response["hallucination_score"]) == 2
    assert len(response["entailment_score"]) == 2
    assert len(response["t_feature_extraction_sec"]) == 2
    assert len(response["t_classification_sec"]) == 2
    assert len(response["t_total_sec"]) == 2
    assert len(response["t_sample_sec"]) == 2


def test_tabular_pipeline_service_dummy_mode_does_not_initialize_llm_backend(monkeypatch: Any) -> None:
    def fail_if_called(self: object, config: object) -> None:
        raise AssertionError("LLM backend should not be initialized in dummy serve_mode")

    monkeypatch.setattr(
        "src.servers.utils.tabular_pipeline_utils.LLMFeatureExtractorBackend.__init__",
        fail_if_called,
    )

    service = TabularPipelineService(TabularPipelineServerConfig(serve_mode="dummy"))
    response = service.predict(TabularPipelineRequest(queries=["q"], model_answers=["a"], threshold=0.5))

    assert response["pred_is_hallucination"] == [0]


def test_dummy_feature_extractor_backend_builds_feature_dataframe() -> None:
    backend = DummyFeatureExtractorBackend()
    features_df = backend.build_features(queries=["q"], model_answers=["a"])

    assert isinstance(features_df, pd.DataFrame)
    assert len(features_df) == 1
    assert "query" in features_df.columns
    assert "model_answer" in features_df.columns
    assert "n_answer_tokens" in features_df.columns


def test_tabular_pipeline_service_uses_dummy_extractor_in_tabular_mode(monkeypatch: Any) -> None:
    def fake_classify(self: object, features_df: pd.DataFrame, threshold: float | None = None) -> pd.DataFrame:
        result = features_df.copy()
        result["hallucination_score"] = [0.1 for _ in range(len(features_df))]
        result["entailment_score"] = [0.9 for _ in range(len(features_df))]
        result["pred_is_hallucination"] = [0 for _ in range(len(features_df))]
        return result

    monkeypatch.setattr(
        "src.servers.utils.tabular_pipeline_utils.TabularClassifierBackend.classify",
        fake_classify,
    )

    config = TabularPipelineServerConfig(serve_mode="tabular_pipeline", feature_extractor_mode="dummy")
    service = TabularPipelineService(config=config)
    response = service.predict(TabularPipelineRequest(queries=["q"], model_answers=["a"], threshold=0.5))

    assert response["pred_is_hallucination"] == [0]
    assert response["hallucination_score"] == [0.1]


def test_tabular_pipeline_service_dummy_mode_uses_classifier_for_precomputed_features(monkeypatch: Any) -> None:
    def fake_classify(self: object, features_df: pd.DataFrame, threshold: float | None = None) -> pd.DataFrame:
        assert "mean_log_prob" in features_df.columns
        result = features_df.copy()
        result["hallucination_score"] = [0.77 for _ in range(len(features_df))]
        result["entailment_score"] = [0.23 for _ in range(len(features_df))]
        result["pred_is_hallucination"] = [1 for _ in range(len(features_df))]
        return result

    monkeypatch.setattr(
        "src.servers.utils.tabular_pipeline_utils.TabularClassifierBackend.classify",
        fake_classify,
    )

    service = TabularPipelineService(TabularPipelineServerConfig(serve_mode="dummy"))
    response = service.predict(
        TabularPipelineRequest(
            queries=["q"],
            model_answers=["a"],
            threshold=0.5,
            precomputed_features=[{"mean_log_prob": -0.1, "n_answer_tokens": 3.0}],
        )
    )

    assert response["pred_is_hallucination"] == [1]
    assert response["hallucination_score"] == [0.77]


def test_llm_feature_extractor_backend_batches_pairs_by_configured_size() -> None:
    backend: LLMFeatureExtractorBackend = LLMFeatureExtractorBackend.__new__(LLMFeatureExtractorBackend)
    backend.config = TabularPipelineServerConfig(feature_batch_size=2)

    batches = list(
        backend._batched_pairs(
            queries=["q1", "q2", "q3", "q4", "q5"],
            model_answers=["a1", "a2", "a3", "a4", "a5"],
        )
    )

    assert [len(batch_queries) for batch_queries, _ in batches] == [2, 2, 1]



