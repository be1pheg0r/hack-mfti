from __future__ import annotations

from typing import *

import pandas as pd
import torch

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


def test_llm_feature_extractor_backend_uses_single_forward_per_batch() -> None:
    class _FakeTokenizer:
        pad_token_id: int = 0
        eos_token_id: int = 0

        def __call__(self, text: str, return_tensors: str = "pt", add_special_tokens: bool = True) -> dict[str, torch.Tensor]:
            _ = (return_tensors, add_special_tokens)
            token_count: int = max(len(str(text).split()), 1)
            return {"input_ids": torch.arange(1, token_count + 1, dtype=torch.long).unsqueeze(0)}

    class _FakeGroups:
        def __init__(self) -> None:
            self.uncertainty = [0.0] * len(TabularPipelineService.__mro__)
            self.internal_scalars = []
            self.probe_vec = []
            self.attention_entropy = []
            self.entropy_drops = []
            self.moe_routing = []

    class _FakeExtractor:
        def __init__(self) -> None:
            self.forward_calls: int = 0
            self._hidden: dict[str, torch.Tensor] = {}

        def __enter__(self) -> "_FakeExtractor":
            return self

        def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
            _ = (exc_type, exc_val, exc_tb)

        def __call__(self, token_ids: torch.Tensor) -> dict[str, torch.Tensor]:
            self.forward_calls += 1
            batch, seq = int(token_ids.shape[0]), int(token_ids.shape[1])
            return {"logits": torch.zeros((batch, seq, 8), dtype=torch.float32)}

        def extract(
            self,
            logits: torch.Tensor,
            input_ids: torch.Tensor,
            answer_start: int,
            hidden_batch_index: int = 0,
            clear_hidden: bool = True,
        ) -> Any:
            _ = (logits, input_ids, answer_start, hidden_batch_index, clear_hidden)
            return _FakeGroups()

    backend: LLMFeatureExtractorBackend = LLMFeatureExtractorBackend.__new__(LLMFeatureExtractorBackend)
    backend.config = TabularPipelineServerConfig(feature_batch_size=2)
    backend._device = torch.device("cpu")
    backend._tokenizer = _FakeTokenizer()
    fake_extractor = _FakeExtractor()
    backend._extractor = fake_extractor

    backend.build_features(
        queries=["q1", "q2", "q3", "q4", "q5"],
        model_answers=["a1", "a2", "a3", "a4", "a5"],
    )

    assert fake_extractor.forward_calls == 3


def test_llm_feature_extractor_backend_load_tokenizer_with_not_implemented_fallback(monkeypatch: Any) -> None:
    calls: list[dict[str, Any]] = []

    def fake_from_pretrained(source: str, **kwargs: Any) -> Any:
        _ = source
        calls.append(dict(kwargs))
        if len(calls) == 1:
            raise NotImplementedError()
        return {"ok": True, "kwargs": kwargs}

    monkeypatch.setattr("src.servers.utils.tabular_pipeline_utils.AutoTokenizer.from_pretrained", fake_from_pretrained)

    backend: LLMFeatureExtractorBackend = LLMFeatureExtractorBackend.__new__(LLMFeatureExtractorBackend)
    tokenizer = backend._load_tokenizer_with_fallback("dummy/model")

    assert tokenizer["ok"] is True
    assert len(calls) == 2


def test_llm_feature_extractor_backend_load_tokenizer_with_all_fallbacks_failed(monkeypatch: Any) -> None:
    def fake_from_pretrained(source: str, **kwargs: Any) -> Any:
        _ = (source, kwargs)
        raise NotImplementedError()

    monkeypatch.setattr("src.servers.utils.tabular_pipeline_utils.AutoTokenizer.from_pretrained", fake_from_pretrained)

    backend: LLMFeatureExtractorBackend = LLMFeatureExtractorBackend.__new__(LLMFeatureExtractorBackend)

    try:
        backend._load_tokenizer_with_fallback("dummy/model")
        assert False, "Expected RuntimeError"
    except RuntimeError:
        assert True



