from __future__ import annotations

from pathlib import Path
from typing import *

from src.servers.gradio_demo import parse_args, run
from src.servers.utils.gradio_demo_utils import (
    DummyTrainSample,
    GradioDemoConfig,
    build_gradio_predict_fn,
    query_tabular_pipeline,
    run_dummy_pipeline,
    run_pipeline,
)


def test_gradio_demo_parse_args_uses_yaml_and_cli_override(tmp_path: Path) -> None:
    config_path: Path = tmp_path / "gradio_demo.yaml"
    config_path.write_text(
        "gradio_demo:\n"
        "  host: '0.0.0.0'\n"
        "  port: 7860\n"
        "  share: false\n"
        "  title: 'Demo'\n"
        "  pipeline_base_url: 'http://127.0.0.1:8020'\n"
        "  timeout_sec: 30.0\n",
        encoding="utf-8",
    )

    config: GradioDemoConfig = parse_args(
        [
            "--config-path",
            str(config_path),
            "--host",
            "127.0.0.1",
            "--port",
            "7861",
            "--share",
            "--timeout-sec",
            "15",
        ]
    )

    assert config.host == "127.0.0.1"
    assert config.port == 7861
    assert config.share is True
    assert config.timeout_sec == 15.0
    assert config.pipeline_base_url == "http://127.0.0.1:8020"


def test_query_tabular_pipeline_maps_prediction_to_text(monkeypatch: Any) -> None:
    config = GradioDemoConfig()

    def fake_post_json(url: str, payload: Mapping[str, Any], timeout_sec: float) -> dict[str, Any]:
        assert url.endswith("/v1/tabular/predict")
        assert payload["query"] == "q"
        assert payload["model_answer"] == "a"
        assert payload["threshold"] == config.classification_threshold
        assert timeout_sec == config.timeout_sec
        return {
            "pred_is_hallucination": [1],
            "hallucination_score": [0.91],
            "t_feature_extraction_sec": [0.12],
            "t_classification_sec": [0.03],
            "t_total_sec": [0.15],
        }

    monkeypatch.setattr("src.servers.utils.gradio_demo_utils._post_json", fake_post_json)

    label, score, timings = query_tabular_pipeline(query="q", model_answer="a", config=config)

    assert label == "галлюцинация"
    assert score == 0.91
    assert "feature_extraction=0.1200s" in timings


def test_query_tabular_pipeline_normalizes_zero_host(monkeypatch: Any) -> None:
    config = GradioDemoConfig(pipeline_base_url="http://0.0.0.0:8020")

    def fake_post_json(url: str, payload: Mapping[str, Any], timeout_sec: float) -> dict[str, Any]:
        assert url.startswith("http://127.0.0.1:8020")
        return {"pred_is_hallucination": [0], "hallucination_score": [0.11]}

    monkeypatch.setattr("src.servers.utils.gradio_demo_utils._post_json", fake_post_json)

    label, score, _ = query_tabular_pipeline(query="q", model_answer="a", config=config)

    assert label == "не галлюцинация"
    assert score == 0.11


def test_query_tabular_pipeline_interprets_label_by_score_and_threshold(monkeypatch: Any) -> None:
    config = GradioDemoConfig(classification_threshold=0.5)

    def fake_post_json(url: str, payload: Mapping[str, Any], timeout_sec: float) -> dict[str, Any]:
        return {"pred_is_hallucination": [1], "hallucination_score": [0.1]}

    monkeypatch.setattr("src.servers.utils.gradio_demo_utils._post_json", fake_post_json)

    label, score, _ = query_tabular_pipeline(query="q", model_answer="a", config=config)

    assert label == "не галлюцинация"
    assert score == 0.1


def test_run_pipeline_calls_pipeline_backend(monkeypatch: Any) -> None:
    config = GradioDemoConfig()
    captured: dict[str, Any] = {}

    def fake_query_tabular_pipeline(*, query: str, model_answer: str, config: GradioDemoConfig) -> tuple[str, float | None, str]:
        captured["query"] = query
        captured["model_answer"] = model_answer
        return "не галлюцинация", 0.02, "feature_extraction=0.0100s"

    monkeypatch.setattr("src.servers.utils.gradio_demo_utils.query_tabular_pipeline", fake_query_tabular_pipeline)

    classifier_result, details, timings = run_pipeline(query="как дела", model_answer="ответ", config=config)

    assert captured["query"] == "как дела"
    assert captured["model_answer"] == "ответ"
    assert classifier_result.startswith("не галлюцинация")
    assert "model_answer=ответ" in details
    assert "feature_extraction" in timings


def test_build_gradio_predict_fn_returns_error_text_on_exception(monkeypatch: Any) -> None:
    config = GradioDemoConfig()

    def fake_pipeline(query: str, model_answer: str, config: GradioDemoConfig) -> tuple[str, str, str]:
        raise ValueError("broken")

    monkeypatch.setattr("src.servers.utils.gradio_demo_utils.run_pipeline", fake_pipeline)

    predict = build_gradio_predict_fn(config)
    classifier_result, status, timings = predict("test", "answer")

    assert classifier_result == ""
    assert "Ошибка:" in status
    assert timings == ""


def test_run_dummy_pipeline_uses_random_train_sample(monkeypatch: Any) -> None:
    config = GradioDemoConfig(dummy_mode=True)
    captured: dict[str, Any] = {}

    monkeypatch.setattr(
        "src.servers.utils.gradio_demo_utils.pick_dummy_train_sample",
        lambda _: DummyTrainSample(
            query="q_train",
            model_answer="a_train",
            correct_answer="c_train",
            features={"mean_log_prob": -0.1, "n_answer_tokens": 3.0},
        ),
    )

    def fake_query_tabular_pipeline(**kwargs: Any) -> tuple[str, float | None, str]:
        captured.update(kwargs)
        return "не галлюцинация", 0.07, "classification=0.0200s"

    monkeypatch.setattr("src.servers.utils.gradio_demo_utils.query_tabular_pipeline", fake_query_tabular_pipeline)

    query, model_answer, correct_answer, classifier_result, details, timings = run_dummy_pipeline(config=config)

    assert query == "q_train"
    assert model_answer == "a_train"
    assert correct_answer == "c_train"
    assert "score=0.0700" in classifier_result
    assert "dummy_mode=true" in details
    assert captured["precomputed_features"]["mean_log_prob"] == -0.1
    assert "classification" in timings


def test_run_launches_gradio_app(monkeypatch: Any) -> None:
    config = GradioDemoConfig(host="127.0.0.1", port=7862, share=False)
    captured: dict[str, Any] = {}

    class DummyApp:
        def launch(self, *, server_name: str, server_port: int, share: bool) -> None:
            captured["server_name"] = server_name
            captured["server_port"] = server_port
            captured["share"] = share

    def fake_build_gradio_app(app_config: GradioDemoConfig) -> DummyApp:
        assert app_config.host == "127.0.0.1"
        return DummyApp()

    monkeypatch.setattr("src.servers.gradio_demo.build_gradio_app", fake_build_gradio_app)

    run(config)

    assert captured["server_name"] == "127.0.0.1"
    assert captured["server_port"] == 7862
    assert captured["share"] is False


def test_run_preloads_dummy_samples_in_dummy_mode(monkeypatch: Any) -> None:
    config = GradioDemoConfig(host="127.0.0.1", port=7863, share=False, dummy_mode=True)
    captured: dict[str, Any] = {"preloaded": False}

    class DummyApp:
        def launch(self, *, server_name: str, server_port: int, share: bool) -> None:
            return None

    monkeypatch.setattr("src.servers.gradio_demo.build_gradio_app", lambda _: DummyApp())
    monkeypatch.setattr(
        "src.servers.gradio_demo.preload_dummy_samples",
        lambda _: captured.update({"preloaded": True}),
    )

    run(config)

    assert captured["preloaded"] is True


