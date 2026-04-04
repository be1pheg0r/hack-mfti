from __future__ import annotations

from pathlib import Path
from typing import *

from src.servers.gradio_demo import parse_args, run
from src.servers.utils.gradio_demo_utils import (
    GradioDemoConfig,
    build_gradio_predict_fn,
    query_hf_nli_label,
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
        "  vllm_base_url: 'http://127.0.0.1:8000'\n"
        "  vllm_model_name: 'dummy-vllm'\n"
        "  hf_nli_base_url: 'http://127.0.0.1:8010'\n"
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
    assert config.vllm_model_name == "dummy-vllm"


def test_query_hf_nli_label_maps_prediction_to_text(monkeypatch: Any) -> None:
    config = GradioDemoConfig()

    def fake_post_json(url: str, payload: Mapping[str, Any], timeout_sec: float) -> dict[str, Any]:
        assert url.endswith("/v1/hf-nli/predict")
        assert payload["premises"] == ["q"]
        assert payload["hypotheses"] == ["a"]
        assert timeout_sec == config.timeout_sec
        return {"pred_is_hallucination": [1], "hallucination_score": [0.91]}

    monkeypatch.setattr("src.servers.utils.gradio_demo_utils._post_json", fake_post_json)

    label, score = query_hf_nli_label(premise="q", hypothesis="a", config=config)

    assert label == "галлюцинация"
    assert score == 0.91


def test_run_pipeline_calls_both_backends(monkeypatch: Any) -> None:
    config = GradioDemoConfig()
    captured: dict[str, Any] = {}

    def fake_query_vllm_answer(*, query: str, config: GradioDemoConfig) -> str:
        captured["query"] = query
        return "vllm-answer"

    def fake_query_hf_nli_label(*, premise: str, hypothesis: str, config: GradioDemoConfig) -> tuple[str, float | None]:
        captured["premise"] = premise
        captured["hypothesis"] = hypothesis
        return "не галлюцинация", 0.02

    monkeypatch.setattr("src.servers.utils.gradio_demo_utils.query_vllm_answer", fake_query_vllm_answer)
    monkeypatch.setattr("src.servers.utils.gradio_demo_utils.query_hf_nli_label", fake_query_hf_nli_label)

    vllm_answer, nli_result = run_pipeline(query="как дела", config=config)

    assert captured["query"] == "как дела"
    assert captured["premise"] == "как дела"
    assert captured["hypothesis"] == "vllm-answer"
    assert vllm_answer == "vllm-answer"
    assert nli_result.startswith("не галлюцинация")


def test_build_gradio_predict_fn_returns_error_text_on_exception(monkeypatch: Any) -> None:
    config = GradioDemoConfig()

    def fake_pipeline(query: str, config: GradioDemoConfig) -> tuple[str, str]:
        raise ValueError("broken")

    monkeypatch.setattr("src.servers.utils.gradio_demo_utils.run_pipeline", fake_pipeline)

    predict = build_gradio_predict_fn(config)
    vllm_answer, status = predict("test")

    assert vllm_answer == ""
    assert "Ошибка:" in status


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

