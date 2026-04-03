from __future__ import annotations

import argparse
from pathlib import Path
from typing import *

from src.sber.constants import DEFAULT_HF_NLI_SERVER_CONFIG_FPATH
from src.servers.hf_nli_server import parse_args, run
from src.servers.utils.hf_nli_utils import (
    DummyHFNLIModel,
    HFNLIServerConfig,
    apply_cli_overrides,
    load_hf_nli_server_config,
)


def test_hf_nli_server_config_from_yaml_reads_nested_section(tmp_path: Path) -> None:
    config_path: Path = tmp_path / "hf_nli_server.yaml"
    config_path.write_text(
        "hf_nli_server:\n"
        "  serve_mode: 'hf_nli'\n"
        "  host: '127.0.0.1'\n"
        "  port: 9010\n"
        "  repo_id: 'be1pheg0r/hack-mfti-sbercase'\n"
        "  nli_config_path: 'configs/sber/hf_nli_clf.yaml'\n"
        "  compute_dtype: 'float32'\n"
        "  batch_size: 4\n"
        "  trust_remote_code: false\n",
        encoding="utf-8",
    )

    config: HFNLIServerConfig = load_hf_nli_server_config(config_path)

    assert config.serve_mode == "hf_nli"
    assert config.host == "127.0.0.1"
    assert config.port == 9010
    assert config.repo_id == "be1pheg0r/hack-mfti-sbercase"
    assert Path(config.nli_config_path) == Path("configs/sber/hf_nli_clf.yaml")
    assert config.compute_dtype == "float32"
    assert config.batch_size == 4
    assert config.trust_remote_code is False


def test_dummy_hf_nli_model_predict_returns_expected_shape() -> None:
    model: DummyHFNLIModel = DummyHFNLIModel()

    payload = model.predict(
        premises=["Москва - столица России", "2+2=4"],
        hypotheses=["Москва - столица России", "2+2=5"],
    )

    assert payload["pred_is_hallucination"] == [0, 1]
    assert len(payload["hallucination_score"]) == 2
    assert len(payload["entailment_score"]) == 2


def test_apply_cli_overrides_ignores_none_values() -> None:
    config = HFNLIServerConfig(
        serve_mode="hf_nli",
        host="0.0.0.0",
        port=8010,
        repo_id="base/model",
        nli_config_path="configs/sber/hf_nli_clf.yaml",
        compute_dtype="float32",
        batch_size=8,
        trust_remote_code=True,
    )

    namespace = argparse.Namespace(
        config_path=str(DEFAULT_HF_NLI_SERVER_CONFIG_FPATH),
        serve_mode=None,
        host="127.0.0.1",
        port=None,
        repo_id=None,
        nli_config_path=None,
        compute_dtype=None,
        batch_size=16,
        trust_remote_code=None,
    )

    merged = apply_cli_overrides(config=config, namespace=namespace)

    assert merged.host == "127.0.0.1"
    assert merged.port == 8010
    assert merged.batch_size == 16
    assert merged.repo_id == "base/model"


def test_parse_args_uses_yaml_defaults_and_cli_overrides(tmp_path: Path) -> None:
    config_path: Path = tmp_path / "hf_nli_server.yaml"
    config_path.write_text(
        "hf_nli_server:\n"
        "  serve_mode: 'hf_nli'\n"
        "  host: '0.0.0.0'\n"
        "  port: 8010\n"
        "  repo_id: 'base/model'\n"
        "  nli_config_path: 'configs/sber/hf_nli_clf.yaml'\n"
        "  compute_dtype: 'float32'\n"
        "  batch_size: 8\n"
        "  trust_remote_code: true\n",
        encoding="utf-8",
    )

    config: HFNLIServerConfig = parse_args(
        [
            "--config-path",
            str(config_path),
            "--host",
            "127.0.0.1",
            "--port",
            "9011",
            "--batch-size",
            "16",
            "--no-trust-remote-code",
        ]
    )

    assert config.host == "127.0.0.1"
    assert config.port == 9011
    assert config.batch_size == 16
    assert config.trust_remote_code is False


def test_run_dispatches_dummy_mode(monkeypatch: Any) -> None:
    config = HFNLIServerConfig(
        serve_mode="dummy",
        host="127.0.0.1",
        port=8010,
        repo_id="dummy-hf-nli",
        nli_config_path="configs/sber/hf_nli_clf.yaml",
        compute_dtype="float32",
        batch_size=8,
    )
    captured: dict[str, Any] = {}

    def fake_dummy_server(dummy_config: HFNLIServerConfig) -> None:
        captured["config"] = dummy_config

    monkeypatch.setattr("src.servers.hf_nli_server.run_dummy_hf_nli_server", fake_dummy_server)

    run(config=config)

    assert captured["config"].serve_mode == "dummy"


def test_run_dispatches_hf_mode(monkeypatch: Any) -> None:
    config = HFNLIServerConfig(
        serve_mode="hf_nli",
        host="127.0.0.1",
        port=8010,
        repo_id="be1pheg0r/hack-mfti-sbercase",
        nli_config_path="configs/sber/hf_nli_clf.yaml",
        compute_dtype="float32",
        batch_size=8,
    )
    captured: dict[str, Any] = {}

    def fake_hf_server(prod_config: HFNLIServerConfig) -> None:
        captured["config"] = prod_config

    monkeypatch.setattr("src.servers.hf_nli_server.run_hf_nli_server", fake_hf_server)

    run(config=config)

    assert captured["config"].serve_mode == "hf_nli"
    assert captured["config"].repo_id == "be1pheg0r/hack-mfti-sbercase"

