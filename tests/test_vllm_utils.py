from __future__ import annotations

import argparse
from pathlib import Path
from typing import *

import pytest

from src.sber.constants import DEFAULT_VLLM_SERVE_CONFIG_FPATH
from src.servers.utils.vllm_utils import (
    DummyVLLMModel,
    VLLMServerConfig,
    apply_cli_overrides,
    build_dummy_vllm_model,
    build_vllm_serve_command,
    load_vllm_server_config,
)
from src.servers.vllm_server import parse_args, run


def test_vllm_server_config_from_yaml_reads_nested_section(tmp_path: Path) -> None:
    config_path: Path = tmp_path / "vllm_server.yaml"
    config_path.write_text(
        "vllm_server:\n"
        "  model_name: 'custom/model'\n"
        "  host: '127.0.0.1'\n"
        "  port: 9000\n"
        "  dtype: 'float16'\n"
        "  tensor_parallel_size: 2\n"
        "  gpu_memory_utilization: 0.75\n"
        "  max_model_len: 4096\n"
        "  trust_remote_code: false\n"
        "  download_dir: 'C:/models'\n"
        "  served_model_name: 'custom-serving-name'\n",
        encoding="utf-8",
    )

    config: VLLMServerConfig = load_vllm_server_config(config_path)

    assert config.model_name == "custom/model"
    assert config.host == "127.0.0.1"
    assert config.port == 9000
    assert config.dtype == "float16"
    assert config.tensor_parallel_size == 2
    assert config.gpu_memory_utilization == pytest.approx(0.75)
    assert config.max_model_len == 4096
    assert config.trust_remote_code is False
    assert Path(config.download_dir) == Path("C:/models")
    assert config.served_model_name == "custom-serving-name"


def test_dummy_vllm_model_builds_openai_like_completion() -> None:
    model = DummyVLLMModel(model_name="dummy-vllm", response_text="Dummy reply", served_model_name="dummy-alias")

    payload = model.build_chat_completion(
        messages=[
            {"role": "system", "content": "You are a helpful bot."},
            {"role": "user", "content": "Hello world"},
        ]
    )

    assert payload["object"] == "chat.completion"
    assert payload["model"] == "dummy-alias"
    assert payload["choices"][0]["message"]["content"] == "Dummy reply Echo: Hello world"
    assert payload["usage"]["total_tokens"] >= payload["usage"]["completion_tokens"]


def test_build_dummy_vllm_model_uses_dummy_defaults_for_dummy_mode() -> None:
    config = VLLMServerConfig(
        model_name="ai-sage/GigaChat3-10B-A1.8B-bf16",
        serve_mode="dummy",
        served_model_name="dummy-alias",
    )

    dummy_model = build_dummy_vllm_model(config=config)

    assert dummy_model.model_name == "dummy-vllm"
    assert dummy_model.api_model_name == "dummy-alias"


def test_apply_cli_overrides_ignores_none_values() -> None:
    config = VLLMServerConfig(
        model_name="base/model",
        host="0.0.0.0",
        port=8000,
        dtype="bfloat16",
    )
    namespace = argparse.Namespace(
        config_path=str(DEFAULT_VLLM_SERVE_CONFIG_FPATH),
        model_name=None,
        host="127.0.0.1",
        port=None,
        dtype=None,
        tensor_parallel_size=4,
        gpu_memory_utilization=None,
        max_model_len=None,
        download_dir=None,
        served_model_name=None,
        trust_remote_code=False,
    )

    merged = apply_cli_overrides(config=config, namespace=namespace)

    assert merged.model_name == "base/model"
    assert merged.host == "127.0.0.1"
    assert merged.port == 8000
    assert merged.tensor_parallel_size == 4
    assert merged.trust_remote_code is False


def test_build_vllm_serve_command_omits_unset_optional_args() -> None:
    config = VLLMServerConfig(
        model_name="ai-sage/GigaChat3-10B-A1.8B-bf16",
        host="0.0.0.0",
        port=8000,
        dtype="bfloat16",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
        max_model_len=None,
        trust_remote_code=True,
        download_dir=None,
        served_model_name=None,
    )

    command = build_vllm_serve_command(config=config)

    assert command[:3] == ["vllm", "serve", "ai-sage/GigaChat3-10B-A1.8B-bf16"]
    assert "--max-model-len" not in command
    assert "--download-dir" not in command
    assert "--served-model-name" not in command
    assert "--trust-remote-code" in command


def test_parse_args_uses_yaml_defaults_and_cli_overrides(tmp_path: Path) -> None:
    config_path: Path = tmp_path / "vllm_server.yaml"
    config_path.write_text(
        "vllm_server:\n"
        "  model_name: 'base/model'\n"
        "  host: '0.0.0.0'\n"
        "  port: 8000\n"
        "  dtype: 'bfloat16'\n"
        "  tensor_parallel_size: 1\n"
        "  gpu_memory_utilization: 0.9\n"
        "  max_model_len: 4096\n"
        "  trust_remote_code: true\n",
        encoding="utf-8",
    )

    config = parse_args(
        [
            "--config-path",
            str(config_path),
            "--host",
            "127.0.0.1",
            "--port",
            "9000",
            "--no-trust-remote-code",
        ]
    )

    assert config.model_name == "base/model"
    assert config.host == "127.0.0.1"
    assert config.port == 9000
    assert config.max_model_len == 4096
    assert config.trust_remote_code is False


def test_parse_args_accepts_dummy_mode(tmp_path: Path) -> None:
    config_path: Path = tmp_path / "vllm_server.yaml"
    config_path.write_text(
        "vllm_server:\n"
        "  serve_mode: 'dummy'\n"
        "  model_name: 'dummy-vllm'\n"
        "  host: '127.0.0.1'\n"
        "  port: 8000\n"
        "  dtype: 'bfloat16'\n",
        encoding="utf-8",
    )

    config = parse_args(["--config-path", str(config_path), "--serve-mode", "dummy"])

    assert config.serve_mode == "dummy"
    assert config.model_name == "dummy-vllm"


def test_run_builds_command_without_invoking_real_subprocess(monkeypatch: Any) -> None:
    config = VLLMServerConfig(
        model_name="base/model",
        host="0.0.0.0",
        port=8000,
        dtype="bfloat16",
    )
    captured: dict[str, Any] = {}

    def fake_run(command: list[str], check: bool, text: bool) -> Any:
        captured["command"] = command
        captured["check"] = check
        captured["text"] = text
        return argparse.Namespace(returncode=0)

    monkeypatch.setattr("src.servers.vllm_server.subprocess.run", fake_run)

    result = run(config=config)

    assert captured["command"][:3] == ["vllm", "serve", "base/model"]
    assert captured["check"] is True
    assert captured["text"] is True
    assert result.returncode == 0


def test_run_dispatches_dummy_mode_without_subprocess(monkeypatch: Any) -> None:
    config = VLLMServerConfig(
        model_name="dummy-vllm",
        serve_mode="dummy",
        host="127.0.0.1",
        port=8000,
        dtype="bfloat16",
    )
    captured: dict[str, Any] = {}

    def fake_dummy_server(dummy_config: VLLMServerConfig) -> None:
        captured["config"] = dummy_config

    monkeypatch.setattr("src.servers.vllm_server.run_dummy_vllm_server", fake_dummy_server)

    result = run(config=config)

    assert captured["config"].serve_mode == "dummy"
    assert result is None





