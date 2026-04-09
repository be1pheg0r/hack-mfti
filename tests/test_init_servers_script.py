from __future__ import annotations

import argparse

from src.servers.utils.gradio_demo_utils import GradioDemoConfig
from src.servers.utils.tabular_pipeline_utils import TabularPipelineServerConfig
from scripts.init_servers import _apply_dummy_mode, build_stack_info, parse_args


def test_build_stack_info_contains_expected_urls() -> None:
    tabular_config = TabularPipelineServerConfig(host="127.0.0.1", port=8021, feature_extractor_mode="dummy")
    gradio_config = GradioDemoConfig(host="127.0.0.1", port=7861)

    info = build_stack_info(tabular_config=tabular_config, gradio_config=gradio_config, with_gradio=True)

    assert info["tabular_server_url"] == "http://127.0.0.1:8021"
    assert info["tabular_health"].endswith("/health")
    assert info["tabular_predict"].endswith("/v1/tabular/predict")
    assert info["feature_extractor_mode"] == "dummy"
    assert info["gradio_url"] == "http://127.0.0.1:7861"


def test_parse_args_supports_dummy_flag() -> None:
    args: argparse.Namespace = parse_args(["--dummy", "--without-gradio"])

    assert bool(args.dummy) is True
    assert bool(args.without_gradio) is True


def test_parse_args_supports_share_flag() -> None:
    args: argparse.Namespace = parse_args(["--share"])

    assert bool(args.share) is True


def test_apply_dummy_mode_overrides_tabular_and_gradio_pipeline_url() -> None:
    tabular_config = TabularPipelineServerConfig(
        host="127.0.0.1",
        port=8021,
        serve_mode="tabular_pipeline",
        feature_extractor_mode="real",
    )
    gradio_config = GradioDemoConfig(host="127.0.0.1", port=7861, pipeline_base_url="http://example.local:9999")

    effective_tabular, effective_gradio = _apply_dummy_mode(
        tabular_config=tabular_config,
        gradio_config=gradio_config,
        enabled=True,
    )

    assert effective_tabular.serve_mode == "dummy"
    assert effective_tabular.feature_extractor_mode == "dummy"
    assert effective_gradio.pipeline_base_url == "http://127.0.0.1:8021"
    assert effective_gradio.dummy_mode is True


def test_apply_dummy_mode_maps_0_0_0_0_to_localhost_for_gradio_client() -> None:
    tabular_config = TabularPipelineServerConfig(host="0.0.0.0", port=8020)
    gradio_config = GradioDemoConfig(host="0.0.0.0", port=7860)

    _, effective_gradio = _apply_dummy_mode(
        tabular_config=tabular_config,
        gradio_config=gradio_config,
        enabled=True,
    )

    assert effective_gradio.pipeline_base_url == "http://127.0.0.1:8020"


