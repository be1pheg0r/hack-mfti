from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import *

from common.configs import load_pydantic_config
from common.logger import SBER_EVALUATION_LOGGER as logger
from common.paths import get_servers_configs_dpath
from src.servers.utils.gradio_demo_utils import GradioDemoConfig
from src.servers.utils.tabular_pipeline_utils import TabularPipelineServerConfig


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Парсит аргументы orchestration-скрипта серверов."""
    parser = argparse.ArgumentParser(description="Инициализация серверного стека проекта")
    parser.add_argument(
        "--tabular-config-path",
        type=str,
        default=str(Path(get_servers_configs_dpath()) / "tabular_pipeline_server.yaml"),
    )
    parser.add_argument(
        "--gradio-config-path",
        type=str,
        default=str(Path(get_servers_configs_dpath()) / "gradio_demo.yaml"),
    )
    parser.add_argument("--without-gradio", action="store_true", help="Не запускать Gradio UI")
    parser.add_argument(
        "--dummy",
        action="store_true",
        help="Запустить tabular и Gradio поверх dummy tabular backend без тяжелых моделей",
    )
    parser.add_argument("--dry-run", action="store_true", help="Только вывести конфиги и команды запуска")
    return parser.parse_args(argv)


def build_stack_info(
    tabular_config: TabularPipelineServerConfig,
    gradio_config: GradioDemoConfig,
    *,
    with_gradio: bool,
) -> dict[str, str]:
    """Формирует сводку по серверному стеку."""
    info: dict[str, str] = {
        "tabular_server_url": f"http://{tabular_config.host}:{tabular_config.port}",
        "tabular_health": f"http://{tabular_config.host}:{tabular_config.port}/health",
        "tabular_predict": f"http://{tabular_config.host}:{tabular_config.port}/v1/tabular/predict",
        "feature_extractor_mode": tabular_config.feature_extractor_mode,
        "tabular_serve_mode": tabular_config.serve_mode,
        "tabular_checkpoint": str(tabular_config.checkpoint_dir),
    }
    if with_gradio:
        info["gradio_url"] = f"http://{gradio_config.host}:{gradio_config.port}"
    return info


def print_stack_info(info: Mapping[str, str], *, dry_run: bool, with_gradio: bool) -> None:
    """Печатает в консоль ключевую информацию по запуску серверов."""
    print("=" * 88)
    print("ИНИЦИАЛИЗАЦИЯ СЕРВЕРОВ")
    print("=" * 88)
    print(f"Tabular server        : {info['tabular_server_url']}")
    print(f"Health endpoint       : {info['tabular_health']}")
    print(f"Predict endpoint      : {info['tabular_predict']}")
    print(f"Serve mode            : {info['tabular_serve_mode']}")
    print(f"Feature extractor mode: {info['feature_extractor_mode']}")
    print(f"Checkpoint dir        : {info['tabular_checkpoint']}")
    if with_gradio:
        print(f"Gradio UI             : {info['gradio_url']}")
    print(f"Dry run               : {dry_run}")
    print("=" * 88)


def _start_process(command: list[str], process_name: str) -> subprocess.Popen[Any]:
    logger.info("Запускаю %s: %s", process_name, " ".join(command))
    return subprocess.Popen(command)


def _apply_dummy_mode(
    tabular_config: TabularPipelineServerConfig,
    gradio_config: GradioDemoConfig,
    *,
    enabled: bool,
) -> tuple[TabularPipelineServerConfig, GradioDemoConfig]:
    """Возвращает эффективные конфиги с принудительным dummy-режимом."""
    if not enabled:
        return tabular_config, gradio_config

    effective_tabular: TabularPipelineServerConfig = tabular_config.model_copy(
        update={
            "serve_mode": "dummy",
            "feature_extractor_mode": "dummy",
        }
    )
    client_host: str = "127.0.0.1" if effective_tabular.host == "0.0.0.0" else effective_tabular.host
    effective_gradio: GradioDemoConfig = gradio_config.model_copy(
        update={
            "pipeline_base_url": f"http://{client_host}:{effective_tabular.port}",
            "dummy_mode": True,
        }
    )
    return effective_tabular, effective_gradio


def run(argv: Sequence[str] | None = None) -> None:
    """Инициализирует серверы и опционально запускает их."""
    args: argparse.Namespace = parse_args(argv)
    with_gradio: bool = not bool(args.without_gradio)

    tabular_config: TabularPipelineServerConfig = load_pydantic_config(
        config_cls=TabularPipelineServerConfig,
        fpath=args.tabular_config_path,
        section_name="tabular_pipeline_server",
    )
    gradio_config: GradioDemoConfig = load_pydantic_config(
        config_cls=GradioDemoConfig,
        fpath=args.gradio_config_path,
        section_name="gradio_demo",
    )

    tabular_config, gradio_config = _apply_dummy_mode(
        tabular_config=tabular_config,
        gradio_config=gradio_config,
        enabled=bool(args.dummy),
    )

    info = build_stack_info(tabular_config=tabular_config, gradio_config=gradio_config, with_gradio=with_gradio)
    print_stack_info(info, dry_run=bool(args.dry_run), with_gradio=with_gradio)

    if args.dry_run:
        return

    processes: list[subprocess.Popen[Any]] = []
    try:
        tabular_cmd: list[str] = [
            sys.executable,
            str(Path("scripts") / "tabular_pipeline_server.py"),
            "--config-path",
            str(args.tabular_config_path),
        ]
        if bool(args.dummy):
            tabular_cmd.extend(["--serve-mode", "dummy", "--feature-extractor-mode", "dummy"])
        processes.append(_start_process(tabular_cmd, "tabular_pipeline_server"))

        if with_gradio:
            gradio_cmd: list[str] = [
                sys.executable,
                "-m",
                "src.servers.gradio_demo",
                "--config-path",
                str(args.gradio_config_path),
            ]
            if bool(args.dummy):
                gradio_cmd.extend(["--pipeline-base-url", gradio_config.pipeline_base_url, "--dummy-mode"])
            processes.append(_start_process(gradio_cmd, "gradio_demo"))

        print("Серверы запущены. Для остановки нажмите Ctrl+C.")
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("Останавливаю серверы...")
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            try:
                process.wait(timeout=10)
            except Exception:
                process.kill()


if __name__ == "__main__":
    run()

