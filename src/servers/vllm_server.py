from __future__ import annotations

import argparse
import subprocess
from typing import *

from common.logger import VLLM_SERVER_LOGGER as logger
from common.configs import load_config_from_namespace
from src.sber.constants import DEFAULT_VLLM_SERVE_CONFIG_FPATH
from src.servers.utils.vllm_utils import (
    VLLMServerConfig,
    build_vllm_serve_command,
    run_dummy_vllm_server,
)


def parse_args(argv: Sequence[str] | None = None) -> VLLMServerConfig:
    """Парсит CLI и возвращает итоговую конфигурацию сервера."""
    parser = argparse.ArgumentParser(description="Запуск vLLM server для Sber-кейса")
    parser.add_argument(
        "--config-path",
        type=str,
        default=str(DEFAULT_VLLM_SERVE_CONFIG_FPATH),
        help="Путь до YAML-конфига сервера",
    )
    parser.add_argument(
        "--serve-mode",
        type=str,
        choices=("vllm", "dummy"),
        default=None,
        help="Режим запуска: production vLLM или lightweight dummy server",
    )
    parser.add_argument("--model-name", type=str, default=None, help="Имя модели или локальный путь")
    parser.add_argument("--host", type=str, default=None, help="Адрес сервера")
    parser.add_argument("--port", type=int, default=None, help="Порт сервера")
    parser.add_argument("--dtype", type=str, default=None, help="Тип чисел для модели")
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=None,
        help="Размер tensor-parallel разбиения",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=None,
        help="Доля GPU-памяти, доступная модели",
    )
    parser.add_argument("--max-model-len", type=int, default=None, help="Максимальная длина контекста")
    parser.add_argument("--download-dir", type=str, default=None, help="Каталог загрузки модели")
    parser.add_argument(
        "--served-model-name",
        type=str,
        default=None,
        help="Имя модели, видимое в API сервера",
    )
    parser.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Разрешить remote code из репозитория модели",
    )
    namespace: argparse.Namespace = parser.parse_args(argv)
    return load_config_from_namespace(
        config_cls=VLLMServerConfig,
        namespace=namespace,
        fpath=namespace.config_path,
        section_name="vllm_server",
    )


def run(config: VLLMServerConfig) -> subprocess.CompletedProcess[str] | None:
    """Запускает vLLM server через CLI-команду."""
    if config.serve_mode == "dummy":
        run_dummy_vllm_server(config)
        return None

    command: list[str] = build_vllm_serve_command(config=config)
    logger.info("Запускаю vLLM server: %s", " ".join(command))
    return subprocess.run(command, check=True, text=True)


def main(argv: Sequence[str] | None = None) -> None:
    """Точка входа CLI."""
    config: VLLMServerConfig = parse_args(argv=argv)
    run(config=config)


if __name__ == "__main__":
    main()







