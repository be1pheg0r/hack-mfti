from __future__ import annotations

import argparse
from typing import *

from common.configs import load_config_from_namespace
from common.logger import HF_NLI_SERVER_LOGGER as logger
from src.sber.constants import DEFAULT_HF_NLI_SERVER_CONFIG_FPATH
from src.servers.utils.hf_nli_utils import (
    HFNLIServerConfig,
    run_dummy_hf_nli_server,
    run_hf_nli_server,
)


def parse_args(argv: Sequence[str] | None = None) -> HFNLIServerConfig:
    """Парсит CLI и возвращает итоговую конфигурацию HF NLI сервера."""
    parser = argparse.ArgumentParser(description="Запуск HF NLI server для Sber-кейса")
    parser.add_argument(
        "--config-path",
        type=str,
        default=str(DEFAULT_HF_NLI_SERVER_CONFIG_FPATH),
        help="Путь до YAML-конфига сервера",
    )
    parser.add_argument(
        "--serve-mode",
        type=str,
        choices=("hf_nli", "dummy"),
        default=None,
        help="Режим запуска: production HF NLI или lightweight dummy server",
    )
    parser.add_argument("--host", type=str, default=None, help="Адрес сервера")
    parser.add_argument("--port", type=int, default=None, help="Порт сервера")
    parser.add_argument("--repo-id", type=str, default=None, help="HF repo id модели")
    parser.add_argument("--nli-config-path", type=str, default=None, help="Путь до YAML-конфига HFNLI")
    parser.add_argument("--compute-dtype", type=str, default=None, help="Тип чисел для инференса")
    parser.add_argument("--batch-size", type=int, default=None, help="Размер батча инференса")
    parser.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Разрешить remote code из репозитория модели",
    )
    namespace: argparse.Namespace = parser.parse_args(argv)
    return load_config_from_namespace(
        config_cls=HFNLIServerConfig,
        namespace=namespace,
        fpath=namespace.config_path,
        section_name="hf_nli_server",
    )


def run(config: HFNLIServerConfig) -> None:
    """Запускает HF NLI server в выбранном режиме."""
    if config.serve_mode == "dummy":
        run_dummy_hf_nli_server(config)
        return

    logger.info(
        "Запускаю HF NLI production server: mode=%s host=%s port=%s repo_id=%s",
        config.serve_mode,
        config.host,
        config.port,
        config.repo_id,
    )
    run_hf_nli_server(config)


def main(argv: Sequence[str] | None = None) -> None:
    """Точка входа CLI."""
    config: HFNLIServerConfig = parse_args(argv=argv)
    run(config=config)


if __name__ == "__main__":
    main()

