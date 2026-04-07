from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import *

from common.configs import load_config_from_namespace
from common.logger import SBER_EVALUATION_LOGGER as logger
from common.paths import get_servers_configs_dpath
from src.servers.utils.tabular_pipeline_utils import (
    TabularPipelineServerConfig,
    TabularPipelineService,
    parse_tabular_pipeline_request,
)


DEFAULT_TABULAR_PIPELINE_SERVER_CONFIG_FPATH = get_servers_configs_dpath() / "tabular_pipeline_server.yaml"


def parse_args(argv: Sequence[str] | None = None) -> TabularPipelineServerConfig:
    """Парсит CLI и возвращает конфиг сервера tabular pipeline."""
    parser = argparse.ArgumentParser(description="HTTP-сервер featureextractor -> tabular classifier")
    parser.add_argument("--config-path", type=str, default=str(DEFAULT_TABULAR_PIPELINE_SERVER_CONFIG_FPATH))
    parser.add_argument("--host", type=str, default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--serve-mode", type=str, default=None)
    parser.add_argument("--feature-model-name", type=str, default=None)
    parser.add_argument("--feature-extractor-mode", type=str, default=None)
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument("--feature-config-path", type=str, default=None)

    namespace: argparse.Namespace = parser.parse_args(argv)
    return load_config_from_namespace(
        config_cls=TabularPipelineServerConfig,
        namespace=namespace,
        fpath=namespace.config_path,
        section_name="tabular_pipeline_server",
    )


def _json_response(handler: BaseHTTPRequestHandler, status_code: int, payload: Mapping[str, Any]) -> None:
    response_body: bytes = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status_code)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(response_body)))
    handler.end_headers()
    handler.wfile.write(response_body)


def run(config: TabularPipelineServerConfig) -> None:
    """Запускает HTTP-сервер для пайплайна featureextractor -> tabular classifier."""
    service = TabularPipelineService(config=config)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/health":
                _json_response(self, 200, {"status": "ok"})
                return
            _json_response(self, 404, {"error": "Not found"})

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/v1/tabular/predict":
                _json_response(self, 404, {"error": "Not found"})
                return

            try:
                content_length: int = int(self.headers.get("Content-Length", "0"))
                body: bytes = self.rfile.read(content_length) if content_length > 0 else b"{}"
                payload_raw: Any = json.loads(body.decode("utf-8")) if body else {}
                if not isinstance(payload_raw, dict):
                    raise ValueError("JSON payload должен быть объектом")

                request = parse_tabular_pipeline_request(payload_raw)
                response = service.predict(request)
                _json_response(self, 200, response)
            except Exception as error:
                logger.error("Ошибка обработки запроса tabular pipeline: %s", error)
                _json_response(self, 400, {"error": str(error)})

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
            logger.debug("HTTP %s", format % args)

    server = ThreadingHTTPServer((config.host, config.port), Handler)  # type: ignore[arg-type]
    logger.info(
        "Запускаю tabular pipeline сервер: host=%s port=%s mode=%s feature_extractor_mode=%s",
        config.host,
        config.port,
        config.serve_mode,
        config.feature_extractor_mode,
    )
    try:
        server.serve_forever()
    finally:
        server.server_close()


def main(argv: Sequence[str] | None = None) -> None:
    """Точка входа CLI."""
    config = parse_args(argv=argv)
    run(config=config)


if __name__ == "__main__":
    main()



