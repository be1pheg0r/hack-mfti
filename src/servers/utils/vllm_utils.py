from __future__ import annotations

import argparse
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import *

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from common.configs import apply_namespace_overrides, load_pydantic_config
from common.logger import VLLM_SERVER_LOGGER as logger
from common.paths import PathLike
from src.sber.constants import (
    DEFAULT_VLLM_DOWNLOAD_DIR,
    DEFAULT_VLLM_DUMMY_MODEL_NAME,
    DEFAULT_VLLM_DUMMY_RESPONSE_TEXT,
    DEFAULT_VLLM_DTYPE,
    DEFAULT_VLLM_GPU_MEMORY_UTILIZATION,
    DEFAULT_VLLM_HOST,
    DEFAULT_VLLM_MAX_MODEL_LEN,
    DEFAULT_VLLM_MODEL_NAME,
    DEFAULT_VLLM_PORT,
    DEFAULT_VLLM_SERVE_CONFIG_FPATH,
    DEFAULT_VLLM_SERVE_MODE,
    DEFAULT_VLLM_TENSOR_PARALLEL_SIZE,
    DEFAULT_VLLM_TRUST_REMOTE_CODE,
)


class VLLMServerConfig(BaseModel):
    """Конфигурация запуска vLLM-сервера.

    Attributes:
        model_name: Имя модели на Hugging Face или локальный путь до чекпоинта.
        host: Адрес, на котором должен слушать сервер.
        port: Порт HTTP-сервера.
        serve_mode: Режим запуска: production vLLM или dummy smoke-сервер.
        dtype: Тип чисел для загрузки модели.
        tensor_parallel_size: Размер tensor-parallel разбиения.
        gpu_memory_utilization: Доля GPU-памяти, доступная для модели.
        max_model_len: Ограничение на длину контекста модели.
        trust_remote_code: Разрешать ли выполнение remote code из репозитория модели.
        download_dir: Каталог для кеша и загрузки модели.
        served_model_name: Имя, под которым модель будет опубликована сервером.
    """

    model_config = ConfigDict(frozen=True)

    model_name: str = DEFAULT_VLLM_MODEL_NAME
    host: str = DEFAULT_VLLM_HOST
    port: int = DEFAULT_VLLM_PORT
    serve_mode: Literal["vllm", "dummy"] = DEFAULT_VLLM_SERVE_MODE
    dtype: str = DEFAULT_VLLM_DTYPE
    tensor_parallel_size: int = DEFAULT_VLLM_TENSOR_PARALLEL_SIZE
    gpu_memory_utilization: float = DEFAULT_VLLM_GPU_MEMORY_UTILIZATION
    max_model_len: int | None = DEFAULT_VLLM_MAX_MODEL_LEN
    trust_remote_code: bool = DEFAULT_VLLM_TRUST_REMOTE_CODE
    download_dir: PathLike | None = Field(default=DEFAULT_VLLM_DOWNLOAD_DIR)
    served_model_name: str | None = None

    @field_validator("model_name", "host", "dtype")
    @classmethod
    def validate_non_empty_string(cls, value: str) -> str:
        """Проверяет, что строковые поля не пустые."""
        normalized: str = value.strip()
        if not normalized:
            raise ValueError("Строковое поле конфигурации не может быть пустым")
        return normalized

    @field_validator("port", "tensor_parallel_size")
    @classmethod
    def validate_positive_int(cls, value: int) -> int:
        """Проверяет, что целочисленные параметры положительные."""
        if value <= 0:
            raise ValueError("Числовой параметр должен быть положительным")
        return value

    @field_validator("gpu_memory_utilization")
    @classmethod
    def validate_gpu_memory_utilization(cls, value: float) -> float:
        """Проверяет долю GPU-памяти."""
        if not 0.0 < value <= 1.0:
            raise ValueError("gpu_memory_utilization должен быть в диапазоне (0, 1]")
        return value

    @field_validator("max_model_len")
    @classmethod
    def validate_optional_positive_int(cls, value: int | None) -> int | None:
        """Проверяет, что максимальная длина контекста положительна."""
        if value is None:
            return value
        if value <= 0:
            raise ValueError("max_model_len должен быть положительным")
        return value

    @field_validator("download_dir")
    @classmethod
    def validate_download_dir(cls, value: PathLike | None) -> PathLike | None:
        """Проверяет путь к каталогу загрузки."""
        if value is None:
            return value
        if not str(value).strip():
            raise ValueError("download_dir не может быть пустым")
        return value

    @field_validator("served_model_name")
    @classmethod
    def validate_optional_non_empty_string(cls, value: str | None) -> str | None:
        """Проверяет алиас модели, если он задан."""
        if value is None:
            return value
        normalized: str = value.strip()
        if not normalized:
            raise ValueError("served_model_name не может быть пустым")
        return normalized

    @model_validator(mode="after")
    def validate_model(self) -> VLLMServerConfig:
        """Проверяет, что alias не пустой для CLI-экспозиции."""
        if self.served_model_name is not None and not self.served_model_name.strip():
            raise ValueError("served_model_name не может быть пустым")
        return self

    @classmethod
    def from_yaml(cls, fpath: PathLike) -> VLLMServerConfig:
        """Создает конфигурацию из YAML-файла.

        Args:
            fpath: Путь до YAML-файла.

        Returns:
            Валидированная конфигурация сервера.
        """
        return load_pydantic_config(config_cls=cls, fpath=fpath, section_name="vllm_server")

    @classmethod
    def from_default_yaml(cls) -> VLLMServerConfig:
        """Создает конфигурацию из штатного YAML-файла."""
        return cls.from_yaml(fpath=DEFAULT_VLLM_SERVE_CONFIG_FPATH)

    def to_vllm_serve_args(self) -> list[str]:
        """Формирует аргументы для команды `vllm serve`."""
        args: list[str] = [self.model_name, "--host", self.host, "--port", str(self.port), "--dtype", self.dtype]
        args.extend(["--tensor-parallel-size", str(self.tensor_parallel_size)])
        args.extend(["--gpu-memory-utilization", str(self.gpu_memory_utilization)])

        if self.max_model_len is not None:
            args.extend(["--max-model-len", str(self.max_model_len)])
        if self.download_dir is not None:
            args.extend(["--download-dir", str(self.download_dir)])
        if self.served_model_name is not None:
            args.extend(["--served-model-name", self.served_model_name])
        if self.trust_remote_code:
            args.append("--trust-remote-code")
        return args


class DummyVLLMModel(BaseModel):
    """Локальная dummy-модель для smoke-теста OpenAI-совместимого API.

    Attributes:
        model_name: Имя dummy-модели, которое будет отображаться в API.
        response_text: Базовый текст ответа, используемый в completions.
        served_model_name: Имя модели в ответе сервера.
    """

    model_config = ConfigDict(frozen=True)

    model_name: str = DEFAULT_VLLM_DUMMY_MODEL_NAME
    response_text: str = DEFAULT_VLLM_DUMMY_RESPONSE_TEXT
    served_model_name: str | None = None

    @field_validator("model_name", "response_text")
    @classmethod
    def validate_non_empty_string(cls, value: str) -> str:
        """Проверяет, что строковые поля не пустые."""
        normalized: str = value.strip()
        if not normalized:
            raise ValueError("Строковое поле dummy-модели не может быть пустым")
        return normalized

    @field_validator("served_model_name")
    @classmethod
    def validate_optional_non_empty_string(cls, value: str | None) -> str | None:
        """Проверяет имя модели, если оно задано."""
        if value is None:
            return value
        normalized: str = value.strip()
        if not normalized:
            raise ValueError("served_model_name не может быть пустым")
        return normalized

    @property
    def api_model_name(self) -> str:
        """Возвращает имя модели, видимое в API."""
        return self.served_model_name or self.model_name

    def _extract_prompt(self, messages: Sequence[Mapping[str, Any]]) -> str:
        """Извлекает последний пользовательский prompt из чата."""
        for message in reversed(list(messages)):
            if str(message.get("role", "")).lower() == "user":
                content: Any = message.get("content", "")
                if isinstance(content, list):
                    return " ".join(str(item) for item in content if item is not None).strip()
                return str(content).strip()

        if messages:
            content = messages[-1].get("content", "")
            if isinstance(content, list):
                return " ".join(str(item) for item in content if item is not None).strip()
            return str(content).strip()
        return ""

    def _count_tokens(self, text: str) -> int:
        """Считает токены примитивным способом для smoke-теста."""
        return len([token for token in text.split() if token])

    def build_models_payload(self) -> dict[str, Any]:
        """Формирует ответ для эндпоинта `/v1/models`."""
        created_at: int = int(time.time())
        return {
            "object": "list",
            "data": [
                {
                    "id": self.api_model_name,
                    "object": "model",
                    "created": created_at,
                    "owned_by": "dummy",
                }
            ],
        }

    def build_chat_completion(self, messages: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        """Формирует OpenAI-совместимый chat completion ответ."""
        prompt: str = self._extract_prompt(messages=messages)
        answer_text: str = self.response_text if not prompt else f"{self.response_text} Echo: {prompt}"
        created_at: int = int(time.time())
        prompt_tokens: int = self._count_tokens(prompt)
        completion_tokens: int = self._count_tokens(answer_text)
        return {
            "id": f"chatcmpl-dummy-{created_at}",
            "object": "chat.completion",
            "created": created_at,
            "model": self.api_model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": answer_text},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }


class _DummyVLLMThreadingHTTPServer(ThreadingHTTPServer):
    """HTTP server для dummy vLLM smoke mode."""

    allow_reuse_address = True


def build_dummy_vllm_model(config: VLLMServerConfig) -> DummyVLLMModel:
    """Создает dummy-модель на основе конфигурации сервера."""
    return DummyVLLMModel(
        model_name=DEFAULT_VLLM_DUMMY_MODEL_NAME if config.serve_mode == "dummy" else config.model_name,
        response_text=DEFAULT_VLLM_DUMMY_RESPONSE_TEXT,
        served_model_name=config.served_model_name or config.model_name,
    )


def _build_dummy_vllm_handler(model: DummyVLLMModel) -> type[BaseHTTPRequestHandler]:
    """Создает HTTP handler для dummy vLLM сервера."""

    class DummyVLLMHandler(BaseHTTPRequestHandler):
        def _send_json(self, status_code: int, payload: dict[str, Any]) -> None:
            body: bytes = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status_code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            if self.path in {"/health", "/v1/health"}:
                self._send_json(200, {"status": "ok", "mode": "dummy"})
                return
            if self.path == "/v1/models":
                self._send_json(200, model.build_models_payload())
                return
            self._send_json(404, {"error": "not found"})

        def do_POST(self) -> None:  # noqa: N802
            if self.path not in {"/v1/chat/completions", "/v1/completions"}:
                self._send_json(404, {"error": "not found"})
                return

            content_length: int = int(self.headers.get("Content-Length", "0"))
            raw_body: bytes = self.rfile.read(content_length)
            payload: Any = json.loads(raw_body.decode("utf-8") or "{}") if raw_body else {}
            messages: Sequence[Mapping[str, Any]] = payload.get("messages", []) if isinstance(payload, dict) else []
            if self.path == "/v1/completions":
                prompt_text: str = str(payload.get("prompt", "")) if isinstance(payload, dict) else ""
                messages = [{"role": "user", "content": prompt_text}]

            response: dict[str, Any] = model.build_chat_completion(messages=messages)
            self._send_json(200, response)

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
            return

    return DummyVLLMHandler


def run_dummy_vllm_server(config: VLLMServerConfig) -> None:
    """Запускает dummy OpenAI-compatible сервер без загрузки реальной модели."""
    model: DummyVLLMModel = build_dummy_vllm_model(config=config)
    handler_cls: type[BaseHTTPRequestHandler] = _build_dummy_vllm_handler(model=model)
    server = _DummyVLLMThreadingHTTPServer((config.host, config.port), handler_cls)
    logger.info("Запускаю dummy vLLM server: http://%s:%s", config.host, config.port)
    try:
        server.serve_forever()
    finally:
        server.server_close()


def load_vllm_server_config(fpath: PathLike | None = None) -> VLLMServerConfig:
    """Загружает конфигурацию сервера из YAML-файла."""
    config_path: PathLike = fpath if fpath is not None else DEFAULT_VLLM_SERVE_CONFIG_FPATH
    return VLLMServerConfig.from_yaml(config_path)


def apply_cli_overrides(config: VLLMServerConfig, namespace: argparse.Namespace) -> VLLMServerConfig:
    """Применяет CLI-значения поверх конфигурации из YAML.

    Аргументы со значением `None` пропускаются, поэтому YAML остается источником
    значений по умолчанию.
    """
    return apply_namespace_overrides(config=config, namespace=namespace)


def build_vllm_serve_command(config: VLLMServerConfig) -> list[str]:
    """Формирует команду запуска `vllm serve`."""
    return ["vllm", "serve", *config.to_vllm_serve_args()]


__all__ = [
    "DummyVLLMModel",
    "VLLMServerConfig",
    "apply_cli_overrides",
    "build_dummy_vllm_model",
    "build_vllm_serve_command",
    "load_vllm_server_config",
    "run_dummy_vllm_server",
]






