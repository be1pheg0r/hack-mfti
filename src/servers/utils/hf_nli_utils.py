from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import *

import torch
from pydantic import BaseModel, ConfigDict, field_validator

from common.configs import apply_namespace_overrides, load_pydantic_config
from common.logger import HF_NLI_SERVER_LOGGER as logger
from common.paths import PathLike
from src.sber.constants import (
    DEFAULT_HF_NLI_SERVER_CONFIG_FPATH,
    DEFAULT_HF_NLI_SERVER_HOST,
    DEFAULT_HF_NLI_SERVER_PORT,
    DEFAULT_HF_NLI_SERVER_SERVE_MODE,
    DEFAULT_SBER_HF_MODEL_REPO_ID,
    DEFAULT_SBER_HF_NLI_BATCH_SIZE,
    DEFAULT_SBER_HF_NLI_COMPUTE_DTYPE,
    DEFAULT_SBER_HF_NLI_CONFIG_FPATH,
)
from src.sber.models.hf_nli_clf import HFNLIClf, HFNLIClfConfig


class HFNLIServerConfig(BaseModel):
    """Конфигурация HTTP-сервера HFNLI-классификатора.

    Attributes:
        serve_mode: Режим запуска: `hf_nli` или `dummy`.
        host: Адрес сервера.
        port: Порт HTTP-сервера.
        repo_id: Идентификатор HF-репозитория модели.
        nli_config_path: Путь до YAML-конфига HFNLI.
        compute_dtype: Тип вычислений (`float32`, `float16`, `bfloat16`).
        batch_size: Размер батча инференса.
        trust_remote_code: Опциональный override для trust_remote_code.
    """

    model_config = ConfigDict(frozen=True)

    serve_mode: Literal["hf_nli", "dummy"] = DEFAULT_HF_NLI_SERVER_SERVE_MODE
    host: str = DEFAULT_HF_NLI_SERVER_HOST
    port: int = DEFAULT_HF_NLI_SERVER_PORT
    repo_id: str = DEFAULT_SBER_HF_MODEL_REPO_ID
    nli_config_path: PathLike = DEFAULT_SBER_HF_NLI_CONFIG_FPATH
    compute_dtype: str = DEFAULT_SBER_HF_NLI_COMPUTE_DTYPE
    batch_size: int = DEFAULT_SBER_HF_NLI_BATCH_SIZE
    trust_remote_code: bool | None = None

    @field_validator("host", "repo_id", "compute_dtype")
    @classmethod
    def validate_non_empty_string(cls, value: str) -> str:
        normalized: str = value.strip()
        if not normalized:
            raise ValueError("Строковое поле конфигурации не может быть пустым")
        return normalized

    @field_validator("port", "batch_size")
    @classmethod
    def validate_positive_int(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("Числовой параметр должен быть положительным")
        return value

    @field_validator("compute_dtype")
    @classmethod
    def validate_compute_dtype(cls, value: str) -> str:
        normalized: str = value.strip().lower()
        allowed: set[str] = {"float32", "float16", "bfloat16"}
        if normalized not in allowed:
            raise ValueError("compute_dtype должен быть одним из: float32, float16, bfloat16")
        return normalized

    @field_validator("nli_config_path")
    @classmethod
    def validate_nli_config_path(cls, value: PathLike) -> PathLike:
        if not str(value).strip():
            raise ValueError("nli_config_path не может быть пустым")
        return value

    @classmethod
    def from_yaml(cls, fpath: PathLike) -> HFNLIServerConfig:
        """Создает конфигурацию из YAML-файла."""
        return load_pydantic_config(config_cls=cls, fpath=fpath, section_name="hf_nli_server")

    @classmethod
    def from_default_yaml(cls) -> HFNLIServerConfig:
        """Создает конфигурацию из штатного YAML-файла."""
        return cls.from_yaml(fpath=DEFAULT_HF_NLI_SERVER_CONFIG_FPATH)


class DummyHFNLIModel(BaseModel):
    """Легковесная dummy-модель HFNLI для smoke-теста HTTP API."""

    model_config = ConfigDict(frozen=True)

    model_name: str = "dummy-hf-nli"

    def predict(self, premises: Sequence[str], hypotheses: Sequence[str]) -> dict[str, list[float] | list[int]]:
        """Возвращает простые детерминированные предсказания для smoke-теста."""
        hallucination_scores: list[float] = []
        entailment_scores: list[float] = []
        preds: list[int] = []

        for premise, hypothesis in zip(premises, hypotheses):
            normalized_premise: str = str(premise).strip().lower()
            normalized_hypothesis: str = str(hypothesis).strip().lower()
            is_hallucination: int = int(normalized_premise != normalized_hypothesis)
            hallucination_score: float = 0.9 if is_hallucination else 0.1
            entailment_score: float = 1.0 - hallucination_score
            preds.append(is_hallucination)
            hallucination_scores.append(hallucination_score)
            entailment_scores.append(entailment_score)

        return {
            "pred_is_hallucination": preds,
            "hallucination_score": hallucination_scores,
            "entailment_score": entailment_scores,
        }


class _HFNLIThreadingHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True


def _parse_predict_payload(payload: Any) -> tuple[list[str], list[str], int | None]:
    if not isinstance(payload, dict):
        raise ValueError("Payload должен быть JSON-объектом")

    premises_raw: Any = payload.get("premises", payload.get("correct_answers"))
    hypotheses_raw: Any = payload.get("hypotheses", payload.get("model_answers"))
    batch_size_raw: Any = payload.get("batch_size")

    if not isinstance(premises_raw, list) or not isinstance(hypotheses_raw, list):
        raise ValueError("Ожидаются list-поля premises и hypotheses")
    if len(premises_raw) != len(hypotheses_raw):
        raise ValueError("premises и hypotheses должны быть одинаковой длины")

    premises: list[str] = [str(value) for value in premises_raw]
    hypotheses: list[str] = [str(value) for value in hypotheses_raw]
    batch_size: int | None = None if batch_size_raw is None else int(batch_size_raw)
    if batch_size is not None and batch_size <= 0:
        raise ValueError("batch_size должен быть положительным")
    return premises, hypotheses, batch_size


def _predict_with_clf(clf: HFNLIClf, premises: Sequence[str], hypotheses: Sequence[str], batch_size: int | None) -> dict[str, list[float] | list[int]]:
    logits: torch.Tensor = clf.predict_logits(
        premises=premises,
        hypotheses=hypotheses,
        batch_size=batch_size,
    )
    if logits.numel() == 0:
        return {
            "pred_is_hallucination": [],
            "hallucination_score": [],
            "entailment_score": [],
        }

    probs: torch.Tensor = torch.softmax(logits, dim=-1)
    pos_idx: int = clf.config.positive_class_index
    if pos_idx >= int(probs.shape[1]):
        raise ValueError("positive_class_index выходит за число классов модели")

    hallucination_probs: list[float] = [float(value) for value in probs[:, pos_idx].tolist()]
    entailment_probs: list[float] = [1.0 - value for value in hallucination_probs]
    predicted_indices: list[int] = [int(value) for value in logits.argmax(dim=-1).tolist()]
    preds: list[int] = [int(index == pos_idx) for index in predicted_indices]

    return {
        "pred_is_hallucination": preds,
        "hallucination_score": hallucination_probs,
        "entailment_score": entailment_probs,
    }


def _build_hf_nli_handler(
    *,
    mode: Literal["hf_nli", "dummy"],
    model_name: str,
    clf: HFNLIClf | None = None,
    dummy_model: DummyHFNLIModel | None = None,
) -> type[BaseHTTPRequestHandler]:
    class HFNLIHandler(BaseHTTPRequestHandler):
        def _send_json(self, status_code: int, payload: dict[str, Any]) -> None:
            body: bytes = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status_code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            if self.path in {"/health", "/v1/health"}:
                self._send_json(200, {"status": "ok", "mode": mode, "model": model_name})
                return
            self._send_json(404, {"error": "not found"})

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/v1/hf-nli/predict":
                self._send_json(404, {"error": "not found"})
                return

            try:
                content_length: int = int(self.headers.get("Content-Length", "0"))
                raw_body: bytes = self.rfile.read(content_length)
                payload: Any = json.loads(raw_body.decode("utf-8") or "{}") if raw_body else {}
                premises, hypotheses, batch_size = _parse_predict_payload(payload=payload)

                if mode == "dummy":
                    assert dummy_model is not None
                    prediction_payload: dict[str, list[float] | list[int]] = dummy_model.predict(
                        premises=premises,
                        hypotheses=hypotheses,
                    )
                else:
                    assert clf is not None
                    prediction_payload = _predict_with_clf(
                        clf=clf,
                        premises=premises,
                        hypotheses=hypotheses,
                        batch_size=batch_size,
                    )

                self._send_json(200, {"model": model_name, **prediction_payload})
            except Exception as error:
                self._send_json(400, {"error": str(error)})

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
            return

    return HFNLIHandler


def run_dummy_hf_nli_server(config: HFNLIServerConfig) -> None:
    """Запускает lightweight dummy HFNLI HTTP server."""
    dummy_model: DummyHFNLIModel = DummyHFNLIModel()
    handler_cls: type[BaseHTTPRequestHandler] = _build_hf_nli_handler(
        mode="dummy",
        model_name=dummy_model.model_name,
        dummy_model=dummy_model,
    )
    handler: Any = cast(Any, handler_cls)
    server = _HFNLIThreadingHTTPServer((config.host, config.port), handler)
    logger.info("Запускаю dummy HF NLI server: http://%s:%s", config.host, config.port)
    try:
        server.serve_forever()
    finally:
        server.server_close()


def run_hf_nli_server(config: HFNLIServerConfig) -> None:
    """Запускает HTTP server поверх HFNLIClf."""
    overrides: dict[str, Any] = {
        "repo_id": config.repo_id,
        "batch_size": config.batch_size,
        "compute_dtype": config.compute_dtype,
    }
    if config.trust_remote_code is not None:
        overrides["trust_remote_code"] = config.trust_remote_code

    nli_config: HFNLIClfConfig = HFNLIClfConfig.from_yaml(config.nli_config_path).model_copy(update=overrides)
    clf: HFNLIClf = HFNLIClf(config=nli_config)
    clf.load()

    handler_cls: type[BaseHTTPRequestHandler] = _build_hf_nli_handler(
        mode="hf_nli",
        model_name=config.repo_id,
        clf=clf,
    )
    handler: Any = cast(Any, handler_cls)
    server = _HFNLIThreadingHTTPServer((config.host, config.port), handler)
    logger.info("Запускаю HF NLI server: http://%s:%s", config.host, config.port)
    try:
        server.serve_forever()
    finally:
        server.server_close()


def load_hf_nli_server_config(fpath: PathLike | None = None) -> HFNLIServerConfig:
    """Загружает конфигурацию сервера из YAML-файла."""
    config_path: PathLike = fpath if fpath is not None else DEFAULT_HF_NLI_SERVER_CONFIG_FPATH
    return HFNLIServerConfig.from_yaml(config_path)


def apply_cli_overrides(config: HFNLIServerConfig, namespace: argparse.Namespace) -> HFNLIServerConfig:
    """Применяет CLI-значения поверх конфигурации из YAML."""
    return apply_namespace_overrides(config=config, namespace=namespace)


__all__ = [
    "DummyHFNLIModel",
    "HFNLIServerConfig",
    "apply_cli_overrides",
    "load_hf_nli_server_config",
    "run_dummy_hf_nli_server",
    "run_hf_nli_server",
]


