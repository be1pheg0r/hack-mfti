from __future__ import annotations

import json
from typing import *
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from pydantic import BaseModel, ConfigDict, field_validator

from common.logger import GRADIO_DEMO_LOGGER as logger
from src.sber.constants import (
    DEFAULT_GRADIO_DEMO_HF_NLI_BASE_URL,
    DEFAULT_GRADIO_DEMO_HOST,
    DEFAULT_GRADIO_DEMO_PORT,
    DEFAULT_GRADIO_DEMO_SHARE,
    DEFAULT_GRADIO_DEMO_TIMEOUT_SEC,
    DEFAULT_GRADIO_DEMO_TITLE,
    DEFAULT_GRADIO_DEMO_VLLM_BASE_URL,
    DEFAULT_VLLM_MODEL_NAME,
)


class GradioDemoConfig(BaseModel):
    """Конфигурация Gradio demo для пайплайна vLLM -> HF NLI.

    Attributes:
        host: Хост, на котором поднимается Gradio UI.
        port: Порт Gradio UI.
        share: Флаг внешнего публичного share-ссылки.
        title: Заголовок UI.
        vllm_base_url: Базовый URL vLLM-сервера.
        vllm_model_name: Имя модели для `/v1/chat/completions`.
        hf_nli_base_url: Базовый URL HF NLI-сервера.
        timeout_sec: Таймаут HTTP-запросов к backend-серверам.
    """

    model_config = ConfigDict(frozen=True)

    host: str = DEFAULT_GRADIO_DEMO_HOST
    port: int = DEFAULT_GRADIO_DEMO_PORT
    share: bool = DEFAULT_GRADIO_DEMO_SHARE
    title: str = DEFAULT_GRADIO_DEMO_TITLE
    vllm_base_url: str = DEFAULT_GRADIO_DEMO_VLLM_BASE_URL
    vllm_model_name: str = DEFAULT_VLLM_MODEL_NAME
    hf_nli_base_url: str = DEFAULT_GRADIO_DEMO_HF_NLI_BASE_URL
    timeout_sec: float = DEFAULT_GRADIO_DEMO_TIMEOUT_SEC

    @field_validator("host", "title", "vllm_base_url", "vllm_model_name", "hf_nli_base_url")
    @classmethod
    def validate_non_empty_string(cls, value: str) -> str:
        normalized: str = value.strip()
        if not normalized:
            raise ValueError("Строковое поле конфигурации не может быть пустым")
        return normalized

    @field_validator("port")
    @classmethod
    def validate_positive_port(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("port должен быть положительным")
        return value

    @field_validator("timeout_sec")
    @classmethod
    def validate_timeout(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("timeout_sec должен быть положительным")
        return value


def _normalize_base_url(url: str) -> str:
    return url.rstrip("/")


def _post_json(url: str, payload: Mapping[str, Any], timeout_sec: float) -> dict[str, Any]:
    body: bytes = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request: Request = Request(
        url=url,
        data=body,
        method="POST",
        headers={"Content-Type": "application/json; charset=utf-8"},
    )
    try:
        with urlopen(request, timeout=timeout_sec) as response:
            raw: str = response.read().decode("utf-8")
            data: Any = json.loads(raw) if raw else {}
            if not isinstance(data, dict):
                raise ValueError("HTTP ответ должен быть JSON-объектом")
            return data
    except HTTPError as error:
        raw_error: str = error.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"HTTP {error.code} при запросе {url}: {raw_error}") from error
    except URLError as error:
        raise RuntimeError(f"Не удалось подключиться к {url}: {error}") from error


def query_vllm_answer(*, query: str, config: GradioDemoConfig) -> str:
    """Запрашивает текстовый ответ у vLLM-сервера через OpenAI-compatible endpoint."""
    endpoint: str = f"{_normalize_base_url(config.vllm_base_url)}/v1/chat/completions"
    payload: dict[str, Any] = {
        "model": config.vllm_model_name,
        "messages": [{"role": "user", "content": query}],
    }
    response: dict[str, Any] = _post_json(endpoint, payload, timeout_sec=config.timeout_sec)
    choices: Any = response.get("choices", [])
    if not isinstance(choices, list) or not choices:
        raise ValueError("Некорректный ответ vLLM: нет choices")

    first_choice: Any = choices[0]
    message: Any = first_choice.get("message", {}) if isinstance(first_choice, dict) else {}
    content: Any = message.get("content") if isinstance(message, dict) else None
    answer: str = str(content or "").strip()
    if not answer:
        raise ValueError("vLLM вернул пустой ответ")
    return answer


def query_hf_nli_label(*, premise: str, hypothesis: str, config: GradioDemoConfig) -> tuple[str, float | None]:
    """Запрашивает классификацию у HF NLI сервера и возвращает человекочитаемый лейбл."""
    endpoint: str = f"{_normalize_base_url(config.hf_nli_base_url)}/v1/hf-nli/predict"
    payload: dict[str, Any] = {"premises": [premise], "hypotheses": [hypothesis]}
    response: dict[str, Any] = _post_json(endpoint, payload, timeout_sec=config.timeout_sec)

    preds_raw: Any = response.get("pred_is_hallucination", [])
    if not isinstance(preds_raw, list) or not preds_raw:
        raise ValueError("Некорректный ответ HF NLI: нет pred_is_hallucination")

    prediction: int = int(preds_raw[0])
    label: str = "галлюцинация" if prediction == 1 else "не галлюцинация"

    score_raw: Any = response.get("hallucination_score", [])
    score: float | None = None
    if isinstance(score_raw, list) and score_raw:
        score = float(score_raw[0])

    return label, score


def run_pipeline(query: str, config: GradioDemoConfig) -> tuple[str, str]:
    """Прогоняет пользовательский запрос через vLLM и HF NLI по HTTP."""
    normalized_query: str = query.strip()
    if not normalized_query:
        raise ValueError("Запрос не должен быть пустым")

    logger.info("Запускаю пайплайн демо: vLLM -> HF NLI")
    vllm_answer: str = query_vllm_answer(query=normalized_query, config=config)
    label, score = query_hf_nli_label(premise=normalized_query, hypothesis=vllm_answer, config=config)

    hf_nli_result: str = label if score is None else f"{label} (score={score:.4f})"
    return vllm_answer, hf_nli_result


def build_gradio_predict_fn(config: GradioDemoConfig) -> Callable[[str], tuple[str, str]]:
    """Возвращает callable для привязки к Gradio UI."""

    def predict(query: str) -> tuple[str, str]:
        try:
            return run_pipeline(query=query, config=config)
        except Exception as error:
            logger.error("Ошибка в Gradio демо пайплайне: %s", error)
            return "", f"Ошибка: {error}"

    return predict


__all__ = [
    "GradioDemoConfig",
    "build_gradio_predict_fn",
    "query_hf_nli_label",
    "query_vllm_answer",
    "run_pipeline",
]



