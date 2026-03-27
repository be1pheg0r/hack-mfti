from __future__ import annotations

import concurrent.futures
import importlib
from dataclasses import dataclass, field
from typing import *

from common.logger import MISTRAL_LOGGER
from common.paths import get_mistral_api_keys_fpath
from common.files import read_file

logger = MISTRAL_LOGGER
T = TypeVar("T")

def _api_keys() -> list[str]:
    """Считывает API-ключи из файла.

    Returns:
        Список API-ключей.
    """
    api_keys_fpath = get_mistral_api_keys_fpath()
    try:
        api_keys: list[str] = read_file(api_keys_fpath)
    except FileNotFoundError:
        logger.warning(f"Файл с ключами Mistral не найден: {api_keys_fpath}")
        return []
    return [key.strip() for key in api_keys if key.strip()]


def _resolve_api_keys(config: MistralCallConfig) -> list[str]:
    """Собирает итоговый список API-ключей для перебора.

    Приоритет: явные ключи из конфига, затем ключи из файла, затем default_api_key.

    Args:
        config: Конфигурация вызова Mistral.

    Returns:
        Список уникальных API-ключей в порядке приоритета.

    Raises:
        ValueError: Если не найдено ни одного валидного ключа.
    """
    merged_keys: list[str] = []
    seen_keys: set[str] = set()

    for key in [*config.api_keys, *_api_keys(), config.default_api_key]:
        normalized_key: str = key.strip()
        if not normalized_key or normalized_key in seen_keys:
            continue
        seen_keys.add(normalized_key)
        merged_keys.append(normalized_key)

    if not merged_keys:
        raise ValueError("Не найдено API-ключей Mistral: проверьте конфиг и файл с ключами.")

    return merged_keys

@dataclass(frozen=True, slots=True)
class MistralCallConfig:
    """Конфигурация вызова Mistral API.

    Attributes:
        models_list: Список моделей в порядке приоритета.
        default_api_key: Основной API-ключ.
        api_keys: Дополнительные API-ключи для фолбэка.
        timeout: Таймаут одного запроса в секундах.
        max_attempts_per_call: Число попыток на один вызов модели.
    """

    models_list: list[str]
    default_api_key: str = "void"
    api_keys: list[str] = field(default_factory=list)
    timeout: int = 240
    max_attempts_per_call: int = 1

    def keys_to_try(self) -> list[str]:
        """Возвращает список ключей для перебора."""
        return self.api_keys if self.api_keys else [self.default_api_key]


def _load_mistral_client_class() -> type[Any]:
    """Ленивая загрузка клиента Mistral.

    Returns:
        Класс клиента Mistral.

    Raises:
        ImportError: Если пакет mistralai недоступен.
        AttributeError: Если в пакете отсутствует класс Mistral.
    """
    module: Any = importlib.import_module("mistralai.client")
    client_class: type[Any] = getattr(module, "Mistral")
    return client_class


def build_client(api_key: str) -> Any:
    """Создает клиент Mistral по API-ключу."""
    client_class: type[Any] = _load_mistral_client_class()
    return client_class(api_key=api_key)


def safe_call(
    function: Callable[..., T],
    *args: Any,
    timeout: int = 30,
    max_attempts: int = 3,
    **kwargs: Any,
) -> T:
    """Безопасно вызывает функцию с таймаутом и повторами."""
    logger.debug(f"safe_call: вызов функции с timeout={timeout}")
    attempt: int = 0
    while attempt < max_attempts:
        attempt += 1
        try:
            logger.debug(f"safe_call: попытка {attempt}")
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(function, *args, **kwargs)
                result = future.result(timeout=timeout)
                logger.debug(f"safe_call: успешное выполнение на попытке {attempt}")
                return result
        except concurrent.futures.TimeoutError:
            logger.warning(f"safe_call: timeout на попытке {attempt}, повтор...")
        except Exception as error:
            logger.error(f"safe_call: ошибка на попытке {attempt}: {str(error)[:20]}...", exc_info=True)
    raise RuntimeError("safe_call: превышено число попыток")


def call_mistral(config: MistralCallConfig, **kwargs: Any) -> str:
    """Вызывает Mistral API с фолбэком по моделям и ключам.

    Args:
        config: Конфигурация вызова API.
        **kwargs: Параметры, которые передаются в `client.chat.complete`.

    Returns:
        Текстовый ответ модели.

    Raises:
        RuntimeError: Если все комбинации моделей и ключей недоступны.
    """
    logger.info("call_mistral: начало вызова API")
    logger.debug(f"call_mistral: параметры - {list(kwargs.keys())}")

    def sub_call(client: Any, model_name: str, **sub_kwargs: Any) -> str:
        logger.debug(f"sub_call: попытка вызова модели {model_name}")
        response: Any = client.chat.complete(model=model_name, **sub_kwargs)
        content: str = response.choices[0].message.content
        logger.debug(f"sub_call: получен ответ длиной {len(content)} символов")
        return content

    last_exception: Exception | None = None
    keys_to_try: list[str] = _resolve_api_keys(config)
    saved_key_index: int = getattr(call_mistral, "current_key_index", 0)
    start_key_index: int = saved_key_index % len(keys_to_try)

    for model_index, model_name in enumerate(config.models_list, 1):
        for offset in range(len(keys_to_try)):
            key_index: int = (start_key_index + offset) % len(keys_to_try)
            api_key: str = keys_to_try[key_index]
            try:
                logger.info(
                    f"call_mistral: попытка {model_index}/{len(config.models_list)} "
                    f"с моделью {model_name}, ключ {key_index + 1}/{len(keys_to_try)}"
                )
                client: Any = build_client(api_key)
                result: str = safe_call(
                    sub_call,
                    client,
                    model_name,
                    timeout=config.timeout,
                    max_attempts=config.max_attempts_per_call,
                    **kwargs,
                )
                setattr(call_mistral, "current_key_index", key_index)
                logger.info(f"call_mistral: успешный вызов модели {model_name}")
                return result
            except Exception as error:
                last_exception = error
                logger.warning(
                    f"call_mistral: ошибка с моделью {model_name} "
                    f"и ключом {key_index + 1}: {error}"
                )
                continue

    logger.error("call_mistral: все модели или ключи недоступны", exc_info=True)
    raise RuntimeError("Все модели или ключи недоступны") from last_exception
