from __future__ import annotations

import logging
import time
from functools import wraps
from typing import *

from colorama import Fore, Style, init
from pydantic import BaseModel, ConfigDict, field_validator

init(autoreset=True)

P = ParamSpec("P")
T = TypeVar("T")

LEVEL_MAP: dict[str, int] = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}


class LoggerConfig(BaseModel):
    """Конфигурация логгера проекта.

    Attributes:
        name: Название логгера.
        level: Уровень логирования в строковом виде.
        prefix: Текстовый префикс перед сообщением логгера.
        log_format: Формат строки лога.
        propagate: Флаг проброса логов к родительскому логгеру.
    """

    model_config = ConfigDict(frozen=True)

    name: str
    level: str = "INFO"
    prefix: str = ""
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    propagate: bool = False

    @field_validator("level")
    @classmethod
    def validate_level(cls, value: str) -> str:
        """Проверяет, что уровень логирования поддерживается."""
        normalized_level: str = value.upper().strip()
        if normalized_level not in LEVEL_MAP:
            raise ValueError(f"Неподдерживаемый уровень логирования: {value}")
        return normalized_level


class ProjectLoggerRegistry(BaseModel):
    """Реестр преднастроенных логгеров проекта.

    Attributes:
        mistral_call: Логгер для безопасного вызова функций в common.mistral.
    """

    model_config = ConfigDict(frozen=True)

    mistral_call: LoggerConfig = LoggerConfig(name="mistral-call", level="DEBUG", prefix="[MISTRAL 🇫🇷] ")
    sber_pt_hooks: LoggerConfig = LoggerConfig(name="sber-pt-hooks", level="DEBUG", prefix="[SBER-HOOKS 🪝] ")


class ColoredFormatter(logging.Formatter):
    """Форматтер с цветовым выделением по уровню логирования."""

    LEVEL_COLORS: dict[int, str] = {
        logging.DEBUG: Fore.BLUE,
        logging.INFO: Fore.GREEN,
        logging.WARNING: Fore.YELLOW,
        logging.ERROR: Fore.RED,
        logging.CRITICAL: Fore.RED + Style.BRIGHT,
    }

    def __init__(self, fmt: str, prefix: str = "") -> None:
        """Инициализирует форматтер.

        Args:
            fmt: Шаблон форматирования лога.
            prefix: Текст, который добавляется перед сообщением.
        """
        super().__init__(fmt)
        self._prefix: str = prefix

    def format(self, record: logging.LogRecord) -> str:
        """Форматирует запись логирования с цветовым выделением.

        Args:
            record: Запись логирования.

        Returns:
            Строка лога с ANSI-цветом для уровня логирования.
        """
        log_color: str = self.LEVEL_COLORS.get(record.levelno, "")
        reset_color: str = Style.RESET_ALL
        original_msg: Any = record.msg
        prefixed_message: str = f"{self._prefix}{record.msg}" if self._prefix else str(record.msg)
        record.msg = f"{log_color}{prefixed_message}{reset_color}"
        result: str = super().format(record)
        record.msg = original_msg
        return result


LOGGERS: ProjectLoggerRegistry = ProjectLoggerRegistry()


def setup_logger(config: LoggerConfig) -> logging.Logger:
    """Создает или возвращает настроенный логгер.

    Args:
        config: Конфигурация логгера.

    Returns:
        Настроенный объект логгера.
    """
    logger: logging.Logger = logging.getLogger(config.name)
    logger.setLevel(LEVEL_MAP.get(config.level.upper(), logging.INFO))
    logger.propagate = config.propagate
    formatter: ColoredFormatter = ColoredFormatter(config.log_format, prefix=config.prefix)

    if logger.handlers:
        for existing_handler in logger.handlers:
            existing_handler.setLevel(LEVEL_MAP.get(config.level.upper(), logging.INFO))
            existing_handler.setFormatter(formatter)
        return logger

    handler: logging.StreamHandler = logging.StreamHandler()
    handler.setLevel(LEVEL_MAP.get(config.level.upper(), logging.INFO))
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def get_project_logger(config_name: str) -> logging.Logger:
    """Возвращает логгер из реестра проектных логгеров.

    Args:
        config_name: Имя поля из реестра LOGGERS.

    Returns:
        Настроенный логгер.

    Raises:
        ValueError: Если имя логгера не найдено в реестре.
    """
    if not hasattr(LOGGERS, config_name):
        raise ValueError(f"Логгер с именем '{config_name}' не найден в реестре.")

    config: LoggerConfig = getattr(LOGGERS, config_name)
    return setup_logger(config)


MISTRAL_LOGGER: logging.Logger = get_project_logger("mistral_call")
SBER_HOOKS_LOGGER: logging.Logger = get_project_logger("sber_pt_hooks")


def log_after_invoke(logger: logging.Logger) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Логирует время выполнения функции и ожидаемые ошибки.

    Args:
        logger: Экземпляр логгера.

    Returns:
        Декоратор для оборачиваемой функции.
    """

    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            try:
                start_time: float = time.time()
                result: T = func(*args, **kwargs)
                duration: float = time.time() - start_time
                logger.debug(f"{func.__name__} выполнена за {duration:.2f} секунд.")
                return result
            except Exception as error:
                logger.error(f"Ошибка при выполнении {func.__name__}: {error}")
                raise

        return wrapper

    return decorator
