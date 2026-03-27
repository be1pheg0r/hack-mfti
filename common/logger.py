import logging
import time
from functools import wraps
from colorama import Fore, Style, init

init(autoreset=True)

level_map = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}


class ColoredFormatter(logging.Formatter):
    """Форматер логирования с цветовым выделением по уровню логирования."""

    LEVEL_COLORS = {
        logging.DEBUG: Fore.BLUE,
        logging.INFO: Fore.GREEN,
        logging.WARNING: Fore.YELLOW,
        logging.ERROR: Fore.RED,
        logging.CRITICAL: Fore.RED + Style.BRIGHT,
    }

    def format(self, record):
        """
        Форматирует запись логирования с цветовым выделением.

        :param record: Запись логирования.
        :return: Отформатированная строка логирования.
        """
        log_color = self.LEVEL_COLORS.get(record.levelno, "")
        reset_color = Style.RESET_ALL
        original_msg = record.msg
        record.msg = f"{log_color}{record.msg}{reset_color}"
        result = super().format(record)
        record.msg = original_msg
        return result


def setup_logger(name: str, level: str = "INFO") -> logging.Logger:
    """
    Инициализирует логгер с цветным форматированием.

    :param name: Название логгера.
    :param level: Уровень логирования (по умолчанию INFO).
    :return: Настроенный объект логгера.
    """
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(level_map.get(level.upper(), logging.INFO))

    handler = logging.StreamHandler()
    handler.setLevel(level_map.get(level.upper(), logging.INFO))
    formatter = ColoredFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    root_logger.addHandler(handler)

    return logging.getLogger(name)


def log_after_invoke(logger: logging.Logger):
    """
    Декоратор для логирования времени выполнения функции и обработки исключений.

    :param logger: Объект логгера для записи информации.
    :return: Декоратор функции.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                t_start = time.time()
                result = func(*args, **kwargs)
                t_end = time.time()
                logger.debug(f"{func.__name__} выполнена за {t_end - t_start:.2f} секунд.")
                return result
            except Exception as e:
                logger.error(f"Ошибка при выполнении {func.__name__}: {e}")
                raise
        return wrapper
    return decorator
