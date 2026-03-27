from pathlib import Path
import os
from typing import *

ANCHOR = "anch"


def fixdir(function: Callable[..., Path]) -> Callable[..., Path]:
    """
    Декоратор для создания директории, если она не существует.
    :param function: Callable: Функция, возвращающая путь к директории.
    :return: Callable: Обернутая функция.
    """

    def wrapper(*args, **kwargs) -> Path:
        dir_path = function(*args, **kwargs)
        dir_path.mkdir(parents=True, exist_ok=True)
        return dir_path

    return wrapper


def avito(function: Callable[..., Path]) -> Callable[..., Path]:
    """
    Декоратор для добавления префикса "avito" к пути.
    :param function: Callable: Функция, возвращающая путь.
    :return: Callable: Обернутая функция.
    """

    def wrapper(*args, **kwargs) -> Path:
        return get_project_root() / "avito" / function(*args, **kwargs)

    return wrapper

def sber(function: Callable[..., Path]) -> Callable[..., Path]:
    """
    Декоратор для добавления префикса "sber" к пути.
    :param function: Callable: Функция, возвращающая путь.
    :return: Callable: Обернутая функция.
    """

    def wrapper(*args, **kwargs) -> Path:
        return get_project_root() / "sber" / function(*args, **kwargs)

    return wrapper


def get_project_root() -> Path:
    """
    Корневой каталог проекта.
    :return: Path: Путь к корневому каталогу проекта.
    """
    current_path = Path(__file__).resolve().parent
    while ANCHOR not in [Path(p).name for p in os.listdir(current_path)]:
        if current_path.parent == current_path:
            raise FileNotFoundError(f"Файл '{ANCHOR}' не найден в корневом каталоге проекта.")
        current_path = current_path.parent
    return current_path


@fixdir
def get_data_dpath() -> Path:
    """
    Каталог для хранения данных проекта.
    :return: Path: Путь к каталогу данных проекта.
    """
    return get_project_root() / "data"

@fixdir
@avito
def get_avito_dpath() -> Path:
    """
    Каталог для сурсов для авито кейса.
    :return: Path: Путь к каталогу сурсов для авито кейса.
    """
    return Path()


@fixdir
@sber
def get_sber_dpath() -> Path:
    """
    Каталог для сурсов для сбер кейса.
    :return: Path: Путь к каталогу сурсов для сбер кейса.
    """
    return Path()

@fixdir
@avito
def get_avito_data_dpath() -> Path:
    """
    Каталог для данных для авито кейса.
    :return: Path: Путь к каталогу данных для авито кейса.
    """
    return Path() / "data"


@fixdir
@sber
def get_sber_data_dpath() -> Path:
    """
    Каталог для данных для сбер кейса.
    :return: Path: Путь к каталогу данных для сбер кейса.
    """
    return Path() / "data"

@fixdir
@avito
def get_avito_configs_dpath() -> Path:
    """
    Каталог для конфигурационных файлов для авито кейса.
    :return: Path: Путь к каталогу конфигурационных файлов для авито кейса.
    """
    return Path() / "configs"

@fixdir
@sber
def get_sber_configs_dpath() -> Path:
    """
    Каталог для конфигурационных файлов для сбер кейса.
    :return: Path: Путь к каталогу конфигурационных файлов для сбер кейса.
    """
    return Path() / "configs"

@fixdir
@avito
def get_avito_cache_dpath() -> Path:
    """
    Каталог для хранения кэша для авито кейса.
    :return: Path: Путь к каталогу кэша для авито кейса.
    """
    return Path() / ".cache"

@fixdir
@sber
def get_sber_cache_dpath() -> Path:
    """
    Каталог для хранения кэша для сбер кейса.
    :return: Path: Путь к каталогу кэша для сбер кейса.
    """
    return Path() / ".cache"

@fixdir
def get_avito_tests_dpath() -> Path:
    """
    Каталог для хранения тестов для авито кейса.
    :return: Path: Путь к каталогу тестов для авито кейса.
    """
    return get_avito_dpath() / "tests"

@fixdir
def get_sber_tests_dpath() -> Path:
    """
    Каталог для хранения тестов для сбер кейса.
    :return: Path: Путь к каталогу тестов для сбер кейса.
    """
    return get_sber_dpath() / "tests"


