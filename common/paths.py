from __future__ import annotations
from functools import wraps
from pathlib import Path
from typing import *

ANCHOR = "anch"
P = ParamSpec("P")
PathLike = str | Path


def fixdir(function: Callable[P, PathLike]) -> Callable[P, PathLike]:
    """Создаёт директорию, если она ещё не существует."""

    @wraps(function)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> PathLike:
        dir_path = Path(function(*args, **kwargs))
        dir_path.mkdir(parents=True, exist_ok=True)
        return dir_path

    return wrapper


def avito(function: Callable[P, PathLike]) -> Callable[P, PathLike]:
    """Добавляет префикс `avito` к пути."""

    @wraps(function)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> PathLike:
        return get_project_root() / "avito" / Path(function(*args, **kwargs))

    return wrapper


@fixdir
def get_project_root() -> PathLike:
    """Возвращает корневой каталог проекта."""
    current_path: PathLike = Path(__file__).resolve().parent
    while not (current_path / ANCHOR).exists():
        if current_path.parent == current_path:
            raise FileNotFoundError(f"Файл '{ANCHOR}' не найден в корневом каталоге проекта.")
        current_path = current_path.parent
    return current_path


@fixdir
def get_data_dpath() -> PathLike:
    """Возвращает корневой каталог данных проекта."""
    return get_project_root() / "data"


@fixdir
def get_data_bench_dpath() -> PathLike:
    """Возвращает каталог benchmark-данных."""
    return get_data_dpath() / "bench"


@fixdir
def get_data_raw_dpath() -> PathLike:
    """Возвращает каталог сырых данных."""
    return get_data_dpath() / "raw"


@fixdir
def get_configs_dpath() -> PathLike:
    """Возвращает каталог конфигураций проекта."""
    return get_project_root() / "configs"


@fixdir
def get_model_dpath() -> PathLike:
    """Возвращает каталог артефактов моделей."""
    return get_project_root() / "model"


@fixdir
def get_src_dpath() -> PathLike:
    """Возвращает каталог исходного кода."""
    return get_project_root() / "src"


@fixdir
def get_scripts_dpath() -> PathLike:
    """Возвращает каталог скриптов."""
    return get_project_root() / "scripts"


@fixdir
def get_notebooks_dpath() -> PathLike:
    """Возвращает каталог ноутбуков."""
    return get_project_root() / "notebooks"


@fixdir
def get_tests_dpath() -> PathLike:
    """Возвращает каталог тестов."""
    return get_project_root() / "tests"


@fixdir
@avito
def get_avito_dpath() -> PathLike:
    """Возвращает каталог кейса Avito."""
    return Path()


@fixdir
@avito
def get_avito_data_dpath() -> PathLike:
    """Возвращает каталог данных Avito-кейса."""
    return Path() / "data"


@fixdir
@avito
def get_avito_configs_dpath() -> PathLike:
    """Возвращает каталог конфигов Avito-кейса."""
    return Path() / "configs"


@fixdir
@avito
def get_avito_cache_dpath() -> PathLike:
    """Возвращает каталог кэша Avito-кейса."""
    return Path() / ".cache"


@fixdir
def get_secrets_dpath() -> PathLike:
    """Возвращает каталог секретов проекта."""
    return get_project_root() / ".credentials"


def get_mistral_api_keys_fpath() -> PathLike:
    """Возвращает путь к файлу с API-ключами для Mistral."""
    return get_secrets_dpath() / "mistral_api_keys"


@fixdir
def get_checkpoints_dpath() -> PathLike:
    """Возвращает каталог общего кэша контрольных точек."""
    return get_project_root() / "checkpoints"
