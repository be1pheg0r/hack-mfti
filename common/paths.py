from functools import wraps
from pathlib import Path
from typing import *

ANCHOR = "anch"
P = ParamSpec("P")
PathLike = str | Path


def fixdir(function: Callable[P, PathLike]) -> Callable[P, PathLike]:
    """Декоратор, создающий директорию, если её ещё нет.

    Args:
        function: Функция, возвращающая путь к директории.

    Returns:
        Обернутая функция, создающая директорию перед возвратом пути.
    """

    @wraps(function)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> PathLike:
        dir_path = Path(function(*args, **kwargs))
        dir_path.mkdir(parents=True, exist_ok=True)
        return dir_path

    return wrapper


def avito(function: Callable[P, PathLike]) -> Callable[P, PathLike]:
    """Декоратор для добавления префикса "avito" к пути.

    Args:
        function: Функция, возвращающая путь.

    Returns:
        Обернутая функция.
    """

    @wraps(function)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> PathLike:
        return get_project_root() / "avito" / Path(function(*args, **kwargs))

    return wrapper

def sber(function: Callable[P, PathLike]) -> Callable[P, PathLike]:
    """Декоратор для добавления префикса "sber" к пути.

    Args:
        function: Функция, возвращающая путь.

    Returns:
        Обернутая функция.
    """

    @wraps(function)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> PathLike:
        return get_project_root() / "sber" / Path(function(*args, **kwargs))

    return wrapper


def get_project_root() -> PathLike:
    """Возвращает корневой каталог проекта.

    Returns:
        Путь к корню проекта.
    """
    current_path: PathLike = Path(__file__).resolve().parent
    while not (current_path / ANCHOR).exists():
        if current_path.parent == current_path:
            raise FileNotFoundError(f"Файл '{ANCHOR}' не найден в корневом каталоге проекта.")
        current_path = current_path.parent
    return current_path


@fixdir
def get_data_dpath() -> PathLike:
    """Каталог для хранения данных проекта.

    Returns:
        Путь к каталогу данных проекта.
    """
    return get_project_root() / "data"


@fixdir
@avito
def get_avito_dpath() -> PathLike:
    """Каталог ресурсов для кейса Авито.

    Returns:
        Путь к каталогу ресурсов Авито.
    """
    return Path()


@fixdir
@sber
def get_sber_dpath() -> PathLike:
    """Каталог ресурсов для кейса Сбер.

    Returns:
        Путь к каталогу ресурсов Сбер.
    """
    return Path()


@fixdir
@avito
def get_avito_data_dpath() -> PathLike:
    """Каталог данных кейса Авито.

    Returns:
        Путь к каталогу данных Авито.
    """
    return Path() / "data"


@fixdir
@avito
def get_avito_gitignore_dpath() -> PathLike:
    """Каталог служебных артефактов Авито, исключенных из git.

    Returns:
        Путь к avito/gitignore.
    """
    return Path() / "gitignore"


@fixdir
@avito
def get_avito_gitignore_data_dpath() -> PathLike:
    """Каталог служебных датасетов Авито, исключенных из git.

    Returns:
        Путь к avito/data/gitignore.
    """
    return Path() / "data" / "gitignore"


@fixdir
@sber
def get_sber_data_dpath() -> PathLike:
    """Каталог данных кейса Сбер.

    Returns:
        Путь к каталогу данных Сбер.
    """
    return Path() / "data"


@fixdir
@avito
def get_avito_configs_dpath() -> PathLike:
    """Каталог конфигурационных файлов кейса Авито.

    Returns:
        Путь к каталогу конфигов Авито.
    """
    return Path() / "configs"

@fixdir
@sber
def get_sber_configs_dpath() -> PathLike:
    """Каталог конфигурационных файлов кейса Сбер.

    Returns:
        Путь к каталогу конфигов Сбер.
    """
    return Path() / "configs"


@fixdir
@avito
def get_avito_cache_dpath() -> PathLike:
    """Каталог для кэша кейса Авито.

    Returns:
        Путь к каталогу кэша Авито.
    """
    return Path() / ".cache"

@fixdir
@sber
def get_sber_cache_dpath() -> PathLike:
    """Каталог для кэша кейса Сбер.

    Returns:
        Путь к каталогу кэша Сбер.
    """
    return Path() / ".cache"


@fixdir
@avito
def get_avito_checkpoints_dpath() -> PathLike:
    """Каталог контрольных точек моделей кейса Авито.

    Returns:
        Путь к каталогу чекпоинтов Авито.
    """
    return Path() / "checkpoints"


@fixdir
def get_avito_tests_dpath() -> PathLike:
    """Каталог тестов кейса Авито.

    Returns:
        Путь к каталогу тестов Авито.
    """
    return get_avito_dpath() / "tests"

@fixdir
def get_sber_tests_dpath() -> PathLike:
    """Каталог тестов кейса Сбер.

    Returns:
        Путь к каталогу тестов Сбер.
    """
    return get_sber_dpath() / "tests"


@fixdir
def get_secrets_dpath() -> PathLike:
    """Каталог для секретов проекта.

    Returns:
        Путь к каталогу секретов.
    """
    return get_project_root() / ".credentials"

def get_mistral_api_keys_fpath() -> PathLike:
    """Путь к файлу с API-ключами для Mistral.

    Returns:
        Путь к файлу ключей.
    """
    return get_secrets_dpath() / "mistral_api_keys"

@fixdir
def get_checkpoints_dpath() -> PathLike:
    """Глобальный каталог контрольных точек моделей.

    Returns:
        Путь к каталогу чекпоинтов.
    """
    return get_project_root() / "checkpoints"