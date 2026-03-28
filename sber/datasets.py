from __future__ import annotations

import subprocess
import zipfile
from pathlib import Path
from typing import *

from pydantic import BaseModel, ConfigDict, Field, field_validator

from common.files import read_yaml
from common.logger import SBER_DATASETS_LOGGER as logger
from common.paths import PathLike, get_sber_gitignore_data_dpath
from sber.constants import (
    DEFAULT_DATASET_NAMES,
    DEFAULT_KAGGLE_API_URL_TEMPLATE,
    DEFAULT_RUBQ_ARCHIVE_NAME,
    DEFAULT_RUBQ_DATASET_SLUG,
    DEFAULT_RUBQ_JSON_NAME,
    DEFAULT_TAPE_CACHE_SUBDIR,
    DEFAULT_TAPE_DATASET_REPO,
)


class SberDatasetsConfig(BaseModel):
    """Конфигурация загрузки датасетов для Sber-кейса.

    Attributes:
        rubq_slug: Идентификатор датасета RuBQ на Kaggle.
        rubq_archive_name: Имя zip-архива, сохраняемого локально.
        rubq_json_name: Имя JSON-файла внутри RuBQ-архива.
        tape_repo: Идентификатор репозитория TAPE в Hugging Face Datasets.
        tape_cache_subdir: Подкаталог кэша TAPE внутри gitignore data.
        dataset_names: Имена датасетов для пакетной загрузки.
        kaggle_api_url_template: Шаблон URL загрузки Kaggle API.
    """

    model_config = ConfigDict(frozen=True)

    rubq_slug: str = DEFAULT_RUBQ_DATASET_SLUG
    rubq_archive_name: str = DEFAULT_RUBQ_ARCHIVE_NAME
    rubq_json_name: str = DEFAULT_RUBQ_JSON_NAME
    tape_repo: str = DEFAULT_TAPE_DATASET_REPO
    tape_cache_subdir: str = DEFAULT_TAPE_CACHE_SUBDIR
    dataset_names: list[str] = Field(default_factory=lambda: list(DEFAULT_DATASET_NAMES))
    kaggle_api_url_template: str = DEFAULT_KAGGLE_API_URL_TEMPLATE

    @field_validator(
        "rubq_slug",
        "rubq_archive_name",
        "rubq_json_name",
        "tape_repo",
        "tape_cache_subdir",
        "kaggle_api_url_template",
    )
    @classmethod
    def validate_non_empty_string(cls, value: str) -> str:
        """Проверяет, что строковые поля не пустые."""
        normalized: str = value.strip()
        if not normalized:
            raise ValueError("Строковое поле конфигурации не может быть пустым")
        return normalized

    @field_validator("dataset_names")
    @classmethod
    def validate_dataset_names(cls, value: list[str]) -> list[str]:
        """Проверяет список имен датасетов."""
        normalized: list[str] = [name.strip() for name in value if name.strip()]
        if not normalized:
            raise ValueError("Список dataset_names не может быть пустым")
        return normalized

    @classmethod
    def from_yaml(cls, fpath: PathLike) -> SberDatasetsConfig:
        """Создает конфиг загрузки датасетов из YAML-файла.

        Args:
            fpath: Путь к YAML-файлу.

        Returns:
            Валидированный конфиг.
        """
        raw_data: Any = read_yaml(fpath)
        if not isinstance(raw_data, dict):
            raise ValueError("YAML-конфиг должен быть словарем")

        payload: Any = raw_data.get("datasets", raw_data)
        if not isinstance(payload, dict):
            raise ValueError("Секция datasets должна быть словарем")
        return cls.model_validate(payload)


def build_kaggle_download_url(config: SberDatasetsConfig) -> str:
    """Собирает URL загрузки архива RuBQ через Kaggle API."""
    return config.kaggle_api_url_template.format(dataset_slug=config.rubq_slug)


def get_rubq_local_dir_name(config: SberDatasetsConfig) -> str:
    """Возвращает имя локального каталога RuBQ, вычисленное из slug."""
    return config.rubq_slug.rsplit("/", maxsplit=1)[-1]


def get_rubq_target_dir(config: SberDatasetsConfig, data_root: PathLike | None = None) -> Path:
    """Возвращает каталог для локального хранения RuBQ."""
    root: Path = Path(data_root) if data_root is not None else Path(get_sber_gitignore_data_dpath()) 
    return root / get_rubq_local_dir_name(config)


def build_curl_download_command(
    url: str,
    archive_fpath: PathLike,
) -> list[str]:
    """Формирует команду curl для скачивания датасета.

    Args:
        url: URL загрузки архива.
        archive_fpath: Локальный путь к zip-архиву.
    Returns:
        Команда для subprocess.run.
    """
    return [
        "curl.exe",
        "-L",
        "-o",
        str(archive_fpath),
        url,
    ]


def download_archive(command: list[str]) -> None:
    """Запускает скачивание архива через subprocess."""
    logger.info("Запускаю скачивание архива датасета")
    subprocess.run(command, check=True)


def extract_archive(archive_fpath: PathLike, target_dir: PathLike, remove_archive: bool = False) -> Path:
    """Распаковывает zip-архив в целевой каталог.

    Args:
        archive_fpath: Путь к архиву.
        target_dir: Каталог для распаковки.
        remove_archive: Флаг удаления архива после распаковки.

    Returns:
        Путь к каталогу распаковки.
    """
    archive_path: Path = Path(archive_fpath)
    output_dir: Path = Path(target_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(archive_path, "r") as zip_file:
        zip_file.extractall(output_dir)

    if remove_archive and archive_path.exists():
        archive_path.unlink()

    logger.info(f"Архив распакован в {output_dir}")
    return output_dir


def ensure_rubq_dataset(
    config: SberDatasetsConfig,
    data_root: PathLike | None = None,
    force_download: bool = False,
) -> Path:
    """Гарантирует наличие локального JSON-файла RuBQ.

    Args:
        config: Конфиг загрузки датасетов.
        data_root: Базовый каталог данных.
        force_download: Принудительно перекачать архив.

    Returns:
        Путь к файлу `RuBQ_2.0_dev.json`.
    """
    target_dir: Path = get_rubq_target_dir(config=config, data_root=data_root)
    target_dir.mkdir(parents=True, exist_ok=True)

    json_fpath: Path = target_dir / config.rubq_json_name
    archive_fpath: Path = target_dir / config.rubq_archive_name

    if json_fpath.exists() and not force_download:
        logger.info(f"Найден локальный файл датасета: {json_fpath}")
        return json_fpath

    url: str = build_kaggle_download_url(config=config)
    command: list[str] = build_curl_download_command(url=url, archive_fpath=archive_fpath)

    download_archive(command=command)
    extract_archive(archive_fpath=archive_fpath, target_dir=target_dir, remove_archive=False)

    if not json_fpath.exists():
        raise FileNotFoundError(f"После распаковки не найден файл {json_fpath}")
    return json_fpath


def load_rubq_dataset(
    config: SberDatasetsConfig,
    data_root: PathLike | None = None,
    force_download: bool = False,
) -> Any:
    """Загружает RuBQ в формате Hugging Face Dataset."""
    from datasets import load_dataset

    json_fpath: Path = ensure_rubq_dataset(
        config=config,
        data_root=data_root,
        force_download=force_download,
    )
    dataset: Any = load_dataset("json", data_files=str(json_fpath))["train"]
    return dataset


def load_tape_dataset(
    config: SberDatasetsConfig,
    tape_name: str,
    data_root: PathLike | None = None,
) -> Any:
    """Загружает датасет из репозитория TAPE."""
    from datasets import load_dataset

    cache_root: Path = Path(data_root) if data_root is not None else Path(get_sber_gitignore_data_dpath()) 
    cache_dir: Path = cache_root / config.tape_cache_subdir
    return load_dataset(
        config.tape_repo,
        name=tape_name,
        trust_remote_code=True,
        cache_dir=str(cache_dir),
    )["train"]


def retrieve_dataset(
    name: str,
    config: SberDatasetsConfig,
    data_root: PathLike | None = None,
    force_download: bool = False,
) -> Any:
    """Загружает один датасет по имени из пайплайна notebook."""
    normalized_name: str = name.strip()
    if normalized_name == "rubq-20":
        return load_rubq_dataset(
            config=config,
            data_root=data_root,
            force_download=force_download,
        )

    if normalized_name.startswith("tape-"):
        tape_name: str = normalized_name.split("-", maxsplit=1)[1]
        return load_tape_dataset(
            config=config,
            tape_name=tape_name,
            data_root=data_root,
        )

    raise ValueError(f"Неподдерживаемое имя датасета: {name}")


def retrieve_all_datasets(
    config: SberDatasetsConfig,
    data_root: PathLike | None = None,
    force_download: bool = False,
) -> dict[str, Any]:
    """Загружает все датасеты из `config.dataset_names`."""
    loaded: dict[str, Any] = {}
    for name in config.dataset_names:
        logger.info(f"Загружаю датасет: {name}")
        loaded[name] = retrieve_dataset(
            name=name,
            config=config,
            data_root=data_root,
            force_download=force_download,
        )
    return loaded


def unpack_dataset(dataset: Iterable[dict[str, Any]], name: str) -> tuple[list[str], list[str]]:
    """Извлекает пары вопрос-ответ из поддерживаемых датасетов.

    Args:
        dataset: Итерируемая коллекция примеров.
        name: Имя датасета из пайплайна.

    Returns:
        Списки `queries` и `answers` одинаковой длины.
    """
    queries: list[str] = []
    answers: list[str] = []

    if name == "rubq-20":
        for item in dataset:
            query: str = str(item["question_text"])
            answer: str = str(item["answer_text"])
            queries.append(query)
            answers.append(answer)
        return queries, answers

    if name == "tape-chegeka.raw":
        for sample in dataset:
            question: str = str(sample["question"])
            correct_answer: str = str(sample["answer"])
            queries.append(question)
            answers.append(correct_answer)
        return queries, answers

    if name == "tape-multiq.raw":
        for sample in dataset:
            question = str(sample["question"])
            correct_answer = str(sample["main_answers"][0]["segment"])
            queries.append(question)
            answers.append(correct_answer)
        return queries, answers

    raise ValueError(f"Неподдерживаемое имя датасета для распаковки: {name}")


def unpack_all_datasets(datasets_map: dict[str, Iterable[dict[str, Any]]]) -> tuple[list[str], list[str]]:
    """Извлекает пары вопрос-ответ для всех датасетов из словаря."""
    all_queries: list[str] = []
    all_answers: list[str] = []

    for name, dataset in datasets_map.items():
        queries, answers = unpack_dataset(dataset=dataset, name=name)
        all_queries.extend(queries)
        all_answers.extend(answers)

    return all_queries, all_answers



