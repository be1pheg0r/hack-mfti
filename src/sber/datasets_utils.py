from __future__ import annotations

import json
import subprocess
import zipfile
from pathlib import Path
from typing import *
import platform

from pydantic import BaseModel, ConfigDict, Field, field_validator

from common.files import read_yaml
from common.logger import SBER_DATASETS_LOGGER as logger
from common.paths import PathLike, get_data_raw_dpath
from .constants import (
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
    root: Path = Path(data_root) if data_root is not None else Path(get_data_raw_dpath())
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
    curl = "curl.exe" if platform.system() == "Windows" else "curl"
    return [curl, "-L", "-o", str(archive_fpath), url]


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


def build_tape_data_dir(tape_name: str) -> str:
    """Возвращает базовый путь до датасета внутри RussianNLP/tape.

    Args:
        tape_name: Имя датасета, например `chegeka.raw`.

    Returns:
        Путь вида `dummy/raw/{dataset_name}`.
    """
    normalized_name: str = tape_name.strip()
    if not normalized_name:
        raise ValueError("tape_name не может быть пустым")

    dataset_name: str = normalized_name.removesuffix(".raw")
    return f"dummy/raw/{dataset_name}"


def build_tape_data_files(tape_name: str) -> dict[str, str]:
    """Формирует пути train/test jsonl внутри репозитория TAPE."""
    data_dir: str = build_tape_data_dir(tape_name=tape_name)
    return {
        "train": f"{data_dir}/train.jsonl",
        "test": f"{data_dir}/test.jsonl",
    }


def download_tape_data_files(
    config: SberDatasetsConfig,
    tape_name: str,
    cache_dir: Path,
) -> dict[str, str]:
    """Скачивает train/test jsonl из Hugging Face Hub для TAPE.

    Args:
        config: Конфиг загрузки датасетов.
        tape_name: Имя датасета, например `chegeka.raw`.
        cache_dir: Каталог кэша Hugging Face.

    Returns:
        Словарь локальных путей к файлам train/test.
    """
    from huggingface_hub import hf_hub_download

    repo_files: dict[str, str] = build_tape_data_files(tape_name=tape_name)
    local_files: dict[str, str] = {}

    for split_name, repo_filename in repo_files.items():
        local_fpath: str = hf_hub_download(
            repo_id=config.tape_repo,
            repo_type="dataset",
            filename=repo_filename,
            cache_dir=str(cache_dir),
        )
        local_files[split_name] = local_fpath

    return local_files


def _normalize_multiq_answers(raw_value: Any) -> list[dict[str, Any]]:
    """Нормализует поле ответов multiq к списку словарей."""
    if raw_value is None:
        return []
    if isinstance(raw_value, list):
        return [item for item in raw_value if isinstance(item, dict)]
    if isinstance(raw_value, dict):
        return [raw_value] if raw_value else []
    return []


def read_jsonl_records(fpath: PathLike, tape_name: str) -> list[dict[str, Any]]:
    """Считывает jsonl-файл в список словарей.

    Args:
        fpath: Путь к jsonl-файлу.
        tape_name: Имя TAPE датасета для условной нормализации полей.

    Returns:
        Список записей датасета.
    """
    records: list[dict[str, Any]] = []
    with open(fpath, "r", encoding="utf-8") as file:
        for line in file:
            payload: str = line.strip()
            if not payload:
                continue
            item: Any = json.loads(payload)
            if not isinstance(item, dict):
                continue

            if tape_name.startswith("multiq"):
                item["bridge_answers"] = _normalize_multiq_answers(item.get("bridge_answers"))
                item["main_answers"] = _normalize_multiq_answers(item.get("main_answers"))

            records.append(item)
    return records


def load_tape_dataset(
    config: SberDatasetsConfig,
    tape_name: str,
    data_root: PathLike | None = None,
) -> list[dict[str, Any]]:
    """Загружает train-часть датасета из RussianNLP/tape.

    Источник: https://huggingface.co/datasets/RussianNLP/tape
    Пути: `dummy/raw/{dataset_name}/train.jsonl` и `dummy/raw/{dataset_name}/test.jsonl`.

    Возвращает список словарей, чтобы избежать проблем нестрогого schema-casting
    в старых версиях `datasets`.
    """
    cache_root: Path = Path(data_root) if data_root is not None else Path(get_data_raw_dpath())
    cache_dir: Path = cache_root / config.tape_cache_subdir
    data_files: dict[str, str] = download_tape_data_files(
        config=config,
        tape_name=tape_name,
        cache_dir=cache_dir,
    )

    train_records: list[dict[str, Any]] = read_jsonl_records(
        fpath=data_files["train"],
        tape_name=tape_name,
    )
    logger.info(f"Загружено {len(train_records)} записей для TAPE {tape_name}")
    return train_records


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

