from __future__ import annotations

import json
from pathlib import Path
from typing import *

import yaml

PathLike = str | Path
SerializedData = dict[str, Any] | list[Any]


def read_yaml(fpath: PathLike) -> SerializedData | None:
    """Считывает YAML-файл и возвращает его содержимое.

    Args:
        fpath: Путь к YAML-файлу.

    Returns:
        Содержимое YAML в виде словаря, списка или None.
    """
    with open(fpath, "r", encoding="utf-8") as file:
        content: SerializedData | None = yaml.safe_load(file)
    return content


def parse_yaml_string(yaml_string: str) -> SerializedData | None:
    """Парсит YAML-строку.

    Args:
        yaml_string: YAML-строка.

    Returns:
        Содержимое YAML в виде словаря, списка или None при ошибке разбора.
    """
    try:
        content: SerializedData | None = yaml.safe_load(yaml_string)
        return content
    except yaml.YAMLError:
        return None


def read_json(fpath: PathLike) -> SerializedData:
    """Считывает JSON-файл и возвращает его содержимое.

    Args:
        fpath: Путь к JSON-файлу.

    Returns:
        Содержимое JSON в виде словаря или списка.
    """
    with open(fpath, "r", encoding="utf-8") as file:
        content: SerializedData = json.load(file)
    return content


def read_txt(fpath: PathLike) -> str:
    """Считывает текстовый файл.

    Args:
        fpath: Путь к текстовому файлу.

    Returns:
        Содержимое файла.
    """
    with open(fpath, "r", encoding="utf-8") as file:
        content: str = file.read()
    return content


def write_txt(fpath: PathLike, content: str) -> None:
    """Записывает строковое содержимое в файл.

    Args:
        fpath: Путь к текстовому файлу.
        content: Содержимое для записи.
    """
    with open(fpath, "w", encoding="utf-8") as file:
        file.write(content)


def read_file(fpath: PathLike) -> list[str]:
    """Считывает файл и возвращает его содержимое в виде списка строк.

    Args:
        fpath: Путь к файлу.

    Returns:
        Список строк из файла.
    """
    with open(fpath, "r", encoding="utf-8") as file:
        lines: list[str] = [line.strip() for line in file if line.strip()]
    return lines
