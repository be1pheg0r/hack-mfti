import yaml
import json

from typing import *
from pathlib import Path


def read_yaml(fpath: Union[str, Path]) -> dict[str, Any]:
    """
    Считать YAML файл и вернуть его содержимое в виде словаря.
    :param fpath: Путь к YAML файлу.
    :return: Dict[str, Any]: Содержимое YAML файла в виде словаря.
    """
    with open(fpath, "r", encoding="utf-8") as file:
        content = yaml.safe_load(file)
    return content


def parse_yaml_string(yaml_string: str) -> Union[dict[str, Any], list[Any], None]:
    """
    Распарсить YAML строку и вернуть её содержимое.
    :param yaml_string: YAML строка.
    :return: Содержимое YAML в виде словаря, списка или None.
    """
    try:
        content = yaml.safe_load(yaml_string)
        return content
    except yaml.YAMLError:
        return None


def read_json(fpath: Union[str, Path]) -> Union[dict[str, Any] | list[Any]]:
    """
    Считать JSON файл и вернуть его содержимое в виде словаря.
    :param fpath: Путь к JSON файлу.
    :return: Dict[str, Any]: Содержимое JSON файла в виде словаря.
    """
    with open(fpath, "r", encoding="utf-8") as file:
        content = json.load(file)
    return content

def read_txt(fpath: Union[str, Path]) -> str:
    """
    Считать текстовый файл и вернуть его содержимое в виде строки.
    :param fpath: Путь к текстовому файлу.
    :return: str: Содержимое текстового файла.
    """
    with open(fpath, "r", encoding="utf-8") as file:
        content = file.read()
    return content


def write_txt(fpath: Union[str, Path], content: str) -> None:
    """
    Записать строковое содержимое в текстовый файл.
    :param fpath: Путь к текстовому файлу.
    :param content: str: Содержимое для записи в файл.
    """
    with open(fpath, "w", encoding="utf-8") as file:
        file.write(content)
