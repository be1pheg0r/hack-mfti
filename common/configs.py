from __future__ import annotations

import argparse
from typing import *

from pydantic import BaseModel

from .files import read_yaml
from .paths import PathLike

ConfigT = TypeVar("ConfigT", bound=BaseModel)


def load_yaml_payload(fpath: PathLike, section_name: str | None = None) -> dict[str, Any]:
    """Загружает словарь конфигурации из YAML.

    Args:
        fpath: Путь до YAML-файла.
        section_name: Имя секции верхнего уровня, если конфиг вложен.

    Returns:
        Словарь с конфигурационными значениями.
    """
    raw_data: Any = read_yaml(fpath)
    if not isinstance(raw_data, dict):
        raise ValueError("YAML-конфиг должен быть словарем")

    payload: Any = raw_data.get(section_name, raw_data) if section_name is not None else raw_data
    if not isinstance(payload, dict):
        if section_name is None:
            raise ValueError("YAML-конфиг должен быть словарем")
        raise ValueError(f"Секция {section_name} должна быть словарем")
    return payload


def load_pydantic_config(config_cls: type[ConfigT], fpath: PathLike, section_name: str | None = None) -> ConfigT:
    """Создает Pydantic-конфиг из YAML-файла.

    Args:
        config_cls: Класс Pydantic-модели.
        fpath: Путь до YAML-файла.
        section_name: Имя секции верхнего уровня, если конфиг вложен.

    Returns:
        Валидированный экземпляр конфигурации.
    """
    payload: dict[str, Any] = load_yaml_payload(fpath=fpath, section_name=section_name)
    return config_cls.model_validate(payload)


def apply_namespace_overrides(
    config: ConfigT,
    namespace: argparse.Namespace,
    *,
    skip_keys: Iterable[str] = ("config_path",),
    key_aliases: Mapping[str, str] | None = None,
) -> ConfigT:
    """Применяет значения из CLI поверх Pydantic-конфига.

    Args:
        config: Исходный конфиг из YAML.
        namespace: Аргументы CLI.
        skip_keys: Ключи namespace, которые нужно игнорировать.
        key_aliases: Отображение `namespace_key -> config_field`.

    Returns:
        Новый валидированный конфиг с учтенными override-значениями.
    """
    payload: dict[str, Any] = config.model_dump()
    allowed_fields: set[str] = set(type(config).model_fields.keys())
    excluded_keys: set[str] = set(skip_keys)
    aliases: dict[str, str] = dict(key_aliases or {})

    for key, value in vars(namespace).items():
        if key in excluded_keys or value is None:
            continue
        target_key: str = aliases.get(key, key)
        if target_key not in allowed_fields:
            continue
        payload[target_key] = value

    return type(config).model_validate(payload)


def load_config_from_namespace(
    config_cls: type[ConfigT],
    namespace: argparse.Namespace,
    fpath: PathLike,
    *,
    section_name: str | None = None,
    skip_keys: Iterable[str] = ("config_path",),
    key_aliases: Mapping[str, str] | None = None,
) -> ConfigT:
    """Загружает YAML-конфиг и применяет CLI override за один вызов."""
    config: ConfigT = load_pydantic_config(config_cls=config_cls, fpath=fpath, section_name=section_name)
    return apply_namespace_overrides(
        config=config,
        namespace=namespace,
        skip_keys=skip_keys,
        key_aliases=key_aliases,
    )

