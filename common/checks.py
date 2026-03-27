from __future__ import annotations

import importlib as imp
from typing import *

VERSION_SEPARATORS: tuple[str, ...] = ("==", ">=", "<=", ">", "<")


def check_module(module_name: str) -> bool:
    """Проверяет, установлен ли модуль.

    Args:
        module_name: Название модуля для проверки.

    Returns:
        True, если модуль установлен, иначе False.
    """
    try:
        imp.import_module(module_name)
        return True
    except ImportError:
        return False


def _extract_module_name(requirement: str) -> str:
    """Извлекает имя пакета из строки зависимости.

    Args:
        requirement: Строка зависимости из requirements.

    Returns:
        Имя пакета без версии и пробелов.
    """
    for separator in VERSION_SEPARATORS:
        if separator in requirement:
            package_name: str = requirement.split(separator, maxsplit=1)[0]
            return package_name.strip()
    return requirement.strip()


def check_requirements(requirements_strings: list[str]) -> bool:
    """Проверяет, установлены ли все модули из списка.

    Args:
        requirements_strings: Список строк зависимостей.

    Returns:
        True, если все зависимости доступны для импорта, иначе False.
    """
    for requirement in requirements_strings:
        module_name: str = _extract_module_name(requirement)
        if module_name and not check_module(module_name):
            return False
    return True


