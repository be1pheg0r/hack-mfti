import importlib as imp
from typing import *
from pathlib import Path

def check_module(module_name: str) -> bool:
    """
    Проверяет, установлен ли модуль.
    :param module_name: str: Название модуля для проверки.
    :return: bool: True, если модуль установлен, иначе False.
    """
    try:
        imp.import_module(module_name)
        return True
    except ImportError:
        return False

def check_requirements(requirements_strings: List[str]) -> bool:
    """
    Проверяет, установлены ли все модули из списка.
    :param requirements_strings: List[str]: Список названий модулей для проверки.
    :return: bool: True, если все модули установлены, иначе False.
    """
    sepators = ["==", ">=", "<=", ">", "<"]
    module_name, module_version = None, None
    for req in requirements_strings:
        for sep in sepators:
            if sep in req:
                module_name, module_version = req.split(sep)
                break
        else:
            module_name = req
        if not check_module(module_name.strip()):
            return False
    return True


