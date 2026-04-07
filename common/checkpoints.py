from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib

from common.paths import PathLike, get_avito_checkpoints_dpath


def resolve_checkpoint_path(filename: str | Path, base_dir: PathLike | None = None) -> Path:
    """Возвращает путь к чекпоинту кейса Авито.

    Если `filename` относительный, он будет размещен в каталоге чекпоинтов Авито
    или в каталоге, переданном через `base_dir`. Родительская директория создается автоматически.
    """
    base_dir_resolved = Path(base_dir) if base_dir is not None else Path(get_avito_checkpoints_dpath())
    path = Path(filename)
    if not path.is_absolute():
        path = base_dir_resolved / path
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def save_checkpoint(payload: Any, *, filename: str | Path, base_dir: PathLike | None = None) -> Path:
    """Сохраняет веса или артефакт модели в joblib-файл и возвращает путь."""
    path = resolve_checkpoint_path(filename=filename, base_dir=base_dir)
    joblib.dump(payload, path)
    return path
