from __future__ import annotations

"""Helpers for retrieving Sber HF models into a local gitignored cache."""

from pathlib import Path

from huggingface_hub import snapshot_download

from common.logger import SBER_HOOKS_LOGGER as logger
from common.paths import PathLike, get_model_dpath
from ..constants import DEFAULT_SBER_HF_MODEL_DIRNAME


def _resolve_target_dir(repo_id: str, target_root: PathLike | None = None) -> Path:
    """Строит локальный путь для snapshot-а модели."""
    root_path: Path = Path(target_root) if target_root is not None else Path(get_model_dpath())
    model_dir_name: str = repo_id.rsplit("/", maxsplit=1)[-1] or DEFAULT_SBER_HF_MODEL_DIRNAME
    return root_path / model_dir_name


def retrieve_hf_model(
    repo_id: str,
    target_root: PathLike | None = None,
    revision: str | None = None,
    force_download: bool = False,
) -> Path:
    """Скачивает HF snapshot в `model/`.

    Args:
        repo_id: Идентификатор репозитория на Hugging Face.
        target_root: Корневая директория, куда надо положить snapshot.
        revision: Опциональная ветка/тег/commit.
        force_download: Принудительно скачивать заново, даже если каталог уже есть.

    Returns:
        Локальный путь до snapshot-а модели.
    """
    if not repo_id.strip():
        raise ValueError("repo_id не может быть пустым")

    target_dir: Path = _resolve_target_dir(repo_id=repo_id, target_root=target_root)
    target_dir.mkdir(parents=True, exist_ok=True)

    if target_dir.exists() and any(target_dir.iterdir()) and not force_download:
        logger.info("Использую уже скачанную модель: %s", target_dir)
        return target_dir

    logger.info("Скачиваю модель %s в %s", repo_id, target_dir)
    downloaded_path: str = snapshot_download(
        repo_id=repo_id,
        local_dir=str(target_dir),
        local_dir_use_symlinks=False,
        force_download=force_download,
        revision=revision,
        repo_type="model",
    )
    result_path: Path = Path(downloaded_path)
    logger.info("Модель скачана: %s", result_path)
    return result_path


__all__ = ["retrieve_hf_model"]


