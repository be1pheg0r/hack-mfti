from __future__ import annotations

from pathlib import Path

from common.paths import (
    ANCHOR,
    fixdir,
    get_avito_checkpoints_dpath,
    get_avito_dpath,
    get_checkpoints_dpath,
    get_project_root,
)


def test_project_root_contains_anchor() -> None:
    root = Path(get_project_root())
    assert (root / ANCHOR).exists(), "Файл-метка anch должен существовать в корне проекта"


def test_fixdir_creates_directory(tmp_path: Path) -> None:
    @fixdir
    def _make_dir() -> Path:
        return tmp_path / "nested" / "inner"

    created = Path(_make_dir())
    assert created.is_dir(), "fixdir должен создавать отсутствующую директорию"


def test_avito_paths_use_project_root_prefix(monkeypatch: object, tmp_path: Path) -> None:
    monkeypatch.setattr("common.paths.get_project_root", lambda: tmp_path)

    avito_root = Path(get_avito_dpath())
    checkpoints = Path(get_avito_checkpoints_dpath())

    assert avito_root == tmp_path / "avito"
    assert avito_root.is_dir(), "Должна быть создана директория avito"
    assert checkpoints == tmp_path / "avito" / "checkpoints"
    assert checkpoints.is_dir(), "Должна быть создана директория avito/checkpoints"


def test_global_checkpoints_path(monkeypatch: object, tmp_path: Path) -> None:
    monkeypatch.setattr("common.paths.get_project_root", lambda: tmp_path)

    checkpoints = Path(get_checkpoints_dpath())

    assert checkpoints == tmp_path / "checkpoints"
    assert checkpoints.is_dir(), "Глобальный каталог checkpoints должен быть создан"
