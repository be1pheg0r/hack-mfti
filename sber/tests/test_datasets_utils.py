from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import *

from sber.datasets_utils import SberDatasetsConfig, build_tape_data_dir, load_tape_dataset


class _FakeDatasetDict(dict[str, Any]):
    pass


def test_build_tape_data_dir() -> None:
    assert build_tape_data_dir("chegeka.raw") == "dummy/raw/chegeka"
    assert build_tape_data_dir("multiq.raw") == "dummy/raw/multiq"


def test_load_tape_dataset_passes_name_and_data_dir(monkeypatch: Any, tmp_path: Path) -> None:
    captured: dict[str, Any] = {}

    def fake_load_dataset(*args: Any, **kwargs: Any) -> _FakeDatasetDict:
        captured["args"] = args
        captured["kwargs"] = kwargs
        return _FakeDatasetDict({"train": [{"question": "q", "answer": "a"}]})

    fake_module = SimpleNamespace(load_dataset=fake_load_dataset)
    monkeypatch.setitem(sys.modules, "datasets", fake_module)

    config = SberDatasetsConfig(tape_repo="RussianNLP/tape", tape_cache_subdir="tape")
    train = load_tape_dataset(config=config, tape_name="chegeka.raw", data_root=tmp_path)

    assert train == [{"question": "q", "answer": "a"}]
    assert captured["args"] == ("RussianNLP/tape",)
    assert captured["kwargs"]["name"] == "chegeka.raw"
    assert captured["kwargs"]["data_dir"] == "dummy/raw/chegeka"
    assert captured["kwargs"]["cache_dir"] == str(tmp_path / "tape")

