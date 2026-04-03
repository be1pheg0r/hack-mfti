from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import *

from src.sber.datasets_utils import (
    SberDatasetsConfig,
    build_tape_data_dir,
    build_tape_data_files,
    load_tape_dataset,
    read_jsonl_records,
)


def test_load_tape_dataset_downloads_jsonl_and_reads_train(monkeypatch: Any, tmp_path: Path) -> None:
    captured: dict[str, Any] = {}

    def fake_hf_hub_download(*, repo_id: str, repo_type: str, filename: str, cache_dir: str) -> str:
        captured.setdefault("downloads", []).append(
            {
                "repo_id": repo_id,
                "repo_type": repo_type,
                "filename": filename,
                "cache_dir": cache_dir,
            }
        )
        local_path: Path = Path(cache_dir) / filename
        local_path.parent.mkdir(parents=True, exist_ok=True)

        if filename.endswith("train.jsonl"):
            local_path.write_text('{"question":"q","answer":"a"}\n', encoding="utf-8")
        else:
            local_path.write_text('{"question":"tq","answer":"ta"}\n', encoding="utf-8")
        return str(local_path)

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(hf_hub_download=fake_hf_hub_download))

    config = SberDatasetsConfig(tape_repo="RussianNLP/tape", tape_cache_subdir="tape")
    train = load_tape_dataset(config=config, tape_name="chegeka.raw", data_root=tmp_path)

    assert train == [{"question": "q", "answer": "a"}]

    downloaded_filenames: list[str] = [item["filename"] for item in captured["downloads"]]
    assert "dummy/raw/chegeka/train.jsonl" in downloaded_filenames
    assert "dummy/raw/chegeka/test.jsonl" in downloaded_filenames


def test_read_jsonl_records_normalizes_multiq_answers(tmp_path: Path) -> None:
    jsonl_path: Path = tmp_path / "train.jsonl"
    jsonl_path.write_text(
        '{"question": "q1", "main_answers": {}, "bridge_answers": []}\n'
        '{"question": "q2", "main_answers": [{"segment": "ans"}], "bridge_answers": {"segment": "b"}}\n',
        encoding="utf-8",
    )

    records: list[dict[str, Any]] = read_jsonl_records(jsonl_path, tape_name="multiq.raw")
    assert records[0]["main_answers"] == []
    assert records[0]["bridge_answers"] == []
    assert records[1]["main_answers"] == [{"segment": "ans"}]
    assert records[1]["bridge_answers"] == [{"segment": "b"}]
