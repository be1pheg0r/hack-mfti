from __future__ import annotations

from pathlib import Path
from typing import *
import zipfile

from src.sber.datasets_utils import (
    SberDatasetsConfig,
    build_curl_download_command,
    ensure_rubq_dataset,
    extract_archive,
    unpack_dataset,
)


def test_datasets_config_from_yaml(tmp_path: Path) -> None:
    config_fpath: Path = tmp_path / "datasets_configs.yaml"
    config_fpath.write_text(
        "datasets:\n"
        "  rubq_slug: valentinbiryukov/rubq-20\n"
        "  rubq_archive_name: rubq-20.zip\n"
        "  rubq_json_name: RuBQ_2.0_dev.json\n"
        "  tape_repo: RussianNLP/tape\n"
        "  tape_cache_subdir: tape\n"
        "  dataset_names: [rubq-20, tape-chegeka.raw]\n"
        "  kaggle_api_url_template: https://www.kaggle.com/api/v1/datasets/download/{dataset_slug}\n",
        encoding="utf-8",
    )

    config: SberDatasetsConfig = SberDatasetsConfig.from_yaml(config_fpath)
    assert config.rubq_slug == "valentinbiryukov/rubq-20"
    assert config.rubq_archive_name == "rubq-20.zip"
    assert config.dataset_names == ["rubq-20", "tape-chegeka.raw"]


def test_build_curl_download_command() -> None:
    command: list[str] = build_curl_download_command(
        url="https://example.test/archive.zip",
        archive_fpath=Path("C:/tmp/archive.zip"),
    )

    assert command[0] == "curl.exe"
    assert "-L" in command
    assert "-u" not in command


def test_extract_archive(tmp_path: Path) -> None:
    archive_path: Path = tmp_path / "data.zip"
    output_dir: Path = tmp_path / "unpacked"

    with zipfile.ZipFile(archive_path, "w") as zip_file:
        payload_path: Path = tmp_path / "sample.json"
        payload_path.write_text('{"ok": true}', encoding="utf-8")
        zip_file.write(payload_path, arcname="RuBQ_2.0_dev.json")

    result_dir: Path = extract_archive(archive_fpath=archive_path, target_dir=output_dir, remove_archive=False)
    assert result_dir == output_dir
    assert (output_dir / "RuBQ_2.0_dev.json").exists()


def test_ensure_rubq_dataset_creates_json(tmp_path: Path, monkeypatch: Any) -> None:
    config = SberDatasetsConfig()

    def fake_download_archive(command: list[str]) -> None:
        archive_fpath: Path = Path(command[3])
        archive_fpath.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive_fpath, "w") as zip_file:
            generated_file: Path = archive_fpath.parent / config.rubq_json_name
            generated_file.write_text("[]", encoding="utf-8")
            zip_file.write(generated_file, arcname=config.rubq_json_name)
            generated_file.unlink()

    monkeypatch.setattr("sber.datasets_utils.download_archive", fake_download_archive)

    json_fpath: Path = ensure_rubq_dataset(config=config, data_root=tmp_path, force_download=True)
    assert json_fpath.exists()
    assert json_fpath.name == config.rubq_json_name


def test_unpack_dataset_for_supported_names() -> None:
    rubq_dataset: list[dict[str, Any]] = [{"question_text": "q", "answer_text": "a"}]
    tape_dataset: list[dict[str, Any]] = [{"question": "q1", "answer": "a1", "main_answers": [{"segment": "a2"}]}]

    q1, a1 = unpack_dataset(rubq_dataset, "rubq-20")
    q2, a2 = unpack_dataset(tape_dataset, "tape-chegeka.raw")
    q3, a3 = unpack_dataset(tape_dataset, "tape-multiq.raw")

    assert q1 == ["q"]
    assert a1 == ["a"]
    assert q2 == ["q1"]
    assert a2 == ["a1"]
    assert q3 == ["q1"]
    assert a3 == ["a2"]

