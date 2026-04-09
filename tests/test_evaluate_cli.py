from __future__ import annotations

from pathlib import Path
from typing import *

from src.sber.utils.evaluate_cli import parse_args


def test_evaluate_cli_supports_snake_case_input_csv(monkeypatch: Any, tmp_path: Path) -> None:
    input_csv: Path = tmp_path / "bench.csv"
    input_csv.write_text("query,model_answer\nq,a\n", encoding="utf-8")

    monkeypatch.setattr(
        "sys.argv",
        [
            "evaluate.py",
            "--input_csv",
            str(input_csv),
            "--output_csv",
            str(tmp_path / "out.csv"),
            "--checkpoint_dir",
            "sber_tabular/latest",
            "--report_dir",
            str(tmp_path / "reports"),
            "--save_plots",
        ],
    )

    config = parse_args()

    assert Path(config.input_csv) == input_csv
    assert Path(config.output_csv) == tmp_path / "out.csv"
    assert str(config.checkpoint_dir) == "sber_tabular/latest"
    assert Path(config.report_dir) == tmp_path / "reports"
    assert config.save_plots is True

