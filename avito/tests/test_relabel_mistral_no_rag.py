from __future__ import annotations

import json
from pathlib import Path
from typing import *

from avito.should_split.interfaces.jobs.relabel_mistral_no_rag import RelabelCliConfig, relabel_dataset


def test_relabel_dataset_preserves_signature_and_duplicates_targets(tmp_path: Path) -> None:
    input_path = tmp_path / "input.json"
    output_path = tmp_path / "output.json"

    raw_samples = [
        {
            "itemId": "1",
            "sourceMcId": 101,
            "sourceMcTitle": "Ремонт",
            "description": "делаем отдельно сантехнику",
            "targetDetectedMcIds": "[]",
            "targetSplitMcIds": "[]",
            "shouldSplit": False,
            "caseType": "legacy",
            "split": None,
        },
        {
            "itemId": "2",
            "sourceMcId": 101,
            "sourceMcTitle": "Ремонт",
            "description": "ремонт под ключ",
            "targetDetectedMcIds": "[]",
            "targetSplitMcIds": "[]",
            "shouldSplit": False,
            "caseType": "legacy",
            "split": None,
        },
    ]

    with open(input_path, "w", encoding="utf-8") as file:
        json.dump(raw_samples, file, ensure_ascii=False)

    def fake_mistral_caller(*args: Any, **kwargs: Any) -> str:
        system_prompt = kwargs["messages"][0]["content"]
        user_prompt = kwargs["messages"][1]["content"]

        if '"shouldSplit": true/false' in system_prompt:
            return (
                '{"shouldSplit": true}'
                if "отдельно сантехнику" in user_prompt
                else '{"shouldSplit": false}'
            )
        return "102, 103"

    config = RelabelCliConfig(
        input_path=input_path,
        output_path=output_path,
        model_name="mistral-medium-latest",
        save_every=1,
    )

    result_path = relabel_dataset(config=config, mistral_caller=fake_mistral_caller)
    assert result_path == output_path
    assert output_path.exists()

    with open(output_path, "r", encoding="utf-8") as file:
        relabeled = json.load(file)

    assert len(relabeled) == 2

    first = relabeled[0]
    assert set(first.keys()) == {
        "itemId",
        "sourceMcId",
        "sourceMcTitle",
        "description",
        "targetDetectedMcIds",
        "targetSplitMcIds",
        "shouldSplit",
    }
    assert first["shouldSplit"] is True
    assert first["targetDetectedMcIds"] == "[102, 103]"
    assert first["targetSplitMcIds"] == "[102, 103]"

    second = relabeled[1]
    assert second["shouldSplit"] is False
    assert second["targetDetectedMcIds"] == "[]"
    assert second["targetSplitMcIds"] == "[]"

