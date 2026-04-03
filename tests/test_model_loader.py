from __future__ import annotations

from pathlib import Path
from typing import *

import torch

from src.sber.models.model_loader import SberHFModelLoader

def test_sber_hf_model_loader_loads_bundle_without_network(monkeypatch: Any, tmp_path: Path) -> None:
    class DummyTokenizer:
        def __init__(self) -> None:
            self.padding_side: str | None = None
            self.pad_token_id: int | None = None
            self.eos_token: str = "<eos>"
            self.pad_token: str | None = None

    class DummyModel:
        def __init__(self) -> None:
            self.eval_called: bool = False
            self.to_device: torch.device | None = None

        def eval(self) -> DummyModel:
            self.eval_called = True
            return self

        def to(self, device: torch.device) -> DummyModel:
            self.to_device = device
            return self

    source_dir: Path = tmp_path / "hack-mfti-sbercase"
    source_dir.mkdir(parents=True, exist_ok=True)

    dummy_model = DummyModel()
    dummy_tokenizer = DummyTokenizer()

    monkeypatch.setattr("src.sber.models.model_loader.retrieve_hf_model", lambda **_: source_dir)
    monkeypatch.setattr("src.sber.models.model_loader.AutoModelForSequenceClassification.from_pretrained", lambda *args, **kwargs: dummy_model)
    monkeypatch.setattr("src.sber.models.model_loader.AutoTokenizer.from_pretrained", lambda *args, **kwargs: dummy_tokenizer)
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)

    loader = SberHFModelLoader(repo_id="be1pheg0r/hack-mfti-sbercase", cache_root=tmp_path)
    bundle = loader.load()

    assert bundle.source_path == source_dir
    assert bundle.model is dummy_model
    assert bundle.tokenizer is dummy_tokenizer
    assert bundle.device == torch.device("cpu")
    assert dummy_model.eval_called is True
    assert dummy_model.to_device == torch.device("cpu")
    assert dummy_tokenizer.padding_side == "left"
    assert dummy_tokenizer.pad_token == "<eos>"

