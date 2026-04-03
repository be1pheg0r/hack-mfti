from __future__ import annotations

from pathlib import Path
from typing import *

import torch

from src.sber.models.hf_nli_clf import HFNLIClf, HFNLIClfConfig


def test_hf_nli_clf_loads_bundle_without_network(monkeypatch: Any, tmp_path: Path) -> None:
    class DummyTokenizer:
        def __init__(self) -> None:
            self.padding_side: str | None = None
            self.pad_token_id: int | None = None
            self.eos_token: str = "<eos>"
            self.pad_token: str | None = None

        class _Encoded(dict[str, torch.Tensor]):
            def to(self, _device: torch.device) -> DummyTokenizer._Encoded:
                return self

        def __call__(
            self,
            premises: list[str],
            hypotheses: list[str],
            truncation: bool,
            max_length: int,
            padding: bool,
            return_tensors: str,
        ) -> DummyTokenizer._Encoded:
            assert truncation is True
            assert max_length == 16
            assert padding is True
            assert return_tensors == "pt"
            assert len(premises) == len(hypotheses)
            batch_size: int = len(premises)
            return self._Encoded(
                {
                    "input_ids": torch.ones((batch_size, 2), dtype=torch.long),
                    "attention_mask": torch.ones((batch_size, 2), dtype=torch.long),
                }
            )

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

        def __call__(self, **kwargs: Any) -> Any:
            batch_size: int = int(kwargs["input_ids"].shape[0])
            logits: torch.Tensor = torch.tensor([[0.2, 1.0]] * batch_size, dtype=torch.float32)
            return type("DummyOutput", (), {"logits": logits})

    source_dir: Path = tmp_path / "hack-mfti-sbercase"
    source_dir.mkdir(parents=True, exist_ok=True)

    dummy_model = DummyModel()
    dummy_tokenizer = DummyTokenizer()

    monkeypatch.setattr("src.sber.models.hf_nli_clf.retrieve_hf_model", lambda **_: source_dir)
    monkeypatch.setattr("src.sber.models.hf_nli_clf.AutoModelForSequenceClassification.from_pretrained", lambda *args, **kwargs: dummy_model)
    monkeypatch.setattr("src.sber.models.hf_nli_clf.AutoTokenizer.from_pretrained", lambda *args, **kwargs: dummy_tokenizer)
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)

    clf = HFNLIClf(
        config=HFNLIClfConfig(
            repo_id="be1pheg0r/hack-mfti-sbercase",
            cache_root=tmp_path,
            max_length=16,
            batch_size=2,
            positive_class_index=1,
            hallucination_threshold=0.5,
        )
    )
    bundle = clf.load()

    assert bundle.source_path == source_dir
    assert bundle.model is dummy_model
    assert bundle.tokenizer is dummy_tokenizer
    assert bundle.device == torch.device("cpu")
    assert dummy_model.eval_called is True
    assert dummy_model.to_device == torch.device("cpu")
    assert dummy_tokenizer.padding_side == "left"
    assert dummy_tokenizer.pad_token == "<eos>"

    hallucination_scores: list[float] = clf.predict_hallucination_proba(
        premises=["p1", "p2"],
        hypotheses=["h1", "h2"],
    )
    entailment_scores: list[float] = clf.predict_entailment_proba(
        premises=["p1", "p2"],
        hypotheses=["h1", "h2"],
    )
    predictions: list[int] = clf.predict_is_hallucination(
        premises=["p1", "p2"],
        hypotheses=["h1", "h2"],
    )
    predictions_with_strict_threshold: list[int] = clf.predict_is_hallucination(
        premises=["p1", "p2"],
        hypotheses=["h1", "h2"],
        threshold=0.99,
    )

    assert len(hallucination_scores) == 2
    assert all(score > 0.5 for score in hallucination_scores)
    assert all(0.0 <= score <= 1.0 for score in entailment_scores)
    assert predictions == [1, 1]
    assert predictions_with_strict_threshold == [1, 1]

    checkpoint_path: Path = clf.download_checkpoint()
    assert checkpoint_path == source_dir


def test_hf_nli_clf_config_from_yaml(tmp_path: Path) -> None:
    config_fpath: Path = tmp_path / "hf_nli_clf.yaml"
    config_fpath.write_text(
        "hf_nli_clf:\n"
        "  repo_id: be1pheg0r/hack-mfti-sbercase\n"
        "  max_length: 256\n"
        "  batch_size: 4\n"
        "  hallucination_threshold: 0.4\n"
        "  positive_class_index: 1\n",
        encoding="utf-8",
    )

    config: HFNLIClfConfig = HFNLIClfConfig.from_yaml(config_fpath)

    assert config.repo_id == "be1pheg0r/hack-mfti-sbercase"
    assert config.max_length == 256
    assert config.batch_size == 4
    assert config.hallucination_threshold == 0.4
    assert config.positive_class_index == 1

