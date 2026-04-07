from __future__ import annotations

from typing import *

import pytest

from avito.embeddings import (
    DEFAULT_ENCODER_CONFIG_FPATH,
    EncoderConfig,
    SentenceTransformerEncoder,
    SentenceTransformerLike,
)


class _FakeSentenceTransformer:
    def __init__(self) -> None:
        self.last_call: dict[str, Any] = {}

    def encode(
        self,
        sentences: list[str],
        *,
        prompt: str,
        batch_size: int,
        normalize_embeddings: bool,
        convert_to_numpy: bool,
        show_progress_bar: bool,
    ) -> list[list[float]]:
        self.last_call = {
            "sentences": sentences,
            "prompt": prompt,
            "batch_size": batch_size,
            "normalize_embeddings": normalize_embeddings,
            "convert_to_numpy": convert_to_numpy,
            "show_progress_bar": show_progress_bar,
        }
        return [[0.1, 0.2] for _ in sentences]


def test_sentence_transformer_encoder_uses_constant_prompt() -> None:
    fake_model = _FakeSentenceTransformer()
    encoder = SentenceTransformerEncoder(
        config=EncoderConfig(
            hub_model_name="stub-model",
            prompt="categorize_entailment: ",
            batch_size=32,
            normalize_embeddings=True,
        ),
        backend_model=cast(SentenceTransformerLike, fake_model),
    )

    embeddings = encoder.encode(["электрика отдельно"])

    assert len(embeddings) == 1
    assert fake_model.last_call["prompt"] == "categorize_entailment: "


def test_sentence_transformer_encoder_empty_texts_validation() -> None:
    fake_model = _FakeSentenceTransformer()
    encoder = SentenceTransformerEncoder(
        config=EncoderConfig(
            hub_model_name="stub-model",
            prompt="categorize_entailment: ",
            batch_size=32,
            normalize_embeddings=True,
        ),
        backend_model=cast(SentenceTransformerLike, fake_model),
    )

    with pytest.raises(ValueError):
        encoder.encode([])


def test_encoder_config_from_yaml_path() -> None:
    config = EncoderConfig(config_path=DEFAULT_ENCODER_CONFIG_FPATH)

    assert config.local_model_path == "rubert-mini-frida"
    assert config.hub_model_name == "sergeyzh/rubert-mini-frida"
    assert config.prompt == "categorize_entailment: "
    assert config.batch_size == 32
    assert config.normalize_embeddings is True


def test_encoder_config_yaml_and_explicit_override() -> None:
    config = EncoderConfig(
        config_path=DEFAULT_ENCODER_CONFIG_FPATH,
        batch_size=64,
    )

    assert config.batch_size == 64


def test_build_model_uses_local_path_first(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    checkpoints_dir = tmp_path / "checkpoints"
    local_model_dir = checkpoints_dir / "rubert-mini-frida"
    local_model_dir.mkdir(parents=True)

    monkeypatch.setattr("avito.embeddings.get_avito_checkpoints_dpath", lambda: checkpoints_dir)

    calls: list[dict[str, Any]] = []

    class _FakeFactory:
        def __init__(self, **kwargs: Any) -> None:
            calls.append(kwargs)

        def encode(self, sentences: list[str], **kwargs: Any) -> list[list[float]]:
            return [[0.1] for _ in sentences]

    monkeypatch.setattr(
        SentenceTransformerEncoder,
        "_load_sentence_transformer_class",
        staticmethod(lambda: _FakeFactory),
    )

    config = EncoderConfig(
        local_model_path="rubert-mini-frida",
        hub_model_name="sergeyzh/rubert-mini-frida",
        prompt="categorize_entailment: ",
        batch_size=32,
        normalize_embeddings=True,
    )

    SentenceTransformerEncoder(config=config)

    assert len(calls) == 1
    assert calls[0]["model_name_or_path"] == str(local_model_dir)


def test_build_model_fallbacks_to_hub(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    checkpoints_dir = tmp_path / "checkpoints"
    checkpoints_dir.mkdir(parents=True)

    monkeypatch.setattr("avito.embeddings.get_avito_checkpoints_dpath", lambda: checkpoints_dir)

    calls: list[dict[str, Any]] = []

    class _FakeFactory:
        def __init__(self, **kwargs: Any) -> None:
            calls.append(kwargs)

        def encode(self, sentences: list[str], **kwargs: Any) -> list[list[float]]:
            return [[0.1] for _ in sentences]

    monkeypatch.setattr(
        SentenceTransformerEncoder,
        "_load_sentence_transformer_class",
        staticmethod(lambda: _FakeFactory),
    )

    config = EncoderConfig(
        local_model_path="missing-model",
        hub_model_name="sergeyzh/rubert-mini-frida",
        prompt="categorize_entailment: ",
        batch_size=32,
        normalize_embeddings=True,
    )

    SentenceTransformerEncoder(config=config)

    assert len(calls) == 1
    assert calls[0]["model_name_or_path"] == "sergeyzh/rubert-mini-frida"


def test_build_model_raises_when_both_sources_failed(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    checkpoints_dir = tmp_path / "checkpoints"
    local_model_dir = checkpoints_dir / "rubert-mini-frida"
    local_model_dir.mkdir(parents=True)

    monkeypatch.setattr("avito.embeddings.get_avito_checkpoints_dpath", lambda: checkpoints_dir)

    class _FailFactory:
        def __init__(self, **kwargs: Any) -> None:
            raise RuntimeError("init failed")

    monkeypatch.setattr(
        SentenceTransformerEncoder,
        "_load_sentence_transformer_class",
        staticmethod(lambda: _FailFactory),
    )

    config = EncoderConfig(
        local_model_path="rubert-mini-frida",
        hub_model_name="sergeyzh/rubert-mini-frida",
        prompt="categorize_entailment: ",
        batch_size=32,
        normalize_embeddings=True,
    )

    with pytest.raises(RuntimeError):
        SentenceTransformerEncoder(config=config)





