from __future__ import annotations
from pathlib import Path
from typing import *

import pytest

from common import mistral


def _build_config(keys: list[str]) -> mistral.MistralCallConfig:
    return mistral.MistralCallConfig(
        models_list=["mistral-medium-latest"],
        default_api_key=keys[0],
        api_keys=keys,
        timeout=1,
        max_attempts_per_call=1,
    )


def test_call_mistral_rotates_api_key_after_each_success(monkeypatch) -> None:
    class FakeClient:
        def __init__(self, api_key: str) -> None:
            self.api_key = api_key
            self.chat = self

        def complete(self, model: str, **kwargs: Any) -> Any:
            content = f"ok:{self.api_key}"
            return type("Response", (), {"choices": [type("Choice", (), {"message": type("Message", (), {"content": content})()})()]})()

    monkeypatch.setattr(mistral, "_api_keys", lambda: [])
    monkeypatch.setattr(mistral, "build_client", lambda api_key: FakeClient(api_key))
    monkeypatch.setattr(
        mistral,
        "safe_call",
        lambda function, *args, timeout, max_attempts, **kwargs: function(*args, **kwargs),
    )
    monkeypatch.setattr(mistral.call_mistral, "current_key_index", 0, raising=False)

    config = _build_config(["k1", "k2"])

    first = mistral.call_mistral(config=config, messages=[])
    second = mistral.call_mistral(config=config, messages=[])
    third = mistral.call_mistral(config=config, messages=[])

    assert first == "ok:k1"
    assert second == "ok:k2"
    assert third == "ok:k1"


def test_call_mistral_next_call_starts_after_successful_fallback(monkeypatch) -> None:
    class FakeClient:
        def __init__(self, api_key: str) -> None:
            self.api_key = api_key
            self.chat = self

        def complete(self, model: str, **kwargs: Any) -> Any:
            if self.api_key == "k1":
                raise RuntimeError("boom")
            content = f"ok:{self.api_key}"
            return type("Response", (), {"choices": [type("Choice", (), {"message": type("Message", (), {"content": content})()})()]})()

    monkeypatch.setattr(mistral, "_api_keys", lambda: [])
    monkeypatch.setattr(mistral, "build_client", lambda api_key: FakeClient(api_key))
    monkeypatch.setattr(
        mistral,
        "safe_call",
        lambda function, *args, timeout, max_attempts, **kwargs: function(*args, **kwargs),
    )
    monkeypatch.setattr(mistral.call_mistral, "current_key_index", 0, raising=False)

    config = _build_config(["k1", "k2"])

    first = mistral.call_mistral(config=config, messages=[])
    second = mistral.call_mistral(config=config, messages=[])

    assert first == "ok:k2"
    assert second == "ok:k2"


@pytest.mark.parametrize("demo_filename", ["demo_keys", "demo_api_keys"])
def test_api_keys_reads_demo_file_when_primary_absent(tmp_path: Path, monkeypatch, demo_filename: str) -> None:
    secrets_dpath: Path = tmp_path / ".credentials"
    secrets_dpath.mkdir(parents=True, exist_ok=True)
    demo_fpath: Path = secrets_dpath / demo_filename
    demo_fpath.write_text("k_demo_1\n\n k_demo_2 \n", encoding="utf-8")

    monkeypatch.setattr(mistral, "get_mistral_api_keys_fpath", lambda: secrets_dpath / "mistral_api_keys")
    monkeypatch.setattr(mistral, "get_secrets_dpath", lambda: secrets_dpath)

    assert mistral._api_keys() == ["k_demo_1", "k_demo_2"]


def test_api_keys_returns_empty_when_no_files(tmp_path: Path, monkeypatch) -> None:
    secrets_dpath: Path = tmp_path / ".credentials"
    secrets_dpath.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(mistral, "get_mistral_api_keys_fpath", lambda: secrets_dpath / "mistral_api_keys")
    monkeypatch.setattr(mistral, "get_secrets_dpath", lambda: secrets_dpath)

    assert mistral._api_keys() == []


