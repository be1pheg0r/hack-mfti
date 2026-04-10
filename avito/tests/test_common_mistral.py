from __future__ import annotations
from typing import *

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

