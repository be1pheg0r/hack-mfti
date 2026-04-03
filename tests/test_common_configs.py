from __future__ import annotations

import argparse
from pathlib import Path

import pytest
from pydantic import BaseModel

from common.configs import apply_namespace_overrides, load_config_from_namespace, load_pydantic_config, load_yaml_payload


class DemoConfig(BaseModel):
    """Минимальный конфиг для тестов общего helper'а.

    Attributes:
        foo: Строковое поле.
        bar: Числовое поле.
        baz: Опциональный флаг.
    """

    foo: str = "alpha"
    bar: int = 1
    baz: bool | None = None


def test_load_yaml_payload_reads_nested_section(tmp_path: Path) -> None:
    config_path: Path = tmp_path / "config.yaml"
    config_path.write_text("demo:\n  foo: 'beta'\n  bar: 7\n", encoding="utf-8")

    payload = load_yaml_payload(config_path, section_name="demo")

    assert payload == {"foo": "beta", "bar": 7}


def test_load_pydantic_config_falls_back_to_root_dict(tmp_path: Path) -> None:
    config_path: Path = tmp_path / "config.yaml"
    config_path.write_text("foo: 'root'\nbar: 3\n", encoding="utf-8")

    config = load_pydantic_config(DemoConfig, config_path)

    assert config.foo == "root"
    assert config.bar == 3
    assert config.baz is None


def test_apply_namespace_overrides_ignores_none_and_unknown_fields() -> None:
    config = DemoConfig(foo="base", bar=1, baz=True)
    namespace = argparse.Namespace(foo=None, bar=10, baz=None, ignored="value")

    merged = apply_namespace_overrides(config, namespace)

    assert merged.foo == "base"
    assert merged.bar == 10
    assert merged.baz is True


def test_load_config_from_namespace_supports_key_aliases(tmp_path: Path) -> None:
    config_path: Path = tmp_path / "config.yaml"
    config_path.write_text("demo:\n  foo: 'base'\n  bar: 2\n", encoding="utf-8")
    namespace = argparse.Namespace(config_path=str(config_path), alias_value="override", bar=None)

    config = load_config_from_namespace(
        DemoConfig,
        namespace,
        fpath=config_path,
        section_name="demo",
        key_aliases={"alias_value": "foo"},
    )

    assert config.foo == "override"
    assert config.bar == 2


def test_load_yaml_payload_rejects_non_mapping_section(tmp_path: Path) -> None:
    config_path: Path = tmp_path / "config.yaml"
    config_path.write_text("demo: [1, 2, 3]\n", encoding="utf-8")

    with pytest.raises(ValueError, match="должна быть словарем"):
        load_yaml_payload(config_path, section_name="demo")



