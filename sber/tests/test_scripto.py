from __future__ import annotations
from typing import *

import random

import pytest
from pydantic import ValidationError

from sber.scripto import ScriptConfig, _sample_temperature


def test_sample_temperature_is_deterministic_for_fixed_seed() -> None:
    rng_a: random.Random = random.Random(17)
    rng_b: random.Random = random.Random(17)

    sampled_a: list[float] = [
        _sample_temperature(rng=rng_a, mean=1.0, std=0.25, min_value=0.3, max_value=1.7)
        for _ in range(5)
    ]
    sampled_b: list[float] = [
        _sample_temperature(rng=rng_b, mean=1.0, std=0.25, min_value=0.3, max_value=1.7)
        for _ in range(5)
    ]

    assert sampled_a == sampled_b
    assert all(0.3 <= value <= 1.7 for value in sampled_a)


def test_sample_temperature_with_zero_std_returns_mean_with_clipping() -> None:
    rng: random.Random = random.Random(1)

    assert _sample_temperature(rng=rng, mean=1.1, std=0.0, min_value=0.3, max_value=1.7) == pytest.approx(1.1)
    assert _sample_temperature(rng=rng, mean=0.1, std=0.0, min_value=0.3, max_value=1.7) == pytest.approx(0.3)
    assert _sample_temperature(rng=rng, mean=2.5, std=0.0, min_value=0.3, max_value=1.7) == pytest.approx(1.7)


def test_script_config_rejects_temperature_bounds_in_wrong_order(tmp_path: Any) -> None:
    with pytest.raises(ValidationError):
        ScriptConfig(
            n=2,
            output_csv=tmp_path / "features.csv",
            temperature_min=1.5,
            temperature_max=1.0,
        )

