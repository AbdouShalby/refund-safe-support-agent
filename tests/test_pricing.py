"""Exact integer pricing and decimal USD conversions."""

import pytest

from app.model import Usage
from app.pricing import cost_nano, nano_to_usd, usd_to_nano, worst_case_nano


def test_spike_cost() -> None:
    assert cost_nano(Usage(6640, 1491, 13419, 894)) == 1_431_565
    assert nano_to_usd(1_431_565) == "0.001432"
    assert nano_to_usd(1_430_000) == "0.001430"


def test_worst_case() -> None:
    assert worst_case_nano(1000, 400) == 325_000


def test_usd_conversion() -> None:
    assert usd_to_nano("0.12") == 120_000_000
    assert usd_to_nano("0") == 0
    assert usd_to_nano("0.000000001") == 1
    with pytest.raises(TypeError):
        usd_to_nano(0.12)


@pytest.mark.parametrize("value", ["-0.01", "NaN", "Infinity", "invalid", "0.0000000001"])
def test_invalid_usd(value: str) -> None:
    with pytest.raises(ValueError):
        usd_to_nano(value)
