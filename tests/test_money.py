from decimal import Decimal

import pytest

from app.money import AMOUNT_PATTERN, fmt_amount, parse_amount


def test_parse_whole_amount() -> None:
    assert parse_amount("30") == Decimal("30.00")


def test_format_amount() -> None:
    assert fmt_amount(parse_amount("7.5")) == "7.50"


@pytest.mark.parametrize(
    "value",
    ["0", "0.00", "-5", "1e3", " 10", "10.001", "123456789", "٢٠", "20,5", ""],
)
def test_reject_invalid_amount(value: str) -> None:
    with pytest.raises(ValueError):
        parse_amount(value)


def test_reject_non_string_amount() -> None:
    with pytest.raises(ValueError):
        parse_amount(30)


def test_normalize_leading_zeros() -> None:
    assert parse_amount("00012.5") == Decimal("12.50")


def test_amount_pattern() -> None:
    assert AMOUNT_PATTERN == r"^[0-9]{1,8}(\.[0-9]{1,2})?$"
