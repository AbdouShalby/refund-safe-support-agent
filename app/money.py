"""Parse and format positive decimal money amounts."""

import re
from decimal import Decimal

AMOUNT_PATTERN = r"^[0-9]{1,8}(\.[0-9]{1,2})?$"


def parse_amount(value: str) -> Decimal:
    """Return a positive amount with two decimal places, or raise ValueError."""
    if not isinstance(value, str) or re.fullmatch(AMOUNT_PATTERN, value) is None:
        raise ValueError("Amount must be a decimal string with at most two decimal places")
    amount = Decimal(value).quantize(Decimal("0.01"))
    if amount <= 0:
        raise ValueError("Amount must be strictly positive")
    return amount


def fmt_amount(value: Decimal) -> str:
    """Format an amount with exactly two decimal places."""
    return f"{value:.2f}"
