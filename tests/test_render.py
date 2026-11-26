"""Canonical rendering remains independent of DOPS timestamp encoding."""

from datetime import UTC, datetime, timedelta

import pytest

from app.dops.models import Order, Refund
from app.render import order_line, refund_line
from tests.dops.test_http_client import ORDER, REFUND


@pytest.mark.parametrize("days,label", [(1, "1 day"), (30, "30 days"), (0, "0 days")])
def test_order_rendering_parity(days: int, label: str) -> None:
    real = Order.model_validate(ORDER)
    mock = Order(**{**ORDER, "created_at": datetime(2026, 11, 17, 10, tzinfo=UTC)})
    now = real.created_at + timedelta(days=days, hours=1)
    expected = f"Order 1004: PAID, total 100.00, refunded 0.00, reserved 0.00, {label} old, 1 items"
    assert order_line(real, now) == order_line(mock, now) == expected
    assert "+00:00" not in expected
    assert "2026-" not in expected
    assert "t-" not in expected


def test_refund_rendering() -> None:
    assert refund_line(Refund.model_validate(REFUND)) == "Refund 30.00: SUCCEEDED (auto)"
