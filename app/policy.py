"""Decide whether a refund is auto, needs_approval, or reject without I/O.

DOPS enforces the hard cap on total reservations and its own auto ceiling as
defence in depth, independently of this policy decision.
"""

from dataclasses import dataclass
from decimal import Decimal
from typing import Literal

ORDER_NOT_PAID = "ORDER_NOT_PAID"
AMOUNT_INVALID = "AMOUNT_INVALID"
EXCEEDS_REMAINING = "EXCEEDS_REMAINING"
OUTSIDE_WINDOW = "OUTSIDE_WINDOW"
OVER_AUTO_LIMIT = "OVER_AUTO_LIMIT"
REPEAT_WITHIN_24H = "REPEAT_WITHIN_24H"
VELOCITY_COUNT = "VELOCITY_COUNT"
VELOCITY_AMOUNT = "VELOCITY_AMOUNT"
WITHIN_POLICY = "WITHIN_POLICY"


def _require_decimal(value: Decimal, field: str) -> None:
    if not isinstance(value, Decimal):
        raise TypeError(f"{field} must be a Decimal")


@dataclass(frozen=True)
class RefundFacts:
    """Refund inputs with order age supplied as truncated whole days."""

    order_status: str
    order_total: Decimal
    reserved: Decimal
    order_age_days: int
    refunds_on_order_24h: int
    customer_auto_count_24h: int
    customer_auto_amount_24h: Decimal

    def __post_init__(self) -> None:
        _require_decimal(self.order_total, "order_total")
        _require_decimal(self.reserved, "reserved")
        _require_decimal(self.customer_auto_amount_24h, "customer_auto_amount_24h")


@dataclass(frozen=True)
class PolicyConfig:
    """Inclusive thresholds for automatic refunds and the refund window."""

    auto_limit: Decimal = Decimal("50.00")
    window_days: int = 30
    velocity_count: int = 3
    velocity_amount: Decimal = Decimal("100.00")

    def __post_init__(self) -> None:
        _require_decimal(self.auto_limit, "auto_limit")
        _require_decimal(self.velocity_amount, "velocity_amount")


@dataclass(frozen=True)
class Decision:
    """Policy outcome with applicable reasons in their definition order."""

    outcome: Literal["auto", "needs_approval", "reject"]
    reasons: tuple[str, ...]


def decide(facts: RefundFacts, amount: Decimal, config: PolicyConfig) -> Decision:
    """Reject invalid requests before considering any human approval gates."""
    _require_decimal(amount, "amount")
    reject_reasons: list[str] = []
    if facts.order_status != "PAID":
        reject_reasons.append(ORDER_NOT_PAID)
    if amount <= 0:
        reject_reasons.append(AMOUNT_INVALID)
    if amount > 0 and facts.reserved + amount > facts.order_total:
        reject_reasons.append(EXCEEDS_REMAINING)
    if facts.order_age_days > config.window_days:
        reject_reasons.append(OUTSIDE_WINDOW)
    if reject_reasons:
        return Decision("reject", tuple(reject_reasons))

    approval_reasons: list[str] = []
    if facts.reserved + amount > config.auto_limit:
        approval_reasons.append(OVER_AUTO_LIMIT)
    if facts.refunds_on_order_24h >= 1:
        approval_reasons.append(REPEAT_WITHIN_24H)
    if facts.customer_auto_count_24h + 1 > config.velocity_count:
        approval_reasons.append(VELOCITY_COUNT)
    if facts.customer_auto_amount_24h + amount > config.velocity_amount:
        approval_reasons.append(VELOCITY_AMOUNT)
    if approval_reasons:
        return Decision("needs_approval", tuple(approval_reasons))
    return Decision("auto", (WITHIN_POLICY,))
