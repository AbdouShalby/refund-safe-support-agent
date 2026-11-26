from dataclasses import FrozenInstanceError, replace
from decimal import Decimal

import pytest

from app.policy import (
    AMOUNT_INVALID,
    EXCEEDS_REMAINING,
    ORDER_NOT_PAID,
    OUTSIDE_WINDOW,
    OVER_AUTO_LIMIT,
    REPEAT_WITHIN_24H,
    VELOCITY_AMOUNT,
    VELOCITY_COUNT,
    WITHIN_POLICY,
    Decision,
    PolicyConfig,
    RefundFacts,
    decide,
)


@pytest.fixture
def facts() -> RefundFacts:
    return RefundFacts(
        order_status="PAID",
        order_total=Decimal("200.00"),
        reserved=Decimal("0.00"),
        order_age_days=2,
        refunds_on_order_24h=0,
        customer_auto_count_24h=0,
        customer_auto_amount_24h=Decimal("0.00"),
    )


def test_default_config() -> None:
    assert PolicyConfig() == PolicyConfig(Decimal("50.00"), 30, 3, Decimal("100.00"))


@pytest.mark.parametrize("amount", ["0.01", "20.00", "50.00"])
def test_amount_within_auto_limit(facts: RefundFacts, amount: str) -> None:
    assert decide(facts, Decimal(amount), PolicyConfig()) == Decision("auto", (WITHIN_POLICY,))


def test_amount_above_auto_limit(facts: RefundFacts) -> None:
    assert decide(facts, Decimal("50.01"), PolicyConfig()) == Decision(
        "needs_approval", (OVER_AUTO_LIMIT,)
    )


@pytest.mark.parametrize("amount, outcome", [("20.00", "auto"), ("20.01", "needs_approval")])
def test_auto_limit_includes_reserved(facts: RefundFacts, amount: str, outcome: str) -> None:
    result = decide(replace(facts, reserved=Decimal("30.00")), Decimal(amount), PolicyConfig())
    assert result.outcome == outcome
    assert result.reasons == ((WITHIN_POLICY,) if outcome == "auto" else (OVER_AUTO_LIMIT,))


@pytest.mark.parametrize("status", ["PENDING", "CANCELLED", "REFUNDED", "paid", ""])
def test_order_must_be_paid(facts: RefundFacts, status: str) -> None:
    assert decide(replace(facts, order_status=status), Decimal("10.00"), PolicyConfig()) == (
        Decision("reject", (ORDER_NOT_PAID,))
    )


@pytest.mark.parametrize("amount", ["0.00", "-0.01", "-50.00"])
def test_amount_must_be_positive(facts: RefundFacts, amount: str) -> None:
    assert decide(facts, Decimal(amount), PolicyConfig()) == Decision("reject", (AMOUNT_INVALID,))


def test_exact_remaining_amount_is_allowed(facts: RefundFacts) -> None:
    facts = replace(facts, order_total=Decimal("40.00"), reserved=Decimal("25.00"))
    assert decide(facts, Decimal("15.00"), PolicyConfig()) == Decision("auto", (WITHIN_POLICY,))


def test_amount_above_remaining_is_rejected(facts: RefundFacts) -> None:
    facts = replace(facts, order_total=Decimal("40.00"), reserved=Decimal("25.00"))
    assert decide(facts, Decimal("15.01"), PolicyConfig()) == Decision(
        "reject", (EXCEEDS_REMAINING,)
    )


@pytest.mark.parametrize("amount", ["0.01", "50.00"])
def test_reserved_above_total_rejects_positive_amounts(facts: RefundFacts, amount: str) -> None:
    facts = replace(facts, order_total=Decimal("40.00"), reserved=Decimal("40.01"))
    assert decide(facts, Decimal(amount), PolicyConfig()) == Decision(
        "reject", (EXCEEDS_REMAINING,)
    )


@pytest.mark.parametrize("amount", ["0.00", "-0.01"])
def test_nonpositive_amount_does_not_evaluate_remaining(facts: RefundFacts, amount: str) -> None:
    facts = replace(facts, order_total=Decimal("40.00"), reserved=Decimal("100.00"))
    assert decide(facts, Decimal(amount), PolicyConfig()) == Decision("reject", (AMOUNT_INVALID,))


@pytest.mark.parametrize(
    "age, expected",
    [(30, Decision("auto", (WITHIN_POLICY,))), (31, Decision("reject", (OUTSIDE_WINDOW,)))],
)
def test_refund_window_boundary(facts: RefundFacts, age: int, expected: Decision) -> None:
    assert decide(replace(facts, order_age_days=age), Decimal("10.00"), PolicyConfig()) == expected


@pytest.mark.parametrize("count", [1, 2])
def test_repeat_on_same_order_requires_approval(facts: RefundFacts, count: int) -> None:
    assert decide(replace(facts, refunds_on_order_24h=count), Decimal("10.00"), PolicyConfig()) == (
        Decision("needs_approval", (REPEAT_WITHIN_24H,))
    )


@pytest.mark.parametrize(
    "count, expected",
    [(2, Decision("auto", (WITHIN_POLICY,))), (3, Decision("needs_approval", (VELOCITY_COUNT,)))],
)
def test_fourth_auto_refund_requires_approval(
    facts: RefundFacts, count: int, expected: Decision
) -> None:
    assert (
        decide(replace(facts, customer_auto_count_24h=count), Decimal("10.00"), PolicyConfig())
        == expected
    )


@pytest.mark.parametrize(
    "amount, expected",
    [
        ("10.00", Decision("auto", (WITHIN_POLICY,))),
        ("10.01", Decision("needs_approval", (VELOCITY_AMOUNT,))),
    ],
)
def test_velocity_amount_boundary(facts: RefundFacts, amount: str, expected: Decision) -> None:
    facts = replace(facts, customer_auto_amount_24h=Decimal("90.00"))
    assert decide(facts, Decimal(amount), PolicyConfig()) == expected


def test_all_approval_reasons_in_definition_order(facts: RefundFacts) -> None:
    facts = replace(
        facts,
        reserved=Decimal("40.00"),
        refunds_on_order_24h=1,
        customer_auto_count_24h=3,
        customer_auto_amount_24h=Decimal("90.00"),
    )
    assert decide(facts, Decimal("20.00"), PolicyConfig()) == Decision(
        "needs_approval", (OVER_AUTO_LIMIT, REPEAT_WITHIN_24H, VELOCITY_COUNT, VELOCITY_AMOUNT)
    )


def test_reject_reasons_override_approval_reasons(facts: RefundFacts) -> None:
    facts = replace(
        facts,
        order_status="CANCELLED",
        order_total=Decimal("20.00"),
        reserved=Decimal("20.00"),
        order_age_days=31,
        refunds_on_order_24h=1,
        customer_auto_count_24h=3,
        customer_auto_amount_24h=Decimal("100.00"),
    )
    assert decide(facts, Decimal("50.00"), PolicyConfig()) == Decision(
        "reject", (ORDER_NOT_PAID, EXCEEDS_REMAINING, OUTSIDE_WINDOW)
    )


def test_multiple_reject_reasons_with_invalid_amount(facts: RefundFacts) -> None:
    facts = replace(facts, order_status="PENDING", order_age_days=31)
    assert decide(facts, Decimal("0.00"), PolicyConfig()) == Decision(
        "reject", (ORDER_NOT_PAID, AMOUNT_INVALID, OUTSIDE_WINDOW)
    )


def test_custom_config_is_used(facts: RefundFacts) -> None:
    config = PolicyConfig(Decimal("10.00"), 5, 1, Decimal("20.00"))
    facts = replace(
        facts,
        order_age_days=5,
        customer_auto_count_24h=1,
        customer_auto_amount_24h=Decimal("10.00"),
    )
    assert decide(facts, Decimal("10.01"), config) == Decision(
        "needs_approval", (OVER_AUTO_LIMIT, VELOCITY_COUNT, VELOCITY_AMOUNT)
    )
    assert decide(replace(facts, order_age_days=6), Decimal("10.01"), config) == Decision(
        "reject", (OUTSIDE_WINDOW,)
    )


@pytest.mark.parametrize("field", ["order_total", "reserved", "customer_auto_amount_24h"])
def test_facts_reject_float_fields_at_construction(facts: RefundFacts, field: str) -> None:
    with pytest.raises(TypeError):
        replace(facts, **{field: float("1.00")})


@pytest.mark.parametrize("field", ["auto_limit", "velocity_amount"])
def test_config_rejects_float_fields_at_construction(field: str) -> None:
    with pytest.raises(TypeError):
        PolicyConfig(**{field: float("1.00")})


def test_decide_rejects_float_amount(facts: RefundFacts) -> None:
    with pytest.raises(TypeError):
        decide(facts, float("1.00"), PolicyConfig())


def test_dataclasses_are_frozen(facts: RefundFacts) -> None:
    with pytest.raises(FrozenInstanceError):
        facts.reserved = Decimal("1.00")
    config = PolicyConfig()
    with pytest.raises(FrozenInstanceError):
        config.auto_limit = Decimal("1.00")
    decision = Decision("auto", (WITHIN_POLICY,))
    with pytest.raises(FrozenInstanceError):
        decision.reasons = ()
