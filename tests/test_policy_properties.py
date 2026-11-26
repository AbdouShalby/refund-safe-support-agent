from dataclasses import replace
from decimal import Decimal

from hypothesis import given, settings
from hypothesis import strategies as st

from app.policy import PolicyConfig, RefundFacts, decide

AMOUNTS = st.decimals(min_value=Decimal("0.01"), max_value=Decimal("1000.00"), places=2)
TOTALS = st.decimals(min_value=Decimal("0.00"), max_value=Decimal("1000.00"), places=2)


@st.composite
def refund_facts(draw: st.DrawFn) -> RefundFacts:
    total = draw(TOTALS)
    reserved = draw(st.decimals(min_value=Decimal("0.00"), max_value=total, places=2))
    return RefundFacts(
        order_status=draw(st.sampled_from(["PAID", "PENDING", "CANCELLED", "REFUNDED"])),
        order_total=total,
        reserved=reserved,
        order_age_days=draw(st.integers(min_value=0, max_value=60)),
        refunds_on_order_24h=draw(st.integers(min_value=0, max_value=4)),
        customer_auto_count_24h=draw(st.integers(min_value=0, max_value=5)),
        customer_auto_amount_24h=draw(TOTALS),
    )


@settings(max_examples=300, deadline=None, derandomize=True)
@given(facts=refund_facts(), amount=AMOUNTS, auto_limit=TOTALS)
def test_auto_never_exceeds_total_or_auto_limit(
    facts: RefundFacts, amount: Decimal, auto_limit: Decimal
) -> None:
    # Keep unrelated gates open so the generated amounts exercise the cumulative cap.
    facts = replace(
        facts,
        order_status="PAID",
        order_age_days=0,
        refunds_on_order_24h=0,
        customer_auto_count_24h=0,
        customer_auto_amount_24h=Decimal("0.00"),
    )
    config = PolicyConfig(auto_limit=auto_limit, velocity_amount=Decimal("1000.00"))
    if decide(facts, amount, config).outcome == "auto":
        assert facts.reserved + amount <= min(facts.order_total, config.auto_limit)


@settings(max_examples=300, deadline=None, derandomize=True)
@given(facts=refund_facts(), amount=AMOUNTS)
def test_auto_requires_paid_order(facts: RefundFacts, amount: Decimal) -> None:
    # Every other gate permits auto; order status alone must still block it.
    facts = replace(
        facts,
        order_total=Decimal("1000.00"),
        reserved=Decimal("0.00"),
        order_age_days=0,
        refunds_on_order_24h=0,
        customer_auto_count_24h=0,
        customer_auto_amount_24h=Decimal("0.00"),
    )
    config = PolicyConfig(auto_limit=Decimal("1000.00"), velocity_amount=Decimal("1000.00"))
    result = decide(facts, amount, config)
    if result.outcome == "auto":
        assert facts.order_status == "PAID"
    if facts.order_status != "PAID":
        assert result.outcome == "reject"


@settings(max_examples=300, deadline=None, derandomize=True)
@given(facts=refund_facts(), amounts=st.lists(AMOUNTS, min_size=1, max_size=50))
def test_auto_sequence_never_exceeds_cumulative_limit(
    facts: RefundFacts, amounts: list[Decimal]
) -> None:
    facts = replace(
        facts,
        order_status="PAID",
        order_age_days=0,
        refunds_on_order_24h=0,
        customer_auto_count_24h=0,
        customer_auto_amount_24h=Decimal("0.00"),
    )
    config = PolicyConfig()
    initial_reserved = facts.reserved
    auto_total = Decimal("0.00")
    for amount in amounts:
        if decide(facts, amount, config).outcome == "auto":
            auto_total += amount
            facts = replace(facts, reserved=facts.reserved + amount)
            assert facts.reserved <= min(facts.order_total, config.auto_limit)
        assert auto_total <= config.auto_limit
        assert auto_total <= max(Decimal("0.00"), config.auto_limit - initial_reserved)
        assert facts.reserved <= facts.order_total


@settings(max_examples=300, deadline=None, derandomize=True)
@given(
    facts=refund_facts(),
    amount=AMOUNTS,
    auto_limit=TOTALS,
    velocity_amount=TOTALS,
    window_days=st.integers(min_value=0, max_value=60),
    velocity_count=st.integers(min_value=0, max_value=5),
)
def test_decisions_are_deterministic(
    facts: RefundFacts,
    amount: Decimal,
    auto_limit: Decimal,
    velocity_amount: Decimal,
    window_days: int,
    velocity_count: int,
) -> None:
    config = PolicyConfig(auto_limit, window_days, velocity_count, velocity_amount)
    facts_before = replace(facts)
    config_before = replace(config)
    first = decide(facts, amount, config)
    assert decide(facts, amount, config) == first
    assert decide(facts_before, amount, config_before) == first
    assert facts == facts_before
    assert config == config_before
