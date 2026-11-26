"""Verify refund invariants, persistence, ownership and simulated execution."""

import json
from datetime import UTC, datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

import pytest

from app.dops.errors import (
    IdempotencyConflict,
    NotFound,
    NotPendingApproval,
    NotRefundable,
    RefundExceedsPaid,
    ValidationFailed,
)
from app.dops.mock import MockDops
from app.dops.models import CreateRefundResult, Order, Refund, RefundMode

NOW = datetime(2026, 11, 18, 10, tzinfo=UTC)
TRACE = "t-0123456789abcdef"
SEED = Path(__file__).resolve().parents[2] / "app/dops/seed.json"


@pytest.fixture
def mock() -> MockDops:
    return MockDops.seeded(now=lambda: NOW)


def create(
    mock: MockDops,
    amount: str = "10.00",
    *,
    order: int = 1004,
    customer: int = 1,
    key: str = "mock-key-1",
    reason: str = "Requested",
    mode: RefundMode = "auto",
) -> CreateRefundResult:
    return mock.create_refund(customer, order, amount, reason, key, mode, TRACE)


def test_seed_ids_item_sums_and_exact_rows(mock: MockDops) -> None:
    seed = json.loads(SEED.read_text(encoding="utf-8"))
    assert seed["customers"] == [
        {"id": 1, "name": "Customer A"},
        {"id": 2, "name": "Customer B"},
        {"id": 3, "name": "Customer C"},
    ]
    expected = [
        (1, "PAID", "20.00", 2),
        (1, "PAID", "80.00", 5),
        (1, "PAID", "450.00", 3),
        (1, "PAID", "100.00", 1),
        (1, "PENDING", "60.00", 1),
        (1, "CANCELLED", "40.00", 1),
        (1, "REFUNDED", "30.00", 10),
        (1, "PAID", "70.00", 45),
        (2, "PAID", "120.00", 2),
        (2, "PAID", "25.00", 1),
        (3, "PAID", "300.00", 4),
    ]
    assert [row["id"] for row in seed["orders"]] == list(range(1001, 1012))
    for row, (customer, status, total, days) in zip(seed["orders"], expected, strict=True):
        assert (row["user_id"], row["status"], row["total_amount"]) == (customer, status, total)
        assert 1 <= len(row["items"]) <= 3
        assert sum(Decimal(i["unit_price"]) * i["quantity"] for i in row["items"]) == Decimal(total)
        assert mock.get_order(customer, row["id"], TRACE).created_at == NOW - timedelta(days=days)
    refund = mock.list_order_refunds(1, 1007, TRACE)[0]
    assert (refund.id, refund.status, refund.mode, refund.amount) == (
        1,
        "SUCCEEDED",
        "auto",
        "30.00",
    )
    assert refund.reason == "Seeded refund"
    assert refund.created_at == NOW - timedelta(days=9)
    assert refund.decided_by is None


@pytest.mark.parametrize(
    "status", ["PENDING_APPROVAL", "PENDING", "SUCCEEDED", "FAILED", "REJECTED"]
)
def test_refund_rows_are_authority_for_totals(mock: MockDops, status: str) -> None:
    create(mock, mode="needs_approval")
    state = mock.to_state()
    state["refunds"][-1]["status"] = status
    rebuilt = MockDops(state, now=lambda: NOW)
    order = rebuilt.get_order(1, 1004, TRACE)
    assert order.reserved_amount == (
        "10.00" if status in {"PENDING_APPROVAL", "PENDING", "SUCCEEDED"} else "0.00"
    )
    assert order.refunded_amount == ("10.00" if status == "SUCCEEDED" else "0.00")


@pytest.mark.parametrize("order", [1005, 1006, 1007])
def test_only_paid_orders_accept_new_refunds(mock: MockDops, order: int) -> None:
    with pytest.raises(NotRefundable):
        create(mock, order=order)


def test_reserved_plus_amount_cannot_exceed_total(mock: MockDops) -> None:
    create(mock, "70", mode="needs_approval")
    with pytest.raises(RefundExceedsPaid):
        create(mock, "30.01", key="mock-key-2")
    assert create(mock, "30", key="mock-key-3").refund.status == "PENDING_APPROVAL"


@pytest.mark.parametrize("order,customer", [(9999, 1), (1009, 1), (1004, 2)])
def test_missing_and_foreign_orders_hide_reads_and_writes(
    mock: MockDops, order: int, customer: int
) -> None:
    for action in (
        lambda: mock.get_order(customer, order, TRACE),
        lambda: mock.list_order_refunds(customer, order, TRACE),
        lambda: create(mock, order=order, customer=customer),
    ):
        with pytest.raises(NotFound):
            action()


def test_ownership_precedes_idempotency_lookup(mock: MockDops) -> None:
    create(mock)
    with pytest.raises(NotFound):
        create(mock, customer=2)
    with pytest.raises(NotFound):
        create(mock, order=9999)
    assert mock.customer_refunds_since(2, NOW - timedelta(days=20), TRACE) == []


def test_normalized_amount_replay_keeps_original_reason_and_trace(mock: MockDops) -> None:
    original = create(mock, "10", reason="Original")
    replay = mock.create_refund(1, 1004, "10.00", "Changed", "mock-key-1", "auto", "other-trace")
    assert replay.replayed is True
    assert replay.refund == original.refund
    assert len(mock.list_order_refunds(1, 1004, TRACE)) == 1


@pytest.mark.parametrize("change", ["amount", "mode", "order", "customer"])
def test_global_key_fingerprint_conflicts(mock: MockDops, change: str) -> None:
    create(mock)
    args = dict(
        customer_id=1,
        order_id=1004,
        amount="10.00",
        reason="Requested",
        idempotency_key="mock-key-1",
        mode="auto",
        trace_id=TRACE,
    )
    args.update(
        {
            "amount": {"amount": "11.00"},
            "mode": {"mode": "needs_approval"},
            "order": {"order_id": 1002},
            "customer": {"customer_id": 2, "order_id": 1009},
        }[change]
    )
    with pytest.raises(IdempotencyConflict):
        mock.create_refund(**args)


@pytest.mark.parametrize("length", [0, 7, 256])
def test_key_length_is_validated(mock: MockDops, length: int) -> None:
    with pytest.raises(ValidationFailed):
        create(mock, key="x" * length)


@pytest.mark.parametrize("length", [8, 255])
def test_key_length_boundaries_are_allowed(mock: MockDops, length: int) -> None:
    assert create(mock, key="x" * length).replayed is False


def test_auto_limit_is_cumulative_and_inclusive(mock: MockDops) -> None:
    assert create(mock, "30").refund.status == "SUCCEEDED"
    assert create(mock, "20", key="mock-key-2").refund.status == "SUCCEEDED"
    assert create(mock, "0.01", key="mock-key-3").refund.status == "PENDING_APPROVAL"


@pytest.mark.parametrize("limit", ["", "bad", "NaN", "Infinity", "-1", "50.001", " 50"])
def test_invalid_limit_holds_everything(limit: str) -> None:
    mock = MockDops.seeded(now=lambda: NOW, auto_approve_limit=limit)
    assert create(mock, "0.01").refund.status == "PENDING_APPROVAL"
    assert mock.to_state()["gateway_side_effects"] == {}


def test_needs_approval_always_holds(mock: MockDops) -> None:
    assert create(mock, mode="needs_approval").refund.status == "PENDING_APPROVAL"
    assert mock.to_state()["gateway_side_effects"] == {}


@pytest.mark.parametrize(
    "outcome,status", [("success", "SUCCEEDED"), ("decline", "FAILED"), ("timeout", "PENDING")]
)
def test_gateway_outcomes(outcome: str, status: str) -> None:
    mock = MockDops.seeded(now=lambda: NOW, gateway_outcome=outcome)
    refund = create(mock).refund
    assert refund.status == status
    assert bool(refund.failure_message) is (outcome == "decline")


def test_successful_full_refund_marks_order_refunded_and_still_replays(mock: MockDops) -> None:
    first = create(mock, "20", order=1001)
    order = mock.get_order(1, 1001, TRACE)
    assert (order.status, order.refunded_amount, order.reserved_amount) == (
        "REFUNDED",
        "20.00",
        "20.00",
    )
    assert create(mock, "20.00", order=1001).refund.id == first.refund.id


def test_decline_releases_reservation() -> None:
    mock = MockDops.seeded(now=lambda: NOW, gateway_outcome="decline")
    assert create(mock, "50").refund.status == "FAILED"
    assert mock.get_order(1, 1004, TRACE).reserved_amount == "0.00"
    assert create(mock, "50", key="mock-key-2").refund.status == "FAILED"


@pytest.mark.parametrize(
    "outcome,status", [("success", "SUCCEEDED"), ("decline", "FAILED"), ("timeout", "PENDING")]
)
def test_approve_executes_and_records_decider(outcome: str, status: str) -> None:
    mock = MockDops.seeded(now=lambda: NOW, gateway_outcome=outcome)
    held = create(mock, mode="needs_approval").refund
    approved = mock.approve_refund(held.id, "test-ops", TRACE)
    assert (approved.status, approved.decided_by) == (status, "test-ops")
    assert approved.decision_reason is None


@pytest.mark.parametrize("first_action", ["approve", "reject"])
def test_decisions_only_from_pending_approval(mock: MockDops, first_action: str) -> None:
    held = create(mock, mode="needs_approval").refund
    if first_action == "approve":
        mock.approve_refund(held.id, "test-ops", TRACE)
    else:
        rejected = mock.reject_refund(held.id, "test-ops", "Declined by reviewer", TRACE)
        assert (rejected.status, rejected.decided_by, rejected.decision_reason) == (
            "REJECTED",
            "test-ops",
            "Declined by reviewer",
        )
        assert mock.get_order(1, 1004, TRACE).reserved_amount == "0.00"
    with pytest.raises(NotPendingApproval):
        mock.approve_refund(held.id, "test-ops", TRACE)
    with pytest.raises(NotPendingApproval):
        mock.reject_refund(held.id, "test-ops", "Again", TRACE)


def test_missing_refund_is_not_found(mock: MockDops) -> None:
    with pytest.raises(NotFound):
        mock.approve_refund(999, "test-ops", TRACE)
    with pytest.raises(NotFound):
        mock.reject_refund(999, "test-ops", "Requested", TRACE)


@pytest.mark.parametrize("status", ["SUCCEEDED", "FAILED", "REJECTED", "pending", ""])
def test_queue_rejects_non_pending_status(mock: MockDops, status: str) -> None:
    with pytest.raises(ValidationFailed):
        mock.pending_refunds(status, TRACE)


@pytest.mark.parametrize(
    "status,outcome,mode",
    [("PENDING_APPROVAL", "success", "needs_approval"), ("PENDING", "timeout", "auto")],
)
def test_queue_oldest_first_max_50(status: str, outcome: str, mode: RefundMode) -> None:
    ticks = iter(NOW + timedelta(seconds=i) for i in range(100))
    mock = MockDops.seeded(now=lambda: next(ticks), gateway_outcome=outcome)
    refunds = [create(mock, "0.01", key=f"queue-key-{i:03}", mode=mode).refund for i in range(55)]
    queue = mock.pending_refunds(status, TRACE)
    assert [r.id for r in queue] == [r.id for r in refunds[:50]]


def test_customer_history_all_statuses_inclusive_cutoff_and_newest_first(mock: MockDops) -> None:
    for i, status in enumerate(("PENDING_APPROVAL", "PENDING", "SUCCEEDED", "FAILED", "REJECTED")):
        create(mock, "1", key=f"history-key-{i}", mode="needs_approval")
        state = mock.to_state()
        state["refunds"][-1]["status"] = status
        state["refunds"][-1]["created_at"] = (NOW + timedelta(seconds=i)).isoformat()
        mock = MockDops(state, now=lambda: NOW)
    history = mock.customer_refunds_since(1, NOW, TRACE)
    assert [r.status for r in history] == [
        "REJECTED",
        "FAILED",
        "SUCCEEDED",
        "PENDING",
        "PENDING_APPROVAL",
    ]
    assert len(mock.customer_refunds_since(1, NOW + timedelta(seconds=1), TRACE)) == 4
    assert mock.customer_refunds_since(2, NOW, TRACE) == []


def test_orders_newest_20_and_owned(mock: MockDops) -> None:
    state = mock.to_state()
    template = state["orders"][0]
    for i in range(25):
        state["orders"].append(
            {**template, "id": 2000 + i, "created_at": (NOW + timedelta(days=i)).isoformat()}
        )
    rebuilt = MockDops(state, now=lambda: NOW)
    assert [o.id for o in rebuilt.list_orders(1, TRACE)] == list(range(2024, 2004, -1))
    assert all(o.user_id == 2 for o in rebuilt.list_orders(2, TRACE))
    assert rebuilt.list_orders(999, TRACE) == []


def test_order_refunds_newest_first() -> None:
    ticks = iter(NOW + timedelta(seconds=i) for i in range(10))
    mock = MockDops.seeded(now=lambda: next(ticks))
    first = create(mock)
    second = create(mock, key="mock-key-2")
    assert [r.id for r in mock.list_order_refunds(1, 1004, TRACE)] == [
        second.refund.id,
        first.refund.id,
    ]


def test_zero_aggregates_and_return_types_trace_ids_and_incrementing_ids(mock: MockDops) -> None:
    order = mock.get_order(1, 1004, TRACE)
    assert isinstance(order, Order)
    assert (order.reserved_amount, order.refunded_amount) == ("0.00", "0.00")
    first = create(mock)
    second = create(mock, key="mock-key-2")
    assert isinstance(first, CreateRefundResult)
    assert isinstance(first.refund, Refund)
    assert (first.refund.id, second.refund.id) == (2, 3)
    assert first.refund.trace_id == TRACE
    assert first.refund.created_at == first.refund.updated_at == NOW
    assert first.refund.created_at.tzinfo is UTC


def test_times_normalized_to_aware_utc() -> None:
    local_now = NOW.astimezone(timezone(timedelta(hours=2)))
    mock = MockDops.seeded(now=lambda: local_now)
    assert mock.get_order(1, 1004, TRACE).created_at.tzinfo is UTC
    assert create(mock).refund.created_at == NOW


def test_state_json_roundtrip_and_copy_isolation(mock: MockDops) -> None:
    create(mock)
    saved = json.loads(json.dumps(mock.to_state()))
    original = json.loads(json.dumps(saved))
    rebuilt = MockDops(saved, now=lambda: NOW)
    create(rebuilt, key="mock-key-2")
    assert saved == original
    saved["orders"][0]["status"] = "CANCELLED"
    assert rebuilt.get_order(1, 1001, TRACE).status == "PAID"
    exported = rebuilt.to_state()
    exported["refunds"].clear()
    assert len(rebuilt.list_order_refunds(1, 1004, TRACE)) == 2
    response = rebuilt.get_order(1, 1004, TRACE)
    response.items.clear()
    assert rebuilt.get_order(1, 1004, TRACE).items
    assert MockDops(original, now=lambda: NOW).to_state() == original


def test_timeout_survives_state_rebuild() -> None:
    mock = MockDops.seeded(now=lambda: NOW, gateway_outcome="timeout", auto_approve_limit="12.00")
    first = create(mock)
    rebuilt = MockDops(json.loads(json.dumps(mock.to_state())), now=lambda: NOW)
    assert rebuilt.to_state()["gateway_outcome"] == "timeout"
    assert rebuilt.to_state()["auto_approve_limit"] == "12.00"
    replay = create(rebuilt)
    assert replay.replayed is True
    assert replay.refund.status == "PENDING"
    assert replay.refund.id == first.refund.id
    assert rebuilt.get_order(1, 1004, TRACE).reserved_amount == "10.00"


def test_pending_replay_after_gateway_recovers_executes_once() -> None:
    mock = MockDops.seeded(now=lambda: NOW, gateway_outcome="timeout")
    first = create(mock)
    state = mock.to_state()
    state["gateway_outcome"] = "success"
    rebuilt = MockDops(state, now=lambda: NOW)
    assert create(rebuilt).refund.status == "SUCCEEDED"
    assert create(rebuilt).refund.id == first.refund.id
    assert len(rebuilt.to_state()["gateway_side_effects"]) == 1
    assert rebuilt.get_order(1, 1004, TRACE).refunded_amount == "10.00"


def test_two_successes_on_same_provider_key_count_once(mock: MockDops) -> None:
    create(mock)
    state = mock.to_state()
    provider_keys = set(state["gateway_side_effects"])
    # Model a process dying after gateway success, before the row records success.
    state["refunds"][-1]["status"] = "PENDING"
    rebuilt = MockDops(state, now=lambda: NOW)
    assert create(rebuilt).refund.status == "SUCCEEDED"
    assert set(rebuilt.to_state()["gateway_side_effects"]) == provider_keys
    assert len(provider_keys) == 1
    assert rebuilt.get_order(1, 1004, TRACE).refunded_amount == "10.00"


@pytest.mark.parametrize(
    "amount", ["0", "-1", "1e2", "1.001", "123456789", "١٠", "1,5", "", " 1", 10, None]
)
def test_invalid_amount_is_domain_validation_error(mock: MockDops, amount: str) -> None:
    with pytest.raises(ValidationFailed):
        create(mock, amount)
    assert mock.list_order_refunds(1, 1004, TRACE) == []


@pytest.mark.parametrize("reason", ["", "x" * 501, None])
def test_create_reason_validation(mock: MockDops, reason: str) -> None:
    with pytest.raises(ValidationFailed):
        create(mock, reason=reason)


@pytest.mark.parametrize("action", ["approve", "reject"])
@pytest.mark.parametrize("decider", ["", "x" * 101, None])
def test_decider_validation(mock: MockDops, action: str, decider: str) -> None:
    held = create(mock, mode="needs_approval").refund
    with pytest.raises(ValidationFailed):
        if action == "approve":
            mock.approve_refund(held.id, decider, TRACE)
        else:
            mock.reject_refund(held.id, decider, "Reason", TRACE)
    assert mock.pending_refunds("PENDING_APPROVAL", TRACE)[0].id == held.id


@pytest.mark.parametrize("reason", ["", "x" * 501, None])
def test_reject_reason_validation(mock: MockDops, reason: str) -> None:
    held = create(mock, mode="needs_approval").refund
    with pytest.raises(ValidationFailed):
        mock.reject_refund(held.id, "test-ops", reason, TRACE)


def test_text_length_boundaries_allowed(mock: MockDops) -> None:
    first = create(mock, reason="x", mode="needs_approval").refund
    assert mock.approve_refund(first.id, "x", TRACE).decided_by == "x"
    second = create(mock, key="mock-key-2", reason="x" * 500, mode="needs_approval").refund
    assert mock.reject_refund(second.id, "x" * 100, "x" * 500, TRACE).status == "REJECTED"


def test_invalid_mode_is_validation_error(mock: MockDops) -> None:
    with pytest.raises(ValidationFailed):
        create(mock, mode="unknown")


@pytest.mark.parametrize("order", [True, 0, -1, "1004"])
def test_order_id_requires_positive_real_int(mock: MockDops, order: int) -> None:
    with pytest.raises(ValidationFailed):
        create(mock, order=order)
