"""Run the same 11 cases through DopsApi, with no implementation inspection.

Real DOPS must be seeded from app/dops/seed.json before each case (Task 4's
AgentEvalSeeder); DOPS_RESET_CMD is responsible for applying that identical seed.
HTTP 201/200/404/409/422 are asserted through the protocol's result/error mapping.
"""

from uuid import uuid4

import pytest

from app.dops.api import DopsApi
from app.dops.errors import (
    IdempotencyConflict,
    NotFound,
    NotPendingApproval,
    NotRefundable,
    RefundExceedsPaid,
)

TRACE = "t-0123456789abcdef"


def key() -> str:
    """Avoid previous local-run keys even when exercising real DOPS."""
    return "parity-" + uuid4().hex


def test_list_orders_matches_seed(dops: DopsApi) -> None:
    expected = {
        1: [
            (1001, "PAID", "20.00"),
            (1002, "PAID", "80.00"),
            (1003, "PAID", "450.00"),
            (1004, "PAID", "100.00"),
            (1005, "PENDING", "60.00"),
            (1006, "CANCELLED", "40.00"),
            (1007, "REFUNDED", "30.00"),
            (1008, "PAID", "70.00"),
        ],
        2: [(1009, "PAID", "120.00"), (1010, "PAID", "25.00")],
        3: [(1011, "PAID", "300.00")],
    }
    for customer_id, rows in expected.items():
        orders = dops.list_orders(customer_id, TRACE)
        assert sorted((o.id, o.status, o.total_amount) for o in orders) == rows
        assert [o.created_at for o in orders] == sorted(
            (o.created_at for o in orders), reverse=True
        )
        assert all(o.refunded_amount == ("30.00" if o.id == 1007 else "0.00") for o in orders)
        assert all(o.reserved_amount == ("30.00" if o.id == 1007 else "0.00") for o in orders)


def test_foreign_order_is_not_found(dops: DopsApi) -> None:
    with pytest.raises(NotFound):
        dops.get_order(1, 1009, TRACE)


def test_create_is_new(dops: DopsApi) -> None:
    result = dops.create_refund(1, 1001, "10.00", "Requested", key(), "auto", TRACE)
    assert result.replayed is False
    assert result.refund.status == "SUCCEEDED"
    assert result.refund.amount == "10.00"


def test_exact_replay_returns_same_refund(dops: DopsApi) -> None:
    request_key = key()
    first = dops.create_refund(1, 1001, "10.00", "Requested", request_key, "auto", TRACE)
    replay = dops.create_refund(1, 1001, "10.00", "Requested", request_key, "auto", TRACE)
    assert replay.replayed is True
    assert replay.refund.id == first.refund.id
    assert len(dops.list_order_refunds(1, 1001, TRACE)) == 1


def test_reason_only_change_replays(dops: DopsApi) -> None:
    request_key = key()
    first = dops.create_refund(1, 1001, "10.00", "Original", request_key, "auto", TRACE)
    replay = dops.create_refund(1, 1001, "10.00", "Changed", request_key, "auto", TRACE)
    assert replay.replayed is True
    assert replay.refund.id == first.refund.id
    assert replay.refund.reason == "Original"


def test_amount_change_conflicts(dops: DopsApi) -> None:
    request_key = key()
    dops.create_refund(1, 1001, "10.00", "Requested", request_key, "auto", TRACE)
    with pytest.raises(IdempotencyConflict):
        dops.create_refund(1, 1001, "11.00", "Requested", request_key, "auto", TRACE)


def test_pending_order_not_refundable(dops: DopsApi) -> None:
    with pytest.raises(NotRefundable):
        dops.create_refund(1, 1005, "10.00", "Requested", key(), "auto", TRACE)


def test_over_total_is_rejected(dops: DopsApi) -> None:
    with pytest.raises(RefundExceedsPaid):
        dops.create_refund(1, 1001, "20.01", "Requested", key(), "auto", TRACE)


def test_cumulative_over_limit_is_held(dops: DopsApi) -> None:
    first = dops.create_refund(1, 1004, "30.00", "Requested", key(), "auto", TRACE)
    second = dops.create_refund(1, 1004, "20.01", "Requested", key(), "auto", TRACE)
    assert first.refund.status == "SUCCEEDED"
    assert second.refund.status == "PENDING_APPROVAL"
    assert dops.get_order(1, 1004, TRACE).reserved_amount == "50.01"


def test_second_approval_conflicts(dops: DopsApi) -> None:
    held = dops.create_refund(1, 1002, "60.00", "Requested", key(), "auto", TRACE).refund
    assert dops.approve_refund(held.id, "test-ops", TRACE).status == "SUCCEEDED"
    with pytest.raises(NotPendingApproval):
        dops.approve_refund(held.id, "test-ops", TRACE)


def test_reject_releases_reservation(dops: DopsApi) -> None:
    held = dops.create_refund(1, 1002, "60.00", "Requested", key(), "auto", TRACE).refund
    rejected = dops.reject_refund(held.id, "test-ops", "Not eligible", TRACE)
    assert rejected.status == "REJECTED"
    assert rejected.decided_by == "test-ops"
    assert rejected.decision_reason == "Not eligible"
    assert dops.get_order(1, 1002, TRACE).reserved_amount == "0.00"
    replacement = dops.create_refund(1, 1002, "60.00", "Again", key(), "auto", TRACE)
    assert replacement.refund.status == "PENDING_APPROVAL"
