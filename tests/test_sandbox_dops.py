"""Verify sandbox isolation, atomic refund updates and persisted gateway settings."""

from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from threading import Barrier
from unittest.mock import patch

import pytest

from app.dops.api import DopsApi
from app.dops.errors import NotFound, RefundExceedsPaid
from app.dops.mock import MockDops
from app.sandbox_dops import SandboxDops
from app.store import MemoryStore

NOW = datetime(2026, 11, 18, 10, tzinfo=UTC)
TRACE = "t-0123456789abcdef"


def test_parallel_refunds_never_over_reserve() -> None:
    store = MemoryStore()
    barrier = Barrier(10)

    def create(index: int) -> bool:
        # Separate instances model independent requests to the same sandbox.
        sandbox = SandboxDops(store, "visitor", lambda: NOW)
        barrier.wait(timeout=10)
        try:
            sandbox.create_refund(1, 1004, "30.00", "Requested", f"parallel-{index}", "auto", TRACE)
        except RefundExceedsPaid:
            return False
        return True

    with ThreadPoolExecutor(max_workers=10) as pool:
        results = list(pool.map(create, range(10)))
    sandbox = SandboxDops(store, "visitor", lambda: NOW)
    assert results.count(True) == 3
    assert len(sandbox.list_order_refunds(1, 1004, TRACE)) == 3
    assert Decimal(sandbox.get_order(1, 1004, TRACE).reserved_amount) == Decimal("90.00")


def test_gateway_timeout_persists_after_reload() -> None:
    store = MemoryStore()
    sandbox = SandboxDops(
        store, "visitor", lambda: NOW, seed_options={"gateway_outcome": "timeout"}
    )
    first = sandbox.create_refund(1, 1004, "30.00", "Requested", "timeout-key", "auto", TRACE)
    assert first.refund.status == "PENDING"
    reloaded = SandboxDops(store, "visitor", lambda: NOW)
    replay = reloaded.create_refund(1, 1004, "30.00", "Requested", "timeout-key", "auto", TRACE)
    assert replay.replayed
    assert replay.refund.id == first.refund.id
    assert replay.refund.status == "PENDING"
    assert store.get_doc("sbx:visitor:dops")["gateway_outcome"] == "timeout"


def test_not_found_does_not_seed_or_write() -> None:
    store = MemoryStore()
    sandbox = SandboxDops(store, "visitor", lambda: NOW)
    with pytest.raises(NotFound):
        sandbox.create_refund(1, 1009, "10.00", "Requested", "foreign-key", "auto", TRACE)
    assert store.get_doc("sbx:visitor:dops") is None
    sandbox.create_refund(1, 1004, "10.00", "Requested", "valid-key", "auto", TRACE)
    before = store._read_document("sbx:visitor:dops")
    compare_and_set = store._compare_and_set
    with patch.object(store, "_compare_and_set", wraps=compare_and_set) as write:
        with pytest.raises(NotFound):
            sandbox.create_refund(1, 1009, "10.00", "Requested", "foreign-key", "auto", TRACE)
        with pytest.raises(NotFound):
            sandbox.get_order(1, 1009, TRACE)
        write.assert_not_called()
    after = store._read_document("sbx:visitor:dops")
    assert after == before
    assert after["_version"] == before["_version"]


def test_sandboxes_are_isolated() -> None:
    store = MemoryStore()
    first = SandboxDops(store, "one", lambda: NOW)
    second = SandboxDops(store, "two", lambda: NOW)
    first.create_refund(1, 1004, "30.00", "Requested", "shared-key", "auto", TRACE)
    assert second.list_order_refunds(1, 1004, TRACE) == []
    assert second.get_order(1, 1004, TRACE).reserved_amount == "0.00"
    assert first.get_order(1, 1004, TRACE).reserved_amount == "30.00"


def test_seed_is_lazy_and_only_initializes_once() -> None:
    store = MemoryStore()
    with patch.object(MockDops, "seeded", wraps=MockDops.seeded) as seeded:
        sandbox = SandboxDops(store, "visitor", lambda: NOW)
        seeded.assert_not_called()
        sandbox.list_orders(1, TRACE)
        seeded.assert_called_once()
        reloaded = SandboxDops(store, "visitor", lambda: NOW + timedelta(days=1))
        assert reloaded.get_order(1, 1004, TRACE).created_at == NOW - timedelta(days=1)
        seeded.assert_called_once()


def test_all_dops_operations_round_trip_through_store() -> None:
    store = MemoryStore()
    sandbox: DopsApi = SandboxDops(store, "visitor", lambda: NOW)
    assert len(sandbox.list_orders(1, TRACE)) == 8
    held = sandbox.create_refund(1, 1003, "60.00", "Requested", "approve-key", "auto", TRACE)
    queue = sandbox.pending_refunds("PENDING_APPROVAL", TRACE)
    assert [r.id for r in queue] == [held.refund.id]
    approved = sandbox.approve_refund(held.refund.id, "ops:visitor", TRACE)
    assert approved.status == "SUCCEEDED"
    reloaded = SandboxDops(store, "visitor", lambda: NOW)
    assert reloaded.get_order(1, 1003, TRACE).refunded_amount == "60.00"
    held = reloaded.create_refund(1, 1003, "60.00", "Requested", "reject-key", "auto", TRACE)
    assert (
        reloaded.reject_refund(held.refund.id, "ops:visitor", "Declined", TRACE).status
        == "REJECTED"
    )
    assert reloaded.pending_refunds("PENDING_APPROVAL", TRACE) == []
    recent = reloaded.customer_refunds_since(1, NOW, TRACE)
    assert {r.status for r in recent} == {"SUCCEEDED", "REJECTED"}
    assert store.ttl("sbx:visitor:dops") == 86_400
