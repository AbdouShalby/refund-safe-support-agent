"""Persist each visitor's simulated DOPS operations through atomic store updates."""

from collections.abc import Callable
from datetime import datetime
from functools import cached_property
from typing import TypeVar

from app.dops.mock import MockDops
from app.dops.models import CreateRefundResult, Order, Refund, RefundMode
from app.store import Store

_T = TypeVar("_T")


class SandboxDops:
    """Expose the complete DOPS interface with one independent state per visitor."""

    def __init__(
        self,
        store: Store,
        sandbox_id: str,
        now: Callable[[], datetime],
        *,
        seed_options: dict[str, str] | None = None,
    ) -> None:
        self._store = store
        self._key = f"sbx:{sandbox_id}:dops"
        self._now = now
        # Constructor-only switches support tests; requests have no settings API.
        self._seed_options = dict(seed_options or {})

    @cached_property
    def _default(self) -> dict:
        return MockDops.seeded(self._now, **self._seed_options).to_state()

    def _apply(self, operation: Callable[[MockDops], _T]) -> _T:
        result: _T

        def mutate(state: dict) -> dict:
            nonlocal result
            mock = MockDops(state, self._now)
            result = operation(mock)
            return mock.to_state()

        self._store.update_doc(self._key, mutate, default=self._default)
        # A result is returned only after its matching snapshot commits, including
        # the final CAS attempt. Domain exceptions leave the stored state untouched.
        return result

    def _read(self, operation: Callable[[MockDops], _T]) -> _T:
        state = self._store.get_doc(self._key)
        if state is None:
            return self._apply(operation)
        return operation(MockDops(state, self._now))

    def list_orders(self, customer_id: int, trace_id: str) -> list[Order]:
        return self._read(lambda mock: mock.list_orders(customer_id, trace_id))

    def get_order(self, customer_id: int, order_id: int, trace_id: str) -> Order:
        return self._read(lambda mock: mock.get_order(customer_id, order_id, trace_id))

    def create_refund(
        self,
        customer_id: int,
        order_id: int,
        amount: str,
        reason: str,
        idempotency_key: str,
        mode: RefundMode,
        trace_id: str,
    ) -> CreateRefundResult:
        return self._apply(
            lambda mock: mock.create_refund(
                customer_id, order_id, amount, reason, idempotency_key, mode, trace_id
            )
        )

    def list_order_refunds(self, customer_id: int, order_id: int, trace_id: str) -> list[Refund]:
        return self._read(lambda mock: mock.list_order_refunds(customer_id, order_id, trace_id))

    def approve_refund(self, refund_id: int, decided_by: str, trace_id: str) -> Refund:
        return self._apply(lambda mock: mock.approve_refund(refund_id, decided_by, trace_id))

    def reject_refund(self, refund_id: int, decided_by: str, reason: str, trace_id: str) -> Refund:
        return self._apply(lambda mock: mock.reject_refund(refund_id, decided_by, reason, trace_id))

    def pending_refunds(self, status: str, trace_id: str) -> list[Refund]:
        return self._read(lambda mock: mock.pending_refunds(status, trace_id))

    def customer_refunds_since(
        self, customer_id: int, since: datetime, trace_id: str
    ) -> list[Refund]:
        return self._read(lambda mock: mock.customer_refunds_since(customer_id, since, trace_id))
