"""Shared protocol for HTTP and in-process DOPS implementations."""

from datetime import datetime
from typing import Protocol

from app.dops.models import CreateRefundResult, Order, Refund, RefundMode


class DopsApi(Protocol):
    """Customer-scoped reads and writes plus separate Ops routes."""

    def list_orders(self, customer_id: int, trace_id: str) -> list[Order]: ...

    def get_order(self, customer_id: int, order_id: int, trace_id: str) -> Order: ...

    def create_refund(
        self,
        customer_id: int,
        order_id: int,
        amount: str,
        reason: str,
        idempotency_key: str,
        mode: RefundMode,
        trace_id: str,
    ) -> CreateRefundResult: ...

    def list_order_refunds(
        self, customer_id: int, order_id: int, trace_id: str
    ) -> list[Refund]: ...

    def approve_refund(self, refund_id: int, decided_by: str, trace_id: str) -> Refund: ...

    def reject_refund(
        self, refund_id: int, decided_by: str, reason: str, trace_id: str
    ) -> Refund: ...

    def pending_refunds(self, status: str, trace_id: str) -> list[Refund]: ...

    def customer_refunds_since(
        self, customer_id: int, since: datetime, trace_id: str
    ) -> list[Refund]: ...
