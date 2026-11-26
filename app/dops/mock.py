"""JSON-backed, in-process DOPS with row-authoritative refund accounting."""

import json
from collections.abc import Callable
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

from app.dops.errors import (
    IdempotencyConflict,
    NotFound,
    NotPendingApproval,
    NotRefundable,
    RefundExceedsPaid,
    ValidationFailed,
)
from app.dops.models import CreateRefundResult, Order, Refund, RefundMode
from app.money import fmt_amount, parse_amount

_RESERVED = frozenset({"PENDING_APPROVAL", "PENDING", "SUCCEEDED"})


class MockDops:
    """Own a private copy of state; callers persist snapshots explicitly.

    The caller supplies an aware clock and serializes concurrent access. Gateway
    settings belong to sandbox state and cannot be changed by API requests.
    """

    def __init__(self, state: dict, now: Callable[[], datetime]) -> None:
        self._state = deepcopy(state)
        self._now = now

    @classmethod
    def seeded(
        cls,
        now: Callable[[], datetime],
        *,
        auto_approve_limit: str = "50.00",
        gateway_outcome: str = "success",
    ) -> "MockDops":
        """Resolve relative seed ages once and create isolated JSON state."""
        if gateway_outcome not in {"success", "decline", "timeout"}:
            raise ValueError("Unknown simulated gateway outcome")
        seed = json.loads(Path(__file__).with_name("seed.json").read_text(encoding="utf-8"))
        instant = cls._utc(now())
        state = {
            "customers": seed["customers"],
            "auto_approve_limit": auto_approve_limit,
            "gateway_outcome": gateway_outcome,
            "next_refund_id": 1,
            "orders": [],
            "refunds": [],
            "idempotency_keys": {},
            "gateway_side_effects": {},
        }
        for source in seed["orders"]:
            order = {k: v for k, v in source.items() if k not in {"created_days_ago", "refunds"}}
            order["created_at"] = (instant - timedelta(days=source["created_days_ago"])).isoformat()
            state["orders"].append(order)
            for seeded_refund in source.get("refunds", []):
                created_at = (
                    instant - timedelta(days=seeded_refund["created_days_ago"])
                ).isoformat()
                state["refunds"].append(
                    {
                        "id": state["next_refund_id"],
                        "order_id": order["id"],
                        "customer_id": order["user_id"],
                        "amount": seeded_refund["amount"],
                        "reason": seeded_refund["reason"],
                        "mode": seeded_refund["mode"],
                        "status": seeded_refund["status"],
                        "decided_by": seeded_refund["decided_by"],
                        "decision_reason": None,
                        "failure_message": None,
                        "trace_id": "t-0000000000000000",
                        "created_at": created_at,
                        "updated_at": created_at,
                    }
                )
                state["next_refund_id"] += 1
        return cls(state, now)

    def to_state(self) -> dict:
        """Return a detached, JSON-serialisable snapshot, including gateway state."""
        return deepcopy(self._state)

    @staticmethod
    def _utc(value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("The mock clock must return an aware datetime")
        return value.astimezone(UTC)

    @staticmethod
    def _text(value: str, field: str, maximum: int) -> None:
        if not isinstance(value, str) or not 1 <= len(value) <= maximum:
            raise ValidationFailed({field: [f"Must contain 1 to {maximum} characters"]})

    def _owned_order(self, customer_id: int, order_id: int) -> dict:
        if type(order_id) is not int or order_id <= 0:
            raise ValidationFailed({"order_id": ["Must be a positive integer"]})
        for order in self._state["orders"]:
            if order["id"] == order_id and order["user_id"] == customer_id:
                return order
        raise NotFound("Order not found")

    def _refund(self, refund_id: int) -> dict:
        for refund in self._state["refunds"]:
            if refund["id"] == refund_id:
                return refund
        raise NotFound("Refund not found")

    def _total(self, order_id: int, statuses: frozenset[str]) -> Decimal:
        return sum(
            (
                Decimal(r["amount"])
                for r in self._state["refunds"]
                if r["order_id"] == order_id and r["status"] in statuses
            ),
            start=Decimal("0.00"),
        )

    def _order_model(self, order: dict) -> Order:
        return Order.model_validate(
            {
                **deepcopy(order),
                "refunded_amount": fmt_amount(self._total(order["id"], frozenset({"SUCCEEDED"}))),
                "reserved_amount": fmt_amount(self._total(order["id"], _RESERVED)),
            }
        )

    @staticmethod
    def _ordered(rows: list[dict], *, newest: bool) -> list[dict]:
        return sorted(
            rows,
            key=lambda row: (datetime.fromisoformat(row["created_at"]), row["id"]),
            reverse=newest,
        )

    def list_orders(self, customer_id: int, trace_id: str) -> list[Order]:
        """Return the newest 20 owned orders with freshly derived totals."""
        owned = [o for o in self._state["orders"] if o["user_id"] == customer_id]
        return [self._order_model(o) for o in self._ordered(owned, newest=True)[:20]]

    def get_order(self, customer_id: int, order_id: int, trace_id: str) -> Order:
        """Hide missing and foreign orders behind the same error."""
        return self._order_model(self._owned_order(customer_id, order_id))

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
        """Reserve once; replay before checking the order's current paid state."""
        order = self._owned_order(customer_id, order_id)
        try:
            parsed = parse_amount(amount)
        except ValueError:
            raise ValidationFailed({"amount": ["Must be a positive decimal amount"]}) from None
        self._text(reason, "reason", 500)
        self._text(idempotency_key, "idempotency_key", 255)
        if len(idempotency_key) < 8:
            raise ValidationFailed({"idempotency_key": ["Must contain 8 to 255 characters"]})
        if mode not in {"auto", "needs_approval"}:
            raise ValidationFailed({"mode": ["Must be auto or needs_approval"]})
        normalized = fmt_amount(parsed)
        fingerprint = [order_id, customer_id, normalized, mode]
        existing = self._state["idempotency_keys"].get(idempotency_key)
        if existing is not None:
            if existing["fingerprint"] != fingerprint:
                raise IdempotencyConflict("Key already used for a different refund")
            refund = self._refund(existing["refund_id"])
            if refund["status"] == "PENDING":
                self._execute(refund)
            return CreateRefundResult(refund=Refund.model_validate(refund), replayed=True)
        if order["status"] != "PAID":
            raise NotRefundable("Order is not paid")
        reserved_after = self._total(order_id, _RESERVED) + parsed
        if reserved_after > Decimal(order["total_amount"]):
            raise RefundExceedsPaid("Refund exceeds the remaining paid amount")
        try:
            limit = parse_amount(self._state["auto_approve_limit"])
        except ValueError:
            limit = Decimal("0.00")
        status = "PENDING" if mode == "auto" and reserved_after <= limit else "PENDING_APPROVAL"
        timestamp = self._utc(self._now()).isoformat()
        refund = {
            "id": self._state["next_refund_id"],
            "order_id": order_id,
            "customer_id": customer_id,
            "amount": normalized,
            "reason": reason,
            "mode": mode,
            "status": status,
            "decided_by": None,
            "decision_reason": None,
            "failure_message": None,
            "trace_id": trace_id,
            "created_at": timestamp,
            "updated_at": timestamp,
        }
        self._state["next_refund_id"] += 1
        self._state["refunds"].append(refund)
        self._state["idempotency_keys"][idempotency_key] = {
            "fingerprint": fingerprint,
            "refund_id": refund["id"],
        }
        if status == "PENDING":
            self._execute(refund)
        return CreateRefundResult(refund=Refund.model_validate(refund), replayed=False)

    def _execute(self, refund: dict) -> None:
        # This stable provider key survives snapshots and ambiguous gateway timeouts.
        provider_key = f"refund-{refund['id']}"
        effects = self._state["gateway_side_effects"]
        outcome = "success" if provider_key in effects else self._state["gateway_outcome"]
        if outcome == "timeout":
            return
        if outcome == "decline":
            refund["status"] = "FAILED"
            refund["failure_message"] = "Simulated gateway declined the refund"
        elif outcome == "success":
            effects.setdefault(
                provider_key, {"order_id": refund["order_id"], "amount": refund["amount"]}
            )
            refund["status"] = "SUCCEEDED"
            refund["failure_message"] = None
            order = self._owned_order(refund["customer_id"], refund["order_id"])
            if self._total(order["id"], frozenset({"SUCCEEDED"})) == Decimal(order["total_amount"]):
                order["status"] = "REFUNDED"
        else:
            raise ValueError("Unknown simulated gateway outcome in saved state")
        refund["updated_at"] = self._utc(self._now()).isoformat()

    def list_order_refunds(self, customer_id: int, order_id: int, trace_id: str) -> list[Refund]:
        """Return newest refunds only after verifying ownership of their order."""
        self._owned_order(customer_id, order_id)
        rows = [r for r in self._state["refunds"] if r["order_id"] == order_id]
        return [Refund.model_validate(r) for r in self._ordered(rows, newest=True)]

    def approve_refund(self, refund_id: int, decided_by: str, trace_id: str) -> Refund:
        """Record the reviewer then execute a refund awaiting approval."""
        self._text(decided_by, "decided_by", 100)
        refund = self._refund(refund_id)
        if refund["status"] != "PENDING_APPROVAL":
            raise NotPendingApproval("Refund is not awaiting approval")
        refund.update(
            status="PENDING", decided_by=decided_by, updated_at=self._utc(self._now()).isoformat()
        )
        self._execute(refund)
        return Refund.model_validate(refund)

    def reject_refund(self, refund_id: int, decided_by: str, reason: str, trace_id: str) -> Refund:
        """Reject a held refund, releasing its reservation immediately."""
        self._text(decided_by, "decided_by", 100)
        self._text(reason, "reason", 500)
        refund = self._refund(refund_id)
        if refund["status"] != "PENDING_APPROVAL":
            raise NotPendingApproval("Refund is not awaiting approval")
        refund.update(
            status="REJECTED",
            decided_by=decided_by,
            decision_reason=reason,
            updated_at=self._utc(self._now()).isoformat(),
        )
        return Refund.model_validate(refund)

    def pending_refunds(self, status: str, trace_id: str) -> list[Refund]:
        """Return the oldest 50 refunds in either supported pending queue."""
        if status not in {"PENDING_APPROVAL", "PENDING"}:
            raise ValidationFailed({"status": ["Must be PENDING_APPROVAL or PENDING"]})
        rows = [r for r in self._state["refunds"] if r["status"] == status]
        return [Refund.model_validate(r) for r in self._ordered(rows, newest=False)[:50]]

    def customer_refunds_since(
        self, customer_id: int, since: datetime, trace_id: str
    ) -> list[Refund]:
        """Return all statuses, newest first, including the cutoff instant."""
        cutoff = self._utc(since)
        rows = [
            r
            for r in self._state["refunds"]
            if r["customer_id"] == customer_id and datetime.fromisoformat(r["created_at"]) >= cutoff
        ]
        return [Refund.model_validate(r) for r in self._ordered(rows, newest=True)]
