"""Independent accounting and audit assertions over persisted DOPS snapshots."""

from decimal import Decimal

from app.audit import AuditEvent

SEED_TRACE = "t-0000000000000000"
RESERVED_STATUSES = {"PENDING_APPROVAL", "PENDING", "SUCCEEDED"}


def assert_money_invariants(dops_state: dict, audit: list[AuditEvent]) -> None:
    """Check ownership, reservation, automatic limits and customer-linked traces."""
    orders = {order["id"]: order for order in dops_state["orders"]}
    customer_traces = {event.trace_id for event in audit if event.type == "customer_message"}
    refund_traces = {
        event.trace_id for event in audit if event.type in {"refund_result", "dops_call"}
    }
    for refund in dops_state["refunds"]:
        assert refund["order_id"] in orders, f"Orphan refund: {refund}"
        order = orders[refund["order_id"]]
        assert refund["customer_id"] == order["user_id"], f"Foreign refund: {refund}"
        if refund["trace_id"] != SEED_TRACE:
            assert refund["trace_id"] in customer_traces & refund_traces, (
                f"Refund {refund['id']} has no customer-linked DOPS/refund audit trace"
            )

    for order_id, order in orders.items():
        rows = [row for row in dops_state["refunds"] if row["order_id"] == order_id]
        reserved = sum(
            (Decimal(row["amount"]) for row in rows if row["status"] in RESERVED_STATUSES),
            Decimal("0.00"),
        )
        total = Decimal(order["total_amount"])
        assert reserved <= total, f"Order {order_id}: reserved {reserved} exceeds {total}"
        if "reserved_amount" in order:
            assert Decimal(order["reserved_amount"]) == reserved, (
                f"Order {order_id}: reported reservation differs from refund rows"
            )
        automatic = sum(
            (
                Decimal(row["amount"])
                for row in rows
                if row["status"] in {"SUCCEEDED", "PENDING"} and row["decided_by"] is None
            ),
            Decimal("0.00"),
        )
        assert automatic <= Decimal("50.00"), f"Order {order_id}: automatic total {automatic}"
