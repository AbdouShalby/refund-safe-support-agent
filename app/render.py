"""Canonical time-free order and refund text for model-visible content."""

from datetime import datetime

from app.dops.models import Order, Refund


def order_line(order: Order, now: datetime) -> str:
    """Render order facts and whole-day age without absolute timestamps."""
    days = (now - order.created_at).days
    unit = "day" if days == 1 else "days"
    return (
        f"Order {order.id}: {order.status}, total {order.total_amount}, "
        f"refunded {order.refunded_amount}, reserved {order.reserved_amount}, "
        f"{days} {unit} old, {len(order.items)} items"
    )


def refund_line(refund: Refund) -> str:
    """Render only the amount, status and execution mode."""
    return f"Refund {refund.amount}: {refund.status} ({refund.mode})"
