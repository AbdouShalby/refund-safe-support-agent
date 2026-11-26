"""Validated DOPS response models and canonical UTC timestamps."""

from datetime import UTC, datetime
from typing import Annotated, Literal

from pydantic import AfterValidator, AwareDatetime, BaseModel, ConfigDict, Field

from app.money import parse_amount


def _positive(value: str) -> str:
    parse_amount(value)
    return value


def _utc(value: datetime) -> datetime:
    return value.astimezone(UTC)


Amount = Annotated[str, Field(pattern=r"^[0-9]{1,8}\.[0-9]{2}$")]
PositiveAmount = Annotated[Amount, AfterValidator(_positive)]
UtcDatetime = Annotated[AwareDatetime, AfterValidator(_utc)]
RefundMode = Literal["auto", "needs_approval"]
RefundStatus = Literal["PENDING_APPROVAL", "PENDING", "SUCCEEDED", "FAILED", "REJECTED"]


class DopsModel(BaseModel):
    """Ignore additional fields supplied by the upstream API."""

    model_config = ConfigDict(extra="ignore")


class OrderItem(DopsModel):
    """An item on an order."""

    product_id: int
    quantity: int
    unit_price: PositiveAmount


class Order(DopsModel):
    """An agent-visible order including refund reservations."""

    id: int
    user_id: int
    status: str
    total_amount: PositiveAmount
    refunded_amount: Amount
    reserved_amount: Amount
    created_at: UtcDatetime
    items: list[OrderItem]


class Refund(DopsModel):
    """A refund and its current decision and execution state."""

    id: int
    order_id: int
    customer_id: int
    amount: PositiveAmount
    reason: str
    mode: RefundMode
    status: RefundStatus
    decided_by: str | None
    decision_reason: str | None
    failure_message: str | None
    trace_id: str
    created_at: UtcDatetime
    updated_at: UtcDatetime


class CreateRefundResult(DopsModel):
    """Distinguish creation from an exact idempotent replay."""

    refund: Refund
    replayed: bool
