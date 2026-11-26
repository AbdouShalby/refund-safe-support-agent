"""Verify the DOPS HTTP contract without network access."""

import json
from datetime import UTC, datetime

import httpx
import pytest
from pydantic import ValidationError

from app.dops.api import DopsApi
from app.dops.errors import (
    DopsError,
    DopsUnauthorized,
    DopsUnavailable,
    IdempotencyConflict,
    NotFound,
    NotPendingApproval,
    NotRefundable,
    RefundExceedsPaid,
    ValidationFailed,
)
from app.dops.http import HttpDopsClient
from app.dops.models import Order, OrderItem, Refund

TOKEN = "test-token-not-secret"  # noqa: S105 -- Deliberately fake contract-test token.
ORDER = {
    "id": 1004,
    "user_id": 1,
    "status": "PAID",
    "total_amount": "100.00",
    "refunded_amount": "0.00",
    "reserved_amount": "0.00",
    "created_at": "2026-11-17T10:00:00Z",
    "items": [{"product_id": 1, "quantity": 2, "unit_price": "50.00"}],
    "idempotency_key": "order-key",
    "cancelled_at": None,
    "updated_at": "2026-11-17T10:00:00Z",
}
REFUND = {
    "id": 7,
    "order_id": 1004,
    "customer_id": 1,
    "amount": "30.00",
    "reason": "Requested",
    "mode": "auto",
    "status": "SUCCEEDED",
    "decided_by": None,
    "decision_reason": None,
    "failure_message": None,
    "trace_id": "t-0123456789abcdef",
    "created_at": "2026-11-18T10:00:00+00:00",
    "updated_at": "2026-11-18T10:00:00+00:00",
}


def client_for(status: int, body: object) -> HttpDopsClient:
    return HttpDopsClient(
        "https://dops.test/",
        TOKEN,
        transport=httpx.MockTransport(lambda request: httpx.Response(status, json=body)),
    )


@pytest.mark.parametrize("status,replayed", [(201, False), (200, True)])
def test_create_headers_body_and_replay(status: int, replayed: bool) -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/api/agent/orders/1004/refunds"
        assert request.headers["X-Service-Token"] == TOKEN
        assert request.headers["X-Customer-Id"] == "1"
        assert request.headers["X-Trace-Id"] == "trace"
        assert request.headers["Accept"] == "application/json"
        assert request.headers["Content-Type"] == "application/json"
        assert json.loads(request.content) == {
            "amount": "30.00",
            "reason": "Requested",
            "idempotency_key": "key",
            "mode": "auto",
        }
        return httpx.Response(status, json={"data": REFUND})

    client: DopsApi = HttpDopsClient(
        "https://dops.test/", TOKEN, transport=httpx.MockTransport(respond)
    )
    result = client.create_refund(1, 1004, "30.00", "Requested", "key", "auto", "trace")
    assert result.replayed is replayed
    assert result.refund == Refund.model_validate(REFUND)


@pytest.mark.parametrize(
    "status,code,expected",
    [
        (404, "NOT_FOUND", NotFound),
        (409, "IDEMPOTENCY_CONFLICT", IdempotencyConflict),
        (422, "ORDER_NOT_REFUNDABLE", NotRefundable),
        (422, "REFUND_EXCEEDS_PAID", RefundExceedsPaid),
        (409, "REFUND_NOT_PENDING_APPROVAL", NotPendingApproval),
        (422, "VALIDATION_ERROR", ValidationFailed),
        (400, "UNKNOWN", DopsError),
        (401, "UNAUTHORIZED", DopsUnauthorized),
    ],
)
def test_error_mapping(status: int, code: str, expected: type[DopsError]) -> None:
    client = client_for(status, {"message": TOKEN, "error_code": code, "errors": {"mode": ["bad"]}})
    with pytest.raises(expected) as caught:
        client.get_order(1, 1004, "trace")
    assert type(caught.value) is expected
    assert TOKEN not in str(caught.value)
    if isinstance(caught.value, ValidationFailed):
        assert caught.value.errors == {"mode": ["bad"]}


def test_rate_limit() -> None:
    with pytest.raises(DopsUnavailable) as caught:
        client_for(429, {"error_code": "RATE_LIMIT_EXCEEDED", "retry_after": 7}).list_orders(1, "t")
    assert caught.value.retry_after == 7


@pytest.mark.parametrize(
    "status,body",
    [
        (500, {}),
        (200, {}),
        (200, {"data": None}),
        (200, {"data": [ORDER, {**ORDER, "total_amount": "30.5"}]}),
        (202, {"data": [ORDER]}),
        (302, {}),
    ],
)
def test_unavailable_status_or_shape(status: int, body: object) -> None:
    with pytest.raises(DopsUnavailable):
        client_for(status, body).list_orders(1, "t")


@pytest.mark.parametrize("error", [httpx.ReadTimeout, httpx.ConnectError])
def test_transport_failure_is_not_retried(error: type[httpx.RequestError]) -> None:
    calls = []

    def respond(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        raise error(TOKEN, request=request)

    client = HttpDopsClient("https://dops.test", TOKEN, transport=httpx.MockTransport(respond))
    with pytest.raises(DopsUnavailable) as caught:
        client.create_refund(1, 1004, "30.00", "reason", "key", "auto", "trace")
    assert len(calls) == 1
    assert TOKEN not in str(caught.value)


def test_all_read_and_ops_routes() -> None:
    requests = []
    replies = [[ORDER], ORDER, [REFUND], REFUND, REFUND, [REFUND], [REFUND]]

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"data": replies[len(requests) - 1]})

    client = HttpDopsClient(
        "https://dops.test/", TOKEN, timeout_s=3, transport=httpx.MockTransport(respond)
    )
    assert client.list_orders(1, "t") == [Order.model_validate(ORDER)]
    assert client.get_order(1, 1004, "t").id == 1004
    assert client.list_order_refunds(1, 1004, "t")[0].id == 7
    assert client.approve_refund(7, "ops", "t").id == 7
    assert client.reject_refund(7, "ops", "reason", "t").id == 7
    assert client.pending_refunds("PENDING_APPROVAL", "t")[0].id == 7
    assert client.customer_refunds_since(1, datetime(2026, 11, 18, tzinfo=UTC), "t")[0].id == 7
    assert [r.url.path for r in requests] == [
        "/api/agent/orders",
        "/api/agent/orders/1004",
        "/api/agent/orders/1004/refunds",
        "/api/agent/refunds/7/approve",
        "/api/agent/refunds/7/reject",
        "/api/agent/refunds",
        "/api/agent/customers/1/refunds",
    ]
    for index, request in enumerate(requests):
        assert request.headers["Accept"] == "application/json"
        assert request.headers["X-Service-Token"] == TOKEN
        assert request.headers["X-Trace-Id"] == "t"
        assert ("X-Customer-Id" in request.headers) is (index < 3)
        assert request.extensions["timeout"]["connect"] == 3
        assert request.extensions["timeout"]["read"] == 3
    assert json.loads(requests[3].content) == {"decided_by": "ops"}
    assert json.loads(requests[4].content) == {"decided_by": "ops", "reason": "reason"}
    assert requests[5].url.params["status"] == "PENDING_APPROVAL"
    assert "%2B00%3A00" in str(requests[6].url)


def test_pending_status_is_validated_by_dops() -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        assert request.url.params["status"] == "OTHER"
        return httpx.Response(
            422, json={"error_code": "VALIDATION_ERROR", "errors": {"status": []}}
        )

    client = HttpDopsClient("https://dops.test", TOKEN, transport=httpx.MockTransport(respond))
    with pytest.raises(ValidationFailed):
        client.pending_refunds("OTHER", "t")


@pytest.mark.parametrize("field", ["total_amount", "refunded_amount", "reserved_amount"])
@pytest.mark.parametrize("amount", ["30.5", "1e2", "-1.00", "123456789.00", "٣٠.00", 30, "30.00\n"])
def test_order_amount_format(field: str, amount: object) -> None:
    with pytest.raises(ValidationError):
        Order.model_validate({**ORDER, field: amount})


def test_positive_amounts_and_utc_datetimes() -> None:
    for amount in ["0.00", "30.5"]:
        with pytest.raises(ValidationError):
            Refund.model_validate({**REFUND, "amount": amount})
        with pytest.raises(ValidationError):
            OrderItem(product_id=1, quantity=1, unit_price=amount)
    with pytest.raises(ValidationError):
        Order.model_validate({**ORDER, "total_amount": "0.00"})
    with pytest.raises(ValidationError):
        Order.model_validate({**ORDER, "created_at": "2026-11-18T10:00:00"})
    refund = Refund.model_validate({**REFUND, "created_at": "2026-11-18T12:00:00+02:00"})
    assert refund.created_at == datetime(2026, 11, 18, 10, tzinfo=UTC)
    assert refund.created_at.tzinfo is UTC


@pytest.mark.parametrize("status", [429, 500])
@pytest.mark.parametrize("operation", ["create", "approve", "reject"])
def test_post_error_responses_are_not_retried(status: int, operation: str) -> None:
    calls = []

    def respond(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(status, json={"error_code": "RATE_LIMIT_EXCEEDED", "retry_after": 7})

    client = HttpDopsClient("https://dops.test", TOKEN, transport=httpx.MockTransport(respond))
    with pytest.raises(DopsUnavailable):
        if operation == "create":
            client.create_refund(1, 1004, "30.00", "Requested", "key", "auto", "t")
        elif operation == "approve":
            client.approve_refund(7, "ops", "t")
        else:
            client.reject_refund(7, "ops", "reason", "t")
    assert len(calls) == 1


@pytest.mark.parametrize("body", [b"not-json", b"[]", b'{"data": {}}'])
def test_invalid_success_envelopes(body: bytes) -> None:
    client = HttpDopsClient(
        "https://dops.test",
        TOKEN,
        transport=httpx.MockTransport(lambda request: httpx.Response(200, content=body)),
    )
    with pytest.raises(DopsUnavailable):
        client.get_order(1, 1004, "t")
