"""Synchronous DOPS client with strict response validation and single attempts."""

from datetime import datetime
from typing import Any, TypeVar

import httpx
from pydantic import BaseModel, TypeAdapter, ValidationError

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
from app.dops.models import CreateRefundResult, Order, Refund, RefundMode

Model = TypeVar("Model", bound=BaseModel)


class HttpDopsClient:
    """Reuse one HTTP client; every operation makes exactly one request."""

    def __init__(
        self,
        base_url: str,
        token: str,
        timeout_s: float = 5.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._client = httpx.Client(
            headers={"Accept": "application/json", "X-Service-Token": token},
            timeout=httpx.Timeout(timeout_s),
            transport=transport,
            follow_redirects=False,
        )

    def close(self) -> None:
        """Release the instance's connection pool."""
        self._client.close()

    def _request(
        self,
        method: str,
        path: str,
        trace_id: str,
        customer_id: int | None = None,
        *,
        body: dict[str, str] | None = None,
        params: dict[str, str] | None = None,
        statuses: tuple[int, ...] = (200,),
    ) -> tuple[Any, int]:
        headers = {"X-Trace-Id": trace_id}
        if customer_id is not None:
            headers["X-Customer-Id"] = str(customer_id)
        try:
            response = self._client.request(
                method,
                f"{self.base_url}/api/agent{path}",
                headers=headers,
                json=body,
                params=params,
            )
        except httpx.RequestError:
            raise DopsUnavailable("DOPS request failed") from None

        status = response.status_code
        if status == 401:
            raise DopsUnauthorized("DOPS service authorization failed")
        if status >= 500:
            raise DopsUnavailable(f"DOPS returned HTTP {status}")
        try:
            payload = response.json()
        except ValueError:
            raise DopsUnavailable("DOPS returned invalid JSON") from None
        if not isinstance(payload, dict):
            raise DopsUnavailable("DOPS returned an invalid response envelope")
        if status == 429:
            delay = payload.get("retry_after")
            raise DopsUnavailable(
                "DOPS rate limit exceeded",
                retry_after=delay if type(delay) is int and delay >= 0 else None,
            )
        if 400 <= status < 500:
            code = payload.get("error_code")
            if not isinstance(code, str):
                raise DopsUnavailable("DOPS returned an invalid error envelope")
            if code == "VALIDATION_ERROR":
                errors = payload.get("errors", {})
                if not isinstance(errors, dict):
                    raise DopsUnavailable("DOPS returned invalid validation errors")
                raise ValidationFailed(errors)
            error_types: dict[str, type[DopsError]] = {
                "NOT_FOUND": NotFound,
                "IDEMPOTENCY_CONFLICT": IdempotencyConflict,
                "ORDER_NOT_REFUNDABLE": NotRefundable,
                "REFUND_EXCEEDS_PAID": RefundExceedsPaid,
                "REFUND_NOT_PENDING_APPROVAL": NotPendingApproval,
                "UNAUTHORIZED": DopsUnauthorized,
            }
            error_type = error_types.get(code, DopsError)
            raise error_type("DOPS rejected the request")
        if status not in statuses or "data" not in payload:
            raise DopsUnavailable("DOPS returned an unexpected status or missing data")
        return payload["data"], status

    @staticmethod
    def _parse(model: type[Model], data: Any) -> Model:
        try:
            return model.model_validate(data)
        except ValidationError:
            raise DopsUnavailable("DOPS returned invalid resource data") from None

    @staticmethod
    def _parse_list(model: type[Model], data: Any) -> list[Model]:
        try:
            return TypeAdapter(list[model]).validate_python(data)
        except ValidationError:
            raise DopsUnavailable("DOPS returned invalid resource list") from None

    def list_orders(self, customer_id: int, trace_id: str) -> list[Order]:
        """List orders visible to a customer."""
        data, _ = self._request("GET", "/orders", trace_id, customer_id)
        return self._parse_list(Order, data)

    def get_order(self, customer_id: int, order_id: int, trace_id: str) -> Order:
        """Fetch one customer's order."""
        data, _ = self._request("GET", f"/orders/{order_id}", trace_id, customer_id)
        return self._parse(Order, data)

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
        """Submit one refund request and distinguish creation from replay."""
        data, status = self._request(
            "POST",
            f"/orders/{order_id}/refunds",
            trace_id,
            customer_id,
            body={
                "amount": amount,
                "reason": reason,
                "idempotency_key": idempotency_key,
                "mode": mode,
            },
            statuses=(200, 201),
        )
        return CreateRefundResult(refund=self._parse(Refund, data), replayed=status == 200)

    def list_order_refunds(self, customer_id: int, order_id: int, trace_id: str) -> list[Refund]:
        """List a customer's refunds on one order."""
        data, _ = self._request("GET", f"/orders/{order_id}/refunds", trace_id, customer_id)
        return self._parse_list(Refund, data)

    def approve_refund(self, refund_id: int, decided_by: str, trace_id: str) -> Refund:
        """Approve a held refund using the service route."""
        data, _ = self._request(
            "POST", f"/refunds/{refund_id}/approve", trace_id, body={"decided_by": decided_by}
        )
        return self._parse(Refund, data)

    def reject_refund(self, refund_id: int, decided_by: str, reason: str, trace_id: str) -> Refund:
        """Reject a held refund using the service route."""
        data, _ = self._request(
            "POST",
            f"/refunds/{refund_id}/reject",
            trace_id,
            body={"decided_by": decided_by, "reason": reason},
        )
        return self._parse(Refund, data)

    def pending_refunds(self, status: str, trace_id: str) -> list[Refund]:
        """List the service queue; DOPS validates the requested status."""
        data, _ = self._request("GET", "/refunds", trace_id, params={"status": status})
        return self._parse_list(Refund, data)

    def customer_refunds_since(
        self, customer_id: int, since: datetime, trace_id: str
    ) -> list[Refund]:
        """Read customer history via a service route with an encoded cutoff."""
        data, _ = self._request(
            "GET",
            f"/customers/{customer_id}/refunds",
            trace_id,
            params={"since": since.isoformat()},
        )
        return self._parse_list(Refund, data)
