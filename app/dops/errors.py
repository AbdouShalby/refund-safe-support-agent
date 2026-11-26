"""Stable domain errors independent of the DOPS transport."""


class DopsError(Exception):
    """Base DOPS failure."""


class NotFound(DopsError):
    """The resource is missing or belongs to another customer."""


class IdempotencyConflict(DopsError):
    """An idempotency key was reused with a different fingerprint."""


class NotRefundable(DopsError):
    """The order is not refundable."""


class RefundExceedsPaid(DopsError):
    """The requested refund exceeds the paid balance."""


class NotPendingApproval(DopsError):
    """The refund is no longer waiting for approval."""


class ValidationFailed(DopsError):
    """DOPS rejected request fields."""

    def __init__(self, errors: dict) -> None:
        super().__init__("DOPS request validation failed")
        self.errors = errors


class DopsUnauthorized(DopsError):
    """The service credentials were rejected."""


class DopsUnavailable(DopsError):
    """DOPS could not provide a valid outcome; a write may have happened."""

    def __init__(
        self, message: str = "DOPS is unavailable", retry_after: int | None = None
    ) -> None:
        super().__init__(message)
        self.retry_after = retry_after
