"""Sign sandbox identities and reject malformed, tampered or expired cookies."""

import base64
import binascii
import hashlib
import hmac
import json
import re
from datetime import datetime


def _encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def _decode(value: str) -> bytes:
    if not re.fullmatch(r"[A-Za-z0-9_-]+", value):
        raise ValueError("Invalid base64url")
    decoded = base64.b64decode(value + "=" * (-len(value) % 4), altchars=b"-_", validate=True)
    if _encode(decoded) != value:
        raise ValueError("Noncanonical base64url")
    return decoded


def sign(payload: dict, secret: str) -> str:
    """Return base64url JSON and an HMAC-SHA256 over its encoded representation."""
    body = _encode(json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8"))
    signature = hmac.new(secret.encode("utf-8"), body.encode("ascii"), hashlib.sha256).digest()
    return body + "." + _encode(signature)


def verify(token: str, secret: str, now: datetime) -> dict | None:
    """Authenticate and validate every identity field; malformed input never raises."""
    try:
        if not isinstance(token, str) or len(token) > 4096:
            return None
        body, signature = token.split(".")
        expected = hmac.new(secret.encode("utf-8"), body.encode("ascii"), hashlib.sha256).digest()
        if not hmac.compare_digest(expected, _decode(signature)):
            return None
        payload = json.loads(_decode(body))
        if not isinstance(payload, dict) or payload.keys() != {
            "sandbox_id",
            "customer_id",
            "role",
            "exp",
        }:
            return None
        sandbox_id = payload["sandbox_id"]
        if not isinstance(sandbox_id, str) or not re.fullmatch(r"s-[a-f0-9]{24}", sandbox_id):
            return None
        if type(payload["customer_id"]) is not int or payload["customer_id"] not in (1, 2, 3):
            return None
        if payload["role"] not in ("customer", "ops"):
            return None
        if type(payload["exp"]) is not int or payload["exp"] <= now.timestamp():
            return None
        return payload
    except (ValueError, TypeError, UnicodeError, binascii.Error, RecursionError, OverflowError):
        return None
