"""Check hashed visitor limits, atomic increments and UTC day rollover."""

import hashlib
import hmac
from datetime import date, timedelta

from app.limits import Limits, ip_hash
from app.store import MemoryStore


def test_ip_hash() -> None:
    expected = hmac.new(b"test-hash-secret", b"192.0.2.1", hashlib.sha256).hexdigest()[:16]
    assert ip_hash("192.0.2.1", "test-hash-secret") == expected
    assert ip_hash("192.0.2.1", "different") != expected


def test_session_limit_and_day_rollover() -> None:
    store = MemoryStore()
    day = date(2026, 11, 18)
    limits = Limits(store, lambda: day)
    assert all(limits.allow_session("visitor") for _ in range(20))
    assert not limits.allow_session("visitor")
    assert limits.allow_session("other")
    day += timedelta(days=1)
    assert limits.allow_session("visitor")


def test_live_limits_by_sandbox_and_ip() -> None:
    store = MemoryStore()
    day = date(2026, 11, 18)
    limits = Limits(store, lambda: day)
    assert all(limits.start_live("one", "ip-one") for _ in range(3))
    assert not limits.start_live("one", "ip-other")
    assert not limits.start_live("two", "ip-one")
    assert limits.start_live("two", "ip-two")
    assert limits.start_live("two", "ip-two")
    # The refused IP attempt above already consumed one sandbox allowance.
    assert not limits.start_live("two", "ip-two")
    day += timedelta(days=1)
    assert limits.start_live("one", "ip-one")
    assert all(store.ttl(key) == 86_400 for key in store._values)
