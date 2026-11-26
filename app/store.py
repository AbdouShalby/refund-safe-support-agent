"""Atomic, expiring demo state backed by memory or prefixed Upstash Redis keys."""

import json
import secrets
import threading
import time
from collections.abc import Callable
from copy import deepcopy
from math import ceil
from types import TracebackType
from typing import Protocol, TypeVar

from upstash_redis import Redis

TTL_SECONDS = 86_400
_MAX_ATTEMPTS = 20
_T = TypeVar("_T")

# Adapted from the spike: the store owns the private version and the key's TTL.
CAS_SCRIPT = """
local current = redis.call('GET', KEYS[1])
local version = 0
if current then version = cjson.decode(current)['_version'] or 0 end
if tostring(version) ~= ARGV[1] then return 0 end
redis.call('SET', KEYS[1], ARGV[2], 'EX', tonumber(ARGV[3]))
return 1
"""

RESERVE_SCRIPT = """
local spent = tonumber(redis.call('GET', KEYS[1]) or '0')
if spent + tonumber(ARGV[1]) > tonumber(ARGV[2]) then return 0 end
redis.call('INCRBY', KEYS[1], ARGV[1])
redis.call('EXPIRE', KEYS[1], tonumber(ARGV[3]))
return 1
"""

_SETTLE_SCRIPT = """
redis.call('INCRBY', KEYS[1], ARGV[1])
redis.call('EXPIRE', KEYS[1], tonumber(ARGV[2]))
return 1
"""

_INCR_BELOW_SCRIPT = """
local current = tonumber(redis.call('GET', KEYS[1]) or '0')
if current >= tonumber(ARGV[1]) then return 0 end
redis.call('INCRBY', KEYS[1], 1)
redis.call('EXPIRE', KEYS[1], tonumber(ARGV[2]))
return 1
"""

_APPEND_SCRIPT = """
redis.call('RPUSH', KEYS[1], unpack(ARGV, 2))
redis.call('EXPIRE', KEYS[1], tonumber(ARGV[1]))
return 1
"""

_RELEASE_SCRIPT = """
if redis.call('GET', KEYS[1]) == ARGV[1] then
    return redis.call('DEL', KEYS[1])
end
return 0
"""


class StoreConflict(Exception):
    """A document could not be committed after bounded CAS retries."""


class LockBusy(Exception):
    """Another owner holds the requested lease."""


class StoreUnavailable(Exception):
    """The shared backend failed; callers must stop dependent operations."""


class Lock:
    """Acquire on entry and release only the lease bearing this owner's token."""

    def __init__(self, acquire: Callable[[str], bool], release: Callable[[str], None]) -> None:
        self._acquire = acquire
        self._release = release
        self._token = secrets.token_hex(16)

    def __enter__(self) -> "Lock":
        if not self._acquire(self._token):
            raise LockBusy("Store lock is busy")
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self._release(self._token)


class Store(Protocol):
    """The state operations required by sandboxes, audit, limits and budget."""

    def get_doc(self, key: str) -> dict | None: ...

    def update_doc(self, key: str, mutate: Callable[[dict], dict], default: dict) -> dict: ...

    def append_many(self, key: str, items: list[dict]) -> None: ...

    def read_list(self, key: str) -> list[dict]: ...

    def reserve(self, key: str, amount: int, cap: int) -> bool: ...

    def settle(self, key: str, reserved: int, actual: int) -> None: ...

    def remaining(self, key: str, cap: int) -> int: ...

    def incr_below(self, key: str, limit: int) -> bool: ...

    def lock(self, key: str, ttl_s: int) -> Lock: ...


def _without_version(doc: dict) -> dict:
    result = deepcopy(doc)
    result.pop("_version", None)
    return result


class _DocumentStore:
    """Share optimistic retries while leaving backend CAS operations atomic."""

    def _read_document(self, key: str) -> dict | None:
        raise NotImplementedError

    def _compare_and_set(self, key: str, expected: int, doc: dict) -> bool:
        raise NotImplementedError

    def get_doc(self, key: str) -> dict | None:
        doc = self._read_document(key)
        return None if doc is None else _without_version(doc)

    def update_doc(self, key: str, mutate: Callable[[dict], dict], default: dict) -> dict:
        """Retry pure mutations on detached snapshots; never expose internal versions."""
        for _ in range(_MAX_ATTEMPTS):
            current = self._read_document(key)
            expected = current.get("_version", 0) if current is not None else 0
            candidate = mutate(_without_version(default if current is None else current))
            new_doc = _without_version(candidate)
            new_doc["_version"] = expected + 1
            if self._compare_and_set(key, expected, new_doc):
                return _without_version(new_doc)
        raise StoreConflict("Document update conflicted after 20 attempts")


class MemoryStore(_DocumentStore):
    """Thread-safe local state with the same CAS, lease and expiry semantics."""

    def __init__(self, now: Callable[[], float] = time.monotonic) -> None:
        self._now = now
        self._mutex = threading.RLock()
        self._values: dict[str, object] = {}
        self._expires: dict[str, float] = {}

    def _get(self, key: str) -> object:
        # All callers hold the one store mutex, including expiry and release.
        if key in self._expires and self._now() >= self._expires[key]:
            del self._expires[key]
            self._values.pop(key, None)
        return self._values.get(key)

    def _put(self, key: str, value: object, ttl_s: int = TTL_SECONDS) -> None:
        self._values[key] = deepcopy(value)
        self._expires[key] = self._now() + ttl_s

    def _read_document(self, key: str) -> dict | None:
        with self._mutex:
            value = self._get(key)
            if value is not None and not isinstance(value, dict):
                raise StoreUnavailable("Store key is not a document")
            return deepcopy(value)

    def _compare_and_set(self, key: str, expected: int, doc: dict) -> bool:
        with self._mutex:
            current = self._get(key)
            version = current.get("_version", 0) if current is not None else 0
            if version != expected:
                return False
            self._put(key, doc)
            return True

    def append_many(self, key: str, items: list[dict]) -> None:
        if not items:
            return
        with self._mutex:
            current = self._get(key)
            self._put(key, (current or []) + deepcopy(items))

    def read_list(self, key: str) -> list[dict]:
        with self._mutex:
            return deepcopy(self._get(key) or [])

    def reserve(self, key: str, amount: int, cap: int) -> bool:
        with self._mutex:
            spent = self._get(key) or 0
            if spent + amount > cap:
                return False
            self._put(key, spent + amount)
            return True

    def settle(self, key: str, reserved: int, actual: int) -> None:
        with self._mutex:
            self._put(key, (self._get(key) or 0) + actual - reserved)

    def remaining(self, key: str, cap: int) -> int:
        with self._mutex:
            return max(0, cap - (self._get(key) or 0))

    def incr_below(self, key: str, limit: int) -> bool:
        with self._mutex:
            current = self._get(key) or 0
            if current >= limit:
                return False
            self._put(key, current + 1)
            return True

    def lock(self, key: str, ttl_s: int) -> Lock:
        if type(ttl_s) is not int or ttl_s <= 0:
            raise ValueError("Lock TTL must be a positive integer")

        def acquire(token: str) -> bool:
            with self._mutex:
                if self._get(key) is not None:
                    return False
                self._put(key, token, ttl_s)
                return True

        def release(token: str) -> None:
            with self._mutex:
                if self._get(key) == token:
                    self._values.pop(key, None)
                    self._expires.pop(key, None)

        return Lock(acquire, release)

    def ttl(self, key: str) -> int | None:
        """Return remaining whole seconds, rounding up; absent keys return None."""
        with self._mutex:
            if self._get(key) is None:
                return None
            # Subtracting large monotonic floats can put an exact whole second
            # microscopically above its boundary; do not report an extra second.
            return max(0, ceil(round(self._expires[key] - self._now(), 6)))


class RedisStore(_DocumentStore):
    """Upstash REST storage with atomic Lua writes and sanitized backend errors."""

    def __init__(self, redis: Redis, prefix: str = "") -> None:
        self.redis = redis
        self.prefix = prefix

    @staticmethod
    def _call(operation: Callable[[], _T]) -> _T:
        try:
            return operation()
        except Exception:
            raise StoreUnavailable("Store backend unavailable") from None

    def _eval(self, script: str, key: str, args: list[str]) -> object:
        return self._call(lambda: self.redis.eval(script, keys=[self.prefix + key], args=args))

    def _read_document(self, key: str) -> dict | None:
        raw = self._call(lambda: self.redis.get(self.prefix + key))
        return None if raw is None else self._call(lambda: json.loads(raw))

    def _compare_and_set(self, key: str, expected: int, doc: dict) -> bool:
        return self._eval(CAS_SCRIPT, key, [str(expected), json.dumps(doc), str(TTL_SECONDS)]) == 1

    def append_many(self, key: str, items: list[dict]) -> None:
        if items:
            self._eval(
                _APPEND_SCRIPT, key, [str(TTL_SECONDS), *(json.dumps(item) for item in items)]
            )

    def read_list(self, key: str) -> list[dict]:
        rows = self._call(lambda: self.redis.lrange(self.prefix + key, 0, -1))
        return self._call(lambda: [json.loads(row) for row in rows])

    def reserve(self, key: str, amount: int, cap: int) -> bool:
        return self._eval(RESERVE_SCRIPT, key, [str(amount), str(cap), str(TTL_SECONDS)]) == 1

    def settle(self, key: str, reserved: int, actual: int) -> None:
        self._eval(_SETTLE_SCRIPT, key, [str(actual - reserved), str(TTL_SECONDS)])

    def remaining(self, key: str, cap: int) -> int:
        value = self._call(lambda: self.redis.get(self.prefix + key))
        current = 0 if value is None else self._call(lambda: int(value))
        return max(0, cap - current)

    def incr_below(self, key: str, limit: int) -> bool:
        return self._eval(_INCR_BELOW_SCRIPT, key, [str(limit), str(TTL_SECONDS)]) == 1

    def lock(self, key: str, ttl_s: int) -> Lock:
        if type(ttl_s) is not int or ttl_s <= 0:
            raise ValueError("Lock TTL must be a positive integer")

        def acquire(token: str) -> bool:
            return bool(
                self._call(lambda: self.redis.set(self.prefix + key, token, nx=True, ex=ttl_s))
            )

        def release(token: str) -> None:
            self._eval(_RELEASE_SCRIPT, key, [token])

        return Lock(acquire, release)
