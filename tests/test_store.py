"""Exercise shared-store atomicity, expiry and fail-closed backend handling."""

import json
import os
import threading
import time
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock
from uuid import uuid4

import pytest
from upstash_redis import Redis

from app.store import (
    LockBusy,
    MemoryStore,
    RedisStore,
    Store,
    StoreConflict,
    StoreUnavailable,
)


class Clock:
    """Advance store expiry without sleeping in memory tests."""

    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value


@pytest.fixture(params=["memory", "redis"])
def store(request: pytest.FixtureRequest) -> Iterator[Store]:
    if request.param == "memory":
        yield MemoryStore()
        return
    url = os.environ.get("KV_REST_API_URL")
    token = os.environ.get("KV_REST_API_TOKEN")
    if not url or not token:
        pytest.skip("Redis requires KV_REST_API_URL and KV_REST_API_TOKEN")
    with Redis(url=url, token=token) as redis:
        yield RedisStore(redis, prefix=f"test:{uuid4()}:")


def parallel(call: Callable[[int], object], count: int = 20) -> list:
    """Start workers together so tests exercise competing operations."""
    barrier = threading.Barrier(count)

    def run(index: int) -> object:
        barrier.wait(timeout=10)
        return call(index)

    with ThreadPoolExecutor(max_workers=count) as pool:
        return list(pool.map(run, range(count)))


def test_parallel_document_updates_lose_nothing(store: Store) -> None:
    def append(index: int) -> dict:
        def mutate(doc: dict) -> dict:
            doc["items"].append(index)
            return doc

        return store.update_doc("doc", mutate, {"items": []})

    parallel(append)
    assert sorted(store.get_doc("doc")["items"]) == list(range(20))


def test_parallel_reservations_respect_cap(store: Store) -> None:
    results = parallel(lambda _: store.reserve("budget", 1, 10))
    assert results.count(True) == 10
    assert store.remaining("budget", 10) == 0
    assert store.remaining("budget", 20) == 10


def test_parallel_increment_respects_limit(store: Store) -> None:
    results = parallel(lambda _: store.incr_below("counter", 3))
    assert results.count(True) == 3
    assert store.remaining("counter", 20) == 17


def test_settlement_reconciles_reserved_and_actual(store: Store) -> None:
    assert store.remaining("budget", 100) == 100
    assert store.reserve("budget", 40, 100)
    store.settle("budget", reserved=40, actual=15)
    assert store.remaining("budget", 100) == 85
    store.settle("budget", reserved=0, actual=200)
    assert store.remaining("budget", 100) == 0


def test_documents_are_detached_and_version_is_private(store: Store) -> None:
    default = {"items": [], "_version": 999}
    assert store.get_doc("missing") is None

    def mutate(doc: dict) -> dict:
        assert "_version" not in doc
        doc["items"].append({"value": 1})
        doc["_version"] = 999
        return doc

    result = store.update_doc("doc", mutate, default)
    assert result == {"items": [{"value": 1}]}
    assert default == {"items": [], "_version": 999}
    result["items"][0]["value"] = 2
    read = store.get_doc("doc")
    read["items"].clear()
    assert store.get_doc("doc") == {"items": [{"value": 1}]}
    assert store.update_doc("doc", lambda doc: doc, {}) == {"items": [{"value": 1}]}


def test_mutation_exception_writes_nothing(store: Store) -> None:
    store.update_doc("doc", lambda _: {"items": [1]}, {})

    def fail(doc: dict) -> dict:
        doc["items"].append(2)
        raise LookupError("domain failure")

    with pytest.raises(LookupError, match="domain failure"):
        store.update_doc("doc", fail, {})
    assert store.get_doc("doc") == {"items": [1]}
    with pytest.raises(LookupError):
        store.update_doc("absent", fail, {"items": []})
    assert store.get_doc("absent") is None


def test_append_many_preserves_order_and_copies(store: Store) -> None:
    assert store.read_list("events") == []
    store.append_many("events", [])
    items = [{"index": 1, "nested": ["original"]}, {"index": 2}]
    store.append_many("events", items)
    store.append_many("events", [{"index": 3}])
    items[0]["nested"].clear()
    result = store.read_list("events")
    assert result == [
        {"index": 1, "nested": ["original"]},
        {"index": 2},
        {"index": 3},
    ]
    result.clear()
    assert len(store.read_list("events")) == 3


@pytest.mark.parametrize("operation", ["document", "append", "reserve", "increment", "settle"])
def test_writes_set_day_ttl(store: Store, operation: str) -> None:
    if operation == "document":
        store.update_doc("ttl", lambda _: {}, {})
    elif operation == "append":
        store.append_many("ttl", [{"event": "test"}])
    elif operation == "reserve":
        assert store.reserve("ttl", 1, 10)
    elif operation == "increment":
        assert store.incr_below("ttl", 10)
    else:
        store.settle("ttl", 0, 1)
    if isinstance(store, MemoryStore):
        assert store.ttl("ttl") == 86_400
    else:
        assert isinstance(store, RedisStore)
        # Only the isolated test prefix is inspected; no shared keys are touched.
        assert 86_395 <= store.redis.ttl(f"{store.prefix}ttl") <= 86_400


def test_lock_excludes_second_holder_and_releases(store: Store) -> None:
    with store.lock("lock", 10):
        with pytest.raises(LockBusy), store.lock("lock", 10):
            pytest.fail("A competing lock was acquired")
    with store.lock("lock", 10):
        pass


def test_expired_holder_cannot_release_new_lock(store: Store) -> None:
    clock = Clock()
    if isinstance(store, MemoryStore):
        store = MemoryStore(now=clock)
    old = store.lock("lease", 1)
    old.__enter__()
    if isinstance(store, MemoryStore):
        assert store.ttl("lease") == 1
        clock.value = 1.0
    else:
        time.sleep(1.1)
    with store.lock("lease", 10):
        old.__exit__(None, None, None)
        with pytest.raises(LockBusy), store.lock("lease", 10):
            pytest.fail("The expired holder released someone else's lock")


def test_memory_expiry_and_write_refresh() -> None:
    clock = Clock()
    store = MemoryStore(now=clock)
    store.update_doc("doc", lambda _: {"value": 1}, {})
    store.append_many("events", [{"value": 1}])
    store.reserve("budget", 1, 10)
    store.incr_below("counter", 3)
    clock.value = 86_399
    assert store.ttl("doc") == 1
    assert store.get_doc("doc") == {"value": 1}
    store.update_doc("doc", lambda doc: doc, {})
    store.append_many("events", [])
    clock.value = 86_400
    assert store.ttl("doc") == 86_399
    assert store.read_list("events") == []
    assert store.ttl("events") is None
    assert store.remaining("budget", 10) == 10
    assert store.remaining("counter", 3) == 3
    assert store.incr_below("counter", 3)
    clock.value = 172_799
    assert store.get_doc("doc") is None
    assert store.ttl("doc") is None


def test_conflict_raises_after_twenty_attempts() -> None:
    store = MemoryStore()
    attempts = 0

    def conflict(doc: dict) -> dict:
        nonlocal attempts
        attempts += 1
        store.update_doc("doc", lambda _: {"concurrent": attempts}, {})
        return {"losing": True}

    with pytest.raises(StoreConflict):
        store.update_doc("doc", conflict, {})
    assert attempts == 20
    assert store.get_doc("doc") == {"concurrent": 20}


@pytest.mark.parametrize(
    "operation",
    [
        lambda s: s.get_doc("key"),
        lambda s: s.update_doc("key", lambda doc: doc, {}),
        lambda s: s.append_many("key", [{}]),
        lambda s: s.read_list("key"),
        lambda s: s.reserve("key", 1, 2),
        lambda s: s.settle("key", 1, 0),
        lambda s: s.remaining("key", 2),
        lambda s: s.incr_below("key", 2),
        lambda s: s.lock("key", 2).__enter__(),
    ],
)
def test_redis_backend_errors_are_sanitized(operation: Callable[[RedisStore], object]) -> None:
    redis = Mock(spec=Redis)
    for name in ("get", "eval", "lrange", "set"):
        getattr(redis, name).side_effect = RuntimeError("test-token-not-secret")
    with pytest.raises(StoreUnavailable) as error:
        operation(RedisStore(redis, prefix="test:fake:"))
    assert "test-token-not-secret" not in str(error.value)
    assert error.value.__suppress_context__


def test_redis_cas_conflicts_and_versions() -> None:
    redis = Mock(spec=Redis)
    redis.get.return_value = json.dumps({"value": 1, "_version": 7})
    redis.eval.return_value = 0
    store = RedisStore(redis, prefix="test:fake:")
    with pytest.raises(StoreConflict):
        store.update_doc("doc", lambda doc: {"value": doc["value"] + 1}, {})
    assert redis.eval.call_count == 20
    assert redis.eval.call_args.kwargs["keys"] == ["test:fake:doc"]
    assert json.loads(redis.eval.call_args.kwargs["args"][1]) == {"value": 2, "_version": 8}


def test_redis_lock_release_errors_are_sanitized() -> None:
    redis = Mock(spec=Redis)
    redis.set.return_value = True
    lock = RedisStore(redis, prefix="test:fake:").lock("lease", 5)
    lock.__enter__()
    redis.eval.side_effect = RuntimeError("test-token-not-secret")
    with pytest.raises(StoreUnavailable, match="Store backend unavailable"):
        lock.__exit__(None, None, None)


def test_redis_append_is_one_batch_with_expiry() -> None:
    redis = Mock(spec=Redis)
    store = RedisStore(redis, prefix="test:fake:")
    store.append_many("events", [])
    assert not redis.mock_calls
    store.append_many("events", [{"index": 1}, {"index": 2}])
    assert redis.eval.call_count == 1
    assert redis.eval.call_args.kwargs["keys"] == ["test:fake:events"]
    args = redis.eval.call_args.kwargs["args"]
    assert args[0] == "86400"
    assert [json.loads(item) for item in args[1:]] == [{"index": 1}, {"index": 2}]
