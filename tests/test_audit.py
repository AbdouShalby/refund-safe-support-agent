"""JSON audit batching and conversation isolation."""

import json
from datetime import UTC, datetime

from app.audit import AuditEvent, AuditLog
from app.store import MemoryStore


class CountingStore(MemoryStore):
    def __init__(self) -> None:
        super().__init__()
        self.appends = 0

    def append_many(self, key: str, items: list[dict]) -> None:
        self.appends += 1
        super().append_many(key, items)


def test_audit_is_plain_json_batched_and_filterable() -> None:
    store = CountingStore()
    audit = AuditLog(store, "s1")
    now = datetime(2026, 11, 18, tzinfo=UTC)
    events = [
        AuditEvent(now, "t-0123456789abcdef", cid, "system", "incident", {"kind": "test"})
        for cid in ("c1", "c2")
    ]
    audit.record_many(events)
    assert store.appends == 1
    rows = audit.list()
    assert json.loads(json.dumps(rows)) == rows
    assert datetime.fromisoformat(rows[0]["ts"]) == now
    assert audit.list("c1") == rows[:1]
    assert AuditLog(store, "s2").list() == []
    assert store.ttl("sbx:s1:audit") == 86_400
