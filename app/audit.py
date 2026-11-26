"""Batch context-bound audit events into a sandbox's expiring JSON list."""

from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Literal

from app.store import Store


@dataclass(frozen=True)
class AuditEvent:
    """A timestamped event; data must contain only sanitized JSON values."""

    ts: datetime
    trace_id: str
    conversation_id: str
    actor: Literal["customer", "model", "policy", "human", "system"]
    type: str
    data: dict

    def to_dict(self) -> dict:
        """Serialize the timestamp as ISO-8601 for plain JSON storage."""
        return {**asdict(self), "ts": self.ts.isoformat()}


class AuditLog:
    """Append once per caller-controlled step and optionally filter by conversation."""

    def __init__(self, store: Store, sandbox_id: str) -> None:
        self.store = store
        self.key = f"sbx:{sandbox_id}:audit"

    def record_many(self, events: Iterable[AuditEvent]) -> None:
        self.store.append_many(self.key, [event.to_dict() for event in events])

    def list(self, conversation_id: str | None = None) -> list[dict]:
        rows = self.store.read_list(self.key)
        return [
            row
            for row in rows
            if conversation_id is None or row["conversation_id"] == conversation_id
        ]
