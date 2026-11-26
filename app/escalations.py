"""Store model-written handoff text without providing any approval action."""

from app.store import Store


class EscalationRepo:
    """A sandbox-local queue of requests for human attention, never decisions."""

    def __init__(self, store: Store, sandbox_id: str) -> None:
        self.store = store
        self.key = f"sbx:{sandbox_id}:escalations"

    def open(self, conversation_id: str, customer_id: int, summary: str) -> dict:
        row = {
            "conversation_id": conversation_id,
            "customer_id": customer_id,
            "summary": summary,
            "status": "OPEN",
        }
        self.store.append_many(self.key, [row])
        return row

    def list(self) -> list[dict]:
        return self.store.read_list(self.key)
