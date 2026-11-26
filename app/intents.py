"""Persist refund fingerprints and tool-use aliases in one atomic document."""

from dataclasses import asdict, dataclass, replace
from hashlib import sha256
from typing import Literal

from app.dops.models import Refund, RefundMode
from app.money import fmt_amount, parse_amount
from app.store import Store


def intent_key(conversation_id: str, order_id: int, amount: str) -> str:
    """Hash a conversation, order and canonical amount, never a tool-use id."""
    normalized = fmt_amount(parse_amount(amount))
    return (
        "intent-" + sha256(f"{conversation_id}|{order_id}|{normalized}".encode()).hexdigest()[:40]
    )


@dataclass(frozen=True)
class RefundIntent:
    """The immutable request fingerprint and its latest submission state."""

    key: str
    conversation_id: str
    customer_id: int
    order_id: int
    amount: str
    reason: str
    mode: RefundMode
    decision_reasons: tuple[str, ...]
    status: Literal["DECIDED", "SUBMITTED", "UNKNOWN"]
    refund_id: int | None
    tool_use_id: str
    attempts: tuple[tuple[str, RefundMode, tuple[str, ...]], ...] = ()
    unconfirmed_attempt: int = 0
    unconfirmed_turn: str | None = None


def _decode(row: dict) -> RefundIntent:
    return RefundIntent(
        **{
            **row,
            "decision_reasons": tuple(row["decision_reasons"]),
            "attempts": tuple(
                (attempt[0], attempt[1], tuple(attempt[2]) if len(attempt) > 2 else ())
                for attempt in row.get("attempts", [])
            ),
        }
    )


def _encode(intent: RefundIntent) -> dict:
    return {
        **asdict(intent),
        "decision_reasons": list(intent.decision_reasons),
        "attempts": [[trace, mode, list(reasons)] for trace, mode, reasons in intent.attempts],
    }


class IntentRepo:
    """Keep intent creation and every tool-use alias atomic and sandbox-local."""

    def __init__(self, store: Store, sandbox_id: str) -> None:
        self.store = store
        self.key = f"sbx:{sandbox_id}:intents"

    def get(self, key: str) -> RefundIntent | None:
        doc = self.store.get_doc(self.key) or {}
        row = doc.get("intents", {}).get(key)
        return None if row is None else _decode(row)

    def snapshot(
        self, conversation_id: str | None = None
    ) -> tuple[dict[str, RefundIntent], dict[str, str]]:
        """Read intent states and their tool aliases from the same document version."""
        doc = self.store.get_doc(self.key) or {}
        intents = {
            key: _decode(row)
            for key, row in doc.get("intents", {}).items()
            if conversation_id is None or row["conversation_id"] == conversation_id
        }
        aliases = {use: key for use, key in doc.get("by_tool_use", {}).items() if key in intents}
        return intents, aliases

    def attempt(self, intent: RefundIntent, trace: str, attempt_mode: RefundMode) -> None:
        """Commit the decision and its submission evidence before contacting DOPS."""

        def mutate(doc: dict) -> dict:
            current = _decode(doc["intents"][intent.key])
            reasons = (
                intent.decision_reasons
                if intent.mode == attempt_mode
                else next(
                    (
                        reasons
                        for _, mode, reasons in reversed(current.attempts)
                        if mode == attempt_mode
                    ),
                    (),
                )
            )
            doc["intents"][intent.key] = _encode(
                replace(
                    current,
                    mode=intent.mode,
                    decision_reasons=intent.decision_reasons,
                    attempts=(*current.attempts, (trace, attempt_mode, reasons)),
                )
            )
            return doc

        self.store.update_doc(self.key, mutate, default={})

    def adopt(self, key: str, refund: Refund) -> tuple[RefundIntent | None, bool]:
        """Claim a DOPS row once, checking both guards on each CAS attempt."""
        adopted = False

        def mutate(doc: dict) -> dict:
            nonlocal adopted
            rows = doc.get("intents", {})
            if key not in rows:
                adopted = False
                return doc
            current = _decode(rows[key])
            adopted = current.status in {"DECIDED", "UNKNOWN"} and not any(
                row["refund_id"] == refund.id for other, row in rows.items() if other != key
            )
            if adopted:
                reasons = next(
                    (
                        reasons
                        for trace, mode, reasons in current.attempts
                        if trace == refund.trace_id and mode == refund.mode
                    ),
                    (),
                )
                rows[key] = _encode(
                    replace(
                        current,
                        status="SUBMITTED",
                        refund_id=refund.id,
                        mode=refund.mode,
                        decision_reasons=reasons,
                    )
                )
            return doc

        doc = self.store.update_doc(self.key, mutate, default={})
        row = doc.get("intents", {}).get(key)
        return (_decode(row) if row else None), adopted

    def note_unconfirmed(self, key: str, turn: str) -> bool:
        """Claim the first unresolved observation of a new attempt, once per turn."""
        emit = False

        def mutate(doc: dict) -> dict:
            nonlocal emit
            current = _decode(doc["intents"][key])
            changed = (
                current.status != "SUBMITTED"
                and len(current.attempts) > current.unconfirmed_attempt
            )
            emit = changed and turn != current.unconfirmed_turn
            if changed:
                doc["intents"][key] = _encode(
                    replace(
                        current, unconfirmed_attempt=len(current.attempts), unconfirmed_turn=turn
                    )
                )
            return doc

        self.store.update_doc(self.key, mutate, default={})
        return emit

    def create(self, intent: RefundIntent) -> tuple[RefundIntent, bool]:
        """Preserve the first fingerprint while indexing retry tool-use ids too."""
        created = False

        def mutate(doc: dict) -> dict:
            nonlocal created
            rows = doc.setdefault("intents", {})
            created = intent.key not in rows
            if created:
                rows[intent.key] = _encode(intent)
            doc.setdefault("by_tool_use", {})[intent.tool_use_id] = intent.key
            return doc

        doc = self.store.update_doc(self.key, mutate, default={})
        return _decode(doc["intents"][intent.key]), created

    def mark(self, key: str, **fields: object) -> None:
        """Update only submission state, leaving the persisted fingerprint intact."""
        if not fields.keys() <= {"status", "refund_id"}:
            raise ValueError("Only submission state can be changed")
        if "status" in fields and fields["status"] not in {"DECIDED", "SUBMITTED", "UNKNOWN"}:
            raise ValueError("Invalid intent status")
        if "refund_id" in fields and fields["refund_id"] is not None:
            if type(fields["refund_id"]) is not int or fields["refund_id"] <= 0:
                raise ValueError("Invalid refund id")

        def mutate(doc: dict) -> dict:
            current = _decode(doc["intents"][key])
            doc["intents"][key] = _encode(replace(current, **fields))
            return doc

        self.store.update_doc(self.key, mutate, default={})

    def by_tool_use(self, tool_use_id: str) -> RefundIntent | None:
        doc = self.store.get_doc(self.key) or {}
        key = doc.get("by_tool_use", {}).get(tool_use_id)
        row = doc.get("intents", {}).get(key)
        return None if row is None else _decode(row)

    def _delete(self, key: str) -> None:
        """Discard a definitively refused write so the next call re-decides."""

        def mutate(doc: dict) -> dict:
            doc.get("intents", {}).pop(key, None)
            doc["by_tool_use"] = {
                use: target for use, target in doc.get("by_tool_use", {}).items() if target != key
            }
            return doc

        self.store.update_doc(self.key, mutate, default={})
