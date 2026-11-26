"""Intent persistence, normalized keys and tool-use recovery."""

from dataclasses import replace
from hashlib import sha256

from app.intents import IntentRepo, RefundIntent, intent_key
from app.store import MemoryStore


def intent(tool_use_id: str = "use-1") -> RefundIntent:
    return RefundIntent(
        intent_key("c1", 1004, "20.00"),
        "c1",
        1,
        1004,
        "20.00",
        "Damaged",
        "auto",
        ("WITHIN_POLICY",),
        "DECIDED",
        None,
        tool_use_id,
    )


def test_intent_key_normalizes_amount() -> None:
    expected = "intent-" + sha256(b"c1|1004|20.00").hexdigest()[:40]
    assert intent_key("c1", 1004, "20") == expected
    assert intent_key("c1", 1004, "20.0") == expected
    assert intent_key("c2", 1004, "20") != expected


def test_create_preserves_first_intent_and_mark_persists() -> None:
    store = MemoryStore()
    repo = IntentRepo(store, "s1")
    original = intent()
    assert repo.create(original) == (original, True)
    changed = replace(original, reason="Different", mode="needs_approval")
    assert repo.create(changed) == (original, False)
    repo.mark(original.key, status="SUBMITTED", refund_id=7)
    assert IntentRepo(store, "s1").get(original.key) == replace(
        original, status="SUBMITTED", refund_id=7
    )
    assert store.ttl("sbx:s1:intents") == 86_400
    assert IntentRepo(store, "s2").get(original.key) is None


def test_tool_use_index() -> None:
    store = MemoryStore()
    repo = IntentRepo(store, "s1")
    original = intent()
    repo.create(original)
    repo.create(replace(original, tool_use_id="use-2"))
    repo.mark(original.key, status="UNKNOWN")
    rebuilt = IntentRepo(store, "s1")
    assert rebuilt.by_tool_use("use-1") == rebuilt.get(original.key)
    assert rebuilt.by_tool_use("use-2") == rebuilt.get(original.key)
    assert rebuilt.by_tool_use("absent") is None
    assert store.get_doc("sbx:s1:intents")["by_tool_use"]["use-2"] == original.key
