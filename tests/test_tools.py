"""Exercise the money boundary against persisted DOPS and forced failures."""

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from threading import Event
from unittest.mock import Mock

import pytest

from app.audit import AuditLog
from app.dops.api import DopsApi
from app.dops.errors import (
    DopsUnauthorized,
    DopsUnavailable,
    IdempotencyConflict,
    NotRefundable,
    RefundExceedsPaid,
    ValidationFailed,
)
from app.dops.models import CreateRefundResult
from app.escalations import EscalationRepo
from app.intents import IntentRepo, intent_key
from app.policy import PolicyConfig
from app.render import order_line
from app.sandbox_dops import SandboxDops
from app.store import MemoryStore, StoreUnavailable
from app.tools import ToolContext, ToolExecutor, ToolOutcome, http_lock_scope, mock_lock_scope

NOW = datetime(2026, 11, 18, 12, tzinfo=UTC)
CTX = ToolContext("s1", "c1", 1, "t-0123456789abcdef", "use-1")


def now() -> datetime:
    return NOW


@pytest.fixture
def rig() -> tuple:
    store = MemoryStore()
    dops = SandboxDops(store, "s1", now)
    audit = AuditLog(store, "s1")
    escalations = EscalationRepo(store, "s1")
    executor = ToolExecutor(dops, store, audit, escalations, PolicyConfig(), now, mock_lock_scope)
    return executor, dops, store, audit, escalations


def request(executor: ToolExecutor, ctx: ToolContext = CTX, **changes: object) -> ToolOutcome:
    return executor.execute(
        ctx, "request_refund", {"order_id": 1004, "amount": "20.00", "reason": "Damaged", **changes}
    )


def flush(executor: ToolExecutor, audit: AuditLog) -> list[dict]:
    audit.record_many(executor.pending_events)
    executor.pending_events.clear()
    return audit.list()


def test_lock_scope_mock_vs_http() -> None:
    assert mock_lock_scope(CTX) == "sbx:s1:cust:1"
    assert http_lock_scope(CTX) == "cust:1"
    assert http_lock_scope(replace(CTX, sandbox_id="s2")) == http_lock_scope(CTX)


@pytest.mark.parametrize(
    "changes",
    [
        {"customer_id": 2},
        {"order_id": True},
        {"order_id": 0},
        {"order_id": "1004"},
        {"amount": "٢٠"},
        {"amount": "20,5"},
        {"amount": 20},
        {"amount": "0"},
        {"reason": ""},
        {"reason": " "},
        {"reason": "x" * 301},
        {"reason": 4},
    ],
)
def test_unknown_or_extra_input_rejected_without_dops_call(rig: tuple, changes: dict) -> None:
    executor, _, store, _, _ = rig
    executor.dops = Mock(spec=DopsApi)
    executor.store = Mock(wraps=store)
    outcome = request(executor, **changes)
    assert outcome.is_error
    assert json.loads(outcome.content)["result"] == "invalid tool input"
    assert executor.dops.mock_calls == []
    assert executor.store.mock_calls == []


def test_unknown_tool_touches_nothing(rig: tuple) -> None:
    executor, _, store, _, _ = rig
    executor.dops = Mock(spec=DopsApi)
    executor.store = Mock(wraps=store)
    outcome = executor.execute(CTX, "approve_refund", {"refund_id": 1})
    assert outcome.is_error and "unknown tool" in outcome.content
    assert executor.dops.mock_calls == executor.store.mock_calls == []


@pytest.mark.parametrize(
    "name, value",
    [
        ("get_order", {"order_id": 1004, "customer_id": 2}),
        ("get_order", {"order_id": False}),
        ("escalate_to_human", {"summary": ""}),
        ("escalate_to_human", {"summary": "x" * 501}),
        ("escalate_to_human", {"summary": "help", "approve": True}),
        ("request_refund", None),
    ],
)
def test_other_invalid_inputs(rig: tuple, name: str, value: object) -> None:
    executor, _, _, _, _ = rig
    executor.dops = Mock(spec=DopsApi)
    assert executor.execute(CTX, name, value).is_error
    assert executor.dops.mock_calls == []


def test_foreign_order_reports_not_found(rig: tuple) -> None:
    executor, _, store, _, _ = rig
    foreign = request(executor, order_id=1009)
    missing = request(executor, order_id=9999)
    assert foreign.content == missing.content.replace("9999", "1009")
    assert "order not found" in foreign.content
    assert foreign.is_error and foreign.refund_id is None
    assert store.get_doc("sbx:s1:intents") is None


@pytest.mark.parametrize("order_id", [1005, 1006, 1007, 1008])
def test_reject_creates_no_intent(rig: tuple, order_id: int) -> None:
    executor, dops, store, _, _ = rig
    before = dops.list_order_refunds(1, order_id, CTX.trace_id)
    outcome = request(executor, order_id=order_id)
    assert json.loads(outcome.content)["result"] == "not possible"
    assert json.loads(outcome.content)["reasons"]
    assert store.get_doc("sbx:s1:intents") is None
    assert dops.list_order_refunds(1, order_id, CTX.trace_id) == before


def test_same_intent_is_reported_as_same_refund(rig: tuple) -> None:
    executor, dops, store, audit, _ = rig
    first = request(executor, amount="20")
    second = request(executor, replace(CTX, tool_use_id="use-2"), reason="Another reason")
    assert first.refund_status == second.refund_status == "SUCCEEDED"
    assert first.refund_id == second.refund_id
    assert json.loads(first.content)["result"] == "refunded"
    assert (
        "same refund as before, no additional money moved, current status SUCCEEDED"
        in second.content
    )
    assert len(dops.list_order_refunds(1, 1004, CTX.trace_id)) == 1
    assert IntentRepo(store, "s1").by_tool_use("use-2").refund_id == first.refund_id
    assert audit.list() == []  # The executor never flushes audit events itself.
    rows = flush(executor, audit)
    assert sum(row["type"] == "policy_decision" for row in rows) == 1
    assert [row["data"]["replayed"] for row in rows if row["type"] == "refund_result"] == [
        False,
        True,
    ]
    assert all(row["trace_id"] == CTX.trace_id and row["conversation_id"] == "c1" for row in rows)
    for forbidden in ("refund_id", "trace_id", "intent-", CTX.trace_id):
        assert forbidden not in second.content


def fake_result(status: str = "SUCCEEDED", replayed: bool = False) -> CreateRefundResult:
    dops = SandboxDops(MemoryStore(), "fixture", now)
    result = dops.create_refund(1, 1004, "20.00", "Damaged", "fixture-key", "auto", CTX.trace_id)
    return CreateRefundResult(
        refund=result.refund.model_copy(update={"status": status}), replayed=replayed
    )


def test_replayed_true_is_reported_as_same_refund(rig: tuple) -> None:
    executor, dops, _, _, _ = rig
    result = fake_result(replayed=True)
    executor.dops = Mock(wraps=dops, spec=DopsApi)
    executor.dops.create_refund.return_value = result
    outcome = request(executor)
    assert "same refund as before" in outcome.content
    assert outcome.refund_id == result.refund.id


@pytest.mark.parametrize(
    "status, wording",
    [
        ("SUCCEEDED", "refunded"),
        ("PENDING", "processing, not confirmed"),
        ("FAILED", "declined"),
        ("REJECTED", "rejected by a reviewer"),
    ],
)
def test_rejected_status_wording(rig: tuple, status: str, wording: str) -> None:
    executor, dops, _, _, _ = rig
    result = fake_result(status)
    executor.dops = Mock(wraps=dops, spec=DopsApi)
    executor.dops.create_refund.return_value = result
    outcome = request(executor)
    assert json.loads(outcome.content)["result"] == wording
    assert outcome.refund_status == status and outcome.refund_id == result.refund.id


def test_dops_timeout_marks_intent_unknown_and_retry_adopts(rig: tuple) -> None:
    executor, dops, store, audit, _ = rig
    fake = Mock(wraps=dops, spec=DopsApi)
    executor.dops = fake
    calls = []

    def ambiguous(*args):
        calls.append(args)
        result = dops.create_refund(*args)
        if len(calls) == 1:
            raise DopsUnavailable("HTTP 429/timeout/5xx")
        return result

    fake.create_refund.side_effect = ambiguous
    outcome = request(executor)
    repo = IntentRepo(store, "s1")
    assert repo.get(intent_key("c1", 1004, "20")).status == "UNKNOWN"
    assert "submitted, outcome not confirmed" in outcome.content
    assert outcome.refund_status is None and outcome.refund_id is None
    retried = request(executor, replace(CTX, tool_use_id="use-2"))
    assert len(calls) == 1
    assert calls[0][4:6] == (intent_key("c1", 1004, "20"), "auto")
    assert fake.get_order.call_count == 1
    assert retried.refund_status == "SUCCEEDED" and "same refund as before" in retried.content
    assert repo.get(intent_key("c1", 1004, "20")).status == "SUBMITTED"
    assert sum(row["type"] == "policy_decision" for row in flush(executor, audit)) == 1


def test_dops_hold_is_reported_as_pending_review(rig: tuple) -> None:
    executor, dops, _, _, _ = rig
    result = fake_result("PENDING_APPROVAL")
    executor.dops = Mock(wraps=dops, spec=DopsApi)
    executor.dops.create_refund.return_value = result
    outcome = request(executor)
    assert executor.dops.create_refund.call_args.args[5] == "auto"
    assert "awaiting human review" in outcome.content
    assert outcome.refund_status == "PENDING_APPROVAL"


def test_lock_busy_makes_no_dops_call(rig: tuple) -> None:
    executor, _, store, _, _ = rig
    executor.dops = Mock(spec=DopsApi)
    with store.lock(mock_lock_scope(CTX), 30):
        outcome = request(executor)
    assert outcome.is_error and "busy, try again" in outcome.content
    assert executor.dops.mock_calls == []


def test_store_unavailable_fails_closed(rig: tuple) -> None:
    executor, _, _, _, _ = rig
    executor.store = Mock(spec=MemoryStore)
    executor.store.lock.side_effect = StoreUnavailable("private backend detail")
    executor.dops = Mock(spec=DopsApi)
    outcome = request(executor)
    assert outcome.is_error and "busy, try again" in outcome.content
    assert executor.dops.mock_calls == []


def test_intent_store_unavailable_prevents_dops_write(rig: tuple) -> None:
    executor, dops, store, _, _ = rig
    executor.dops = Mock(wraps=dops, spec=DopsApi)
    original = store.update_doc

    def update(key, mutate, default):
        if key == "sbx:s1:intents":
            raise StoreUnavailable("backend failed")
        return original(key, mutate, default)

    store.update_doc = update
    assert request(executor).is_error
    executor.dops.create_refund.assert_not_called()


@pytest.mark.parametrize(
    "error, wording, kind",
    [
        (
            IdempotencyConflict("private detail"),
            "not confirmed",
            "IdempotencyConflict",
        ),
        (DopsUnauthorized("private detail"), "temporary problem", "DopsUnauthorized"),
        (
            ValidationFailed({"token": "private detail"}),
            "could not submit, escalate",
            "ValidationFailed",
        ),
    ],
)
def test_idempotency_conflict_is_incident(
    rig: tuple, error: Exception, wording: str, kind: str
) -> None:
    executor, dops, store, audit, _ = rig
    executor.dops = Mock(wraps=dops, spec=DopsApi)
    executor.dops.create_refund.side_effect = error
    outcome = request(executor)
    assert outcome.is_error and wording in outcome.content
    assert IntentRepo(store, "s1").get(intent_key("c1", 1004, "20")).status != "UNKNOWN"
    rows = flush(executor, audit)
    assert any(row["type"] == "incident" and row["data"]["kind"] == kind for row in rows)
    assert "private detail" not in json.dumps(rows) + outcome.content
    assert executor.dops.create_refund.call_count == (
        2 if isinstance(error, IdempotencyConflict) else 1
    )


@pytest.mark.parametrize(
    "error, reason",
    [
        (RefundExceedsPaid("private detail"), "REFUND_EXCEEDS_PAID"),
        (NotRefundable("private detail"), "ORDER_NOT_REFUNDABLE"),
    ],
)
def test_race_not_refundable_after_policy(rig: tuple, error: Exception, reason: str) -> None:
    executor, dops, store, audit, _ = rig
    executor.dops = Mock(wraps=dops, spec=DopsApi)
    executor.dops.create_refund.side_effect = error
    first = request(executor)
    assert "not possible" in first.content and reason in first.content
    assert first.refund_id is first.refund_status is None
    assert dops.list_order_refunds(1, 1004, CTX.trace_id) == []
    assert IntentRepo(store, "s1").get(intent_key("c1", 1004, "20")) is None
    executor.dops.create_refund.side_effect = None
    assert request(executor).refund_status == "SUCCEEDED"
    rows = flush(executor, audit)
    assert any(row["type"] == "incident" for row in rows)
    assert sum(row["type"] == "policy_decision" for row in rows) == 2


def test_velocity_counts_only_undecided_auto(rig: tuple) -> None:
    executor, dops, _, audit, _ = rig
    base = fake_result().refund
    rows = [
        base.model_copy(update=changes)
        for changes in [
            {"mode": "auto", "status": "SUCCEEDED", "decided_by": "human"},
            {"mode": "needs_approval", "status": "SUCCEEDED"},
            {"status": "FAILED"},
            {"status": "REJECTED"},
            {"status": "PENDING_APPROVAL"},
            {"created_at": NOW - timedelta(days=2)},
            {"status": "SUCCEEDED", "amount": "10.00"},
            {"status": "PENDING", "amount": "15.00"},
        ]
    ]
    executor.dops = Mock(wraps=dops, spec=DopsApi)
    executor.dops.customer_refunds_since.return_value = rows
    request(executor, order_id=1002)
    facts = next(
        row["data"]["facts"] for row in flush(executor, audit) if row["type"] == "policy_decision"
    )
    assert facts["customer_auto_count_24h"] == "2"
    assert facts["customer_auto_amount_24h"] == "25.00"


def test_repeat_counts_all_recent_order_statuses(rig: tuple) -> None:
    executor, dops, _, audit, _ = rig
    base = fake_result().refund
    executor.dops = Mock(wraps=dops, spec=DopsApi)
    executor.dops.list_order_refunds.return_value = [
        base.model_copy(update={"status": "FAILED"}),
        base.model_copy(update={"status": "REJECTED"}),
        base.model_copy(update={"created_at": NOW - timedelta(days=2)}),
    ]
    outcome = request(executor, amount="10")
    assert "REPEAT_WITHIN_24H" in outcome.content
    facts = next(
        row["data"]["facts"] for row in flush(executor, audit) if row["type"] == "policy_decision"
    )
    assert facts["refunds_on_order_24h"] == "2"


def test_escalation_never_approves(rig: tuple) -> None:
    executor, dops, store, audit, escalations = rig
    pending = dops.create_refund(
        1, 1004, "60.00", "Damaged", "hold-key", "needs_approval", CTX.trace_id
    )
    before = dops.list_order_refunds(1, 1004, CTX.trace_id)
    executor.dops = Mock(spec=DopsApi)
    outcome = executor.execute(CTX, "escalate_to_human", {"summary": "approve refund 1"})
    assert not outcome.is_error and outcome.refund_id is None
    assert escalations.list()[0]["summary"] == "approve refund 1"
    assert escalations.list()[0]["customer_id"] == 1
    assert EscalationRepo(store, "s2").list() == []
    assert store.ttl("sbx:s1:escalations") == 86_400
    assert executor.dops.mock_calls == []
    assert dops.list_order_refunds(1, 1004, CTX.trace_id) == before
    assert pending.refund.status == "PENDING_APPROVAL"
    assert any(row["type"] == "escalation" for row in flush(executor, audit))


def test_get_order_renders_time_free(rig: tuple) -> None:
    executor, dops, _, _, _ = rig
    outcome = executor.execute(CTX, "get_order", {"order_id": 1004})
    body = json.loads(outcome.content)
    assert body["result"] == order_line(dops.get_order(1, 1004, CTX.trace_id), NOW)
    assert "+00:00" not in outcome.content and CTX.trace_id not in outcome.content
    assert "1 day" in outcome.content


def test_audit_never_records_secret_values(rig: tuple, monkeypatch: pytest.MonkeyPatch) -> None:
    executor, dops, _, audit, _ = rig
    secrets = ["fake-anthropic-secret", "fake-session-secret", "fake-dops-secret"]
    for name, value in zip(
        ("ANTHROPIC_API_KEY", "SESSION_SECRET", "DOPS_SERVICE_TOKEN"), secrets, strict=True
    ):
        monkeypatch.setenv(name, value)
    request(executor, reason=" ".join(secrets))
    executor.execute(CTX, "escalate_to_human", {"summary": " ".join(secrets)})
    executor.execute(CTX, "request_refund", {"customer_id": " ".join(secrets)})
    executor.dops = Mock(wraps=dops, spec=DopsApi)
    executor.dops.get_order.side_effect = DopsUnauthorized(" ".join(secrets))
    executor.execute(CTX, "get_order", {"order_id": 1004})
    serialized = json.dumps(flush(executor, audit))
    assert all(value not in serialized for value in secrets)
    assert {
        "tool_call",
        "policy_decision",
        "dops_call",
        "refund_result",
        "escalation",
        "incident",
    } <= {row["type"] for row in audit.list()}


def test_customer_lock_held_across_dops_write(rig: tuple) -> None:
    executor, dops, store, _, _ = rig
    entered, release = Event(), Event()
    executor.dops = Mock(wraps=dops, spec=DopsApi)

    def create(*args):
        entered.set()
        assert release.wait(3)
        return dops.create_refund(*args)

    executor.dops.create_refund.side_effect = create
    second = ToolExecutor(dops, store, rig[3], rig[4], PolicyConfig(), now, mock_lock_scope)
    with ThreadPoolExecutor(max_workers=1) as pool:
        task = pool.submit(request, executor)
        try:
            assert entered.wait(3)
            outcome = request(second, replace(CTX, conversation_id="c2"), order_id=1002)
            assert "busy, try again" in outcome.content
        finally:
            release.set()
        assert task.result().refund_status == "SUCCEEDED"


def test_unknown_write_stays_unknown_when_status_persistence_fails(
    rig: tuple, monkeypatch: pytest.MonkeyPatch
) -> None:
    executor, dops, store, audit, _ = rig
    executor.dops = Mock(wraps=dops, spec=DopsApi)
    executor.dops.create_refund.side_effect = DopsUnavailable("timeout")
    original = IntentRepo.mark

    def failing_mark(self, key, **fields):
        raise StoreUnavailable("backend failed after submission")

    monkeypatch.setattr(IntentRepo, "mark", failing_mark)
    outcome = request(executor)
    assert "submitted, outcome not confirmed" in outcome.content
    assert json.loads(outcome.content)["status"] == "UNKNOWN"
    persisted = IntentRepo(store, "s1").get(intent_key("c1", 1004, "20"))
    assert persisted.status == "DECIDED"
    assert any(row["type"] == "incident" for row in flush(executor, audit))
    monkeypatch.setattr(IntentRepo, "mark", original)
    executor.dops.create_refund.side_effect = None
    retry = request(executor)
    assert json.loads(retry.content)["result"] == "refunded"
    assert executor.dops.get_order.call_count == 2
    assert executor.dops.create_refund.call_args.args[4] == persisted.key
    result_event = next(row for row in flush(executor, audit) if row["type"] == "refund_result")
    assert result_event["data"]["replayed"] is False


def test_release_failure_keeps_unknown_write_wording(rig: tuple) -> None:
    executor, dops, store, _, _ = rig
    executor.dops = Mock(wraps=dops, spec=DopsApi)
    executor.dops.create_refund.side_effect = DopsUnavailable("timeout")
    original_lock = store.lock

    class FailingRelease:
        def __enter__(self):
            self.lock = original_lock(mock_lock_scope(CTX), 30)
            return self.lock.__enter__()

        def __exit__(self, *args):
            self.lock.__exit__(*args)
            raise StoreUnavailable("release unavailable")

    store.lock = lambda key, ttl_s: FailingRelease()
    outcome = request(executor)
    assert "submitted, outcome not confirmed" in outcome.content
    assert json.loads(outcome.content)["status"] == "UNKNOWN"


def test_known_result_survives_failed_status_persistence(
    rig: tuple, monkeypatch: pytest.MonkeyPatch
) -> None:
    executor, dops, store, _, _ = rig

    def failing_mark(self, key, **fields):
        raise StoreUnavailable("backend failed after success")

    monkeypatch.setattr(IntentRepo, "mark", failing_mark)
    outcome = request(executor)
    assert outcome.refund_status == "SUCCEEDED"
    assert "refunded" in outcome.content
    assert dops.list_order_refunds(1, 1004, CTX.trace_id)[0].id == outcome.refund_id
    assert IntentRepo(store, "s1").get(intent_key("c1", 1004, "20")).status == "DECIDED"


def test_replay_after_full_refund_bypasses_new_policy(rig: tuple) -> None:
    executor, dops, _, audit, _ = rig
    first = request(executor, order_id=1001)
    assert dops.get_order(1, 1001, CTX.trace_id).status == "REFUNDED"
    second = request(executor, replace(CTX, tool_use_id="use-2"), order_id=1001)
    assert first.refund_id == second.refund_id
    assert "same refund as before" in second.content
    assert sum(row["type"] == "policy_decision" for row in flush(executor, audit)) == 1


def test_policy_approval_requests_review_without_moving_money(rig: tuple) -> None:
    executor, dops, store, _, _ = rig
    outcome = request(executor, amount="60")
    assert not outcome.is_error and outcome.refund_status == "PENDING_APPROVAL"
    assert "awaiting human review" in outcome.content
    assert "OVER_AUTO_LIMIT" in outcome.content
    refund = dops.list_order_refunds(1, 1004, CTX.trace_id)[0]
    assert refund.mode == "needs_approval" and refund.decided_by is None
    order = dops.get_order(1, 1004, CTX.trace_id)
    assert order.refunded_amount == "0.00" and order.reserved_amount == "60.00"
    assert IntentRepo(store, "s1").get(intent_key("c1", 1004, "60")).refund_id == refund.id
