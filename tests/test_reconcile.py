"""Recover refund submissions across process interruptions using persisted evidence."""

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import UTC, datetime
from threading import Event
from unittest.mock import Mock

import pytest

from app.agent import AgentConfig, AgentDeps, run_customer_turn, start_conversation
from app.audit import AuditLog
from app.budget import Budget
from app.dops.errors import DopsUnavailable
from app.escalations import EscalationRepo
from app.intents import IntentRepo, intent_key
from app.model import ModelResponse, ScriptedModel, TextBlock, Usage
from app.policy import PolicyConfig
from app.sandbox_dops import SandboxDops
from app.store import MemoryStore
from app.tools import ToolContext, ToolExecutor, mock_lock_scope


def now():
    return datetime(2026, 11, 18, 12, tzinfo=UTC)


def restart(store):
    audit = AuditLog(store, "s1")
    executor = ToolExecutor(
        SandboxDops(store, "s1", now),
        store,
        audit,
        EscalationRepo(store, "s1"),
        PolicyConfig(),
        now,
        mock_lock_scope,
    )
    return AgentDeps(
        "s1",
        ScriptedModel([ModelResponse("end_turn", [TextBlock("Ready.")], Usage(1, 0, 0, 1))]),
        executor,
        Budget(store, 120_000_000, lambda: now().date()),
        audit,
        store,
        AgentConfig(),
        now,
    )


def request(executor, ctx, amount="20.00"):
    return executor.execute(
        ctx, "request_refund", {"order_id": 1004, "amount": amount, "reason": "Damaged"}
    )


@pytest.fixture
def setup():
    deps = restart(MemoryStore())
    conv = start_conversation(deps.store, "s1", 1, deps.executor.dops, now)
    ctx = ToolContext("s1", conv.id, 1, "t-0000000000000001", "use-1")
    return deps, ctx


@pytest.mark.parametrize("committed", [False, True], ids=["before-commit", "after-commit"])
@pytest.mark.parametrize("interruption", ["crash", "lost-response"])
def test_restart_repairs_from_dops_without_writing(setup, committed, interruption):
    deps, ctx = setup
    store = deps.store
    use = {
        "type": "tool_use",
        "id": ctx.tool_use_id,
        "name": "request_refund",
        "input": {"order_id": 1004, "amount": "20.00", "reason": "Damaged"},
    }
    store.update_doc(
        f"sbx:s1:conv:{ctx.conversation_id}",
        lambda doc: {**doc, "messages": [{"role": "assistant", "content": [use]}]},
        {},
    )
    dops = deps.executor.dops
    deps.executor.dops = Mock(wraps=dops)

    def interrupt(*args):
        if committed:
            dops.create_refund(*args)
        if interruption == "crash":
            raise SystemExit
        raise DopsUnavailable("response lost")

    deps.executor.dops.create_refund.side_effect = interrupt
    if interruption == "crash":
        with pytest.raises(SystemExit):
            request(deps.executor, ctx)
    else:
        assert "not confirmed" in request(deps.executor, ctx).content

    fresh = restart(store)
    fresh.executor.dops = Mock(wraps=fresh.executor.dops)
    turn = run_customer_turn(ctx.conversation_id, "Any update?", fresh)
    body = json.loads(fresh.model.requests[0]["messages"][1]["content"][0]["content"])
    fresh.executor.dops.create_refund.assert_not_called()
    assert turn.stopped == "end_turn"
    rows = dops.list_order_refunds(1, 1004, ctx.trace_id)
    assert len(rows) == int(committed)
    intent = IntentRepo(store, "s1").get(intent_key(ctx.conversation_id, 1004, "20"))
    if committed:
        assert body["status"] == "SUCCEEDED"
        assert turn.refunds[0]["refund_id"] == intent.refund_id == rows[0].id
        assert "1004 / 20.00: refunded" in turn.reply
        assert intent.status == "SUBMITTED"
    else:
        assert "not confirmed" in body["result"] and "again" in body["result"]
        assert not turn.refunds and intent.refund_id is None
        assert intent.status == ("DECIDED" if interruption == "crash" else "UNKNOWN")
        assert sum(e["type"] == "intent_unconfirmed" for e in fresh.audit.list()) == int(
            interruption == "crash"
        )
    assert intent.attempts == ((ctx.trace_id, "auto", ("WITHIN_POLICY",)),)


def test_retry_after_lost_request_checks_velocity_with_same_key(setup):
    deps, ctx = setup
    dops = deps.executor.dops
    deps.executor.dops = Mock(wraps=dops)
    deps.executor.dops.create_refund.side_effect = DopsUnavailable("lost")
    assert "not confirmed" in request(deps.executor, ctx).content
    for order in (1001, 1002, 1003):
        dops.create_refund(1, order, "1.00", "Other", f"other-key-{order}", "auto", "other")
    fresh = restart(deps.store)
    retry = replace(ctx, trace_id="t-0000000000000002", tool_use_id="again")
    outcome = request(fresh.executor, retry)
    assert outcome.refund_status == "PENDING_APPROVAL"
    intent = IntentRepo(deps.store, "s1").by_tool_use("again")
    assert intent.key == deps.executor.dops.create_refund.call_args.args[4]
    assert intent.mode == "needs_approval" and "VELOCITY_COUNT" in intent.decision_reasons
    assert intent.attempts == (
        (ctx.trace_id, "auto", ("WITHIN_POLICY",)),
        (retry.trace_id, "needs_approval", intent.decision_reasons),
    )
    assert len(dops.list_order_refunds(1, 1004, ctx.trace_id)) == 1


def lose_request(deps, ctx, amount="20.00", *, committed=False):
    dops = deps.executor.dops
    proxy = Mock(wraps=dops)

    def lose(*args):
        if committed:
            dops.create_refund(*args)
        raise DopsUnavailable("lost")

    proxy.create_refund.side_effect = lose
    deps.executor.dops = proxy
    try:
        assert "not confirmed" in request(deps.executor, ctx, amount).content
    finally:
        deps.executor.dops = dops
    return proxy.create_refund.call_args.args


@pytest.mark.parametrize("legacy", [None, "no-attempts", "pairs"])
def test_first_write_lands_between_read_and_changed_mode_submission(setup, legacy):
    deps, ctx = setup
    original = lose_request(deps, ctx)
    repo = IntentRepo(deps.store, "s1")
    if legacy:

        def remove_attempts(doc):
            row = doc["intents"][original[4]]
            if legacy == "no-attempts":
                del row["attempts"]
            else:
                row["attempts"] = [[ctx.trace_id, "auto"]]
            return doc

        deps.store.update_doc(repo.key, remove_attempts, {})
        assert repo.get(original[4]).attempts == (
            () if legacy == "no-attempts" else ((ctx.trace_id, "auto", ()),)
        )
    fresh = restart(deps.store)
    dops = fresh.executor.dops
    fresh.executor.config = PolicyConfig(velocity_count=0)
    proxy = Mock(wraps=dops)

    def late_write(*args):
        row = repo.get(original[4])
        assert row.attempts[-1][:2] == (args[6], args[5])
        if args[5] == "needs_approval":
            assert row.mode == "needs_approval" and "VELOCITY_COUNT" in row.decision_reasons
            dops.create_refund(*original)
        return dops.create_refund(*args)

    proxy.create_refund.side_effect = late_write
    fresh.executor.dops = proxy
    retry = replace(ctx, trace_id="retry", tool_use_id="retry")
    outcome = request(fresh.executor, retry)
    assert outcome.refund_status == "SUCCEEDED" and "same refund" in outcome.content
    rows = dops.list_order_refunds(1, 1004, "read")
    assert len(rows) == 1 and rows[0].trace_id == ctx.trace_id
    intent = repo.by_tool_use("retry")
    assert intent.mode == "auto" and intent.refund_id == rows[0].id == outcome.refund_id
    assert intent.decision_reasons == (() if legacy else ("WITHIN_POLICY",))
    assert json.loads(outcome.content)["reasons"] == list(intent.decision_reasons)
    events = fresh.executor.pending_events
    assert sum(e.type == "intent_adopted" for e in events) == 1
    decision = next(e.data for e in events if e.type == "intent_redecided")
    assert (decision["old_mode"], decision["new_mode"]) == ("auto", "needs_approval")
    assert [c.args[4:6] for c in proxy.create_refund.call_args_list] == [
        (original[4], "needs_approval"),
        (original[4], "auto"),
    ]


def test_repair_preserves_existing_reject_wording_until_confirmed(setup):
    deps, ctx = setup
    lose_request(deps, ctx)
    deps.executor.config = PolicyConfig(window_days=0)
    outcome = request(deps.executor, ctx)
    assert "do not allow a new request" in outcome.content
    history = [
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": ctx.tool_use_id,
                    "name": "request_refund",
                    "input": {"order_id": 1004, "amount": "20.00", "reason": "Damaged"},
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": ctx.tool_use_id,
                    "content": outcome.content,
                    "is_error": outcome.is_error,
                }
            ],
        },
    ]
    deps.store.update_doc(
        f"sbx:s1:conv:{ctx.conversation_id}", lambda doc: {**doc, "messages": history}, {}
    )
    fresh = restart(deps.store)
    run_customer_turn(ctx.conversation_id, "Status?", fresh)
    assert fresh.model.requests[0]["messages"][1]["content"][0]["content"] == outcome.content


def test_known_id_recovers_demoted_legacy_intent_without_redecision(setup):
    deps, ctx = setup
    first = request(deps.executor, ctx)
    repo = IntentRepo(deps.store, "s1")
    key = repo.by_tool_use(ctx.tool_use_id).key

    def make_legacy(doc):
        doc["intents"][key].pop("attempts")
        return doc

    deps.store.update_doc(repo.key, make_legacy, {})
    lose_request(deps, replace(ctx, trace_id="outage"))
    assert repo.get(key).status == "UNKNOWN" and repo.get(key).refund_id == first.refund_id
    fresh = restart(deps.store)
    fresh.executor.dops = Mock(wraps=fresh.executor.dops)
    outcome = request(fresh.executor, replace(ctx, trace_id="retry"))
    assert outcome.refund_id == first.refund_id and outcome.refund_status == "SUCCEEDED"
    fresh.executor.dops.get_order.assert_not_called()
    fresh.executor.dops.create_refund.assert_not_called()
    assert repo.get(key).decision_reasons == ()


def test_unconfirmed_audited_once_per_attempt_and_turn(setup):
    deps, ctx = setup
    lose_request(deps, ctx)
    lose_request(deps, ctx)
    assert sum(e.type == "intent_unconfirmed" for e in deps.executor.pending_events) == 1
    fresh = restart(deps.store)
    run_customer_turn(ctx.conversation_id, "Status?", fresh)
    assert not any(e["type"] == "intent_unconfirmed" for e in fresh.audit.list())
    lose_request(fresh, replace(ctx, trace_id="another-turn"))
    assert sum(e.type == "intent_unconfirmed" for e in fresh.executor.pending_events) == 1


def test_late_write_after_policy_reject_is_recovered_next_turn(setup):
    deps, ctx = setup
    original = lose_request(deps, ctx)
    fresh = restart(deps.store)
    fresh.executor.config = PolicyConfig(window_days=0)
    dops = fresh.executor.dops
    fresh.executor.dops = Mock(wraps=dops)
    outcome = request(fresh.executor, replace(ctx, trace_id="retry", tool_use_id="retry"))
    assert "not confirmed" in outcome.content and "do not allow a new request" in outcome.content
    fresh.executor.dops.create_refund.assert_not_called()
    assert dops.list_order_refunds(1, 1004, "read") == []
    repo = IntentRepo(deps.store, "s1")
    assert repo.by_tool_use("retry").key == original[4]
    late = dops.create_refund(*original).refund
    turn = run_customer_turn(ctx.conversation_id, "Any news?", fresh)
    assert repo.by_tool_use("retry").refund_id == late.id
    assert turn.refunds[0]["refund_id"] == late.id and "refunded" in turn.reply
    fresh.executor.dops.create_refund.assert_not_called()


@pytest.mark.parametrize(
    "status,wording",
    [
        ("FAILED", "declined"),
        ("REJECTED", "rejected by a reviewer"),
        ("PENDING", "processing, not confirmed yet"),
    ],
)
def test_repair_reports_actual_adopted_status(setup, status, wording):
    deps, ctx = setup
    dops = deps.executor.dops

    def gateway(doc):
        doc["gateway_outcome"] = "decline" if status == "FAILED" else "timeout"
        return doc

    deps.store.update_doc("sbx:s1:dops", gateway, {})
    amount = "60.00" if status == "REJECTED" else "20.00"
    lose_request(deps, ctx, amount, committed=True)
    if status == "REJECTED":
        (row,) = dops.list_order_refunds(1, 1004, ctx.trace_id)
        dops.reject_refund(row.id, "Ops", "Review", "ops-trace")
    fresh = restart(deps.store)
    fresh.executor.dops = Mock(wraps=fresh.executor.dops)
    turn = run_customer_turn(ctx.conversation_id, "Status?", fresh)
    assert turn.refunds[0]["status"] == status and wording in turn.reply
    fresh.executor.dops.create_refund.assert_not_called()


def test_repair_matches_amount_and_trace_and_reads_order_once(setup):
    deps, ctx = setup
    lose_request(deps, ctx, "20.00")
    lose_request(deps, replace(ctx, tool_use_id="second"), "10.00", committed=True)
    dops = deps.executor.dops
    dops.create_refund(1, 1004, "20.00", "Other", "ops-created", "auto", "ops")
    other = replace(ctx, conversation_id="elsewhere", tool_use_id="other", trace_id="other")
    request(deps.executor, other)
    fresh = restart(deps.store)
    fresh.executor.dops = Mock(wraps=fresh.executor.dops)
    turn = run_customer_turn(ctx.conversation_id, "Status?", fresh)
    fresh.executor.dops.list_order_refunds.assert_called_once()
    assert [r["amount"] for r in turn.refunds] == ["10.00"]
    repo = IntentRepo(deps.store, "s1")
    assert repo.by_tool_use(ctx.tool_use_id).refund_id is None
    assert repo.by_tool_use("second").refund_id == turn.refunds[0]["refund_id"]


def test_parallel_retries_share_one_refund(setup):
    deps, ctx = setup
    lose_request(deps, ctx)
    first, second = restart(deps.store).executor, restart(deps.store).executor
    dops = first.dops
    entered, release = Event(), Event()
    first.dops = Mock(wraps=dops)

    def write(*args):
        entered.set()
        assert release.wait(5)
        return dops.create_refund(*args)

    first.dops.create_refund.side_effect = write
    with ThreadPoolExecutor(max_workers=1) as pool:
        task = pool.submit(request, first, replace(ctx, trace_id="first"))
        try:
            assert entered.wait(5)
            assert "busy" in request(second, replace(ctx, trace_id="second")).content
        finally:
            release.set()
        first_result = task.result()
    assert first_result.refund_id == request(second, ctx).refund_id
    assert len(dops.list_order_refunds(1, 1004, "read")) == 1


def test_near_deadline_skips_repair_read(setup):
    deps, ctx = setup
    lose_request(deps, ctx, committed=True)
    fresh = restart(deps.store)
    fresh.config = replace(fresh.config, turn_deadline_s=1)
    fresh.monotonic = lambda: 0
    fresh.executor.dops = Mock(wraps=fresh.executor.dops)
    turn = run_customer_turn(ctx.conversation_id, "Status?", fresh)
    assert turn.stopped == "end_turn" and not turn.refunds
    fresh.executor.dops.list_order_refunds.assert_not_called()
    assert not any(e["type"] == "intent_unconfirmed" for e in fresh.audit.list())


def test_crash_after_decision_before_first_attempt(setup, monkeypatch):
    deps, ctx = setup
    with monkeypatch.context() as patch:
        patch.setattr(IntentRepo, "attempt", Mock(side_effect=SystemExit))
        deps.executor.dops = Mock(wraps=deps.executor.dops)
        with pytest.raises(SystemExit):
            request(deps.executor, ctx)
        deps.executor.dops.create_refund.assert_not_called()
    repo = IntentRepo(deps.store, "s1")
    intent = repo.by_tool_use(ctx.tool_use_id)
    assert intent.status == "DECIDED" and intent.attempts == ()
    fresh = restart(deps.store)
    fresh.executor.config = PolicyConfig(velocity_count=0)
    outcome = request(fresh.executor, replace(ctx, tool_use_id="retry"))
    assert (
        outcome.refund_status == "PENDING_APPROVAL" and "awaiting human review" in outcome.content
    )
    assert repo.by_tool_use("retry").key == intent.key
    assert len(fresh.executor.dops.list_order_refunds(1, 1004, "read")) == 1


def test_pending_match_is_driven_with_its_actual_mode(setup):
    deps, ctx = setup
    deps.store.update_doc("sbx:s1:dops", lambda doc: {**doc, "gateway_outcome": "timeout"}, {})
    original = lose_request(deps, ctx, committed=True)
    repo = IntentRepo(deps.store, "s1")
    intent = repo.get(original[4])
    # Another attempt was recorded in the interval before the first write landed.
    repo.attempt(replace(intent, mode="needs_approval"), "later", "needs_approval")
    deps.store.update_doc("sbx:s1:dops", lambda doc: {**doc, "gateway_outcome": "success"}, {})
    fresh = restart(deps.store)
    fresh.executor.dops = Mock(wraps=fresh.executor.dops)
    outcome = request(fresh.executor, ctx)
    assert outcome.refund_status == "SUCCEEDED"
    assert fresh.executor.dops.create_refund.call_args.args[4:6] == (intent.key, "auto")
    assert repo.get(intent.key).mode == "auto"
    assert len(fresh.executor.dops.list_order_refunds(1, 1004, "read")) == 1


def test_adoption_checks_current_owner_inside_update(setup):
    deps, ctx = setup
    original = lose_request(deps, ctx, committed=True)
    (row,) = deps.executor.dops.list_order_refunds(1, 1004, "read")
    repo = IntentRepo(deps.store, "s1")
    first = repo.get(original[4])
    other = replace(
        first, conversation_id="other", key=intent_key("other", 1004, "20"), tool_use_id="other"
    )
    repo.create(other)
    assert repo.adopt(first.key, row)[1]
    assert not repo.adopt(first.key, row)[1]
    assert not repo.adopt(first.key, row.model_copy(update={"id": row.id + 1}))[1]
    assert not repo.adopt(other.key, row)[1]
    assert repo.get(first.key).refund_id == row.id
    assert repo.get(other.key).refund_id is None
