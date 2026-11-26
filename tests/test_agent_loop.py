"""Verify bounded, persisted turns with the real sandboxed money boundary."""

import json
import re
from dataclasses import asdict, replace
from datetime import UTC, datetime, timedelta
from unittest.mock import Mock

import pytest

from app.agent import (
    AgentConfig,
    AgentDeps,
    Conversation,
    load_conversation,
    run_customer_turn,
    start_conversation,
)
from app.audit import AuditLog
from app.budget import Budget
from app.customers import customer_name
from app.escalations import EscalationRepo
from app.intents import IntentRepo, RefundIntent, intent_key
from app.model import ModelResponse, ModelUnavailable, ScriptedModel, TextBlock, ToolUseBlock, Usage
from app.policy import PolicyConfig
from app.pricing import cost_nano, worst_case_nano
from app.sandbox_dops import SandboxDops
from app.store import MemoryStore, StoreConflict, StoreUnavailable
from app.tools import ToolExecutor, mock_lock_scope

NOW = datetime(2026, 11, 18, 12, tzinfo=UTC)
USAGE = Usage(10, 20, 30, 40)
CAP = 120_000_000


def response(stop: str = "end_turn", *blocks: TextBlock | ToolUseBlock) -> ModelResponse:
    return ModelResponse(stop, list(blocks), USAGE)


def tool(use: str = "use-1", name: str = "get_order", **data: object) -> ToolUseBlock:
    return ToolUseBlock(use, name, data or {"order_id": 1001})


@pytest.fixture
def rig() -> tuple:
    store = MemoryStore()
    now = lambda: NOW  # noqa: E731
    dops = SandboxDops(store, "s1", now)
    audit = AuditLog(store, "s1")
    executor = ToolExecutor(
        dops, store, audit, EscalationRepo(store, "s1"), PolicyConfig(), now, mock_lock_scope
    )
    deps = AgentDeps(
        "s1",
        ScriptedModel([]),
        executor,
        Budget(store, CAP, NOW.date),
        audit,
        store,
        AgentConfig(),
        now,
    )
    conv = start_conversation(store, "s1", 1, dops, now)
    return deps, conv, dops


def saved(deps: AgentDeps, conv: Conversation) -> Conversation:
    return load_conversation(deps.store, "s1", conv.id)


def test_customer_names_and_conversation_round_trip(rig: tuple) -> None:
    deps, conv, _ = rig
    assert customer_name(1) == "Customer A"
    assert customer_name(987) == "Customer 987"
    assert re.fullmatch(r"c-[0-9a-f]{16}", conv.id)
    assert saved(deps, conv) == conv
    assert load_conversation(deps.store, "other", conv.id) is None
    assert conv.messages == [] and conv.turns == 0 and conv.busy_until is None


def test_small_refund_and_ordered_results_fed_back(rig: tuple) -> None:
    deps, conv, dops = rig
    deps.model = ScriptedModel(
        [
            response(
                "tool_use",
                TextBlock("Checking."),
                tool(),
                tool("use-2", "request_refund", order_id=1001, amount="10.00", reason="Damaged"),
            ),
            response("end_turn", TextBlock("Refunded 10.00.")),
        ]
    )
    result = run_customer_turn(conv.id, "Refund 10.00", deps)
    assert result.stopped == "end_turn"
    assert result.reply == "Refunded 10.00.\n\nRefund system status:\n- 1001 / 10.00: refunded"
    assert result.refunds == [
        {"refund_id": 2, "status": "SUCCEEDED", "order_id": 1001, "amount": "10.00"}
    ]
    assert dops.get_order(1, 1001, result.trace_id).refunded_amount == "10.00"
    request = deps.model.requests[1]
    assert [b["tool_use_id"] for b in request["messages"][-1]["content"]] == ["use-1", "use-2"]
    assert request["messages"][-2]["content"][0] == {"type": "text", "text": "Checking."}
    assert deps.budget.remaining() == CAP - 2 * cost_nano(USAGE)
    assert request["model"] == "claude-haiku-5-5"
    assert request["output_config"] == {"effort": "low"}
    assert saved(deps, conv).busy_until is None
    assert re.fullmatch(r"t-[0-9a-f]{16}", result.trace_id)
    events = deps.audit.list(conv.id)
    assert all(e["trace_id"] == result.trace_id for e in events)
    assert [e["data"] for e in events if e["type"] == "model_call"] == [
        {"usage": asdict(USAGE), "cost_nano": cost_nano(USAGE), "stop_reason": stop}
        for stop in ["tool_use", "end_turn"]
    ]
    assert deps.executor.pending_events == []


def test_step_limit_after_four_tool_only_responses(rig: tuple) -> None:
    deps, conv, _ = rig
    deps.model = ScriptedModel([response("tool_use", tool(f"u{i}")) for i in range(4)])
    result = run_customer_turn(conv.id, "My order?", deps)
    assert result.stopped == "step_limit"
    assert len(deps.model.requests) == 4
    assert len(saved(deps, conv).messages) == 9


@pytest.mark.parametrize("message", ["My order?", "طلب استرجاع"])
def test_model_error_keeps_reservation_and_clears_busy(rig: tuple, message: str) -> None:
    deps, conv, _ = rig
    deps.model = ScriptedModel([ModelUnavailable("private exception text")])
    result = run_customer_turn(conv.id, message, deps)
    assert result.stopped == "model_error"
    assert ("\u0600" <= result.reply[0] <= "\u06ff") == (message != "My order?")
    request = deps.model.requests[0]
    worst = worst_case_nano(len(json.dumps(request, ensure_ascii=False).encode()), 400)
    assert deps.budget.remaining() == CAP - worst
    assert saved(deps, conv).busy_until is None
    events = deps.audit.list()
    assert any(e["type"] == "incident" for e in events)
    assert not any(e["type"] == "tool_call" for e in events)
    assert "private exception text" not in json.dumps(events)


def test_budget_stop_runs_no_further_tools(rig: tuple) -> None:
    deps, conv, _ = rig
    deps.budget = Budget(deps.store, 0, NOW.date)
    deps.model = ScriptedModel([response("tool_use", tool())])
    result = run_customer_turn(conv.id, "My order?", deps)
    assert result.stopped == "budget" and deps.model.requests == []
    assert any(e["type"] == "budget_refused" for e in deps.audit.list())


def test_budget_can_run_out_between_steps(rig: tuple) -> None:
    deps, conv, _ = rig
    deps.model = ScriptedModel([response("tool_use", tool())])
    original = deps.budget.reserve
    deps.budget.reserve = Mock(side_effect=[True, False])
    result = run_customer_turn(conv.id, "My order?", deps)
    assert result.stopped == "budget" and len(deps.model.requests) == 1
    assert len([e for e in deps.audit.list() if e["type"] == "tool_call"]) == 1
    deps.budget.reserve = original


def test_turn_limit_on_ninth_message(rig: tuple) -> None:
    deps, conv, _ = rig
    deps.model = ScriptedModel([response("end_turn", TextBlock("Hello"))] * 8)
    for _ in range(8):
        assert run_customer_turn(conv.id, "Hello", deps).stopped == "end_turn"
    assert run_customer_turn(conv.id, "Hello", deps).stopped == "turn_limit"
    assert len(deps.model.requests) == 8 and saved(deps, conv).turns == 8


@pytest.mark.parametrize("with_tool", [False, True])
def test_max_tokens_executes_no_tool(rig: tuple, with_tool: bool) -> None:
    deps, conv, _ = rig
    blocks = [TextBlock("Partial answer")]
    if with_tool:
        blocks.append(tool())
    deps.model = ScriptedModel([response("max_tokens", *blocks)])
    result = run_customer_turn(conv.id, "My order?", deps)
    assert result.reply == "Partial answer"
    assert len(saved(deps, conv).messages) == (1 if with_tool else 2)
    assert not any(e["type"] == "tool_call" for e in deps.audit.list())


def test_busy_when_a_turn_is_in_flight(rig: tuple) -> None:
    deps, conv, _ = rig
    nested = []

    def respond(request: dict) -> ModelResponse:
        claimed = saved(deps, conv)
        assert claimed.busy_until == NOW + timedelta(seconds=50)
        nested.append(run_customer_turn(conv.id, "Another message", deps))
        assert saved(deps, conv).busy_until == claimed.busy_until
        return response("end_turn", TextBlock("Hello"))

    deps.model = ScriptedModel([respond])
    run_customer_turn(conv.id, "Hello", deps)
    assert nested[0].stopped == "busy" and saved(deps, conv).turns == 1


@pytest.mark.parametrize("visible", [True, False])
def test_dangling_tool_use_is_repaired(rig: tuple, visible: bool) -> None:
    deps, conv, dops = rig
    repo = IntentRepo(deps.store, "s1")
    key = intent_key(conv.id, 1001, "10.00")
    if visible:
        assert dops.create_refund(1, 1001, "10.00", "Damaged", key, "auto", "old").refund.id == 2
    repo.create(
        RefundIntent(
            key,
            conv.id,
            1,
            1001,
            "10.00",
            "Damaged",
            "auto",
            ("WITHIN_POLICY",),
            "SUBMITTED",
            2,
            "recorded",
        )
    )
    key = f"sbx:s1:conv:{conv.id}"
    deps.store.update_doc(
        key,
        lambda doc: {
            **doc,
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": use,
                            "name": "request_refund",
                            "input": {"order_id": 1001, "amount": "10.00", "reason": "Damaged"},
                        }
                        for use in ["recorded", "missing"]
                    ],
                }
            ],
        },
        {},
    )
    deps.model = ScriptedModel([response("end_turn", TextBlock("Recovered"))])
    run_customer_turn(conv.id, "Continue", deps)
    repaired = deps.model.requests[0]["messages"][1]["content"]
    assert [b["tool_use_id"] for b in repaired] == ["recorded", "missing"]
    body = json.loads(repaired[0]["content"])
    assert body["status"] == ("SUCCEEDED" if visible else "UNKNOWN")
    assert ("refunded" if visible else "not confirmed") in body["result"]
    assert repaired[1]["is_error"] is True
    assert json.loads(repaired[1]["content"]) == {"result": "interrupted, please retry"}


def test_system_is_frozen_after_refund(rig: tuple) -> None:
    deps, conv, _ = rig
    before = json.dumps(conv.system, ensure_ascii=False)
    deps.model = ScriptedModel(
        [
            response(
                "tool_use", tool("r", "request_refund", order_id=1001, amount="10.00", reason="Bad")
            ),
            response("end_turn", TextBlock("Refunded")),
            response("end_turn", TextBlock("Hello")),
        ]
    )
    run_customer_turn(conv.id, "Refund", deps)
    run_customer_turn(conv.id, "Hello", deps)
    assert all(json.dumps(r["system"], ensure_ascii=False) == before for r in deps.model.requests)
    assert json.dumps(saved(deps, conv).system, ensure_ascii=False) == before


@pytest.mark.parametrize("stop", ["tool_use", "end_turn"])
def test_deadline_honoured_with_fake_clock(rig: tuple, stop: str) -> None:
    deps, conv, _ = rig
    ticks = [0.0]
    deps.monotonic = lambda: ticks[0]

    def respond(request: dict) -> ModelResponse:
        ticks[0] = 45.0
        return response(stop, tool() if stop == "tool_use" else TextBlock("Late answer"))

    deps.model = ScriptedModel([respond])
    result = run_customer_turn(conv.id, "My order?", deps)
    assert result.stopped == "model_error" and len(deps.model.requests) == 1
    assert not any(e["type"] == "tool_call" for e in deps.audit.list())
    assert deps.budget.remaining() == CAP - cost_nano(USAGE)


@pytest.mark.parametrize("message", ["", " \n\t", "x" * 1001])
def test_invalid_message_makes_no_model_call_and_consumes_turn(rig: tuple, message: str) -> None:
    deps, conv, _ = rig
    result = run_customer_turn(conv.id, message, deps)
    assert result.reply and deps.model.requests == []
    assert saved(deps, conv).turns == 1
    event = next(e for e in deps.audit.list() if e["type"] == "customer_message")
    assert event["data"] == {"length": len(message), "language": "en"}
    assert saved(deps, conv).messages == []


def test_refused_long_message_never_reaches_the_model_later(rig: tuple) -> None:
    deps, conv, _ = rig
    run_customer_turn(conv.id, "x" * 1001, deps)
    deps.model = ScriptedModel([response("end_turn", TextBlock("Hello."))])
    run_customer_turn(conv.id, "Hi", deps)
    sent = json.dumps(deps.model.requests[0]["messages"])
    assert "x" * 1001 not in sent and len(deps.model.requests[0]["messages"]) == 1


@pytest.mark.parametrize("failure", [StoreUnavailable, StoreConflict])
@pytest.mark.parametrize("save_number", [1, 2, 3])
def test_store_failure_on_save_stops_safely(rig: tuple, failure: type, save_number: int) -> None:
    deps, conv, _ = rig
    deps.model = ScriptedModel([response("tool_use", tool())])
    original = deps.store.update_doc
    calls = 0

    def update(*args, **kwargs) -> dict:
        nonlocal calls
        calls += 1
        if calls == save_number:
            raise failure("private backend exception")
        return original(*args, **kwargs)

    deps.store.update_doc = update
    result = run_customer_turn(conv.id, "My order?", deps)
    assert result.stopped == "model_error"
    assert not any(e["type"] == "tool_call" for e in deps.audit.list())
    assert saved(deps, conv).busy_until is None


def test_failed_audit_flush_stops_before_more_tools(rig: tuple) -> None:
    deps, conv, _ = rig
    deps.model = ScriptedModel([response("tool_use", tool("u1")), response("tool_use", tool("u2"))])
    deps.audit.record_many = Mock(side_effect=StoreUnavailable("private backend exception"))
    result = run_customer_turn(conv.id, "My order?", deps)
    assert result.stopped == "model_error"
    assert len(deps.model.requests) <= 1
    assert saved(deps, conv).busy_until is None


def test_audit_never_contains_customer_text(rig: tuple) -> None:
    deps, conv, _ = rig
    deps.model = ScriptedModel([response("end_turn", TextBlock("Hello"))])
    message = "Unique private customer message مرحبا"
    run_customer_turn(conv.id, message, deps)
    events = deps.audit.list()
    assert message not in json.dumps(events, ensure_ascii=False)
    assert next(e for e in events if e["type"] == "customer_message")["data"] == {
        "length": len(message),
        "language": "ar",
    }


def test_refusal_is_safe_and_audited(rig: tuple) -> None:
    deps, conv, _ = rig
    deps.model = ScriptedModel([response("refusal", TextBlock("Unsafe response"), tool())])
    result = run_customer_turn(conv.id, "My order?", deps)
    assert result.stopped == "refusal" and result.reply != "Unsafe response"
    assert any(e["data"].get("kind") == "refusal" for e in deps.audit.list())
    assert not any(e["type"] == "tool_call" for e in deps.audit.list())


def test_tool_limit_returns_ordered_errors_without_execution(rig: tuple) -> None:
    deps, conv, _ = rig
    deps.model = ScriptedModel(
        [
            response("tool_use", *(tool(f"u{i}") for i in range(5))),
            response("end_turn", TextBlock("Done")),
        ]
    )
    run_customer_turn(conv.id, "My order?", deps)
    results = deps.model.requests[1]["messages"][-1]["content"]
    assert [b["tool_use_id"] for b in results] == [f"u{i}" for i in range(5)]
    assert all(
        b["is_error"] and json.loads(b["content"]) == {"result": "too many tool calls in one step"}
        for b in results[3:]
    )
    assert len([e for e in deps.audit.list() if e["type"] == "tool_call"]) == 3


def test_final_empty_response_uses_fallback(rig: tuple) -> None:
    deps, conv, _ = rig
    deps.model = ScriptedModel([response()])
    assert run_customer_turn(conv.id, "Hello", deps).reply == (
        "I could not finish that request; please try again."
    )


def test_release_failure_is_swallowed_and_audited(rig: tuple) -> None:
    deps, conv, _ = rig
    original = deps.store.update_doc

    def respond(request: dict) -> ModelResponse:
        return response("end_turn", TextBlock("Hello"))

    def update(key: str, mutate, default: dict) -> dict:
        current = deps.store.get_doc(key) or default
        was_busy = current.get("busy_until") is not None
        candidate = mutate(current)
        if was_busy and candidate.get("busy_until") is None:
            raise StoreUnavailable("release failed")
        return original(key, mutate, default)

    deps.store.update_doc = update
    deps.model = ScriptedModel([respond])
    result = run_customer_turn(conv.id, "Hello", deps)
    assert result.reply == "Hello"
    assert any(e["data"].get("kind") == "release" for e in deps.audit.list())


def test_deadline_before_first_call(rig: tuple) -> None:
    deps, conv, _ = rig
    deps.config = replace(deps.config, turn_deadline_s=0)
    result = run_customer_turn(conv.id, "Hello", deps)
    assert result.stopped == "model_error" and deps.model.requests == []
    assert deps.budget.remaining() == CAP


@pytest.mark.parametrize("count", [1, 2])
def test_deadline_during_tool_stops_further_work(rig: tuple, count: int) -> None:
    deps, conv, _ = rig
    ticks = [0.0]
    deps.monotonic = lambda: ticks[0]
    deps.config = replace(deps.config, max_steps=1)
    execute = deps.executor.execute

    def slow_execute(*args):
        outcome = execute(*args)
        ticks[0] = 45.0
        return outcome

    deps.executor.execute = slow_execute
    deps.model = ScriptedModel([response("tool_use", *(tool(f"u{i}") for i in range(count)))])
    result = run_customer_turn(conv.id, "My order?", deps)
    assert result.stopped == "model_error"
    assert len([e for e in deps.audit.list() if e["type"] == "tool_call"]) == 1


def test_store_failure_inside_tool_stops_remaining_tools(rig: tuple) -> None:
    deps, conv, _ = rig
    deps.model = ScriptedModel(
        [
            response(
                "tool_use",
                tool("r", "request_refund", order_id=1001, amount="10.00", reason="Bad"),
                tool("later"),
            ),
        ]
    )
    original = deps.store.update_doc

    def update(key: str, mutate, default: dict) -> dict:
        if key == "sbx:s1:intents":
            raise StoreUnavailable("private backend exception")
        return original(key, mutate, default)

    deps.store.update_doc = update
    result = run_customer_turn(conv.id, "Refund", deps)
    assert result.stopped == "model_error"
    assert len([e for e in deps.audit.list() if e["type"] == "tool_call"]) == 1
    assert deps.executor.pending_events == []


def test_step_audit_is_one_batch_with_executor_events(rig: tuple) -> None:
    deps, conv, _ = rig
    deps.audit.record_many = Mock(wraps=deps.audit.record_many)
    deps.model = ScriptedModel(
        [response("tool_use", tool()), response("end_turn", TextBlock("Done"))]
    )
    run_customer_turn(conv.id, "My order?", deps)
    batches = [call.args[0] for call in deps.audit.record_many.call_args_list]
    assert len(batches) == 2
    assert [e.type for e in batches[0]] == [
        "customer_message",
        "model_call",
        "tool_call",
        "dops_call",
    ]
    assert [e.type for e in batches[1]] == ["model_call"]


def test_partial_dangling_results_are_repaired_in_order(rig: tuple) -> None:
    deps, conv, _ = rig
    existing = {"type": "tool_result", "tool_use_id": "u2", "content": "known", "is_error": False}
    deps.store.update_doc(
        f"sbx:s1:conv:{conv.id}",
        lambda doc: {
            **doc,
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": use,
                            "name": "get_order",
                            "input": {"order_id": 1001},
                        }
                        for use in ["u1", "u2"]
                    ],
                },
                {"role": "user", "content": [existing]},
            ],
        },
        {},
    )
    deps.model = ScriptedModel([response("end_turn", TextBlock("Done"))])
    run_customer_turn(conv.id, "Continue", deps)
    results = deps.model.requests[0]["messages"][1]["content"]
    assert [r["tool_use_id"] for r in results] == ["u1", "u2"]
    assert results[1] == existing


def test_status_footer_uses_latest_status_and_customer_language() -> None:
    from app.agent import status_footer

    refunds = [
        {"refund_id": 7, "status": "PENDING", "order_id": 1004, "amount": "10.00"},
        {"refund_id": 7, "status": "SUCCEEDED", "order_id": 1004, "amount": "10.00"},
        {"refund_id": 8, "status": "PENDING_APPROVAL", "order_id": 1003, "amount": "60.00"},
    ]
    assert status_footer(refunds, "en") == (
        "Refund system status:\n- 1004 / 10.00: refunded\n- 1003 / 60.00: awaiting human review"
    )
    arabic = status_footer(refunds, "ar")
    assert arabic.startswith("حالة الاسترداد من النظام:")
    assert "- 1003 / 60.00: في انتظار مراجعة موظف" in arabic
    assert status_footer([], "en") == ""
