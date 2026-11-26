"""Verify exact fixture serialization, request matching and complete loop replay."""

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from app.agent import AgentConfig, AgentDeps, run_customer_turn, start_conversation
from app.audit import AuditLog
from app.budget import Budget
from app.escalations import EscalationRepo
from app.model import ModelResponse, ScriptedModel, TextBlock, ToolUseBlock, Usage, request_hash
from app.policy import PolicyConfig
from app.replay import (
    FixtureExhausted,
    FixtureMismatch,
    RecordingModel,
    ReplayModel,
    response_from_json,
    response_to_json,
)
from app.sandbox_dops import SandboxDops
from app.store import MemoryStore
from app.tools import ToolExecutor, mock_lock_scope

RESPONSE = ModelResponse(
    "tool_use",
    [
        TextBlock("Checking \u0627\u0644\u0637\u0644\u0628"),
        ToolUseBlock("u1", "get_order", {"order_id": 1001}),
    ],
    Usage(1, 2, 3, 4),
)
REQUEST = {"system": "Be helpful.", "messages": []}


@pytest.mark.parametrize("response", [RESPONSE, ModelResponse("end_turn", [], Usage(0, 0, 0, 0))])
def test_json_round_trip(response: ModelResponse) -> None:
    assert response_from_json(json.loads(json.dumps(response_to_json(response)))) == response


@given(
    stop=st.text(),
    blocks=st.lists(
        st.one_of(
            st.builds(TextBlock, st.text()),
            st.builds(
                ToolUseBlock,
                st.text(),
                st.text(),
                st.dictionaries(
                    st.text(),
                    st.recursive(
                        st.none() | st.booleans() | st.integers() | st.text(),
                        lambda values: st.lists(values) | st.dictionaries(st.text(), values),
                        max_leaves=10,
                    ),
                ),
            ),
        ),
        max_size=5,
    ),
    usage=st.builds(Usage, *(st.integers(min_value=0, max_value=1_000_000) for _ in range(4))),
)
def test_json_round_trip_property(
    stop: str, blocks: list[TextBlock | ToolUseBlock], usage: Usage
) -> None:
    response = ModelResponse(stop, blocks, usage)
    assert response_from_json(json.loads(json.dumps(response_to_json(response)))) == response


def test_record_and_replay_in_order(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "recording.jsonl"
    other = ModelResponse("end_turn", [TextBlock("Done.")], Usage(5, 6, 7, 8))
    recorder = RecordingModel(ScriptedModel([RESPONSE, other]), path)
    assert recorder.create(REQUEST) == RESPONSE
    assert recorder.create({"system": "Next"}) == other
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert "\u0627\u0644\u0637\u0644\u0628" in lines[0]
    assert json.loads(lines[0]) == {
        "request_hash": request_hash(REQUEST),
        "response": response_to_json(RESPONSE),
    }
    replay = ReplayModel(path)
    assert replay.create(REQUEST) == RESPONSE
    assert replay.create({"system": "Next"}) == other
    with pytest.raises(FixtureExhausted):
        replay.create(REQUEST)


def test_changed_prompt_fails_loudly(tmp_path: Path) -> None:
    path = tmp_path / "recording.jsonl"
    RecordingModel(ScriptedModel([RESPONSE]), path).create(REQUEST)
    replay = ReplayModel(path)
    changed = {**REQUEST, "system": "Be helpful!"}
    with pytest.raises(FixtureMismatch) as raised:
        replay.create(changed)
    assert isinstance(raised.value, AssertionError)
    assert raised.value.expected == request_hash(REQUEST)
    assert raised.value.actual == request_hash(changed)
    assert replay.create(REQUEST) == RESPONSE


def test_empty_fixture_is_exhausted(tmp_path: Path) -> None:
    path = tmp_path / "empty.jsonl"
    path.write_text("", encoding="utf-8")
    with pytest.raises(FixtureExhausted) as raised:
        ReplayModel(path).create(REQUEST)
    assert isinstance(raised.value, AssertionError)


def test_secret_guard_preserves_existing_recording(tmp_path: Path) -> None:
    path = tmp_path / "recording.jsonl"
    secret = ModelResponse("end_turn", [TextBlock("sk-ant-test-not-secret")], Usage(0, 0, 0, 0))
    recorder = RecordingModel(ScriptedModel([RESPONSE, secret]), path)
    recorder.create(REQUEST)
    original = path.read_bytes()
    with pytest.raises(ValueError, match="credential"):
        recorder.create(REQUEST)
    assert path.read_bytes() == original


def test_complete_agent_turn_replays_identically(tmp_path: Path) -> None:
    path = tmp_path / "turn.jsonl"
    steps = [
        ModelResponse(
            "tool_use",
            [
                ToolUseBlock(
                    "u1",
                    "request_refund",
                    {"order_id": 1001, "amount": "10.00", "reason": "Damaged"},
                )
            ],
            Usage(10, 20, 30, 40),
        ),
        ModelResponse("end_turn", [TextBlock("Refunded 10.00.")], Usage(10, 0, 30, 40)),
    ]

    def run(model: RecordingModel | ReplayModel) -> object:
        now_value = datetime(2026, 11, 18, 12, tzinfo=UTC)

        def now() -> datetime:
            return now_value

        store = MemoryStore()
        dops = SandboxDops(store, "s1", now)
        audit = AuditLog(store, "s1")
        executor = ToolExecutor(
            dops, store, audit, EscalationRepo(store, "s1"), PolicyConfig(), now, mock_lock_scope
        )
        deps = AgentDeps(
            "s1",
            model,
            executor,
            Budget(store, 10_000_000, now_value.date),
            audit,
            store,
            AgentConfig(),
            now,
            new_trace_id=lambda: "t-0000000000000001",
        )
        conv = start_conversation(store, "s1", 1, dops, now)
        result = run_customer_turn(conv.id, "Refund 10.00 for damage.", deps)
        assert result.stopped == "end_turn"
        assert result.refunds == [
            {"refund_id": 2, "status": "SUCCEEDED", "order_id": 1001, "amount": "10.00"}
        ]
        return result

    recorded = run(RecordingModel(ScriptedModel(steps), path))
    assert run(ReplayModel(path)) == recorded
