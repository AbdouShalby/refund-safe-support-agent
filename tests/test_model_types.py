"""Provider-neutral responses and deterministic scripted model behavior."""

from dataclasses import FrozenInstanceError

import pytest

from app.model import (
    ModelResponse,
    ModelUnavailable,
    ScriptedModel,
    TextBlock,
    ToolUseBlock,
    Usage,
    request_hash,
)


def test_scripted_steps_and_requests() -> None:
    first = ModelResponse("end_turn", [TextBlock("Hello")], Usage(1, 2, 3, 4))
    second = ModelResponse(
        "tool_use", [ToolUseBlock("tool-1", "get_order", {"order_id": 1001})], Usage(4, 3, 2, 1)
    )
    requests = [{"message": "hello"}, {"message": "order"}, {}, {}]

    def respond(request: dict) -> ModelResponse:
        assert request == requests[1]
        return second

    failure = ModelUnavailable("offline")
    model = ScriptedModel([first, respond, failure])
    assert model.create(requests[0]) == first
    assert model.create(requests[1]) == second
    with pytest.raises(ModelUnavailable) as caught:
        model.create(requests[2])
    assert caught.value is failure
    with pytest.raises(AssertionError, match="^ScriptedModel exhausted$"):
        model.create(requests[3])
    assert model.requests == requests


@pytest.mark.parametrize(
    "value,field",
    [
        (TextBlock("hi"), "text"),
        (ToolUseBlock("a", "get_order", {}), "name"),
        (Usage(1, 2, 3, 4), "input"),
        (ModelResponse("end_turn", [], Usage(0, 0, 0, 0)), "stop_reason"),
    ],
)
def test_response_dataclasses_are_frozen(value: object, field: str) -> None:
    with pytest.raises(FrozenInstanceError):
        setattr(value, field, None)


def test_request_hash() -> None:
    import hashlib

    expected = hashlib.sha256('{"a":"مرحبا","b":{"x":1}}'.encode()).hexdigest()
    assert request_hash({"b": {"x": 1}, "a": "مرحبا"}) == expected
    assert request_hash({"a": "مرحبا!", "b": {"x": 1}}) != expected
