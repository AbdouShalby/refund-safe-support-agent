"""Exercise the installed SDK boundary without credentials or network calls."""

import logging
from types import SimpleNamespace
from unittest.mock import Mock

import anthropic
import httpx2
import pytest

from app.model import (
    AnthropicModel,
    ModelResponse,
    ModelUnavailable,
    TextBlock,
    ToolUseBlock,
    Usage,
)


def sdk_response() -> SimpleNamespace:
    return SimpleNamespace(
        stop_reason="tool_use",
        content=[
            SimpleNamespace(type="text", text="Checking."),
            SimpleNamespace(type="thinking"),
            SimpleNamespace(type="future_block"),
            SimpleNamespace(type="tool_use", id="u1", name="get_order", input={"order_id": 1001}),
        ],
        usage=SimpleNamespace(
            input_tokens=12,
            cache_creation_input_tokens=None,
            cache_read_input_tokens=None,
            output_tokens=8,
        ),
    )


def test_conversion_and_request_passthrough() -> None:
    client = Mock()
    client.messages.create.return_value = sdk_response()
    request = {"model": "example", "messages": [], "system": [{"text": "Hello"}]}
    assert AnthropicModel(client).create(request) == ModelResponse(
        "tool_use",
        [TextBlock("Checking."), ToolUseBlock("u1", "get_order", {"order_id": 1001})],
        Usage(12, 0, 0, 8),
    )
    client.messages.create.assert_called_once_with(**request)
    assert client.messages.create.call_args.kwargs["messages"] is request["messages"]


def test_nonzero_cache_usage() -> None:
    client = Mock()
    response = sdk_response()
    response.usage.cache_creation_input_tokens = 20
    response.usage.cache_read_input_tokens = 30
    client.messages.create.return_value = response
    assert AnthropicModel(client).create({}).usage == Usage(12, 20, 30, 8)


def sdk_errors() -> list[anthropic.APIError]:
    request = httpx2.Request("POST", "https://example.invalid/messages")
    return [
        anthropic.APIError("private original", request, body=None),
        anthropic.APIConnectionError(message="private original", request=request),
        anthropic.APITimeoutError(request),
        *[
            cls("private original", response=httpx2.Response(status, request=request), body=None)
            for cls, status in [
                (anthropic.RateLimitError, 429),
                (anthropic.InternalServerError, 500),
                (anthropic.AuthenticationError, 401),
                (anthropic.BadRequestError, 400),
            ]
        ],
    ]


@pytest.mark.parametrize("error", sdk_errors(), ids=lambda error: type(error).__name__)
def test_sdk_errors_are_sanitized(
    error: anthropic.APIError, caplog: pytest.LogCaptureFixture
) -> None:
    client = Mock()
    client.messages.create.side_effect = error
    with caplog.at_level(logging.WARNING), pytest.raises(ModelUnavailable) as raised:
        AnthropicModel(client).create({})
    assert str(error) not in str(raised.value)
    assert raised.value.__suppress_context__
    assert [record.getMessage() for record in caplog.records] == [type(error).__name__]


@pytest.mark.parametrize("error", [ValueError("invalid request"), TypeError("invalid request")])
def test_programming_errors_propagate(error: Exception) -> None:
    client = Mock()
    client.messages.create.side_effect = error
    with pytest.raises(type(error)) as raised:
        AnthropicModel(client).create({})
    assert raised.value is error


def test_lazy_client_created_once(monkeypatch: pytest.MonkeyPatch) -> None:
    client = Mock()
    client.messages.create.return_value = sdk_response()
    constructor = Mock(return_value=client)
    monkeypatch.setattr(anthropic, "Anthropic", constructor)
    model = AnthropicModel(timeout_s=3.0)
    constructor.assert_not_called()
    model.create({})
    model.create({})
    constructor.assert_called_once_with(max_retries=0, timeout=3.0)


@pytest.mark.parametrize(
    "error", [RuntimeError("private original"), ValueError("private original")]
)
def test_client_construction_errors_are_sanitized(
    error: Exception, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    constructor = Mock(side_effect=error)
    monkeypatch.setattr(anthropic, "Anthropic", constructor)
    with caplog.at_level(logging.WARNING), pytest.raises(ModelUnavailable) as raised:
        AnthropicModel().create({})
    assert "private original" not in str(raised.value)
    assert [record.getMessage() for record in caplog.records] == [type(error).__name__]
