"""Provider-neutral model responses and a deterministic scripted client."""

import hashlib
import json
import logging
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Protocol

import anthropic


@dataclass(frozen=True)
class TextBlock:
    """A model-written text block."""

    text: str


@dataclass(frozen=True)
class ToolUseBlock:
    """A requested tool invocation, without authorization to move money."""

    id: str
    name: str
    input: dict


@dataclass(frozen=True)
class Usage:
    """Token counts separated by billing category."""

    input: int
    cache_write: int
    cache_read: int
    output: int


@dataclass(frozen=True)
class ModelResponse:
    """The model's stop condition, content and billed usage."""

    stop_reason: str
    content: list[TextBlock | ToolUseBlock]
    usage: Usage


class ModelClient(Protocol):
    """The synchronous model boundary consumed by the agent loop."""

    def create(self, request: dict) -> ModelResponse: ...


class ModelUnavailable(Exception):
    """The model call failed; its budget reservation must remain spent."""


class ScriptedModel:
    """Consume responses, request callbacks or failures in scripted order."""

    def __init__(
        self,
        steps: Iterable[ModelResponse | Callable[[dict], ModelResponse] | Exception],
    ) -> None:
        self._steps = iter(steps)
        self.requests: list[dict] = []

    def create(self, request: dict) -> ModelResponse:
        """Record each request, including failures and exhausted calls."""
        self.requests.append(request)
        try:
            step = next(self._steps)
        except StopIteration:
            raise AssertionError("ScriptedModel exhausted") from None
        if isinstance(step, Exception):
            raise step
        return step(request) if callable(step) else step


def request_hash(request: dict) -> str:
    """Hash canonical compact Unicode JSON for deterministic replay matching."""
    encoded = json.dumps(request, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


class AnthropicModel:
    """Adapt the Anthropic SDK with lazy initialization and sanitized failures."""

    def __init__(self, client: object | None = None, *, timeout_s: float = 20.0) -> None:
        self._client = client
        self._timeout_s = timeout_s

    def create(self, request: dict) -> ModelResponse:
        """Pass the request unchanged and retain only provider-neutral blocks."""
        if self._client is None:
            try:
                self._client = anthropic.Anthropic(max_retries=0, timeout=self._timeout_s)
            except Exception as error:
                logging.getLogger(__name__).warning("%s", type(error).__name__)
                raise ModelUnavailable("Model unavailable") from None
        try:
            response = self._client.messages.create(**request)
        except anthropic.APIError as error:
            logging.getLogger(__name__).warning("%s", type(error).__name__)
            raise ModelUnavailable("Model unavailable") from None

        content: list[TextBlock | ToolUseBlock] = []
        for block in response.content:
            if block.type == "text":
                content.append(TextBlock(block.text))
            elif block.type == "tool_use":
                content.append(ToolUseBlock(block.id, block.name, block.input))
        usage = response.usage
        return ModelResponse(
            response.stop_reason,
            content,
            Usage(
                usage.input_tokens or 0,
                usage.cache_creation_input_tokens or 0,
                usage.cache_read_input_tokens or 0,
                usage.output_tokens or 0,
            ),
        )
