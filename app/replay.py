"""Record provider-neutral responses and replay strictly matched requests."""

import json
from dataclasses import asdict
from pathlib import Path

from app.model import ModelClient, ModelResponse, TextBlock, ToolUseBlock, Usage, request_hash


class FixtureMismatch(AssertionError):
    """The next recorded request differs from the current request."""

    def __init__(self, expected: str, actual: str) -> None:
        self.expected = expected
        self.actual = actual
        super().__init__(f"Fixture request hash mismatch: expected {expected}, actual {actual}")


class FixtureExhausted(AssertionError):
    """No recorded response remains for the requested call."""


def response_to_json(response: ModelResponse) -> dict:
    """Encode each content block with a discriminator for lossless decoding."""
    return {
        "stop_reason": response.stop_reason,
        "content": [
            {"type": "text" if isinstance(block, TextBlock) else "tool_use", **asdict(block)}
            for block in response.content
        ],
        "usage": asdict(response.usage),
    }


def response_from_json(data: dict) -> ModelResponse:
    """Decode the exact provider-neutral response representation."""
    content: list[TextBlock | ToolUseBlock] = []
    for block in data["content"]:
        if block["type"] == "text":
            content.append(TextBlock(block["text"]))
        elif block["type"] == "tool_use":
            content.append(ToolUseBlock(block["id"], block["name"], block["input"]))
        else:
            raise ValueError("Unknown recorded content block type")
    return ModelResponse(data["stop_reason"], content, Usage(**data["usage"]))


class RecordingModel:
    """Append one UTF-8 JSON line for each successfully returned response."""

    def __init__(self, inner: ModelClient, path: Path) -> None:
        self._inner = inner
        self._path = path

    def create(self, request: dict) -> ModelResponse:
        """Record the request hash before caller-owned conversation state changes."""
        fingerprint = request_hash(request)
        response = self._inner.create(request)
        line = json.dumps(
            {"request_hash": fingerprint, "response": response_to_json(response)},
            ensure_ascii=False,
        )
        if "sk-ant" in line:
            raise ValueError("Recording refused: potential credential detected")
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8", newline="\n") as recording:
            recording.write(line + "\n")
        return response


class ReplayModel:
    """Return fixture responses in order, without consuming mismatched entries."""

    def __init__(self, path: Path) -> None:
        with path.open(encoding="utf-8") as recording:
            self._entries = [json.loads(line) for line in recording]
        self._index = 0

    def create(self, request: dict) -> ModelResponse:
        """Fail loudly if the request changed or the fixture ran out."""
        if self._index >= len(self._entries):
            raise FixtureExhausted("Replay fixture exhausted")
        entry = self._entries[self._index]
        actual = request_hash(request)
        if entry["request_hash"] != actual:
            raise FixtureMismatch(entry["request_hash"], actual)
        response = response_from_json(entry["response"])
        self._index += 1
        return response
