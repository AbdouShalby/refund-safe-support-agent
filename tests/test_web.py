"""Verify isolated HTTP journeys without a provider call or external store."""

import importlib
import json
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from threading import Event
from types import ModuleType
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from app.budget import Budget
from app.config import Settings
from app.dops.errors import DopsUnavailable
from app.escalations import EscalationRepo
from app.intents import IntentRepo, intent_key
from app.model import ModelResponse, ScriptedModel, TextBlock, ToolUseBlock, Usage
from app.policy import PolicyConfig
from app.session import sign, verify
from app.store import MemoryStore, StoreUnavailable

NOW = datetime(2026, 11, 18, 12, tzinfo=UTC)
SECRET = "test-session-secret-with-at-least-32-characters"  # noqa: S105
USAGE = Usage(10, 0, 0, 10)


def answer(text: str = "Ready.") -> ModelResponse:
    return ModelResponse("end_turn", [TextBlock(text)], USAGE)


def held_steps() -> list[ModelResponse]:
    return [
        ModelResponse(
            "tool_use",
            [
                ToolUseBlock(
                    "held-refund",
                    "request_refund",
                    {"order_id": 1004, "amount": "60.00", "reason": "Please review."},
                )
            ],
            USAGE,
        ),
        answer("Your request awaits review."),
    ]


@pytest.fixture
def web(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    monkeypatch.setenv("SESSION_SECRET", SECRET)
    monkeypatch.setenv("IP_HASH_SECRET", "test-ip-secret-with-at-least-32-characters")
    for name in ("VERCEL", "DEMO_FAKE_MODEL", "DOPS_MODE", "DAILY_CAP_USD"):
        monkeypatch.delenv(name, raising=False)
    return importlib.import_module("app.web")


@pytest.fixture
def rig(web: ModuleType) -> tuple:
    store = MemoryStore(now=lambda: 1000)
    deps = web.WebDeps(
        Settings.from_env(), store, ScriptedModel([]), lambda: NOW, lambda: NOW.date()
    )
    app = web.create_app(deps.settings)
    app.state.now = deps.now
    app.dependency_overrides[web.get_deps] = lambda: deps
    with TestClient(app, base_url="https://testserver") as client:
        yield client, deps, web


def session(client: TestClient, customer_id: int = 1) -> dict:
    response = client.post("/api/session", json={"customer_id": customer_id})
    assert response.status_code == 200
    return response.json()


def sandbox(client: TestClient) -> str:
    return verify(client.cookies["sid"], SECRET, NOW)["sandbox_id"]


def live(client: TestClient) -> str:
    response = client.post("/api/live/start")
    assert response.status_code == 200, response.text
    return response.json()["conversation_id"]


def role(client: TestClient, value: str) -> None:
    assert client.post("/api/session/role", json={"role": value}).status_code == 200


def interrupted_chat(rig, monkeypatch, *, commit=False):
    client, deps, web = rig
    session(client)
    conv = live(client)
    body = {"conversation_id": conv, "message": "Please review"}
    deps.model = ScriptedModel([held_steps()[0], answer("I will check."), answer()])
    original = web.SandboxDops.create_refund

    def lose_response(self, *args):
        if commit:
            original(self, *args)
        raise DopsUnavailable("lost response")

    monkeypatch.setattr(web.SandboxDops, "create_refund", lose_response)
    result = client.post("/api/chat", json=body)
    assert result.status_code == 200 and result.json()["refunds"] == []
    return body


@pytest.mark.parametrize("failure", ["outage", "busy", "store"])
def test_http_repair_failure_semantics(rig, monkeypatch, failure):
    client, deps, web = rig
    body = interrupted_chat(rig, monkeypatch)
    store_error = StoreUnavailable("store unavailable")
    # Inject the store failure in the observed store, as the real adapter does.
    if failure == "store":
        original = deps.store.get_doc

        def fail_intents(key):
            if key.endswith(":intents"):
                raise store_error
            return original(key)

        monkeypatch.setattr(deps.store, "get_doc", fail_intents)
    read = Mock(side_effect=DopsUnavailable("private detail"))
    write = Mock(side_effect=AssertionError("Repair attempted a submission"))
    monkeypatch.setattr(web.SandboxDops, "list_order_refunds", read)
    monkeypatch.setattr(web.SandboxDops, "create_refund", write)
    lock = deps.store.lock(f"sbx:{sandbox(client)}:cust:1", 30)
    with lock if failure == "busy" else nullcontext():
        result = client.post("/api/chat", json=body)
    write.assert_not_called()
    if failure == "store":
        assert result.status_code == 503
        return
    assert read.call_count == int(failure == "outage")
    assert result.status_code == 200 and result.json()["stopped"] == "end_turn"
    assert result.json()["refunds"] == []
    history = deps.model.requests[-1]["messages"]
    block = next(b for m in history for b in m["content"] if b["type"] == "tool_result")
    assert "submitted, outcome not confirmed" in block["content"]
    assert "private detail" not in result.text


def test_http_repair_races_ops_list_without_duplicate_adoption(rig, monkeypatch):
    client, deps, web = rig
    body = interrupted_chat(rig, monkeypatch, commit=True)
    original = web.SandboxDops.list_order_refunds
    entered, release = Event(), Event()

    def pause(*args):
        entered.set()
        assert release.wait(5)
        return original(*args)

    monkeypatch.setattr(web.SandboxDops, "list_order_refunds", pause)
    with ThreadPoolExecutor(max_workers=1) as pool:
        chat = pool.submit(client.post, "/api/chat", json=body)
        try:
            assert entered.wait(5)
            role(client, "ops")
            queue = client.get("/api/ops/queue")
            assert queue.status_code == 200
            assert queue.json()["unconfirmed"] == [{"order_id": 1004, "amount": "60.00"}]
        finally:
            release.set()
        assert chat.result().json()["refunds"][0]["status"] == "PENDING_APPROVAL"
    queue = client.get("/api/ops/queue").json()
    assert queue["unconfirmed"] == [] and len(queue["refunds"]) == 1
    events = client.get("/api/audit").json()
    assert sum(e["type"] == "intent_adopted" for e in events) == 1


def test_ops_card_uses_late_held_attempt_reasons(rig, monkeypatch):
    client, deps, web = rig
    create = web.SandboxDops.create_refund
    body = interrupted_chat(rig, monkeypatch)
    repo = IntentRepo(deps.store, sandbox(client))
    first = repo.by_tool_use("held-refund")

    def late(self, *args):
        if args[5] == "auto":
            create(self, *args[:5], first.mode, first.attempts[0][0])
        return create(self, *args)

    monkeypatch.setattr(web.SandboxDops, "create_refund", late)
    monkeypatch.setattr(web, "PolicyConfig", lambda: PolicyConfig(auto_limit=Decimal("100.00")))
    deps.model = ScriptedModel(held_steps())
    result = client.post("/api/chat", json=body)
    assert result.status_code == 200 and result.json()["refunds"][0]["status"] == "PENDING_APPROVAL"
    role(client, "ops")
    (card,) = client.get("/api/ops/queue").json()["refunds"]
    assert card["facts"]["policy_reasons"] == list(first.decision_reasons) == ["OVER_AUTO_LIMIT"]


def test_ops_resolution_is_bounded_and_fails_open_for_lookup_outage(rig, monkeypatch):
    client, deps, web = rig
    interrupted_chat(rig, monkeypatch, commit=True)
    repo = IntentRepo(deps.store, sandbox(client))
    first = repo.by_tool_use("held-refund")
    for index in range(20):
        amount = f"{index + 1}.00"
        repo.create(
            replace(
                first,
                key=intent_key(first.conversation_id, 1004, amount),
                amount=amount,
                tool_use_id=f"other-{index}",
                attempts=((f"other-{index}", first.mode, first.decision_reasons),),
                unconfirmed_attempt=0,
                unconfirmed_turn=None,
            )
        )
    role(client, "ops")
    with monkeypatch.context() as patch:
        unavailable = Mock(side_effect=DopsUnavailable("lookup failed"))
        patch.setattr(web.SandboxDops, "list_order_refunds", unavailable)
        queue = client.get("/api/ops/queue")
        assert queue.status_code == 200 and len(queue.json()["unconfirmed"]) == 20
        assert queue.json()["unconfirmed_remaining"] == 1
        assert queue.json()["unconfirmed"][0]["amount"] == "20.00"
        unavailable.assert_called_once()
    events = client.get("/api/audit").json()
    assert sum(e["type"] == "intent_unconfirmed" for e in events) == 1
    queue = client.get("/api/ops/queue").json()
    assert len(queue["unconfirmed"]) == 20
    assert queue["unconfirmed_remaining"] == 1 and repo.get(first.key).refund_id is None
    assert sum(e["type"] == "intent_unconfirmed" for e in client.get("/api/audit").json()) == 1


def test_chat_refreshes_uncertain_tool_history_after_ops_adoption(rig, monkeypatch):
    client, deps, _ = rig
    body = interrupted_chat(rig, monkeypatch, commit=True)
    role(client, "ops")
    queue = client.get("/api/ops/queue").json()
    assert queue["unconfirmed"] == []
    refund_id = queue["refunds"][0]["refund_id"]
    assert (
        client.post(f"/api/ops/refunds/{refund_id}/reject", json={"reason": "Reviewed"}).status_code
        == 200
    )
    role(client, "customer")
    result = client.post("/api/chat", json=body).json()
    assert result["stopped"] == "end_turn" and "rejected by a reviewer" in result["reply"]
    (block,) = [
        b
        for m in deps.model.requests[-1]["messages"]
        for b in m["content"]
        if b["type"] == "tool_result"
    ]
    assert json.loads(block["content"])["status"] == "REJECTED"


def test_session_cookie_seed_and_lazy_store(rig: tuple) -> None:
    client, deps, _ = rig
    response = client.post("/api/session", json={"customer_id": 1})
    cookie = response.headers["set-cookie"]
    for attribute in ("HttpOnly", "Secure", "SameSite=lax", "Path=/", "Max-Age=86400"):
        assert attribute in cookie
    assert response.json()["customer"] == {"id": 1, "name": "Customer A"}
    assert len(response.json()["orders"]) == 8
    assert all(not key.startswith("sbx:") for key in deps.store._values)
    assert "testclient" not in json.dumps(deps.store._values)
    assert len(sandbox(client)) == 26


@pytest.mark.parametrize("kind", ["missing", "tampered", "expired"])
def test_bad_cookie(rig: tuple, kind: str) -> None:
    client, _, _ = rig
    if kind != "missing":
        token = sign(
            {
                "sandbox_id": "s-" + "a" * 24,
                "customer_id": 1,
                "role": "customer",
                "exp": int((NOW - timedelta(seconds=1)).timestamp()),
            },
            SECRET,
        )
        client.cookies.set("sid", token if kind == "expired" else token + "bad")
    response = client.post("/api/live/start")
    assert response.status_code == 401 and response.json() == {"error": "session"}


def test_role_enforcement(rig: tuple) -> None:
    client, _, _ = rig
    session(client)
    for path in ("/api/ops/queue",):
        assert client.get(path).status_code == 403
    for action in ("approve", "reject"):
        response = client.post(f"/api/ops/refunds/1/{action}", json={"reason": "No"})
        assert response.status_code == 403 and response.json() == {"error": "role"}
    role(client, "ops")
    response = client.post("/api/chat", json={"conversation_id": "x", "message": "Hi"})
    assert response.status_code == 403 and response.json() == {"error": "role"}
    assert client.post("/api/live/start").status_code == 403
    assert client.get("/api/conversation/x/refunds").status_code == 403


def test_cross_sandbox_and_customer_conversation(rig: tuple) -> None:
    client, deps, _ = rig
    session(client)
    conversation_id = live(client)
    original = verify(client.cookies["sid"], SECRET, NOW)
    session(client, 2)
    assert (
        client.post(
            "/api/chat", json={"conversation_id": conversation_id, "message": "Hi"}
        ).status_code
        == 404
    )
    assert client.get(f"/api/conversation/{conversation_id}/refunds").status_code == 404
    assert client.get(f"/api/audit?conversation_id={conversation_id}").json() == []
    client.cookies.clear()
    client.cookies.set("sid", sign({**original, "customer_id": 2}, SECRET))
    assert (
        client.post(
            "/api/chat", json={"conversation_id": conversation_id, "message": "Hi"}
        ).status_code
        == 404
    )
    assert deps.model.requests == []


def test_fourth_live_start_and_new_session_cannot_bypass_ip(rig: tuple) -> None:
    client, _, _ = rig
    session(client)
    for _ in range(3):
        live(client)
    response = client.post("/api/live/start")
    assert response.status_code == 429 and response.json() == {"reason": "visitor_limit"}
    session(client)
    assert client.post("/api/live/start").json() == {"reason": "visitor_limit"}


def test_budget_before_live_limit(rig: tuple) -> None:
    client, deps, _ = rig
    session(client)
    assert Budget(deps.store, deps.settings.daily_cap_nano, deps.today).reserve(116_000_000)
    response = client.post("/api/live/start")
    assert response.status_code == 429 and response.json() == {"reason": "daily_budget"}
    assert deps.model.requests == []
    assert not any("live" in key for key in deps.store._values)


def test_twenty_first_session(rig: tuple) -> None:
    client, _, _ = rig
    for _ in range(20):
        session(client)
    assert client.post("/api/session", json={"customer_id": 1}).status_code == 429


def test_full_approval_journey_and_audit_privacy(rig: tuple) -> None:
    client, deps, _ = rig
    deps.model = ScriptedModel(held_steps())
    session(client)
    conversation_id = live(client)
    message = "Customer private text not for audit"
    result = client.post(
        "/api/chat", json={"conversation_id": conversation_id, "message": message}
    ).json()
    assert result["stopped"] == "end_turn"
    refund_id = result["refunds"][0]["refund_id"]
    assert result["refunds"][0]["status"] == "PENDING_APPROVAL"
    assert message not in json.dumps(result["events"])
    assert all(row["trace_id"] == result["trace_id"] for row in result["events"])
    EscalationRepo(deps.store, sandbox(client)).open(conversation_id, 1, "Review this request")
    role(client, "ops")
    queue = client.get("/api/ops/queue").json()
    pending = queue["refunds"][0]
    assert pending["refund_id"] == refund_id
    assert "OVER_AUTO_LIMIT" in pending["facts"]["policy_reasons"]
    assert pending["facts"] == {
        "amount": "60.00",
        "order_total": "100.00",
        "reserved": "60.00",
        "refunded": "0.00",
        "customer_id": 1,
        "order_id": 1004,
        "policy_reasons": ["OVER_AUTO_LIMIT"],
    }
    assert pending["model_text"] == {"label": "model-written", "text": "Please review."}
    assert queue["escalations"][0]["model_text"]["label"] == "model-written"
    response = client.post(f"/api/ops/refunds/{refund_id}/approve")
    assert response.json() == {"status": "SUCCEEDED"}
    assert client.post(f"/api/ops/refunds/{refund_id}/approve").status_code == 409
    role(client, "customer")
    assert client.get(f"/api/conversation/{conversation_id}/refunds").json() == [
        {"refund_id": refund_id, "order_id": 1004, "amount": "60.00", "status": "SUCCEEDED"}
    ]
    events = client.get(f"/api/audit?conversation_id={conversation_id}").json()
    assert any(row["type"] == "approval" and row["actor"] == "human" for row in events)
    assert all(deps.store.ttl(key) == 86_400 for key in deps.store._values)

    def no_floats(value: object) -> None:
        assert not isinstance(value, float)
        if isinstance(value, dict):
            for item in value.values():
                no_floats(item)
        elif isinstance(value, list):
            for item in value:
                no_floats(item)

    for value in (result, queue, events):
        no_floats(value)


def test_ops_queue_isolation(rig: tuple) -> None:
    client, deps, _ = rig
    deps.model = ScriptedModel(held_steps())
    session(client)
    conversation_id = live(client)
    client.post("/api/chat", json={"conversation_id": conversation_id, "message": "Refund"})
    session(client)
    role(client, "ops")
    assert client.get("/api/ops/queue").json() == {
        "refunds": [],
        "escalations": [],
        "unconfirmed": [],
        "unconfirmed_remaining": 0,
    }
    assert client.post("/api/ops/refunds/2/approve").status_code == 404


def test_rejection_and_missing_refund(rig: tuple) -> None:
    client, deps, _ = rig
    deps.model = ScriptedModel(held_steps())
    session(client)
    conversation_id = live(client)
    result = client.post(
        "/api/chat", json={"conversation_id": conversation_id, "message": "Refund"}
    ).json()
    refund_id = result["refunds"][0]["refund_id"]
    role(client, "ops")
    response = client.post(f"/api/ops/refunds/{refund_id}/reject", json={"reason": "Denied"})
    assert response.json() == {"status": "REJECTED"}
    assert (
        client.post(f"/api/ops/refunds/{refund_id}/reject", json={"reason": "Again"}).status_code
        == 409
    )
    assert client.post("/api/ops/refunds/999/approve").status_code == 404
    assert any(row["type"] == "rejection" for row in client.get("/api/audit").json())


def test_parallel_chat_posts_conflict(rig: tuple) -> None:
    client, deps, _ = rig
    entered, release = Event(), Event()

    def block(request: dict) -> ModelResponse:
        entered.set()
        assert release.wait(5)
        return answer()

    deps.model = ScriptedModel([block])
    session(client)
    conversation_id = live(client)
    body = {"conversation_id": conversation_id, "message": "Hello"}
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(client.post, "/api/chat", json=body)
        try:
            assert entered.wait(5)
            second = client.post("/api/chat", json=body)
            assert second.status_code == 409 and second.json()["stopped"] == "busy"
        finally:
            release.set()
        assert first.result().status_code == 200
    assert len(deps.model.requests) == 1


def test_conversation_survives_new_app_instance(rig: tuple) -> None:
    client, deps, web = rig
    session(client)
    conversation_id = live(client)
    restarted = web.create_app(deps.settings)
    restarted.state.now = deps.now
    restarted.dependency_overrides[web.get_deps] = lambda: deps
    deps.model = ScriptedModel([answer()])
    with TestClient(restarted, base_url="https://testserver") as new_client:
        new_client.cookies.update(client.cookies)
        response = new_client.post(
            "/api/chat", json={"conversation_id": conversation_id, "message": "After restart"}
        )
    assert response.status_code == 200
    assert deps.model.requests[0]["messages"][-1]["content"][0]["text"] == "After restart"


@pytest.mark.parametrize(
    ("path", "body"),
    [
        ("/api/session", {"customer_id": 4}),
        ("/api/session", {"customer_id": True}),
        ("/api/session", {"customer_id": "1"}),
        ("/api/session", {"customer_id": 1, "extra": True}),
        ("/api/session/role", {"role": "admin"}),
        ("/api/chat", {"conversation_id": "x", "message": 1}),
        ("/api/chat", {"conversation_id": "x", "message": "Hi", "customer_id": 2}),
    ],
)
def test_request_validation(rig: tuple, path: str, body: dict) -> None:
    client, _, _ = rig
    session(client)
    assert client.post(path, json=body).status_code == 422


@pytest.mark.parametrize("reason", ["", "x" * 501, 7])
def test_rejection_reason_validation(rig: tuple, reason: object) -> None:
    client, _, _ = rig
    session(client)
    role(client, "ops")
    assert client.post("/api/ops/refunds/2/reject", json={"reason": reason}).status_code == 422


def test_store_failure_and_no_store_endpoints(rig: tuple, monkeypatch: pytest.MonkeyPatch) -> None:
    client, deps, web = rig
    session(client)
    conversation_id = live(client)

    def unavailable(*args: object, **kwargs: object) -> None:
        raise StoreUnavailable("Private backend detail")

    monkeypatch.setattr(deps.store, "get_doc", unavailable)
    response = client.post(
        "/api/chat", json={"conversation_id": conversation_id, "message": "Hello"}
    )
    assert response.status_code == 503 and response.json() == {"error": "unavailable"}
    assert deps.model.requests == []
    client.app.dependency_overrides[web.get_deps] = unavailable
    assert client.get("/api/health").json() == {"ok": True}
    assert client.get("/api/recordings").status_code == 200


def test_unexpected_error_is_sanitized(rig: tuple, monkeypatch: pytest.MonkeyPatch) -> None:
    client, deps, _ = rig

    def broken(*args: object) -> None:
        raise RuntimeError("Private detail")

    monkeypatch.setattr(deps.store, "incr_below", broken)
    response = client.post("/api/session", json={"customer_id": 1})
    assert response.status_code == 500 and response.json() == {"error": "internal"}
    assert response.headers["cache-control"] == "no-store"


def test_recordings_and_traversal(
    rig: tuple, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client, _, web = rig
    monkeypatch.setattr(web, "RECORDINGS_DIR", tmp_path)
    data = {"name": "sample", "title": "Sample recording", "language": "en", "steps": []}
    (tmp_path / "sample.json").write_text(json.dumps(data), encoding="utf-8")
    (tmp_path / "INVALID.json").write_text("{}", encoding="utf-8")
    assert client.get("/api/recordings").json() == [
        {"name": "sample", "title": "Sample recording", "language": "en"}
    ]
    assert client.get("/api/recordings/sample").json() == data
    for name in ("../x", "%2e%2e%2fx", "bad.name", "a" * 41, "missing"):
        assert client.get(f"/api/recordings/{name}").status_code == 404
    assert client.get("/api/recordings/sample").headers["cache-control"] == "no-store"


def test_no_store_cache_header_on_errors(rig: tuple) -> None:
    client, _, _ = rig
    for response in (
        client.get("/api/health"),
        client.post("/api/live/start"),
        client.get("/absent"),
    ):
        assert response.headers["cache-control"] == "no-store"


def test_vercel_ip_header_and_local_socket(rig: tuple) -> None:
    client, deps, web = rig
    from app.limits import ip_hash

    client.post("/api/session", json={"customer_id": 1}, headers={"x-real-ip": "192.0.2.8"})
    local_hash = ip_hash("testclient", deps.settings.ip_hash_secret)
    assert any(local_hash in key for key in deps.store._values)
    settings = replace(deps.settings, on_vercel=True)
    deps.settings = settings
    app = web.create_app(settings)
    app.dependency_overrides[web.get_deps] = lambda: deps
    with TestClient(app, base_url="https://testserver") as hosted:
        hosted.post("/api/session", json={"customer_id": 1}, headers={"x-real-ip": "192.0.2.8"})
    hashed = ip_hash("192.0.2.8", settings.ip_hash_secret)
    assert any(hashed in key for key in deps.store._values)
    assert "192.0.2.8" not in json.dumps(deps.store._values)


def test_transient_store_error_during_turn_returns_503(
    rig: tuple, monkeypatch: pytest.MonkeyPatch
) -> None:
    client, deps, _ = rig
    session(client)
    conversation_id = live(client)
    original = deps.store.reserve

    def fail_once(key: str, amount: int, cap: int) -> bool:
        monkeypatch.setattr(deps.store, "reserve", original)
        raise StoreUnavailable("Private reservation failure")

    monkeypatch.setattr(deps.store, "reserve", fail_once)
    response = client.post(
        "/api/chat", json={"conversation_id": conversation_id, "message": "Hello"}
    )
    assert response.status_code == 503 and response.json() == {"error": "unavailable"}
    assert deps.model.requests == []


def test_mid_conversation_budget_refusal(rig: tuple) -> None:
    client, deps, _ = rig
    deps.model = ScriptedModel([answer()])
    session(client)
    conversation_id = live(client)
    assert (
        client.post(
            "/api/chat", json={"conversation_id": conversation_id, "message": "First"}
        ).status_code
        == 200
    )
    budget = Budget(deps.store, deps.settings.daily_cap_nano, deps.today)
    assert budget.reserve(budget.remaining())
    response = client.post(
        "/api/chat", json={"conversation_id": conversation_id, "message": "Second"}
    )
    assert response.status_code == 200 and response.json()["stopped"] == "budget"
    assert len(deps.model.requests) == 1
    assert all(row["type"] != "tool_call" for row in response.json()["events"])


def test_message_length_is_enforced_by_loop(rig: tuple) -> None:
    client, deps, _ = rig
    session(client)
    conversation_id = live(client)
    for message in ("", " " * 5, "x" * 1001):
        response = client.post(
            "/api/chat", json={"conversation_id": conversation_id, "message": message}
        )
        assert response.status_code == 200 and response.json()["stopped"] == "end_turn"
    assert deps.model.requests == []


def test_local_fake_model_and_shared_memory(web: ModuleType) -> None:
    settings = replace(Settings.from_env(), demo_fake_model=True, kv_url=None, kv_token=None)
    app = web.create_app(settings)
    with TestClient(app, base_url="https://testserver") as client:
        session(client)
        conversation_id = live(client)
        for _ in range(2):
            response = client.post(
                "/api/chat", json={"conversation_id": conversation_id, "message": "Hello"}
            )
            assert response.status_code == 200
            assert response.json()["stopped"] == "end_turn"
            assert "simulated" in response.json()["reply"]


def test_hosted_missing_store_fails_closed(web: ModuleType) -> None:
    settings = replace(Settings.from_env(), on_vercel=True, kv_url=None, kv_token=None)
    app = web.create_app(settings)
    with TestClient(app, base_url="https://testserver") as client:
        assert client.get("/api/health").json() == {"ok": True}
        response = client.post("/api/session", json={"customer_id": 1})
        assert response.status_code == 503 and response.json() == {"error": "unavailable"}


def test_cookie_expiry_uses_injected_clock(rig: tuple) -> None:
    client, _, _ = rig
    session(client)
    client.app.state.now = lambda: NOW + timedelta(days=1)
    assert client.post("/api/live/start").status_code == 401


def test_local_concurrent_initial_requests_share_session_limit(web: ModuleType) -> None:
    settings = replace(Settings.from_env(), kv_url=None, kv_token=None)
    app = web.create_app(settings)
    with TestClient(app, base_url="https://testserver") as client:
        with ThreadPoolExecutor(max_workers=8) as pool:
            responses = list(
                pool.map(lambda _: client.post("/api/session", json={"customer_id": 1}), range(21))
            )
    assert sum(response.status_code == 200 for response in responses) == 20
    assert sum(response.status_code == 429 for response in responses) == 1


def test_unrecorded_transient_store_failure_still_returns_503(
    rig: tuple, monkeypatch: pytest.MonkeyPatch
) -> None:
    client, deps, _ = rig
    session(client)
    conversation_id = live(client)

    def unavailable(*args: object, **kwargs: object) -> None:
        raise StoreUnavailable("Short outage")

    monkeypatch.setattr(deps.store, "reserve", unavailable)
    monkeypatch.setattr(deps.store, "append_many", unavailable)
    response = client.post(
        "/api/chat", json={"conversation_id": conversation_id, "message": "Hello"}
    )
    assert response.status_code == 503 and response.json() == {"error": "unavailable"}
    assert deps.model.requests == []


def test_release_failure_returns_503_even_when_audit_is_unavailable(
    rig: tuple, monkeypatch: pytest.MonkeyPatch
) -> None:
    client, deps, _ = rig
    session(client)
    conversation_id = live(client)
    original = deps.store.update_doc

    def refuse_release(key: str, mutate: object, default: dict) -> dict:
        def apply(doc: dict) -> dict:
            result = mutate(doc)
            if result.get("busy_until", "absent") is None:
                raise StoreUnavailable("Lease release outage")
            return result

        return original(key, apply, default)

    def after_model(request: dict) -> ModelResponse:
        monkeypatch.setattr(deps.store, "update_doc", refuse_release)
        return answer()

    deps.model = ScriptedModel([after_model])
    original_append = deps.store.append_many

    def refuse_incident(key: str, rows: list[dict]) -> None:
        if any(row["type"] == "incident" for row in rows):
            raise StoreUnavailable("Incident audit outage")
        original_append(key, rows)

    monkeypatch.setattr(deps.store, "append_many", refuse_incident)
    response = client.post(
        "/api/chat", json={"conversation_id": conversation_id, "message": "Hello"}
    )
    assert response.status_code == 503 and response.json() == {"error": "unavailable"}


def test_approval_audit_failure_prevents_decision(
    rig: tuple, monkeypatch: pytest.MonkeyPatch
) -> None:
    client, deps, _ = rig
    deps.model = ScriptedModel(held_steps())
    session(client)
    conversation_id = live(client)
    result = client.post(
        "/api/chat", json={"conversation_id": conversation_id, "message": "Refund"}
    ).json()
    refund_id = result["refunds"][0]["refund_id"]
    role(client, "ops")

    def unavailable(*args: object, **kwargs: object) -> None:
        raise StoreUnavailable("Audit write outage")

    monkeypatch.setattr(deps.store, "append_many", unavailable)
    response = client.post(f"/api/ops/refunds/{refund_id}/approve")
    assert response.status_code == 503
    assert client.get("/api/ops/queue").json()["refunds"][0]["refund_id"] == refund_id


def test_completed_decision_audit_survives_result_append_failure(
    rig: tuple, monkeypatch: pytest.MonkeyPatch
) -> None:
    client, deps, _ = rig
    deps.model = ScriptedModel(held_steps())
    session(client)
    conversation_id = live(client)
    result = client.post(
        "/api/chat", json={"conversation_id": conversation_id, "message": "Refund"}
    ).json()
    refund_id = result["refunds"][0]["refund_id"]
    role(client, "ops")
    original = deps.store.append_many

    def fail_outcome(key: str, rows: list[dict]) -> None:
        if any(row["type"] == "approval" for row in rows):
            raise StoreUnavailable("Result append outage")
        original(key, rows)

    monkeypatch.setattr(deps.store, "append_many", fail_outcome)
    assert client.post(f"/api/ops/refunds/{refund_id}/approve").status_code == 503
    events = client.get(f"/api/audit?conversation_id={conversation_id}").json()
    decisions = [row for row in events if row["type"] == "approval"]
    assert len(decisions) == 1 and decisions[0]["data"]["status"] == "SUCCEEDED"
    assert client.post(f"/api/ops/refunds/{refund_id}/approve").status_code == 409


def test_empty_body_endpoints_reject_extra_fields(rig: tuple) -> None:
    client, deps, _ = rig
    deps.model = ScriptedModel(held_steps())
    session(client)
    assert client.post("/api/live/start", json={"customer_id": 2}).status_code == 422
    conversation_id = live(client)
    result = client.post(
        "/api/chat", json={"conversation_id": conversation_id, "message": "Refund"}
    ).json()
    refund_id = result["refunds"][0]["refund_id"]
    role(client, "ops")
    assert (
        client.post(
            f"/api/ops/refunds/{refund_id}/approve", json={"decided_by": "someone"}
        ).status_code
        == 422
    )
    assert client.get("/api/ops/queue").json()["refunds"][0]["refund_id"] == refund_id
