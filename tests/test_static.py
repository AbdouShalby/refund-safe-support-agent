"""Check the static demo boundary and local routing without a browser or network."""

import ast
import importlib
import re
import shutil
import subprocess
from html.parser import HTMLParser
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import Settings

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT / "public"
CSP = (
    "default-src 'self'; style-src 'self'; script-src 'self'; "
    "connect-src 'self'; img-src 'self' data:"
)


class Markup(HTMLParser):
    """Collect actual HTML attributes rather than matching JavaScript variable names."""

    def __init__(self) -> None:
        super().__init__()
        self.tags: list[tuple[str, dict]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.tags.append((tag, dict(attrs)))


def assets() -> list[Path]:
    return sorted(path for path in PUBLIC.rglob("*") if path.is_file())


def markup() -> Markup:
    parsed = Markup()
    parsed.feed((PUBLIC / "index.html").read_text(encoding="utf-8"))
    return parsed


def test_safe_text_only_assets() -> None:
    forbidden = ("innerHTML", "outerHTML", "insertAdjacentHTML", "document.write")
    for path in assets():
        text = path.read_text(encoding="utf-8")
        assert not any(token in text for token in forbidden), path
        assert not re.search(r"\beval\s*\(", text), path
        assert not re.search(r"\bon[a-z]+\s*=", text, re.IGNORECASE), path
        assert "production" not in text.lower(), path
        assert b"\r" not in path.read_bytes(), path
    for _, attrs in markup().tags:
        assert not any(re.fullmatch(r"on[a-z]+", key) for key in attrs)


def test_self_hosted_assets_and_csp() -> None:
    tags = markup().tags
    assert any(
        tag == "meta"
        and attrs.get("http-equiv", "").lower() == "content-security-policy"
        and attrs.get("content") == CSP
        for tag, attrs in tags
    )
    for tag, attrs in tags:
        if tag == "script" or (tag == "link" and attrs.get("rel") == "stylesheet"):
            source = attrs.get("src" if tag == "script" else "href", "")
            assert source.startswith("/") and not source.startswith("//")
            assert not re.search(r"https?://", source)
            assert (PUBLIC / source.lstrip("/")).is_file()
    assert any(tag == "script" and attrs.get("src") == "/app.js" for tag, attrs in tags)


def test_javascript_api_calls_have_matching_python_routes() -> None:
    python = ast.parse((ROOT / "app/web.py").read_text(encoding="utf-8"))
    routes = set()
    for node in ast.walk(python):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name) and node.func.value.id == "router":
                if node.func.attr in {"get", "post"}:
                    routes.add((node.func.attr.upper(), "/api" + ast.literal_eval(node.args[0])))
    js = "\n".join(path.read_text(encoding="utf-8") for path in PUBLIC.rglob("*.js"))
    calls = re.findall(r"api\(\s*([\"'`])(/api/[^\"'`]+)\1(?:,\s*([\"'])(GET|POST)\3)?", js)
    assert len(calls) >= 10, "Every endpoint must be visible in a direct api(...) call"
    for _, path, _, method in calls:
        normalized = re.sub(r"\$\{[^}]+\}", "{value}", path.split("?")[0])
        normalized = re.sub(r"\{[^}]+\}", "{value}", normalized)
        assert any(
            (method or "GET") == verb and normalized == re.sub(r"\{[^}]+\}", "{value}", route)
            for verb, route in routes
        ), (method, path)
    assert 'credentials: "same-origin"' in js


@pytest.fixture
def local_settings(monkeypatch: pytest.MonkeyPatch) -> Settings:
    monkeypatch.setenv("SESSION_SECRET", "test-session-secret-with-at-least-32-characters")
    monkeypatch.setenv("IP_HASH_SECRET", "test-ip-secret-with-at-least-32-characters")
    monkeypatch.setenv("DEMO_FAKE_MODEL", "1")
    monkeypatch.setenv("DOPS_MODE", "mock")
    monkeypatch.setenv("DAILY_CAP_USD", "0.12")
    for name in ("VERCEL", "KV_REST_API_URL", "KV_REST_API_TOKEN"):
        monkeypatch.delenv(name, raising=False)
    return Settings.from_env()


def test_local_page_and_api_precedence(local_settings: Settings) -> None:
    web = importlib.import_module("app.web")
    with TestClient(web.create_app(local_settings), base_url="https://testserver") as client:
        response = client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert "Refund-safe support agent" in response.text
        for name, content_type in (("app.js", "javascript"), ("style.css", "text/css")):
            asset = client.get("/" + name)
            assert asset.status_code == 200
            assert content_type in asset.headers["content-type"]
        assert client.get("/api/health").json() == {"ok": True}
        assert client.get("/api/health").status_code == 200
        assert client.get("/api/recordings").status_code == 200
        assert client.get("/api/ops/queue").status_code == 401


UI_CHECK = r"""
const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");
class Node {
  constructor(tag) {
    this.tagName = tag; this.children = []; this.attributes = {}; this.dataset = {};
    this.listeners = {}; this.value = ""; this._text = ""; this.parent = null;
  }
  set textContent(value) { this._text = String(value); this.children = []; }
  get textContent() { return this._text + this.children.map(n => n.textContent).join(""); }
  append(...nodes) {
    for (const node of nodes) { node.remove(); node.parent = this; this.children.push(node); }
  }
  replaceChildren(...nodes) { this._text = ""; this.children = []; this.append(...nodes); }
  remove() {
    if (this.parent) this.parent.children = this.parent.children.filter(n => n !== this);
    this.parent = null;
  }
  setAttribute(key, value) { this.attributes[key] = String(value); }
  getAttribute(key) { return this.attributes[key]; }
  removeAttribute(key) { delete this.attributes[key]; }
  addEventListener(type, handler) { this.listeners[type] = handler; }
  focus() { this.focused = true; }
  setCustomValidity(value) { this.validityMessage = value; }
  reportValidity() { this.validityReported = true; }
  querySelectorAll(selector) {
    return descendants(this).filter(n => selector === "button" ? n.tagName === "button"
      : selector === "[data-refund-id]" ? n.dataset.refundId !== undefined
      : n.className?.split(" ").includes(selector.slice(1)));
  }
  querySelector(selector) { return this.querySelectorAll(selector)[0] || null; }
}
function descendants(node) { return node.children.flatMap(n => [n, ...descendants(n)]); }
const ids = {};
const html = fs.readFileSync("public/index.html", "utf8");
for (const [, tag, id] of html.matchAll(/<([a-z]+)[^>]*\bid="([^"]+)"/g)) ids[id] = new Node(tag);
ids.customer.value = "1";
ids.customer.selectedOptions = [{ textContent: "Customer A" }];
const calls = [];
let responses = [];
let timer;
const context = vm.createContext({
  console, Map, Set, Date, BigInt,
  document: {
    getElementById: id => ids[id], createElement: tag => new Node(tag),
    querySelectorAll: selector =>
      [...new Set(Object.values(ids).flatMap(n => [n, ...descendants(n)]))]
      .filter(n => selector === "button" && n.tagName === "button"),
  },
  window: { matchMedia: () => ({ matches: true }), addEventListener() {} },
  setTimeout: (callback, delay) => { timer = { callback, delay }; return 1; },
  clearTimeout: () => { timer = null; },
  fetch: async (path, options) => {
    calls.push({ path, options });
    assert.equal(options.credentials, "same-origin");
    if (options.method === "POST") assert.equal(typeof JSON.parse(options.body), "object");
    const expected = responses.shift();
    assert.ok(expected, `Unexpected request: ${path}`);
    assert.equal(path, expected.path);
    const status = expected.status || 200;
    return { ok: status < 400, status, json: async () => expected.data };
  },
});
responses.push({ path: "/api/recordings", data: [] });
vm.runInContext(fs.readFileSync("public/app.js", "utf8"), context);
const run = source => vm.runInContext(source, context);
const queue = (path, data, status) => responses.push({ path, data, status });
const tick = () => new Promise(resolve => setImmediate(resolve));
(async () => {
  await tick();
  assert.equal(calls.length, 1, "Loading must not create a session or call a model");
  assert.equal(ids.message.disabled, true);
  assert.match(ids["recording-list"].textContent, /No recorded runs yet/);
  assert.equal(ids.behind.open, false);
  const refund = { refund_id: 9, order_id: 1002, amount: "60.00", status: "PENDING_APPROVAL" };
  const event = { ts: "2026-11-18T12:00:00Z", actor: "model", type: "model_call",
    trace_id: "t-0000000000000001",
    data: { usage: { input: 1532, output: 87 }, cost_nano: 412000 } };
  const recording = { customer: "Customer A", steps: [
    { role: "customer", text: "Refund order 1002", events: [], refunds: [] },
    { role: "agent", text: "Review requested.\n\nRefund system status:\nAwaiting review",
      events: [event], refunds: [refund] },
  ], ops: [{ action: "approve", refund_id: 9, status_after: "SUCCEEDED" }] };
  queue("/api/recordings", [{ name: "held", title: "Held refund", language: "en" }]);
  await run('action(null, "", loadRecordings)');
  queue("/api/recordings/held", recording);
  await ids["recording-list"].children[0].listeners.click();
  run("play()");
  assert.equal(timer.delay, 1200);
  timer.callback();
  assert.equal(ids["total-cost"].textContent, "$0.000412");
  assert.match(ids.events.textContent, /1,532 in \/ 87 out/);
  assert.equal(ids.messages.querySelectorAll(".system-output").length, 1);
  timer.callback();
  assert.match(ids.messages.textContent, /Ops approved refund/);
  assert.match(ids["refund-statuses"].textContent, /60.00.*refunded/);
  assert.equal(timer, null);
  assert.equal(ids.play.disabled, true);
  run('appendMessage("agent", "<script>bad()</script>")');
  assert.match(ids.messages.textContent, /<script>bad/);
  const arabic = "\u062d\u0627\u0644\u0629 " +
    "\u0627\u0644\u0627\u0633\u062a\u0631\u062f\u0627\u062f " +
    "\u0645\u0646 \u0627\u0644\u0646\u0638\u0627\u0645:";
  context.arabic = arabic;
  run(String.raw`appendMessage("agent", "Arabic reply\n" + arabic + "\nheld",
    [{refund_id: 9, status: "PENDING_APPROVAL"}])`);
  assert.equal(ids.messages.querySelectorAll(".system-output").length, 3);
  queue("/api/session", { customer: { name: "Customer A" }, orders: [] });
  queue("/api/live/start", { conversation_id: "c-demo" });
  const starting = run("startLive()");
  assert.equal(ids["start-live"].disabled, true);
  assert.equal(ids["start-live"].getAttribute("aria-busy"), "true");
  assert.equal(ids.role.disabled, true);
  await starting;
  assert.equal(ids.message.disabled, false);
  assert.equal(ids.mode.textContent, "Live");
  ids.message.value = "Please refund 60.00";
  queue("/api/chat", { reply: "Please wait.\n\nRefund system status:\nheld",
    refunds: [refund], events: [event], stopped: "end_turn" });
  queue("/api/conversation/c-demo/refunds", [refund]);
  await run("sendMessage({preventDefault(){}})");
  assert.match(ids["refund-statuses"].textContent, /awaiting human review/);
  assert.equal(ids.messages.querySelectorAll(".typing").length, 0);
  assert.equal(ids.message.value, "");
  const ops = { refunds: [{ refund_id: 9, facts: { amount: "60.00", order_total: "80.00",
    reserved: "60.00", refunded: "0.00", customer_id: 1, order_id: 1002,
    policy_reasons: ["OVER_AUTO_LIMIT"] }, model_text: { text: "<b>untrusted</b>" } }],
    unconfirmed: [{ order_id: 1004, amount: "<b>20.00</b>" }],
    unconfirmed_remaining: 3,
    escalations: [{ customer_id: 1, status: "OPEN",
      model_text: { text: "Human help requested" } }] };
  queue("/api/session/role", { role: "ops" });
  queue("/api/ops/queue", ops);
  await run('changeRole("ops", $("ops-tab"))');
  assert.equal(ids["ops-panel"].hidden, false);
  assert.match(ids["ops-queue"].textContent, /Facts \(from the refund system\)/);
  assert.match(ids["ops-queue"].textContent, /<b>untrusted<\/b>/);
  const unconfirmed = ids["ops-queue"].children.at(-1);
  assert.match(unconfirmed.textContent, /Unconfirmed refund requests.*Order #1004.*<b>20.00<\/b>/);
  assert.equal(unconfirmed.querySelectorAll("button").length, 0);
  assert.match(unconfirmed.textContent, /3 more unconfirmed/);
  const form = ids["ops-queue"].querySelectorAll(".decision-form")[0];
  const reason = form.children[1];
  reason.value = "   ";
  form.listeners.submit({preventDefault(){}});
  assert.match(reason.validityMessage, /Enter a rejection reason/);
  queue("/api/ops/refunds/9/approve", { status: "SUCCEEDED" });
  queue("/api/ops/queue", { refunds: [], escalations: [] });
  queue("/api/audit?conversation_id=c-demo", [event]);
  await run('decide(9, "approve", "", $("ops-queue").querySelector("button"))');
  assert.match(ids["decision-results"].textContent, /refunded/);
  assert.match(ids["ops-queue"].textContent, /No refunds are waiting/);
  assert.doesNotMatch(ids["ops-queue"].textContent, /Unconfirmed/);
  queue("/api/session/role", { role: "customer" });
  queue("/api/conversation/c-demo/refunds", [{ ...refund, status: "SUCCEEDED" }]);
  queue("/api/audit?conversation_id=c-demo", [event]);
  await run('changeRole("customer", $("customer-tab"))');
  assert.match(ids["refund-statuses"].textContent, /refunded/);
  assert.equal(ids["total-cost"].textContent, "$0.000412",
    "Audit refresh must not double count cost");
  ids.message.value = "Another request";
  queue("/api/chat", { stopped: "busy" }, 409);
  queue("/api/conversation/c-demo/refunds", [{ ...refund, status: "SUCCEEDED" }]);
  queue("/api/audit?conversation_id=c-demo", [event]);
  await run("sendMessage({preventDefault(){}})");
  assert.equal(ids.message.value, "Another request");
  assert.match(ids["turn-notice"].textContent, /already in progress/);
  queue("/api/chat", { reply: "Budget reached", refunds: [], events: [], stopped: "budget" });
  queue("/api/conversation/c-demo/refunds", [{ ...refund, status: "SUCCEEDED" }]);
  await run("sendMessage({preventDefault(){}})");
  assert.equal(ids.message.disabled, true);
  assert.equal(ids.recordings.hidden, false,
    "Budget stop must offer recordings after controls sync");
  for (const [reason, message] of [
    ["daily_budget", /budget is used up/], ["visitor_limit", /live demo limit/],
  ]) {
    queue("/api/session", { customer: { name: "Customer A" }, orders: [] });
    queue("/api/live/start", { reason }, 429);
    await run("startLive()");
    assert.match(ids.notice.textContent, message);
    assert.equal(ids.mode.textContent, "Recorded run");
    assert.equal(ids.recordings.hidden, false);
  }
  queue("/api/session", { customer: { name: "Customer A" }, orders: [] });
  queue("/api/live/start", { error: "internal" }, 500);
  await run("startLive()");
  assert.match(ids.notice.textContent, /temporarily unavailable/);
  assert.equal(ids.message.disabled, true);
  for (const status of [401, 403, 503]) {
    queue("/api/ops/queue", { error: "never show this raw payload" }, status);
    await run('action($("refresh-ops"), "Refreshing", loadQueue)');
    const expected = status === 401 ? /session expired/
      : status === 403 ? /Switch role/ : /temporarily unavailable/;
    assert.match(ids.notice.textContent, expected);
    assert.doesNotMatch(ids.notice.textContent, /raw payload/);
  }
  assert.equal(responses.length, 0);
  console.log("Offline UI state transitions passed; this is not a browser check.");
})().catch(error => { console.error(error); process.exitCode = 1; });
"""


def test_offline_ui_state_transitions() -> None:
    """Exercise status authority, replay, live limits and human decisions with fake HTTP."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is unavailable; the dependency-free UI behavior check needs Node")
    result = subprocess.run(  # noqa: S603
        [node, "-e", UI_CHECK], cwd=ROOT, capture_output=True, text=True, timeout=30, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr
