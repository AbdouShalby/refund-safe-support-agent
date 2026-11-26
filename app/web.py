"""Expose a sandboxed JSON demo API with code-owned roles, limits and approvals."""

import json
import re
import secrets
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from datetime import UTC, date, datetime
from itertools import repeat
from pathlib import Path
from types import TracebackType
from typing import Annotated, Literal, TypeVar

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, StrictInt
from upstash_redis import Redis

from app.agent import (
    AgentConfig,
    AgentDeps,
    Conversation,
    load_conversation,
    run_customer_turn,
    start_conversation,
)
from app.audit import AuditEvent, AuditLog
from app.budget import Budget
from app.config import Settings
from app.customers import customer_name
from app.dops.errors import NotFound, NotPendingApproval
from app.dops.mock import MockDops
from app.escalations import EscalationRepo
from app.intents import IntentRepo
from app.limits import Limits, ip_hash
from app.model import AnthropicModel, ModelClient, ModelResponse, ScriptedModel, TextBlock, Usage
from app.policy import PolicyConfig
from app.sandbox_dops import SandboxDops
from app.session import sign, verify
from app.store import Lock, MemoryStore, RedisStore, Store, StoreConflict, StoreUnavailable
from app.tools import ToolExecutor, mock_lock_scope

RECORDINGS_DIR = Path(__file__).resolve().parents[1] / "demo" / "recordings"
_RECORDING_NAME = re.compile(r"[a-z0-9-]{1,40}")
router = APIRouter(prefix="/api")
_T = TypeVar("_T")


class _ObservedLock(Lock):
    """Latch failures on lease acquisition and release as well as store methods."""

    def __init__(self, inner: Lock, owner: "_ObservedStore") -> None:
        self.inner = inner
        self.owner = owner

    def __enter__(self) -> Lock:
        return self.owner.call(self.inner.__enter__)

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.owner.call(lambda: self.inner.__exit__(exc_type, exc_value, traceback))


class _ObservedStore:
    """Remember request-local store failures even when the agent catches them."""

    def __init__(self, inner: Store) -> None:
        self.inner = inner
        self.failed = False

    def call(self, operation: Callable[[], _T]) -> _T:
        try:
            return operation()
        except (StoreUnavailable, StoreConflict):
            self.failed = True
            raise

    def get_doc(self, key: str) -> dict | None:
        return self.call(lambda: self.inner.get_doc(key))

    def update_doc(self, key: str, mutate: Callable[[dict], dict], default: dict) -> dict:
        return self.call(lambda: self.inner.update_doc(key, mutate, default))

    def append_many(self, key: str, items: list[dict]) -> None:
        self.call(lambda: self.inner.append_many(key, items))

    def read_list(self, key: str) -> list[dict]:
        return self.call(lambda: self.inner.read_list(key))

    def reserve(self, key: str, amount: int, cap: int) -> bool:
        return self.call(lambda: self.inner.reserve(key, amount, cap))

    def settle(self, key: str, reserved: int, actual: int) -> None:
        self.call(lambda: self.inner.settle(key, reserved, actual))

    def remaining(self, key: str, cap: int) -> int:
        return self.call(lambda: self.inner.remaining(key, cap))

    def incr_below(self, key: str, limit: int) -> bool:
        return self.call(lambda: self.inner.incr_below(key, limit))

    def lock(self, key: str, ttl_s: int) -> Lock:
        return _ObservedLock(self.call(lambda: self.inner.lock(key, ttl_s)), self)


class _DecisionStore(_ObservedStore):
    """Commit a human outcome audit with the mock DOPS decision in one CAS.

    Store has no cross-key transaction. This mock-only adapter retains a recovery
    copy in the DOPS document, whose unknown top-level fields MockDops preserves.
    Audit reads merge that copy if the separate audit-list append failed.
    """

    def __init__(self, inner: Store, sandbox_id: str, event: AuditEvent) -> None:
        super().__init__(inner)
        self.key = f"sbx:{sandbox_id}:dops"
        self.event = event

    def update_doc(self, key: str, mutate: Callable[[dict], dict], default: dict) -> dict:
        if key != self.key:
            return super().update_doc(key, mutate, default)

        def with_audit(doc: dict) -> dict:
            result = mutate(doc)
            refund = next(
                row for row in result["refunds"] if row["id"] == self.event.data["refund_id"]
            )
            event = replace(self.event, data={**self.event.data, "status": refund["status"]})
            result.setdefault("web_decision_events", []).append(event.to_dict())
            return result

        return super().update_doc(key, with_audit, default)


@dataclass
class WebDeps:
    """Request services and injectable UTC clocks, shared only through the store."""

    settings: Settings
    store: Store
    model: ModelClient
    now: Callable[[], datetime]
    today: Callable[[], date]


def get_deps(request: Request) -> WebDeps:
    """Build request dependencies; keep local memory stable across HTTP requests.

    Authentication uses app.state.now independently so a missing cookie still
    yields 401 without initializing the backend. Tests replacing these clocks
    should set app.state.now to the same callable as WebDeps.now.
    """
    settings = request.app.state.settings
    store = request.app.state.store
    if store is None:
        if settings.kv_url and settings.kv_token:
            store = RedisStore(Redis(url=settings.kv_url, token=settings.kv_token))
        else:
            raise StoreUnavailable("Shared demo store is not configured")
    model = (
        ScriptedModel(
            repeat(
                ModelResponse(
                    "end_turn",
                    [TextBlock("Local simulated reply. No model call was made.")],
                    Usage(0, 0, 0, 0),
                )
            )
        )
        if settings.demo_fake_model
        else AnthropicModel()
    )
    now = request.app.state.now
    return WebDeps(settings, store, model, now, lambda: now().date())


class StrictRequest(BaseModel):
    """Reject unrecognized fields and implicit scalar conversions."""

    model_config = ConfigDict(extra="forbid", strict=True)


class SessionRequest(StrictRequest):
    """Select a synthetic customer without admitting other identities."""

    customer_id: Annotated[StrictInt, Field(ge=1, le=3)]


class RoleRequest(StrictRequest):
    """The simulated role switch never creates an agent approval tool."""

    role: Literal["customer", "ops"]


class ChatRequest(StrictRequest):
    """Leave customer-message length enforcement to the bounded agent loop."""

    conversation_id: str
    message: str


class RejectionRequest(StrictRequest):
    """Require a bounded human decision reason."""

    reason: Annotated[str, Field(min_length=1, max_length=500)]


def _session(request: Request) -> dict:
    current = request.app.state.now()
    payload = verify(
        request.cookies.get("sid", ""), request.app.state.settings.session_secret, current
    )
    if payload is None:
        raise HTTPException(401, {"error": "session"})
    return payload


def _customer(session: Annotated[dict, Depends(_session)]) -> dict:
    if session["role"] != "customer":
        raise HTTPException(403, {"error": "role"})
    return session


def _ops(session: Annotated[dict, Depends(_session)]) -> dict:
    if session["role"] != "ops":
        raise HTTPException(403, {"error": "role"})
    return session


Deps = Annotated[WebDeps, Depends(get_deps)]
Session = Annotated[dict, Depends(_session)]
Customer = Annotated[dict, Depends(_customer)]
Ops = Annotated[dict, Depends(_ops)]


def _cookie(response: Response, payload: dict, settings: Settings) -> None:
    response.set_cookie(
        "sid",
        sign(payload, settings.session_secret),
        max_age=86_400,
        httponly=True,
        secure=True,
        samesite="lax",
        path="/",
    )


def _visitor(request: Request, settings: Settings) -> str:
    ip = request.headers.get("x-real-ip") if settings.on_vercel else None
    if not ip:
        ip = request.client.host if request.client else "unknown"
    return ip_hash(ip, settings.ip_hash_secret)


def _trace() -> str:
    return "t-" + secrets.token_hex(8)


def _dops(session: dict, deps: WebDeps) -> SandboxDops:
    return SandboxDops(deps.store, session["sandbox_id"], deps.now)


def _budget(deps: WebDeps) -> Budget:
    return Budget(deps.store, deps.settings.daily_cap_nano, deps.today)


def _conversation(session: dict, deps: WebDeps, conversation_id: str) -> Conversation:
    conversation = load_conversation(deps.store, session["sandbox_id"], conversation_id)
    if conversation is None or conversation.customer_id != session["customer_id"]:
        raise HTTPException(404, {"error": "not_found"})
    return conversation


def _intents(session: dict, deps: WebDeps) -> list[dict]:
    repo = IntentRepo(deps.store, session["sandbox_id"])
    # IntentRepo has no list method; read its sandbox-local persisted document.
    return list((deps.store.get_doc(repo.key) or {}).get("intents", {}).values())


@router.get("/health")
def health() -> dict[str, bool]:
    """Provide a health probe with no shared-state dependency."""
    return {"ok": True}


@router.post("/session")
def new_session(body: SessionRequest, request: Request, response: Response, deps: Deps) -> dict:
    """Issue a signed visitor identity, writing only the hashed session counter."""
    if not Limits(deps.store, deps.today).allow_session(_visitor(request, deps.settings)):
        raise HTTPException(429, {"reason": "visitor_limit"})
    payload = {
        "sandbox_id": "s-" + secrets.token_hex(12),
        "customer_id": body.customer_id,
        "role": "customer",
        "exp": int(deps.now().timestamp()) + 86_400,
    }
    orders = MockDops.seeded(deps.now).list_orders(body.customer_id, _trace())
    _cookie(response, payload, deps.settings)
    return {
        "customer": {"id": body.customer_id, "name": customer_name(body.customer_id)},
        "orders": [order.model_dump(mode="json") for order in orders],
    }


@router.post("/session/role")
def switch_role(body: RoleRequest, response: Response, request: Request, session: Session) -> dict:
    """Re-sign the existing sandbox identity without extending its absolute expiry."""
    _cookie(response, {**session, "role": body.role}, request.app.state.settings)
    return {"role": body.role}


@router.post("/live/start")
def live_start(
    request: Request, session: Customer, deps: Deps, body: StrictRequest | None = None
) -> dict:
    """Seed lazily, check budget and visitor caps, then persist a conversation."""
    dops = _dops(session, deps)
    dops.list_orders(session["customer_id"], _trace())
    if _budget(deps).remaining() < 5_000_000:
        raise HTTPException(429, {"reason": "daily_budget"})
    if not Limits(deps.store, deps.today).start_live(
        session["sandbox_id"], _visitor(request, deps.settings)
    ):
        raise HTTPException(429, {"reason": "visitor_limit"})
    conversation = start_conversation(
        deps.store, session["sandbox_id"], session["customer_id"], dops, deps.now
    )
    return {"conversation_id": conversation.id}


@router.post("/chat")
def chat(body: ChatRequest, response: Response, session: Customer, deps: Deps) -> dict:
    """Execute one bounded customer turn and expose only its sanitized audit events."""
    observed = _ObservedStore(deps.store)
    deps = replace(deps, store=observed)
    _conversation(session, deps, body.conversation_id)
    audit = AuditLog(deps.store, session["sandbox_id"])
    executor = ToolExecutor(
        _dops(session, deps),
        deps.store,
        audit,
        EscalationRepo(deps.store, session["sandbox_id"]),
        PolicyConfig(),
        deps.now,
        mock_lock_scope,
    )
    result = run_customer_turn(
        body.conversation_id,
        body.message,
        AgentDeps(
            session["sandbox_id"],
            deps.model,
            executor,
            _budget(deps),
            audit,
            deps.store,
            AgentConfig(),
            deps.now,
        ),
    )
    events = [row for row in audit.list(body.conversation_id) if row["trace_id"] == result.trace_id]
    # The loop catches store errors and may fail to persist their incidents too.
    # Observe the operations themselves, independently of the persisted audit.
    if observed.failed:
        raise StoreUnavailable("Turn state was unavailable")
    if result.stopped == "busy":
        response.status_code = 409
    return {**asdict(result), "events": events}


@router.get("/conversation/{conversation_id}/refunds")
def conversation_refunds(conversation_id: str, session: Customer, deps: Deps) -> list[dict]:
    """Refresh each intent's actual DOPS status instead of replaying model claims."""
    _conversation(session, deps, conversation_id)
    intents = [row for row in _intents(session, deps) if row["conversation_id"] == conversation_id]
    dops = _dops(session, deps)
    found = {}
    trace = _trace()
    for order_id in sorted({row["order_id"] for row in intents}):
        for refund in dops.list_order_refunds(session["customer_id"], order_id, trace):
            found[refund.id] = refund
    return [
        {
            "refund_id": refund.id,
            "order_id": refund.order_id,
            "amount": refund.amount,
            "status": refund.status,
        }
        for refund_id in dict.fromkeys(row["refund_id"] for row in intents)
        if (refund := found.get(refund_id)) is not None
    ]


@router.get("/audit")
def audit_events(session: Session, deps: Deps, conversation_id: str | None = None) -> list[dict]:
    """Read only the signed sandbox's audit log, with an optional conversation filter."""
    rows = AuditLog(deps.store, session["sandbox_id"]).list(conversation_id)
    state = deps.store.get_doc(f"sbx:{session['sandbox_id']}:dops") or {}
    seen = {(row["trace_id"], row["type"]) for row in rows}
    for row in state.get("web_decision_events", []):
        if (conversation_id is None or row["conversation_id"] == conversation_id) and (
            row["trace_id"],
            row["type"],
        ) not in seen:
            rows.append(row)
    return sorted(rows, key=lambda row: row["ts"])


@router.get("/ops/queue")
def ops_queue(session: Ops, deps: Deps) -> dict:
    """Keep DOPS facts separate from labelled, untrusted model-written text."""
    dops = _dops(session, deps)
    trace = _trace()
    sandbox = session["sandbox_id"]
    repo = IntentRepo(deps.store, sandbox)
    rows, _ = repo.snapshot()
    unresolved = [row for row in reversed(rows.values()) if row.status != "SUBMITTED"]
    selected = unresolved[:20]
    if unresolved:
        audit = AuditLog(deps.store, sandbox)
        executor = ToolExecutor(
            dops,
            deps.store,
            audit,
            EscalationRepo(deps.store, sandbox),
            PolicyConfig(),
            deps.now,
            mock_lock_scope,
        )
        try:
            executor.reconcile(repo, sandbox, selected, trace, audit_unconfirmed=False)
            rows, _ = repo.snapshot()
        except (StoreUnavailable, StoreConflict) as error:
            executor.pending_events.append(
                AuditEvent(
                    deps.now(),
                    trace,
                    "",
                    "system",
                    "incident",
                    {
                        "kind": type(error).__name__,
                        "message": "Ops refund lookup could not complete",
                    },
                )
            )
        try:
            audit.record_many(executor.pending_events)
        except (StoreUnavailable, StoreConflict):
            pass  # A best-effort resolution must not take down the review queue.
    intents = {row.refund_id: row for row in rows.values() if row.refund_id is not None}
    remaining = {row.key for row in rows.values() if row.status != "SUBMITTED"}
    refunds = []
    for refund in dops.pending_refunds("PENDING_APPROVAL", trace):
        order = dops.get_order(refund.customer_id, refund.order_id, trace)
        refunds.append(
            {
                "refund_id": refund.id,
                "facts": {
                    "amount": refund.amount,
                    "order_total": order.total_amount,
                    "reserved": order.reserved_amount,
                    "refunded": order.refunded_amount,
                    "customer_id": refund.customer_id,
                    "order_id": refund.order_id,
                    "policy_reasons": intents[refund.id].decision_reasons
                    if refund.id in intents
                    else [],
                },
                "model_text": {"label": "model-written", "text": refund.reason},
            }
        )
    escalations = [
        {
            **{key: value for key, value in row.items() if key != "summary"},
            "model_text": {"label": "model-written", "text": row["summary"]},
        }
        for row in EscalationRepo(deps.store, session["sandbox_id"]).list()
    ]
    return {
        "refunds": refunds,
        "escalations": escalations,
        "unconfirmed": [
            {"order_id": row.order_id, "amount": row.amount}
            for row in selected
            if row.key in remaining
        ],
        "unconfirmed_remaining": len(remaining.difference(row.key for row in selected)),
    }


def _decide(session: dict, deps: WebDeps, refund_id: int, reason: str | None) -> dict:
    trace = _trace()
    actor = "ops:" + session["sandbox_id"][:8]
    intents = _intents(session, deps)
    audit = AuditLog(deps.store, session["sandbox_id"])
    conversation_id = next(
        (row["conversation_id"] for row in intents if row["refund_id"] == refund_id), ""
    )
    event = AuditEvent(
        deps.now(),
        trace,
        conversation_id,
        "human",
        "approval" if reason is None else "rejection",
        {"refund_id": refund_id},
    )
    # Persist the human request before any decision. An audit outage must prevent
    # the write, while the outcome itself is stored atomically with mock DOPS.
    audit.record_many([replace(event, type=event.type + "_requested")])
    store = _DecisionStore(deps.store, session["sandbox_id"], event)
    dops = SandboxDops(store, session["sandbox_id"], deps.now)
    refund = (
        dops.approve_refund(refund_id, actor, trace)
        if reason is None
        else dops.reject_refund(refund_id, actor, reason, trace)
    )
    audit.record_many([replace(event, data={"refund_id": refund_id, "status": refund.status})])
    return {"status": refund.status}


@router.post("/ops/refunds/{refund_id}/approve")
def approve(refund_id: int, session: Ops, deps: Deps, body: StrictRequest | None = None) -> dict:
    """Approve only a pending refund within the Ops visitor's sandbox."""
    return _decide(session, deps, refund_id, None)


@router.post("/ops/refunds/{refund_id}/reject")
def reject(refund_id: int, body: RejectionRequest, session: Ops, deps: Deps) -> dict:
    """Reject a pending refund and release its reservation through DOPS."""
    return _decide(session, deps, refund_id, body.reason)


def _recording(name: str) -> dict:
    if not _RECORDING_NAME.fullmatch(name):
        raise HTTPException(404, {"error": "not_found"})
    path = RECORDINGS_DIR / f"{name}.json"
    if not path.is_file():
        raise HTTPException(404, {"error": "not_found"})
    return json.loads(path.read_text(encoding="utf-8"))


@router.get("/recordings")
def recordings() -> list[dict]:
    """List committed recording metadata without initializing a store or model."""
    return [
        {"name": path.stem, "title": data["title"], "language": data["language"]}
        for path in sorted(RECORDINGS_DIR.glob("*.json"))
        if _RECORDING_NAME.fullmatch(path.stem)
        for data in [_recording(path.stem)]
    ]


@router.get("/recordings/{name}")
def recording(name: str) -> dict:
    """Serve a validated recording filename, never a caller-supplied path."""
    return _recording(name)


def create_app(settings: Settings) -> FastAPI:
    """Assemble an independently restartable API over injected or shared state."""
    application = FastAPI(openapi_url=None, docs_url=None, redoc_url=None)
    application.state.settings = settings
    # Allocate local memory before concurrent requests can race to create it.
    # Construction performs no store operation; health and recordings remain independent.
    application.state.store = (
        MemoryStore()
        if not settings.on_vercel and not settings.kv_url and not settings.kv_token
        else None
    )
    application.state.now = lambda: datetime.now(UTC)

    @application.middleware("http")
    async def safe_response(request: Request, call_next: Callable) -> Response:
        try:
            response = await call_next(request)
        except StoreUnavailable:
            response = JSONResponse({"error": "unavailable"}, status_code=503)
        except NotPendingApproval:
            response = JSONResponse({"error": "not_pending_approval"}, status_code=409)
        except NotFound:
            response = JSONResponse({"error": "not_found"}, status_code=404)
        except Exception:
            response = JSONResponse({"error": "internal"}, status_code=500)
        response.headers["Cache-Control"] = "no-store"
        return response

    @application.exception_handler(HTTPException)
    async def http_error(request: Request, error: HTTPException) -> JSONResponse:
        return JSONResponse(error.detail, status_code=error.status_code)

    @application.exception_handler(RequestValidationError)
    async def invalid_request(request: Request, error: RequestValidationError) -> JSONResponse:
        return JSONResponse({"error": "request"}, status_code=422)

    application.include_router(router)
    if not settings.on_vercel:
        from starlette.staticfiles import StaticFiles

        application.mount(
            "/", StaticFiles(directory=Path(__file__).resolve().parents[1] / "public", html=True)
        )
    return application


app = create_app(Settings.from_env())
