"""1. Validate model input and bind the customer from the session context.
2. Hold the customer lock while reading facts and applying refund policy.
3. Persist a stable intent before calling DOPS with its idempotency key.
4. Replay stored requests and report the actual DOPS status truthfully.
5. Buffer sanitized audit events for the caller to flush once per step.
"""

import json
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timedelta
from decimal import Decimal
from typing import TypeVar

from app.audit import AuditEvent, AuditLog
from app.dops.api import DopsApi
from app.dops.errors import (
    DopsError,
    DopsUnauthorized,
    DopsUnavailable,
    IdempotencyConflict,
    NotFound,
    NotRefundable,
    RefundExceedsPaid,
    ValidationFailed,
)
from app.dops.models import CreateRefundResult, Order, Refund, RefundMode
from app.escalations import EscalationRepo
from app.intents import IntentRepo, RefundIntent, intent_key
from app.money import fmt_amount, parse_amount
from app.policy import PolicyConfig, RefundFacts, decide
from app.render import order_line
from app.store import LockBusy, Store, StoreConflict, StoreUnavailable

_T = TypeVar("_T")
_TOOLS = {"get_order", "request_refund", "escalate_to_human"}
_WORDING = {
    "SUCCEEDED": "refunded",
    "PENDING_APPROVAL": "awaiting human review",
    "PENDING": "processing, not confirmed",
    "FAILED": "declined",
    "REJECTED": "rejected by a reviewer",
}


@dataclass(frozen=True)
class ToolContext:
    """Trusted session identity and identifiers for this tool invocation."""

    sandbox_id: str
    conversation_id: str
    customer_id: int
    trace_id: str
    tool_use_id: str


@dataclass(frozen=True)
class ToolOutcome:
    """Model-visible JSON plus code-derived refund metadata for the caller."""

    content: str
    is_error: bool = False
    refund_id: int | None = None
    refund_status: str | None = None


def mock_lock_scope(ctx: ToolContext) -> str:
    """Serialize a simulated customer's requests within its visitor sandbox."""
    return f"sbx:{ctx.sandbox_id}:cust:{ctx.customer_id}"


def http_lock_scope(ctx: ToolContext) -> str:
    """Serialize a DOPS customer across all visitor sandboxes."""
    return f"cust:{ctx.customer_id}"


def _outcome(
    result: str,
    status: str,
    amount: str | None = None,
    order_id: int | None = None,
    *,
    reasons: tuple[str, ...] | None = None,
    is_error: bool = False,
    refund: Refund | None = None,
) -> ToolOutcome:
    body = {"result": result, "status": status, "amount": amount, "order_id": order_id}
    if reasons is not None:
        body["reasons"] = list(reasons)
    return ToolOutcome(
        json.dumps(body, separators=(",", ":")),
        is_error,
        None if refund is None else refund.id,
        None if refund is None else refund.status,
    )


def _validate(name: str, tool_input: object) -> dict:
    if not isinstance(tool_input, dict):
        raise ValueError("Invalid input")
    keys = {
        "get_order": {"order_id"},
        "request_refund": {"order_id", "amount", "reason"},
        "escalate_to_human": {"summary"},
    }[name]
    if tool_input.keys() != keys:
        raise ValueError("Invalid fields")
    data = dict(tool_input)
    if "order_id" in keys and (type(data["order_id"]) is not int or data["order_id"] <= 0):
        raise ValueError("Invalid order")
    for field, maximum in (("reason", 300), ("summary", 500)):
        if field in keys:
            value = data[field]
            if not isinstance(value, str) or not value.strip() or len(value) > maximum:
                raise ValueError("Invalid text")
    if "amount" in keys:
        data["amount"] = fmt_amount(parse_amount(data["amount"]))
    return data


class ToolExecutor:
    """Enforce policy and idempotency without exposing any approval tool.

    A definitive NotRefundable/RefundExceedsPaid write failure deletes the
    intent and its tool-use aliases, allowing the next identical request to
    gather fresh facts and re-decide. Ambiguous writes retain the original key
    and attempts for reconciliation. Freeform text and exception messages are
    deliberately omitted from audit data so they cannot leak credentials.
    """

    def __init__(
        self,
        dops: DopsApi,
        store: Store,
        audit: AuditLog,
        escalations: EscalationRepo,
        config: PolicyConfig,
        now: Callable[[], datetime],
        lock_scope: Callable[[ToolContext], str],
    ) -> None:
        self.dops = dops
        self.store = store
        self.audit = audit
        self.escalations = escalations
        self.config = config
        self.now = now
        self.lock_scope = lock_scope
        self.pending_events: list[AuditEvent] = []

    def _event(self, ctx: ToolContext, kind: str, data: dict) -> None:
        actor = (
            "model" if kind == "tool_call" else "policy" if kind == "policy_decision" else "system"
        )
        self.pending_events.append(
            AuditEvent(self.now(), ctx.trace_id, ctx.conversation_id, actor, kind, data)
        )

    def _incident(self, ctx: ToolContext, kind: str, message: str) -> None:
        self._event(ctx, "incident", {"kind": kind, "message": message})

    def _call(self, ctx: ToolContext, operation: str, call: Callable[[], _T]) -> _T:
        try:
            result = call()
        except DopsError as error:
            status = {
                NotFound: "4xx",
                DopsUnauthorized: "4xx",
                IdempotencyConflict: "4xx",
                NotRefundable: "4xx",
                RefundExceedsPaid: "4xx",
                ValidationFailed: "4xx",
            }.get(type(error), "unknown")
            self._event(ctx, "dops_call", {"operation": operation, "status": status})
            raise
        status = result.refund.status if isinstance(result, CreateRefundResult) else "2xx"
        self._event(ctx, "dops_call", {"operation": operation, "status": status})
        return result

    def execute(self, ctx: ToolContext, name: str, tool_input: object) -> ToolOutcome:
        """Validate before any I/O; retain audit events in memory for the loop."""
        if name not in _TOOLS:
            self._event(ctx, "tool_call", {"name": "unknown tool", "input": {}})
            return _outcome("unknown tool", "ERROR", is_error=True)
        try:
            data = _validate(name, tool_input)
        except ValueError:
            self._event(ctx, "tool_call", {"name": name, "input": {"invalid": True}})
            return _outcome("invalid tool input", "ERROR", is_error=True)
        sanitized = {
            key: "[redacted]" if key in {"reason", "summary"} else value
            for key, value in data.items()
        }
        self._event(ctx, "tool_call", {"name": name, "input": sanitized})
        outcome = None
        try:
            if name == "get_order":
                order = self._call(
                    ctx,
                    "get_order",
                    lambda: self.dops.get_order(ctx.customer_id, data["order_id"], ctx.trace_id),
                )
                return _outcome(
                    order_line(order, self.now()), order.status, order.total_amount, order.id
                )
            if name == "escalate_to_human":
                self.escalations.open(ctx.conversation_id, ctx.customer_id, data["summary"])
                self._event(ctx, "escalation", {"status": "OPEN"})
                return _outcome("escalated to a human", "OPEN")
            with self.store.lock(self.lock_scope(ctx), 30):
                outcome = self._request(ctx, data)
            return outcome
        except (LockBusy, StoreUnavailable, StoreConflict) as error:
            self._incident(ctx, type(error).__name__, "Shared state is busy or unavailable")
            # A lease-release error must not hide a known or ambiguous result.
            if outcome is not None:
                return outcome
            return _outcome(
                "busy, try again", "ERROR", data.get("amount"), data.get("order_id"), is_error=True
            )
        except NotFound:
            return _outcome(
                "order not found", "ERROR", data.get("amount"), data.get("order_id"), is_error=True
            )
        except DopsError as error:
            self._incident(ctx, type(error).__name__, "DOPS could not complete the operation")
            return _outcome(
                "temporary problem",
                "ERROR",
                data.get("amount"),
                data.get("order_id"),
                is_error=True,
            )

    def _order_refunds(self, ctx: ToolContext, order_id: int) -> list[Refund]:
        return self._call(
            ctx,
            "list_order_refunds",
            lambda: self.dops.list_order_refunds(ctx.customer_id, order_id, ctx.trace_id),
        )

    def _facts(self, ctx: ToolContext, order: Order, order_refunds: list[Refund]) -> RefundFacts:
        current = self.now()
        since = current - timedelta(hours=24)
        customer_refunds = self._call(
            ctx,
            "customer_refunds_since",
            lambda: self.dops.customer_refunds_since(ctx.customer_id, since, ctx.trace_id),
        )
        auto = [
            refund
            for refund in customer_refunds
            if (
                refund.created_at >= since
                and refund.mode == "auto"
                and refund.decided_by is None
                and refund.status in {"SUCCEEDED", "PENDING"}
            )
        ]
        return RefundFacts(
            order.status,
            Decimal(order.total_amount),
            Decimal(order.reserved_amount),
            (current - order.created_at).days,
            sum(refund.created_at >= since for refund in order_refunds),
            len(auto),
            sum((Decimal(refund.amount) for refund in auto), Decimal("0.00")),
        )

    def _adopt(
        self, ctx: ToolContext, repo: IntentRepo, intent: RefundIntent, refund: Refund
    ) -> RefundIntent | None:
        current, adopted = repo.adopt(intent.key, refund)
        if adopted:
            self._event(ctx, "intent_adopted", {"refund_id": refund.id, "status": refund.status})
        if current and current.refund_id == refund.id and current.status == "SUBMITTED":
            return current
        return None

    def _match(
        self, ctx: ToolContext, repo: IntentRepo, intent: RefundIntent, refunds: list[Refund]
    ) -> tuple[Refund, RefundIntent] | None:
        traces = {trace for trace, _, _ in intent.attempts}
        for refund in refunds:
            if (
                refund.order_id != intent.order_id
                or refund.customer_id != intent.customer_id
                or refund.amount != intent.amount
            ):
                continue
            if intent.refund_id is not None:
                if refund.id != intent.refund_id:
                    continue
                if intent.status == "SUBMITTED":
                    return refund, intent
            elif refund.trace_id not in traces:
                continue
            if current := self._adopt(ctx, repo, intent, refund):
                return refund, current
        return None

    def _unconfirmed(self, ctx: ToolContext, repo: IntentRepo, intent: RefundIntent) -> None:
        if repo.note_unconfirmed(intent.key, ctx.trace_id):
            self._event(
                ctx, "intent_unconfirmed", {"order_id": intent.order_id, "amount": intent.amount}
            )

    def reconcile(
        self,
        repo: IntentRepo,
        sandbox: str,
        intents: list[RefundIntent],
        trace: str,
        *,
        near_deadline: Callable[[], bool] = lambda: False,
        audit_unconfirmed: bool = True,
    ) -> dict[str, Refund]:
        """Read each order once under its customer's nonblocking lock. Never submit."""
        groups: dict[tuple[int, int], list[RefundIntent]] = {}
        for intent in intents:
            groups.setdefault((intent.customer_id, intent.order_id), []).append(intent)
        found = {}
        for (customer, order), group in groups.items():
            ctx = ToolContext(sandbox, group[0].conversation_id, customer, trace, "")
            if not near_deadline():
                try:
                    with self.store.lock(self.lock_scope(ctx), 30):
                        rows = self._order_refunds(ctx, order)
                        for intent in group:
                            ctx = replace(ctx, conversation_id=intent.conversation_id)
                            if match := self._match(ctx, repo, intent, rows):
                                found[intent.key] = match[0]
                            elif (
                                audit_unconfirmed
                                and len(intent.attempts) > intent.unconfirmed_attempt
                            ):
                                self._unconfirmed(ctx, repo, intent)
                except (DopsError, LockBusy) as error:
                    self._incident(ctx, type(error).__name__, "Refund lookup could not complete")
        return found

    def _request(self, ctx: ToolContext, data: dict) -> ToolOutcome:
        repo = IntentRepo(self.store, ctx.sandbox_id)
        key = intent_key(ctx.conversation_id, data["order_id"], data["amount"])
        intent = repo.get(key)
        order_refunds = None
        if intent is not None:
            intent, _ = repo.create(replace(intent, tool_use_id=ctx.tool_use_id))
            if intent.status == "SUBMITTED":
                return self._submit(ctx, repo, intent)
            order_refunds = self._order_refunds(ctx, intent.order_id)
            if match := self._match(ctx, repo, intent, order_refunds):
                refund, current = match
                if refund.status == "PENDING":
                    return self._submit(ctx, repo, current)
                return self._refund_outcome(refund, current, same=True)

        order = self._call(
            ctx,
            "get_order",
            lambda: self.dops.get_order(ctx.customer_id, data["order_id"], ctx.trace_id),
        )
        if order_refunds is None:
            order_refunds = self._order_refunds(ctx, order.id)
        facts = self._facts(ctx, order, order_refunds)
        decision = decide(facts, parse_amount(data["amount"]), self.config)
        self._event(
            ctx,
            "policy_decision",
            {
                "facts": {field: str(value) for field, value in asdict(facts).items()},
                "outcome": decision.outcome,
                "reasons": list(decision.reasons),
            },
        )
        if intent is not None:
            self._event(
                ctx,
                "intent_redecided",
                {
                    "old_mode": intent.mode,
                    "new_mode": decision.outcome,
                    "reasons": list(decision.reasons),
                },
            )
        if decision.outcome == "reject":
            if intent is not None:
                self._unconfirmed(ctx, repo, intent)
                return _outcome(
                    "earlier request not confirmed; current rules do not allow a new request",
                    "UNKNOWN",
                    intent.amount,
                    order.id,
                    reasons=decision.reasons,
                )
            return _outcome(
                "not possible", "REJECTED", data["amount"], order.id, reasons=decision.reasons
            )
        retry = intent is not None
        if intent is None:
            intent, _ = repo.create(
                RefundIntent(
                    key,
                    ctx.conversation_id,
                    ctx.customer_id,
                    order.id,
                    data["amount"],
                    data["reason"],
                    decision.outcome,
                    decision.reasons,
                    "DECIDED",
                    None,
                    ctx.tool_use_id,
                )
            )
        return self._submit(
            ctx,
            repo,
            replace(intent, mode=decision.outcome, decision_reasons=decision.reasons),
            retry=retry,
        )

    def _write(
        self, ctx: ToolContext, repo: IntentRepo, intent: RefundIntent, mode: RefundMode
    ) -> CreateRefundResult:
        repo.attempt(intent, ctx.trace_id, mode)
        return self._call(
            ctx,
            "create_refund",
            lambda: self.dops.create_refund(
                ctx.customer_id,
                intent.order_id,
                intent.amount,
                intent.reason,
                intent.key,
                mode,
                ctx.trace_id,
            ),
        )

    def _recover_conflict(
        self,
        ctx: ToolContext,
        repo: IntentRepo,
        intent: RefundIntent,
    ) -> ToolOutcome:
        # A conflict proves the key exists and the attempted fingerprint differs.
        # DOPS has only two modes, even when the earlier trace was never stored.
        earlier_mode: RefundMode = "auto" if intent.mode == "needs_approval" else "needs_approval"
        try:
            replay = self._write(ctx, repo, intent, earlier_mode)
        except DopsError:
            pass
        else:
            if current := self._adopt(ctx, repo, intent, replay.refund):
                return self._refund_outcome(replay.refund, current, same=True)
        self._incident(ctx, "IdempotencyConflict", "Conflicting submission could not be resolved")
        self._unconfirmed(ctx, repo, intent)
        return _outcome(
            "request not confirmed; escalate",
            "UNKNOWN",
            intent.amount,
            intent.order_id,
            is_error=True,
        )

    def _submit(
        self,
        ctx: ToolContext,
        repo: IntentRepo,
        intent: RefundIntent,
        *,
        retry: bool = False,
    ) -> ToolOutcome:
        confirmed = intent.status == "SUBMITTED"
        try:
            result = self._write(ctx, repo, intent, intent.mode)
        except DopsUnavailable:
            self._incident(ctx, "DopsUnavailable", "Refund submission outcome is not confirmed")
            try:
                repo.mark(intent.key, status="UNKNOWN")
                self._unconfirmed(ctx, repo, intent)
            except (StoreUnavailable, StoreConflict) as error:
                # The already-persisted DECIDED intent still makes a retry safe.
                self._incident(ctx, type(error).__name__, "Unknown outcome could not be persisted")
            return _outcome(
                "submitted, outcome not confirmed",
                "UNKNOWN",
                intent.amount,
                intent.order_id,
                reasons=intent.decision_reasons,
            )
        except (NotRefundable, RefundExceedsPaid) as error:
            self._incident(
                ctx, type(error).__name__, "DOPS refused the refund after the policy decision"
            )
            if retry:
                self._unconfirmed(ctx, repo, intent)
                return _outcome(
                    "earlier request not confirmed; new submission refused",
                    "UNKNOWN",
                    intent.amount,
                    intent.order_id,
                )
            repo._delete(intent.key)
            reason = (
                "ORDER_NOT_REFUNDABLE"
                if isinstance(error, NotRefundable)
                else "REFUND_EXCEEDS_PAID"
            )
            return _outcome(
                "not possible", "REJECTED", intent.amount, intent.order_id, reasons=(reason,)
            )
        except NotFound:
            if not retry:
                repo._delete(intent.key)
            return _outcome(
                "order not found", "ERROR", intent.amount, intent.order_id, is_error=True
            )
        except IdempotencyConflict:
            if confirmed:
                self._incident(ctx, "IdempotencyConflict", "DOPS refused the refund submission")
                return _outcome(
                    "could not submit, escalate",
                    "ERROR",
                    intent.amount,
                    intent.order_id,
                    is_error=True,
                )
            return self._recover_conflict(ctx, repo, intent)
        except (ValidationFailed, DopsUnauthorized) as error:
            self._incident(ctx, type(error).__name__, "DOPS refused the refund submission")
            text = (
                "temporary problem"
                if isinstance(error, DopsUnauthorized)
                else "could not submit, escalate"
            )
            return _outcome(text, "ERROR", intent.amount, intent.order_id, is_error=True)
        refund = result.refund
        self._event(
            ctx,
            "refund_result",
            {
                "refund_id": refund.id,
                "status": refund.status,
                "replayed": result.replayed,
            },
        )
        try:
            if result.replayed and not confirmed:
                current = self._adopt(ctx, repo, intent, refund)
                if not current:
                    self._unconfirmed(ctx, repo, intent)
                    return _outcome(
                        "request not confirmed", "UNKNOWN", intent.amount, intent.order_id
                    )
                intent = current
            else:
                repo.mark(intent.key, status="SUBMITTED", refund_id=refund.id)
        except (StoreUnavailable, StoreConflict) as error:
            # The DECIDED intent still carries the same key if acknowledgement fails.
            self._incident(ctx, type(error).__name__, "Refund result could not be persisted")
        return self._refund_outcome(refund, intent, same=confirmed or result.replayed)

    @staticmethod
    def _refund_outcome(refund: Refund, intent: RefundIntent, *, same: bool) -> ToolOutcome:
        wording = (
            f"same refund as before, no additional money moved, current status {refund.status}"
            if same
            else _WORDING[refund.status]
        )
        return _outcome(
            wording,
            refund.status,
            refund.amount,
            refund.order_id,
            reasons=intent.decision_reasons,
            refund=refund,
        )
