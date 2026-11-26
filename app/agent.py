"""1. Atomically claim a bounded customer turn with a short lease.
2. Repair interrupted tool requests using persisted refund intents.
3. Save the customer message and check its length before model calls.
4. Reserve, call, settle, execute bounded tools, save and audit each step.
5. Release the turn lease in a finally block, including on failures.
"""

import json
import secrets
import time
from collections.abc import Callable
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from typing import Literal

from app.audit import AuditEvent, AuditLog
from app.budget import Budget
from app.customers import customer_name
from app.dops.api import DopsApi
from app.intents import IntentRepo
from app.model import ModelClient, ModelUnavailable, TextBlock, ToolUseBlock
from app.policy import PolicyConfig
from app.pricing import cost_nano, worst_case_nano
from app.prompt import TOOLS, build_system
from app.store import Store, StoreConflict, StoreUnavailable
from app.tools import ToolContext, ToolExecutor

TEMPORARY_PROBLEM = {
    "en": "There is a temporary problem; please try again shortly.",
    "ar": "في مشكلة مؤقتة؛ حاول تاني بعد شوية.",
}
BUDGET_REACHED = {
    "en": "The demo budget has been reached; please try again later.",
    "ar": "ميزانية التجربة خلصت؛ حاول تاني لاحقًا.",
}
TURN_LIMIT = {
    "en": "This conversation has reached its turn limit; please start a new one.",
    "ar": "المحادثة وصلت للحد المسموح؛ ابدأ محادثة جديدة.",
}
INVALID_MESSAGE = {
    "en": "Please send a message between 1 and 1,000 characters.",
    "ar": "ابعت رسالة من حرف واحد لحد ١٠٠٠ حرف.",
}
BUSY = {
    "en": "A reply is already in progress; please try again shortly.",
    "ar": "جاري الرد على رسالة سابقة؛ حاول تاني بعد شوية.",
}
FALLBACK = {
    "en": "I could not finish that request; please try again.",
    "ar": "مقدرتش أكمل الطلب؛ حاول تاني.",
}

STATUS_WORDS = {
    "en": {
        "SUCCEEDED": "refunded",
        "PENDING_APPROVAL": "awaiting human review",
        "PENDING": "processing, not confirmed yet",
        "FAILED": "declined",
        "REJECTED": "rejected by a reviewer",
    },
    "ar": {
        "SUCCEEDED": "تم الاسترداد",
        "PENDING_APPROVAL": "في انتظار مراجعة موظف",
        "PENDING": "جاري التنفيذ ولسه مش متأكد",
        "FAILED": "اترفض",
        "REJECTED": "اترفض من الموظف",
    },
}
STATUS_HEADER = {
    "en": "Refund system status:",
    "ar": "حالة الاسترداد من النظام:",
}


def status_footer(refunds: list[dict], language: str) -> str:
    """Render the refund system's own status, so model text can never overstate it."""
    latest: dict[int, dict] = {}
    for refund in refunds:
        latest[refund["refund_id"]] = refund
    if not latest:
        return ""
    words = STATUS_WORDS[language]
    lines = [STATUS_HEADER[language]]
    for refund in latest.values():
        status = words.get(refund["status"], refund["status"])
        lines.append(f"- {refund['order_id']} / {refund['amount']}: {status}")
    return "\n".join(lines)


StopReason = Literal[
    "end_turn", "step_limit", "model_error", "refusal", "budget", "turn_limit", "busy"
]
_STORE_ERRORS = (StoreUnavailable, StoreConflict)


class _DeadlineReached(Exception):
    """The turn clock expired between bounded operations."""


@dataclass
class Conversation:
    """Persisted session identity, frozen system context and ordered messages."""

    id: str
    sandbox_id: str
    customer_id: int
    system: list[dict]
    messages: list[dict]
    turns: int
    busy_until: datetime | None


@dataclass(frozen=True)
class AgentConfig:
    """Bounds on work for one customer turn."""

    max_steps: int = 4
    max_tokens: int = 400
    max_turns: int = 8
    max_tools_per_step: int = 3
    turn_deadline_s: float = 45


def _new_trace_id() -> str:
    return "t-" + secrets.token_hex(8)


@dataclass
class AgentDeps:
    """Injected services and clocks for one visitor sandbox."""

    sandbox_id: str
    model: ModelClient
    executor: ToolExecutor
    budget: Budget
    audit: AuditLog
    store: Store
    config: AgentConfig
    now: Callable[[], datetime]
    monotonic: Callable[[], float] = time.monotonic
    new_trace_id: Callable[[], str] = _new_trace_id


@dataclass(frozen=True)
class TurnResult:
    """Customer reply and code-derived refund metadata for status rendering."""

    reply: str
    stopped: StopReason
    trace_id: str
    refunds: list[dict] = field(default_factory=list)


def _key(sandbox_id: str, conversation_id: str) -> str:
    return f"sbx:{sandbox_id}:conv:{conversation_id}"


def _encode(conversation: Conversation) -> dict:
    return {
        **asdict(conversation),
        "busy_until": (
            conversation.busy_until.isoformat() if conversation.busy_until is not None else None
        ),
    }


def _decode(doc: dict) -> Conversation:
    return Conversation(
        **{
            **doc,
            "busy_until": datetime.fromisoformat(doc["busy_until"])
            if doc["busy_until"] is not None
            else None,
        }
    )


def start_conversation(
    store: Store,
    sandbox_id: str,
    customer_id: int,
    dops: DopsApi,
    now: Callable[[], datetime],
) -> Conversation:
    """Build and persist one frozen prompt from the customer's current orders."""
    conversation_id = "c-" + secrets.token_hex(8)
    trace_id = _new_trace_id()
    orders = dops.list_orders(customer_id, trace_id)
    conversation = Conversation(
        conversation_id,
        sandbox_id,
        customer_id,
        build_system(customer_name(customer_id), orders, PolicyConfig(), now()),
        [],
        0,
        None,
    )
    doc = store.update_doc(_key(sandbox_id, conversation_id), lambda _: _encode(conversation), {})
    return _decode(doc)


def load_conversation(store: Store, sandbox_id: str, conversation_id: str) -> Conversation | None:
    """Load a conversation only from the requested visitor sandbox."""
    doc = store.get_doc(_key(sandbox_id, conversation_id))
    return None if doc is None else _decode(doc)


def _tool_result(tool_use_id: str, content: str, is_error: bool = False) -> dict:
    return {
        "type": "tool_result",
        "tool_use_id": tool_use_id,
        "content": content,
        "is_error": is_error,
    }


def _repair(
    conversation: Conversation, deps: AgentDeps, trace: str, deadline: float, refunds: list[dict]
) -> None:
    """Refresh uncertain tool results and complete a final interrupted tool step."""
    repo = IntentRepo(deps.store, conversation.sandbox_id)
    intents, aliases = repo.snapshot(conversation.id)
    messages = conversation.messages
    pending = {key: intent for key, intent in intents.items() if intent.status != "SUBMITTED"}
    # Ops can confirm an intent before chat resumes. In that case the tool history
    # still needs the real status once, including for an interrupted tool call.
    result_blocks = {
        b["tool_use_id"]: b for m in messages for b in m["content"] if b["type"] == "tool_result"
    }
    for message in messages:
        for use in message["content"]:
            if use["type"] != "tool_use" or use["name"] != "request_refund":
                continue
            intent = intents.get(aliases.get(use["id"]))
            if intent is None or intent.status != "SUBMITTED":
                continue
            block = result_blocks.get(use["id"])
            if block is None or json.loads(block["content"]).get("status") in {
                "DECIDED",
                "UNKNOWN",
            }:
                pending[intent.key] = intent
    found = deps.executor.reconcile(
        repo,
        conversation.sandbox_id,
        list(pending.values()),
        trace,
        near_deadline=lambda: deps.monotonic() + 5 >= deadline,
    )
    for refund in found.values():
        refunds.append(
            {
                "refund_id": refund.id,
                "order_id": refund.order_id,
                "amount": refund.amount,
                "status": refund.status,
            }
        )
    updates = {
        i.key: {
            "result": STATUS_WORDS["en"][found[i.key].status]
            if i.key in found
            else "request not confirmed; it may be requested again",
            "status": found[i.key].status if i.key in found else "UNKNOWN",
            "order_id": i.order_id,
            "amount": i.amount,
        }
        for i in pending.values()
    }
    for use_id, block in result_blocks.items():
        key = aliases.get(use_id)
        if key in found and json.loads(block["content"]).get("status") in {"DECIDED", "UNKNOWN"}:
            block.update(content=json.dumps(updates[key]), is_error=False)
    if not messages:
        return
    assistant = len(messages) - 1
    if messages[assistant]["role"] == "user":
        assistant -= 1
    if assistant < 0 or messages[assistant]["role"] != "assistant":
        return
    uses = [b for b in messages[assistant]["content"] if b["type"] == "tool_use"]
    if not uses:
        return
    existing = messages[assistant + 1]["content"] if assistant + 1 < len(messages) else []
    answered = {b["tool_use_id"]: b for b in existing if b["type"] == "tool_result"}
    results = []
    for use in uses:
        if use["id"] in answered:
            results.append(answered[use["id"]])
            continue
        body = updates.get(aliases.get(use["id"])) if use["name"] == "request_refund" else None
        missing = body is None
        if missing:
            body = {"result": "interrupted, please retry"}
        results.append(_tool_result(use["id"], json.dumps(body), missing))
    if assistant + 1 == len(messages):
        messages.append({"role": "user", "content": results})
    else:
        messages[assistant + 1]["content"] = results + [
            b for b in existing if b["type"] != "tool_result"
        ]


def run_customer_turn(conversation_id: str, message: str, deps: AgentDeps) -> TurnResult:
    """Run one claimed turn, failing closed on model or shared-state errors."""
    trace = deps.new_trace_id()
    language = "ar" if any("\u0600" <= char <= "\u06ff" for char in message) else "en"
    sandbox = deps.sandbox_id
    key = _key(sandbox, conversation_id)
    deadline = deps.monotonic() + deps.config.turn_deadline_s
    events: list[AuditEvent] = []
    refunds: list[dict] = []
    conversation: Conversation | None = None
    claimed = False
    branch = "missing"

    def finish(reply: str, stopped: StopReason) -> TurnResult:
        # Code-derived status is appended whenever a refund was touched this turn.
        footer = status_footer(refunds, language)
        text = f"{reply}\n\n{footer}" if footer else reply
        return TurnResult(text, stopped, trace, refunds)

    def result(texts: dict[str, str], stopped: StopReason) -> TurnResult:
        return finish(texts[language], stopped)

    def event(
        kind: str, data: dict, actor: Literal["system", "customer", "model"] = "system"
    ) -> None:
        events.append(AuditEvent(deps.now(), trace, conversation_id, actor, kind, data))

    def incident(kind: str, text: str) -> None:
        event("incident", {"kind": kind, "message": text})

    def check_deadline() -> None:
        if deps.monotonic() >= deadline:
            raise _DeadlineReached

    def flush() -> None:
        try:
            deps.audit.record_many([*events, *deps.executor.pending_events])
        finally:
            events.clear()
            deps.executor.pending_events.clear()

    def save() -> None:
        def mutate(doc: dict) -> dict:
            if not doc or (
                doc["busy_until"] != conversation.busy_until.isoformat()
                or doc["turns"] != conversation.turns
            ):
                raise StoreConflict("Conversation lease changed")
            return _encode(conversation)

        deps.store.update_doc(key, mutate, {})

    def claim(doc: dict) -> dict:
        nonlocal branch
        current = deps.now()
        if not doc:
            branch = "missing"
        elif doc["busy_until"] and datetime.fromisoformat(doc["busy_until"]) > current:
            branch = "busy"
        elif doc["turns"] >= deps.config.max_turns:
            branch = "turn_limit"
        else:
            branch = "claimed"
            doc["busy_until"] = (current + timedelta(seconds=50)).isoformat()
            doc["turns"] += 1
        return doc

    try:
        doc = deps.store.update_doc(key, claim, {})
        if branch == "busy":
            return result(BUSY, "busy")
        if branch == "turn_limit":
            return result(TURN_LIMIT, "turn_limit")
        if branch == "missing":
            return result(TEMPORARY_PROBLEM, "model_error")
        claimed = True
        conversation = _decode(doc)
        _repair(conversation, deps, trace, deadline, refunds)
        event("customer_message", {"length": len(message), "language": language}, "customer")
        if not message.strip() or len(message) > 1000:
            # A refused message never enters the history, so it can never reach the model.
            save()
            flush()
            return result(INVALID_MESSAGE, "end_turn")
        conversation.messages.append(
            {"role": "user", "content": [{"type": "text", "text": message}]}
        )
        save()
        reply = FALLBACK[language]
        for _ in range(deps.config.max_steps):
            check_deadline()
            request = deepcopy(
                {
                    "model": "claude-haiku-5-5",
                    "max_tokens": deps.config.max_tokens,
                    "system": conversation.system,
                    "tools": TOOLS,
                    "messages": conversation.messages,
                    "output_config": {"effort": "low"},
                }
            )
            worst = worst_case_nano(
                len(json.dumps(request, ensure_ascii=False).encode()), deps.config.max_tokens
            )
            if not deps.budget.reserve(worst):
                event("budget_refused", {})
                flush()
                return result(BUDGET_REACHED, "budget")
            if deps.monotonic() >= deadline:
                deps.budget.settle(worst, 0)
                raise _DeadlineReached
            response = deps.model.create(request)
            actual = cost_nano(response.usage)
            event(
                "model_call",
                {
                    "usage": asdict(response.usage),
                    "cost_nano": actual,
                    "stop_reason": response.stop_reason,
                },
                "model",
            )
            deps.budget.settle(worst, actual)
            check_deadline()
            reply = "\n".join(b.text for b in response.content if isinstance(b, TextBlock))
            reply = reply or FALLBACK[language]
            if response.stop_reason == "refusal":
                incident("refusal", "The model refused the request")
                flush()
                return result(TEMPORARY_PROBLEM, "refusal")
            uses = [b for b in response.content if isinstance(b, ToolUseBlock)]
            content = [
                {"type": "text", "text": b.text}
                if isinstance(b, TextBlock)
                else {"type": "tool_use", "id": b.id, "name": b.name, "input": b.input}
                for b in response.content
            ]
            if response.stop_reason == "max_tokens":
                if not uses:
                    conversation.messages.append({"role": "assistant", "content": content})
                    save()
                flush()
                return finish(reply, "end_turn")
            conversation.messages.append({"role": "assistant", "content": content})
            # Persist tool ids before any side effects so interruption can be repaired.
            save()
            if response.stop_reason == "tool_use":
                results = []
                for index, use in enumerate(uses):
                    check_deadline()
                    if index >= deps.config.max_tools_per_step:
                        results.append(
                            _tool_result(
                                use.id,
                                json.dumps({"result": "too many tool calls in one step"}),
                                True,
                            )
                        )
                        continue
                    ctx = ToolContext(
                        sandbox, conversation_id, conversation.customer_id, trace, use.id
                    )
                    outcome = deps.executor.execute(ctx, use.name, use.input)
                    if outcome.refund_id is not None:
                        body = json.loads(outcome.content)
                        refunds.append(
                            {
                                "refund_id": outcome.refund_id,
                                "status": outcome.refund_status,
                                "order_id": body.get("order_id"),
                                "amount": body.get("amount"),
                            }
                        )
                    # The executor buffers store incidents instead of raising them.
                    if any(
                        e.type == "incident"
                        and e.data.get("kind") in {"StoreUnavailable", "StoreConflict"}
                        for e in deps.executor.pending_events
                    ):
                        raise StoreUnavailable("Tool state could not be persisted")
                    results.append(_tool_result(use.id, outcome.content, outcome.is_error))
                conversation.messages.append({"role": "user", "content": results})
                save()
            check_deadline()
            flush()
            if response.stop_reason != "tool_use":
                return finish(reply, "end_turn")
        return finish(reply, "step_limit")
    except (ModelUnavailable, StoreUnavailable, StoreConflict, _DeadlineReached) as error:
        incident(
            "deadline"
            if isinstance(error, _DeadlineReached)
            else "model"
            if isinstance(error, ModelUnavailable)
            else "store",
            "The turn deadline was reached"
            if isinstance(error, _DeadlineReached)
            else "The turn could not be completed",
        )
        try:
            flush()
        except _STORE_ERRORS:
            pass
        return result(TEMPORARY_PROBLEM, "model_error")
    finally:
        if claimed and conversation is not None:

            def release(doc: dict) -> dict:
                if (
                    doc
                    and doc["busy_until"] == conversation.busy_until.isoformat()
                    and doc["turns"] == conversation.turns
                ):
                    doc["busy_until"] = None
                return doc

            try:
                deps.store.update_doc(key, release, {})
            except _STORE_ERRORS:
                incident("release", "The turn lease could not be released")
                try:
                    flush()
                except _STORE_ERRORS:
                    pass
