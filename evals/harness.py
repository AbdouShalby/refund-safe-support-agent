"""Assemble isolated eval stacks with a shared clock, policy and state inspection."""

import json
import os
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from app.agent import (
    AgentConfig,
    AgentDeps,
    TurnResult,
    load_conversation,
    run_customer_turn,
    start_conversation,
)
from app.audit import AuditEvent, AuditLog
from app.budget import Budget
from app.dops.api import DopsApi
from app.dops.http import HttpDopsClient
from app.escalations import EscalationRepo
from app.model import ModelClient
from app.policy import PolicyConfig
from app.sandbox_dops import SandboxDops
from app.store import MemoryStore
from app.tools import ToolExecutor, http_lock_scope, mock_lock_scope

RUN_CAP_NANO = 500_000_000
INSPECTION_TRACE = "t-ffffffffffffffff"


@dataclass
class EvalStack:
    """Own one run's services and conversation, independent of all other runs."""

    mode: Literal["mock", "http"]
    sandbox_id: str
    store: MemoryStore
    dops: DopsApi
    deps: AgentDeps
    now: Callable[[], datetime]
    seed_refund_ids: list[int]
    conversation_id: str | None = None

    def turn(
        self, customer_id: int, message: str, conversation_id: str | None = None
    ) -> TurnResult:
        """Start a conversation or continue an explicitly supplied conversation."""
        if conversation_id is None:
            conversation = start_conversation(
                self.store, self.sandbox_id, customer_id, self.dops, self.now
            )
            conversation_id = conversation.id
        else:
            conversation = load_conversation(self.store, self.sandbox_id, conversation_id)
            if conversation is None or conversation.customer_id != customer_id:
                raise ValueError("Conversation does not belong to this customer")
        self.conversation_id = conversation_id
        return run_customer_turn(conversation_id, message, self.deps)

    def dops_state(self) -> dict:
        """Reconstruct HTTP state through public customer-scoped APIs, as in parity tests."""
        if self.mode == "mock":
            self.dops.list_orders(1, INSPECTION_TRACE)
            state = self.store.get_doc(f"sbx:{self.sandbox_id}:dops")
            if state is None:
                raise RuntimeError("Mock state was not initialized")
        else:
            seed = json.loads((Path(__file__).parents[1] / "app/dops/seed.json").read_text("utf-8"))
            orders, refunds = [], []
            for customer in seed["customers"]:
                for order in self.dops.list_orders(customer["id"], INSPECTION_TRACE):
                    orders.append(order.model_dump(mode="json"))
                    refunds.extend(
                        refund.model_dump(mode="json")
                        for refund in self.dops.list_order_refunds(
                            customer["id"], order.id, INSPECTION_TRACE
                        )
                    )
            state = {"customers": seed["customers"], "orders": orders, "refunds": refunds}
        return {**state, "_seed_refund_ids": self.seed_refund_ids}

    def events(self) -> list[AuditEvent]:
        """Return typed copies of the complete run audit trail."""
        return [
            AuditEvent(**{**row, "ts": datetime.fromisoformat(row["ts"])})
            for row in self.deps.audit.list()
        ]

    def escalations(self) -> list[dict]:
        """Return all handoffs created in this run."""
        return EscalationRepo(self.store, self.sandbox_id).list()

    def close(self) -> None:
        """Release the HTTP connection pool when the run ends."""
        if isinstance(self.dops, HttpDopsClient):
            self.dops.close()


def build_stack(
    mode: Literal["mock", "http"],
    model: ModelClient,
    *,
    sandbox_id: str,
    reset_cmd: str | None = None,
) -> EvalStack:
    """Reset only explicitly configured eval infrastructure; never silently fall back."""
    if mode not in {"mock", "http"}:
        raise ValueError("DOPS mode must be mock or http")
    store = MemoryStore()

    def now() -> datetime:
        return datetime.now(UTC)

    if mode == "http":
        required = ("DOPS_BASE_URL", "DOPS_SERVICE_TOKEN")
        missing = [name for name in required if not os.environ.get(name, "").strip()]
        if missing or not reset_cmd or not reset_cmd.strip():
            raise ValueError("HTTP evals require DOPS_BASE_URL, DOPS_SERVICE_TOKEN and reset_cmd")
        # Only explicitly configured local test infrastructure may supply this command.
        subprocess.run(reset_cmd, shell=True, check=True, timeout=120)  # noqa: S602
        dops: DopsApi = HttpDopsClient(
            os.environ["DOPS_BASE_URL"], os.environ["DOPS_SERVICE_TOKEN"]
        )
    else:
        dops = SandboxDops(store, sandbox_id, now)
    audit = AuditLog(store, sandbox_id)
    executor = ToolExecutor(
        dops,
        store,
        audit,
        EscalationRepo(store, sandbox_id),
        PolicyConfig(),
        now,
        http_lock_scope if mode == "http" else mock_lock_scope,
    )
    deps = AgentDeps(
        sandbox_id,
        model,
        executor,
        Budget(store, RUN_CAP_NANO, lambda: now().date()),
        audit,
        store,
        AgentConfig(),
        now,
    )
    stack = EvalStack(mode, sandbox_id, store, dops, deps, now, [])
    try:
        stack.seed_refund_ids = [refund["id"] for refund in stack.dops_state()["refunds"]]
    except BaseException:
        stack.close()
        raise
    return stack
