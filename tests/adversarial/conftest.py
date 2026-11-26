"""Build isolated full-stack attack rigs, optionally using explicitly reset local DOPS."""

import json
import os
import subprocess
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import pytest

from app.agent import (
    AgentConfig,
    AgentDeps,
    Conversation,
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
from app.model import ModelResponse, ScriptedModel
from app.policy import PolicyConfig
from app.sandbox_dops import SandboxDops
from app.store import MemoryStore
from app.tools import ToolExecutor, http_lock_scope, mock_lock_scope

SANDBOX = "adversarial"
INSPECTION_TRACE = "t-ffffffffffffffff"
FIXED_NOW = datetime(2026, 11, 18, 12, tzinfo=UTC)


def pytest_configure(config: pytest.Config) -> None:
    """The repository root already registers the shared --real-dops option."""
    config.addinivalue_line("markers", "real_dops_ok: attack supports reset local HTTP DOPS")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Restrict this suite's real runs to attacks with an HTTP-compatible setup."""
    if not config.getoption("--real-dops"):
        return
    for item in items:
        if Path(__file__).parent in item.path.parents and not item.get_closest_marker(
            "real_dops_ok"
        ):
            item.add_marker(
                pytest.mark.skip(reason="Attack requires mock-only setup; not marked real_dops_ok")
            )


@pytest.fixture
def store() -> MemoryStore:
    """Share one real, thread-safe store across all conversations of an attack."""
    return MemoryStore()


@pytest.fixture
def now(request: pytest.FixtureRequest) -> Callable[[], datetime]:
    """Use wall time only for HTTP seed ages and recent-refund queries."""
    if request.config.getoption("--real-dops"):
        return lambda: datetime.now(UTC)
    return lambda: FIXED_NOW


@pytest.fixture
def dops(
    request: pytest.FixtureRequest, store: MemoryStore, now: Callable[[], datetime]
) -> Iterator[DopsApi]:
    """Never silently fall back if a requested real-DOPS configuration is missing."""
    if not request.config.getoption("--real-dops"):
        yield SandboxDops(store, SANDBOX, now, seed_options=getattr(request, "param", None))
        return
    required = ("DOPS_BASE_URL", "DOPS_SERVICE_TOKEN", "DOPS_RESET_CMD")
    missing = [name for name in required if not os.environ.get(name, "").strip()]
    if missing:
        pytest.fail("--real-dops requires environment variables: " + ", ".join(missing))
    # Only explicitly configured local test infrastructure may supply this command.
    subprocess.run(os.environ["DOPS_RESET_CMD"], shell=True, check=True, timeout=120)  # noqa: S602
    client = HttpDopsClient(os.environ["DOPS_BASE_URL"], os.environ["DOPS_SERVICE_TOKEN"])
    try:
        yield client
    finally:
        client.close()


@dataclass
class AttackSession:
    """One trusted conversation, with a private executor and scripted model per turn."""

    conversation: Conversation
    deps: AgentDeps

    def turn(
        self, steps: Sequence[ModelResponse], message: str = "Please process my refund request."
    ) -> TurnResult:
        """Replace only the model; keep all product services and persisted state real."""
        self.deps.model = ScriptedModel(steps)
        return run_customer_turn(self.conversation.id, message, self.deps)

    def tool_results(self) -> list[dict]:
        """Read the actual saved responses delivered back to the hostile model."""
        saved = load_conversation(self.deps.store, self.deps.sandbox_id, self.conversation.id)
        assert saved is not None
        return [
            block
            for message in saved.messages
            for block in message["content"]
            if block["type"] == "tool_result"
        ]


@dataclass
class AttackRig:
    """A sandbox-wide harness; concurrent workers always receive separate sessions."""

    store: MemoryStore
    dops: DopsApi
    now: Callable[[], datetime]
    real_dops: bool
    sessions: dict[int, AttackSession] = field(default_factory=dict)

    def new_conversation(self, customer_id: int = 1) -> AttackSession:
        """Create an independent conversation against the same customer-level lock."""
        audit = AuditLog(self.store, SANDBOX)
        executor = ToolExecutor(
            self.dops,
            self.store,
            audit,
            EscalationRepo(self.store, SANDBOX),
            PolicyConfig(),
            self.now,
            http_lock_scope if self.real_dops else mock_lock_scope,
        )
        deps = AgentDeps(
            SANDBOX,
            ScriptedModel([]),
            executor,
            Budget(self.store, 120_000_000, lambda: self.now().date()),
            audit,
            self.store,
            AgentConfig(),
            self.now,
        )
        conversation = start_conversation(self.store, SANDBOX, customer_id, self.dops, self.now)
        return AttackSession(conversation, deps)

    def turn(
        self,
        customer_id: int,
        steps: Sequence[ModelResponse],
        message: str = "Please process my refund request.",
    ) -> TurnResult:
        """Start or reuse the default conversation for the requested customer."""
        if customer_id not in self.sessions:
            self.sessions[customer_id] = self.new_conversation(customer_id)
        return self.sessions[customer_id].turn(steps, message)

    def state(self) -> dict:
        """Snapshot every customer's orders/refunds without relying on HTTP internals."""
        if not self.real_dops:
            self.dops.list_orders(1, INSPECTION_TRACE)
            state = self.store.get_doc(f"sbx:{SANDBOX}:dops")
            assert state is not None
            return state
        seed = json.loads(
            (Path(__file__).parents[2] / "app/dops/seed.json").read_text(encoding="utf-8")
        )
        orders = []
        refunds = []
        for customer in seed["customers"]:
            for order in self.dops.list_orders(customer["id"], INSPECTION_TRACE):
                orders.append(order.model_dump(mode="json"))
                refunds.extend(
                    refund.model_dump(mode="json")
                    for refund in self.dops.list_order_refunds(
                        customer["id"], order.id, INSPECTION_TRACE
                    )
                )
        return {"customers": seed["customers"], "orders": orders, "refunds": refunds}

    def events(self) -> list[AuditEvent]:
        """Decode all persisted sandbox events into the invariant helper's public type."""
        return [
            AuditEvent(**{**row, "ts": datetime.fromisoformat(row["ts"])})
            for row in AuditLog(self.store, SANDBOX).list()
        ]


@pytest.fixture
def rig(
    request: pytest.FixtureRequest,
    store: MemoryStore,
    dops: DopsApi,
    now: Callable[[], datetime],
) -> AttackRig:
    """Assemble the full agent, tools, policy, intents, audit, store and DOPS stack."""
    return AttackRig(store, dops, now, request.config.getoption("--real-dops"))
