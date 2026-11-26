"""Run with `python -m scripts.smoke_live`; expected cost about 0.0005 USD.

Never run in CI. Uses the real Anthropic API and synthetic in-memory DOPS data.
"""

import os
from datetime import UTC, datetime

from app.agent import AgentConfig, AgentDeps, run_customer_turn, start_conversation
from app.audit import AuditLog
from app.budget import Budget
from app.escalations import EscalationRepo
from app.model import AnthropicModel
from app.policy import PolicyConfig
from app.pricing import nano_to_usd, usd_to_nano
from app.sandbox_dops import SandboxDops
from app.store import MemoryStore
from app.tools import ToolExecutor, mock_lock_scope


def main() -> int:
    """Run two customer turns only when an API key is configured."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        pass
    else:
        load_dotenv()
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY is missing. Set it in the environment before running this script.")
        return 1

    def now() -> datetime:
        return datetime.now(UTC)

    store = MemoryStore()
    sandbox_id = "smoke-live"
    dops = SandboxDops(store, sandbox_id, now)
    audit = AuditLog(store, sandbox_id)
    budget = Budget(store, usd_to_nano("0.01"), lambda: now().date())
    executor = ToolExecutor(
        dops, store, audit, EscalationRepo(store, sandbox_id), PolicyConfig(), now, mock_lock_scope
    )
    deps = AgentDeps(
        sandbox_id, AnthropicModel(), executor, budget, audit, store, AgentConfig(), now
    )
    conversation = start_conversation(store, sandbox_id, 1, dops, now)
    for message in (
        "Where is order 1002?",
        "Please refund 10.00 on order 1001, it arrived damaged.",
    ):
        result = run_customer_turn(conversation.id, message, deps)
        print(f"Customer: {message}")
        print(f"Reply: {result.reply}")
        print(f"stopped: {result.stopped}")
        print(f"refunds: {result.refunds}")
    calls = [event for event in audit.list(conversation.id) if event["type"] == "model_call"]
    for index, event in enumerate(calls, start=1):
        print(f"model_call {index} usage: {event['data']['usage']}")
    total = sum(event["data"]["cost_nano"] for event in calls)
    print(f"Total cost: {nano_to_usd(total)} USD")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
