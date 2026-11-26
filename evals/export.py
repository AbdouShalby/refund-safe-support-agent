"""Turn a strictly replayed fixture into a demo recording without any model/network calls."""

import argparse
import json
import re
from pathlib import Path

from app.customers import customer_name
from app.replay import ReplayModel
from app.sandbox_dops import SandboxDops
from evals.harness import INSPECTION_TRACE, build_stack
from evals.run import FIXTURES_DIR, ROOT, CountedModel, load_scenarios


def export_recording(
    scenario: dict,
    fixture: Path,
    *,
    name: str,
    title: str,
    output_dir: Path,
    approve_held: bool = False,
) -> Path:
    """Preserve observed replies, per-turn events and the original refund status metadata."""
    if not re.fullmatch(r"[a-z0-9-]{1,40}", name):
        raise ValueError("Recording name must match ^[a-z0-9-]{1,40}$")
    model = CountedModel(ReplayModel(fixture))
    stack = build_stack("mock", model, sandbox_id="eval-export")
    recording = {
        "name": name,
        "title": title,
        "language": scenario["language"],
        "customer": customer_name(scenario["customer_id"]),
        "steps": [],
        "ops": [],
    }
    try:
        for message in scenario["messages"]:
            recording["steps"].append(
                {"role": "customer", "text": message, "events": [], "refunds": []}
            )
            before = len(stack.events())
            result = stack.turn(scenario["customer_id"], message, stack.conversation_id)
            if result.stopped != "end_turn":
                raise ValueError(f"Cannot export incomplete turn: {result.stopped}")
            recording["steps"].append(
                {
                    "role": "agent",
                    "text": result.reply,
                    "events": [event.to_dict() for event in stack.events()[before:]],
                    "refunds": result.refunds,
                }
            )
        recorded_calls = len(fixture.read_text("utf-8").splitlines())
        if model.calls != recorded_calls:
            raise ValueError(
                f"Unconsumed fixture responses: expected {recorded_calls}, actual {model.calls}"
            )
        if approve_held:
            if not isinstance(stack.dops, SandboxDops):
                raise TypeError("Exports require SandboxDops")
            for refund in stack.dops_state()["refunds"]:
                if refund["status"] == "PENDING_APPROVAL":
                    approved = stack.dops.approve_refund(
                        refund["id"], "demo-reviewer", INSPECTION_TRACE
                    )
                    recording["ops"].append(
                        {
                            "action": "approve",
                            "refund_id": refund["id"],
                            "status_after": approved.status,
                        }
                    )
        output_dir.mkdir(parents=True, exist_ok=True)
        destination = output_dir / f"{name}.json"
        destination.write_text(
            json.dumps(recording, ensure_ascii=False, indent=2) + "\n", "utf-8", newline="\n"
        )
        return destination
    finally:
        stack.close()


def main(argv: list[str] | None = None) -> int:
    """Resolve a scenario/run fixture and export it only after complete mock replay."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixture", help="Scenario id followed by the run number")
    parser.add_argument("--name", required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument("--approve-held", action="store_true")
    args = parser.parse_args(argv)
    scenario_id, _, run_number = args.fixture.rpartition("-")
    scenarios = {scenario["id"]: scenario for scenario in load_scenarios()}
    if scenario_id not in scenarios or not re.fullmatch(r"[1-9][0-9]*", run_number):
        parser.error("Expected a known <scenario-id>-<positive-run-number>")
    path = export_recording(
        scenarios[scenario_id],
        FIXTURES_DIR / f"{args.fixture}.jsonl",
        name=args.name,
        title=args.title,
        output_dir=ROOT / "demo/recordings",
        approve_held=args.approve_held,
    )
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
