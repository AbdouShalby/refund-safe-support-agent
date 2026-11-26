"""Run recorded evaluations offline or explicitly record live model outcome measurements."""

import argparse
import json
import re
from pathlib import Path
from typing import Literal

import anthropic

from app.model import AnthropicModel, ModelClient, ModelResponse, ModelUnavailable, request_hash
from app.pricing import cost_nano, nano_to_usd, worst_case_nano
from app.replay import FixtureExhausted, FixtureMismatch, RecordingModel, ReplayModel
from evals.graders import grade, new_refunds
from evals.harness import RUN_CAP_NANO, EvalStack, build_stack
from evals.report import write_report

ROOT = Path(__file__).parents[1]
SCENARIOS_DIR = ROOT / "evals/scenarios"
FIXTURES_DIR = ROOT / "evals/fixtures"
REPORTS_DIR = ROOT / "reports"
NON_MONEY_FAMILIES = frozenset({"escalation", "off_topic"})


class CountedModel:
    """Count attempted calls, including unavailable responses that cannot be recorded."""

    def __init__(self, inner: ModelClient) -> None:
        self.inner = inner
        self.calls = 0
        self.attempts: list[dict] = []
        self.turn_index = 0
        self.usage_cost_nano = 0
        self.unconfirmed_cost_nano = 0

    def create(self, request: dict) -> ModelResponse:
        """Count before calling, so provider failures remain visible in the report."""
        self.calls += 1
        attempt = {
            "request_hash": request_hash(request),
            "turn": self.turn_index,
            "infra_error": False,
        }
        self.attempts.append(attempt)
        try:
            response = self.inner.create(request)
        except ModelUnavailable:
            attempt["infra_error"] = True
            self.unconfirmed_cost_nano += worst_case_nano(
                len(json.dumps(request, ensure_ascii=False).encode()), request["max_tokens"]
            )
            raise
        except BaseException:
            self.unconfirmed_cost_nano += worst_case_nano(
                len(json.dumps(request, ensure_ascii=False).encode()), request["max_tokens"]
            )
            raise
        self.usage_cost_nano += cost_nano(response.usage)
        return response


def load_scenarios(directory: Path = SCENARIOS_DIR) -> list[dict]:
    """Load every scenario in stable order and reject duplicate identifiers."""
    scenarios = [json.loads(path.read_text("utf-8")) for path in sorted(directory.glob("*.json"))]
    ids = [scenario["id"] for scenario in scenarios]
    if len(ids) != len(set(ids)):
        raise ValueError("Scenario ids must be unique")
    return scenarios


def _row(scenario: dict, run: int) -> dict:
    return {
        "scenario": scenario["id"],
        "family": scenario["family"],
        "run": run,
        "passed": False,
        "money_safe": False,
        "infra_error": False,
        "failures": [],
        "cost_nano": 0,
        "cost_usd": nano_to_usd(0),
        "model_calls": 0,
        "usage_cost_nano": 0,
        "unconfirmed_cost_nano": 0,
    }


def _execute(scenario: dict, stack: EvalStack, row: dict, model: CountedModel) -> list[str]:
    results = []
    for index, message in enumerate(scenario["messages"], 1):
        model.turn_index = index
        if isinstance(model.inner, OutcomeReplay):
            model.inner.turn_index = index
        results.append(stack.turn(scenario["customer_id"], message, stack.conversation_id))
    state = stack.dops_state()
    outcome = grade(scenario, state, stack.events(), results, stack.escalations())
    safe_refusal = (
        outcome.passed
        and outcome.money_safe
        and all(refund["status"] not in {"SUCCEEDED", "PENDING"} for refund in new_refunds(state))
    )
    row["money_safe"] = outcome.money_safe
    row["infra_error"] = outcome.infra_error
    failures = outcome.failures
    for index, result in enumerate(results, 1):
        if result.stopped != "end_turn" and not (result.stopped == "refusal" and safe_refusal):
            failures.append(f"stopped[{index}]: expected 'end_turn', actual {result.stopped!r}")
    return failures


def _metrics(row: dict, model: CountedModel | None) -> None:
    row["model_calls"] = model.calls if model is not None else 0
    if model is not None:
        row["usage_cost_nano"] = model.usage_cost_nano
        row["unconfirmed_cost_nano"] = model.unconfirmed_cost_nano
        row["cost_nano"] = model.usage_cost_nano + model.unconfirmed_cost_nano
    row["cost_usd"] = nano_to_usd(row["cost_nano"])
    row["passed"] = not row["failures"]


class OutcomeReplay:
    """Reproduce documented provider failures without weakening request matching."""

    def __init__(self, path: Path, metadata: dict) -> None:
        self.replay = ReplayModel(path)
        self.metadata = metadata
        self.turn_index = 0
        self.calls = 0
        self.failed_turns: set[int] = set()
        self.returned = 0

    def create(self, request: dict) -> ModelResponse:
        self.calls += 1
        attempts = self.metadata.get("attempts")
        if attempts is not None:
            if self.calls > len(attempts):
                raise FixtureExhausted("Replay metadata exhausted")
            attempt = attempts[self.calls - 1]
            actual = request_hash(request)
            if attempt["request_hash"] != actual or attempt["turn"] != self.turn_index:
                raise FixtureMismatch(attempt["request_hash"], actual)
            if attempt["infra_error"]:
                raise ModelUnavailable("Recorded infrastructure error")
            if attempt.get("deadline"):
                # The response arrived after the turn deadline (slow network plus eval-only
                # retries), so the loop stopped without using it. Consume the recorded
                # response to keep alignment, then reproduce the same stop.
                self.replay.create(request)
                self.returned += 1
                raise ModelUnavailable("Recorded turn deadline")
        try:
            response = self.replay.create(request)
            self.returned += 1
            return response
        except (FixtureExhausted, FixtureMismatch):
            # Legacy reports identify the failed turn, not the unavailable request's hash.
            # Only its boundary may omit a response; all subsequent hashes and the total
            # attempted-call count must still match the immutable original report.
            if (
                attempts is None
                and self.turn_index in self.metadata.get("error_turns", [])
                and self.turn_index not in self.failed_turns
            ):
                self.failed_turns.add(self.turn_index)
                raise ModelUnavailable("Recorded infrastructure error") from None
            raise


def _mark_deadline_turns(row: dict, attempts: list[dict]) -> None:
    """Mark the last attempt of a turn that stopped with model_error although no call failed.

    That happens when a response arrives after the 45 s turn deadline. Replay cannot
    reproduce wall-clock time, so the marker lets it reproduce the same recorded stop.
    """
    for failure in row["failures"]:
        if not (failure.startswith("stopped[") and failure.endswith("'model_error'")):
            continue
        turn = int(failure[len("stopped[") : failure.index("]")])
        in_turn = [attempt for attempt in attempts if attempt["turn"] == turn]
        if in_turn and not any(attempt["infra_error"] for attempt in in_turn):
            in_turn[-1]["deadline"] = True


def _legacy_metadata(path: Path, fixtures_dir: Path) -> dict | None:
    """Use the original report only for this repository's original fixtures."""
    if fixtures_dir.resolve() != FIXTURES_DIR.resolve():
        return None
    report = json.loads((REPORTS_DIR / "eval-2026-11-19.json").read_text("utf-8"))
    for row in report["runs"]:
        if f"{row['scenario']}-{row['run']}" == path.stem:
            return {
                **row,
                "error_turns": [
                    int(match.group(1))
                    for failure in row["failures"]
                    if (
                        match := re.fullmatch(
                            r"stopped\[(\d+)\]: expected 'end_turn', actual 'model_error'", failure
                        )
                    )
                ],
            }
    return None


def replay_runs(
    scenarios: list[dict],
    fixtures_dir: Path = FIXTURES_DIR,
    *,
    require_coverage: bool = True,
) -> list[dict]:
    """Replay every fixture on mock DOPS; partial money coverage and stale hashes fail."""
    paths = sorted(fixtures_dir.glob("*.jsonl"))
    if not paths:
        print("No fixtures yet (Task 14).")
        return []
    by_id = {scenario["id"]: scenario for scenario in scenarios}
    covered: set[str] = set()
    rows = []
    for path in paths:
        scenario_id, _, number = path.stem.rpartition("-")
        scenario = by_id.get(scenario_id)
        if scenario is None or not number.isascii() or not number.isdigit() or int(number) < 1:
            row = _row({"id": path.stem, "family": "unknown"}, 0)
            row["failures"].append(
                f"fixture: expected known scenario and positive run number, actual {path.name!r}"
            )
            rows.append(row)
            continue
        covered.add(scenario_id)
        row = _row(scenario, int(number))
        stack = None
        model = None
        try:
            sidecar = path.with_suffix(".meta.json")
            metadata = (
                json.loads(sidecar.read_text("utf-8"))
                if sidecar.exists()
                else _legacy_metadata(path, fixtures_dir)
            )
            model = CountedModel(OutcomeReplay(path, metadata) if metadata else ReplayModel(path))
            stack = build_stack("mock", model, sandbox_id="eval-replay")
            row["failures"].extend(_execute(scenario, stack, row, model))
            recorded_calls = (
                metadata["model_calls"]
                if metadata is not None
                else len(path.read_text("utf-8").splitlines())
            )
            if isinstance(model.inner, OutcomeReplay):
                returned = len(path.read_text("utf-8").splitlines())
                if model.inner.returned != returned:
                    row["failures"].append(
                        f"fixture.responses: expected {returned}, actual {model.inner.returned}"
                    )
                if "attempts" in metadata and len(metadata["attempts"]) != recorded_calls:
                    row["failures"].append("fixture.metadata: attempted-call count mismatch")
            if model.calls != recorded_calls:
                row["failures"].append(
                    f"fixture.calls: expected {recorded_calls}, actual {model.calls}"
                )
            if metadata is not None:
                expected = list(metadata["failures"])
                if "error_turns" in metadata:
                    if scenario["family"] in {"not_refundable", "exceed_total", "injection"}:
                        expected = [
                            failure
                            for failure in expected
                            if not failure.startswith(("decision:", "refunds."))
                        ]
                    if metadata["error_turns"]:
                        expected.append("infra_error: expected False, actual True")
                row["replay_verified"] = (
                    model.calls == recorded_calls
                    and sorted(row["failures"]) == sorted(expected)
                    and row["infra_error"]
                    == bool(metadata.get("infra_error") or metadata.get("error_turns"))
                    and (
                        "money_safe" not in metadata or row["money_safe"] == metadata["money_safe"]
                    )
                )
        except Exception as error:
            # FixtureMismatch includes both request hashes; do not hide either hash.
            row["failures"].append(f"replay: expected matching complete run, actual {error}")
            row["replay_verified"] = False
        finally:
            _metrics(row, model)
            if stack is not None:
                stack.close()
        rows.append(row)
    if require_coverage:
        for scenario in scenarios:
            if scenario["family"] not in NON_MONEY_FAMILIES and scenario["id"] not in covered:
                row = _row(scenario, 0)
                row["failures"].append("fixture: expected at least one recorded run, actual none")
                rows.append(row)
    for row in rows:
        print(f"{row['scenario']}-{row['run']}: {'PASS' if row['passed'] else 'FAIL'}")
        for failure in row["failures"]:
            print(f"  {failure}")
    return rows


def live_runs(
    scenarios: list[dict],
    *,
    mode: Literal["mock", "http"],
    reset_cmd: str | None,
    fixtures_dir: Path = FIXTURES_DIR,
) -> list[dict]:
    """Record all attempted runs and stop cleanly after the cumulative 0.50 USD cap."""
    rows: list[dict] = []
    total = 0
    interrupted = False
    fixtures_dir.mkdir(parents=True, exist_ok=True)
    for scenario in scenarios:
        count = 1 if scenario["family"] in NON_MONEY_FAMILIES else 3
        for number in range(1, count + 1):
            # Never append a new run to an older recording or overwrite evidence.
            used = {path.stem for path in fixtures_dir.glob(f"{scenario['id']}-*.jsonl")}
            while f"{scenario['id']}-{number}" in used:
                number += 1
            fixture = fixtures_dir / f"{scenario['id']}-{number}.jsonl"
            row = _row(scenario, number)
            stack = None
            model = None
            try:
                # An empty fixture honestly preserves a run with no returned model responses.
                fixture.touch(exist_ok=False)
                # The agent itself never retries. Eval-only SDK retries separate network
                # noise from model behaviour without changing the agent's failure reply.
                model = CountedModel(
                    RecordingModel(
                        AnthropicModel(anthropic.Anthropic(max_retries=2, timeout=30.0)), fixture
                    )
                )
                stack = build_stack(mode, model, sandbox_id="eval-live", reset_cmd=reset_cmd)
                row["failures"].extend(_execute(scenario, stack, row, model))
            except KeyboardInterrupt:
                row["failures"].append("run: expected completed scenario, actual interrupted")
                interrupted = True
            except Exception as error:
                # Provider wrappers sanitize failures. Avoid echoing arbitrary reset/env errors.
                row["failures"].append(
                    f"run: expected completed scenario, actual {type(error).__name__}"
                )
            finally:
                _metrics(row, model)
                if stack is not None:
                    stack.close()
                if model is not None:
                    _mark_deadline_turns(row, model.attempts)
                fixture.with_suffix(".meta.json").write_text(
                    json.dumps({**row, "attempts": model.attempts if model else []}, indent=2)
                    + "\n",
                    "utf-8",
                    newline="\n",
                )
                rows.append(row)
            total += row["cost_nano"]
            print(f"{scenario['id']}-{number}: {'PASS' if row['passed'] else 'FAIL'}")
            if total >= RUN_CAP_NANO or interrupted:
                print(f"Stopped: cumulative accounted cost {nano_to_usd(total)} USD.")
                return rows
    return rows


def main(argv: list[str] | None = None) -> int:
    """Live exit status reports successful report writing; replay status reports correctness."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("replay", "live"), default="replay")
    parser.add_argument("--dops", choices=("mock", "http"), default="mock")
    parser.add_argument("--reset-cmd")
    parser.add_argument("--record", action="store_true", help="Live runs are always recorded")
    parser.add_argument("--fixtures-dir", type=Path, default=FIXTURES_DIR)
    parser.add_argument("--reports-dir", type=Path, default=REPORTS_DIR)
    args = parser.parse_args(argv)
    scenarios = load_scenarios()
    if args.mode == "replay":
        rows = replay_runs(scenarios, args.fixtures_dir)
        return 0 if all(row.get("replay_verified", row["passed"]) for row in rows) else 1
    rows = live_runs(
        scenarios, mode=args.dops, reset_cmd=args.reset_cmd, fixtures_dir=args.fixtures_dir
    )
    paths = write_report(rows, dops_mode=args.dops, output_dir=args.reports_dir)
    print(f"Reports: {paths[0]} and {paths[1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
