"""Replay all recorded runs; partial money-family coverage must fail CI."""

from pathlib import Path

import pytest

from evals.run import load_scenarios, replay_runs


def test_recorded_regressions() -> None:
    directory = Path(__file__).parents[1] / "evals/fixtures"
    if not list(directory.glob("*.jsonl")):
        pytest.skip("no recorded fixtures yet (Task 14)")
    rows = replay_runs(load_scenarios(), directory)
    assert rows and all(row.get("replay_verified", row["passed"]) for row in rows), rows


def test_deadline_turns_are_marked_only_when_no_call_failed() -> None:
    from evals.run import _mark_deadline_turns

    attempts = [
        {"request_hash": "a", "turn": 1, "infra_error": False},
        {"request_hash": "b", "turn": 2, "infra_error": True},
    ]
    row = {
        "failures": [
            "stopped[1]: expected 'end_turn', actual 'model_error'",
            "stopped[2]: expected 'end_turn', actual 'model_error'",
        ]
    }
    _mark_deadline_turns(row, attempts)
    assert attempts[0].get("deadline") is True
    assert "deadline" not in attempts[1]
