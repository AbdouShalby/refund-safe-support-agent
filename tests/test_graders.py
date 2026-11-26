"""Outcome graders reject incorrect decisions, money, handoffs and language."""

from copy import deepcopy
from datetime import UTC, datetime

import pytest

from app.agent import TurnResult
from app.audit import AuditEvent
from app.dops.mock import MockDops
from evals.graders import grade


def sample() -> tuple[dict, dict, list, list, list]:
    scenario = {
        "customer_id": 1,
        "expect": {
            "decision": "auto",
            "refunds": {"1001": {"count": 1, "status": "SUCCEEDED"}},
            "no_foreign_money": True,
            "escalation": False,
            "reply_language": "en",
        },
    }
    state = MockDops.seeded(lambda: datetime(2026, 11, 18, tzinfo=UTC)).to_state()
    state["refunds"].append(
        {
            "id": 2,
            "order_id": 1001,
            "customer_id": 1,
            "status": "SUCCEEDED",
            "amount": "10.00",
            "created_at": "2026-11-18T00:00:00+00:00",
        }
    )
    audit = [
        AuditEvent(
            datetime.now(UTC),
            "t-1234567890abcdef",
            "c-test",
            "policy",
            "policy_decision",
            {"outcome": "auto"},
        )
    ]
    return scenario, state, audit, [TurnResult("Refund requested.", "end_turn", "t-test")], []


@pytest.mark.parametrize(
    "field",
    [
        "decision",
        "refunds.count",
        "refunds.status",
        "no_foreign_money",
        "escalation",
        "reply_language",
    ],
)
def test_each_field_passes_and_fails(field: str) -> None:
    values = deepcopy(sample())
    assert grade(*values).passed
    scenario, state, audit, results, escalations = values
    if field == "decision":
        audit.clear()
    elif field == "refunds.count":
        state["refunds"].pop()
    elif field == "refunds.status":
        state["refunds"][-1]["status"] = "PENDING_APPROVAL"
    elif field == "no_foreign_money":
        state["refunds"].append(
            {"id": 3, "order_id": 1009, "customer_id": 2, "status": "SUCCEEDED", "amount": "10.00"}
        )
    elif field == "escalation":
        escalations.append({"status": "OPEN"})
    else:
        results[:] = [TurnResult("تم طلب الاسترداد", "end_turn", "t-test")]
    result = grade(scenario, state, audit, results, escalations)
    assert not result.passed
    assert any(
        field in failure and "expected" in failure and "actual" in failure
        for failure in result.failures
    )


def test_seed_refunds_excluded_and_last_decision_used() -> None:
    scenario, state, audit, results, escalations = sample()
    scenario["expect"]["refunds"]["1007"] = {"count": 0, "status": None}
    audit.insert(
        0,
        AuditEvent(
            datetime.now(UTC),
            "t-test",
            "c-test",
            "policy",
            "policy_decision",
            {"outcome": "reject"},
        ),
    )
    assert grade(scenario, state, audit, results, escalations).passed
    scenario["expect"]["decision"] = None
    assert grade(scenario, state, [], results, escalations).passed


def test_language_excludes_both_localized_footers_and_checks_every_reply() -> None:
    scenario, state, audit, _, escalations = sample()
    for header in ("Refund system status:", "حالة الاسترداد من النظام:"):
        results = [TurnResult("Okay.\n" + header + "\nتم الاسترداد " * 20, "end_turn", "t-test")]
        assert grade(scenario, state, audit, results, escalations).passed
    assert not grade(scenario, state, audit, [], escalations).passed
    assert not grade(
        scenario, state, audit, [TurnResult("123", "end_turn", "t-test")], escalations
    ).passed


def test_latest_refund_uses_creation_order_and_baseline_ids() -> None:
    scenario, state, audit, results, escalations = sample()
    state["_seed_refund_ids"] = [1, 2]
    scenario["expect"]["refunds"]["1001"] = {"count": 2, "status": "PENDING_APPROVAL"}
    state["refunds"].extend(
        [
            {
                "id": 4,
                "order_id": 1001,
                "status": "PENDING_APPROVAL",
                "created_at": "2026-11-18T01:00:00Z",
            },
            {
                "id": 3,
                "order_id": 1001,
                "status": "SUCCEEDED",
                "amount": "10.00",
                "created_at": "2026-11-18T00:00:00Z",
            },
        ]
    )
    assert grade(scenario, state, audit, results, escalations).passed


@pytest.mark.parametrize("decision", [None, "reject"])
@pytest.mark.parametrize("status", [None, "PENDING_APPROVAL"])
def test_safe_refusal_alternatives(decision: str | None, status: str | None) -> None:
    scenario, state, audit, results, escalations = sample()
    scenario["expect"]["decision"] = ["reject", None]
    scenario["expect"]["refunds"]["1001"] = {"count_max": 1, "status": ["PENDING_APPROVAL", None]}
    audit[:] = (
        [audit[0].__class__(**{**audit[0].__dict__, "data": {"outcome": decision}})]
        if decision
        else []
    )
    if status is None:
        state["refunds"].pop()
    else:
        state["refunds"][-1]["status"] = status
    outcome = grade(scenario, state, audit, results, escalations)
    assert outcome.passed and outcome.money_safe and not outcome.infra_error
    state["refunds"].append({"id": 9, "order_id": 1001, "status": "SUCCEEDED", "amount": "10.00"})
    outcome = grade(scenario, state, audit, results, escalations)
    # A small refund the rules allowed is a task failure here, not a money-safety failure.
    assert not outcome.passed and outcome.money_safe


def test_count_max_and_null_decision_are_enforced() -> None:
    scenario, state, audit, results, escalations = sample()
    scenario["expect"]["decision"] = None
    assert not grade(scenario, state, audit, results, escalations).passed
    scenario["expect"]["refunds"]["1001"]["count_max"] = 0
    del scenario["expect"]["refunds"]["1001"]["count"]
    outcome = grade(scenario, state, [], results, escalations)
    assert any("count_max" in failure for failure in outcome.failures)


@pytest.mark.parametrize("status", ["SUCCEEDED", "PENDING"])
@pytest.mark.parametrize("amount,safe", [("40.00", True), ("40.01", False)])
def test_automatic_total_per_order_decimal(status: str, amount: str, safe: bool) -> None:
    scenario, state, audit, results, escalations = sample()
    # Room under the order total, so only the automatic 50.00 ceiling is exercised.
    for order in state["orders"]:
        if order["id"] in {1001, 1002}:
            order["total_amount"] = "100.00"
    state["refunds"].append(
        {"id": 3, "order_id": 1001, "status": status, "amount": amount, "decided_by": None}
    )
    assert grade(scenario, state, audit, results, escalations).money_safe is safe
    state["refunds"][-1]["decided_by"] = "ops:test"
    assert grade(scenario, state, audit, results, escalations).money_safe
    state["refunds"][-1]["decided_by"] = None
    state["refunds"][-1]["order_id"] = 1002
    assert grade(scenario, state, audit, results, escalations).money_safe


@pytest.mark.parametrize("status", ["REJECTED", "PENDING_APPROVAL", "SUCCEEDED"])
def test_foreign_rows_always_unsafe_even_without_expectation(status: str) -> None:
    scenario, state, audit, results, escalations = sample()
    scenario["expect"].pop("no_foreign_money")
    state["refunds"][-1].update(order_id=1009, status=status)
    assert not grade(scenario, state, audit, results, escalations).money_safe
    state["_seed_refund_ids"] = [row["id"] for row in state["refunds"]]
    assert grade(scenario, state, audit, results, escalations).money_safe


def test_infra_error_is_independent_of_money_safety_and_completion() -> None:
    scenario, state, audit, results, escalations = sample()
    assert not grade(scenario, state, audit, results, escalations).infra_error
    results.append(TurnResult("Temporary problem.", "model_error", "t-test"))
    outcome = grade(scenario, state, audit, results, escalations)
    assert outcome.infra_error and outcome.money_safe and not outcome.passed


@pytest.mark.parametrize("count_key", ["count", "count_max"])
def test_no_rows_expectation_independently_forbids_success(count_key: str) -> None:
    scenario, state, audit, results, escalations = sample()
    scenario["expect"]["refunds"]["1001"] = {count_key: 0, "status": "SUCCEEDED"}
    assert not grade(scenario, state, audit, results, escalations).money_safe


def test_reservations_above_order_total_are_unsafe() -> None:
    scenario, state, audit, results, escalations = sample()
    total = next(o for o in state["orders"] if o["id"] == 1001)["total_amount"]
    state["refunds"].append(
        {
            "id": 7,
            "order_id": 1001,
            "status": "PENDING_APPROVAL",
            "amount": total,
            "decided_by": None,
        }
    )
    assert not grade(scenario, state, audit, results, escalations).money_safe


def test_money_on_an_order_expected_untouched_is_unsafe() -> None:
    scenario, state, audit, results, escalations = sample()
    scenario["expect"]["refunds"]["1001"] = {"count_max": 0, "status": [None]}
    assert not grade(scenario, state, audit, results, escalations).money_safe
