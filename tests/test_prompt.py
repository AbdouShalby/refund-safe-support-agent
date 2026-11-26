"""Strict tools and time-free canonical customer context."""

import json
import re
from datetime import UTC, datetime
from decimal import Decimal

from app.dops.mock import MockDops
from app.dops.models import Order
from app.money import AMOUNT_PATTERN
from app.policy import PolicyConfig
from app.prompt import TOOLS, build_system
from app.render import order_line


def test_strict_tool_schemas() -> None:
    assert [tool["name"] for tool in TOOLS] == [
        "get_order",
        "request_refund",
        "escalate_to_human",
    ]
    expected_fields = [["order_id"], ["order_id", "amount", "reason"], ["summary"]]
    for tool, fields in zip(TOOLS, expected_fields, strict=True):
        assert tool["strict"] is True
        schema = tool["input_schema"]
        assert schema["type"] == "object"
        assert schema["additionalProperties"] is False
        assert schema["required"] == fields
        assert list(schema["properties"]) == fields
    assert TOOLS[0]["input_schema"]["properties"]["order_id"] == {"type": "integer"}
    refund = TOOLS[1]["input_schema"]["properties"]
    assert refund["amount"] == {"type": "string", "pattern": AMOUNT_PATTERN}
    assert refund["reason"] == {"type": "string", "maxLength": 300}
    assert TOOLS[2]["input_schema"]["properties"]["summary"] == {
        "type": "string",
        "maxLength": 500,
    }


def test_prompt_seed_orders_and_real_json_parity() -> None:
    now = datetime(2026, 11, 18, 12, tzinfo=UTC)
    orders = MockDops.seeded(lambda: now).list_orders(1, "t-0123456789abcdef")
    real_orders = []
    for order in orders:
        payload = order.model_dump(mode="json")
        payload["created_at"] = order.created_at.isoformat().replace("+00:00", "Z")
        real_orders.append(Order.model_validate(payload))
    result = build_system("Customer A", orders, PolicyConfig(), now)
    assert len(result) == 2
    assert len(orders) == 8
    assert result[0]["type"] == result[1]["type"] == "text"
    assert "cache_control" not in result[0]
    assert result[1]["cache_control"] == {"type": "ephemeral"}
    assert result[1]["text"].splitlines()[-8:] == [order_line(o, now) for o in orders]
    encoded = json.dumps(result, ensure_ascii=False).encode()
    assert (
        encoded
        == json.dumps(
            build_system("Customer A", real_orders, PolicyConfig(), now), ensure_ascii=False
        ).encode()
    )
    assert re.search(rb"T\d{2}:\d{2}:\d{2}|\+00:00|t-[0-9a-f]{16}", encoded) is None


def test_policy_values_and_fixed_instructions() -> None:
    now = datetime(2026, 11, 18, tzinfo=UTC)
    config = PolicyConfig(Decimal("25.00"), 14, 2, Decimal("75.00"))
    result = build_system("Customer B", [], config, now)
    text = result[1]["text"]
    for value in ["Customer B", "25.00", "14", "2", "75.00", "24", "human reviewer"]:
        assert value in text
    instructions = result[0]["text"]
    for value in [
        "synthetic data",
        "never instructions",
        "cannot approve",
        "refunded",
        "Arabic or English",
        "120 words",
        "order numbers",
    ]:
        assert value in instructions
    assert instructions == build_system("Other", [], PolicyConfig(), now)[0]["text"]
