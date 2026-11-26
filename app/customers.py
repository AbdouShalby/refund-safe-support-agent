"""Resolve synthetic customer names from the shared DOPS seed."""

import json
from pathlib import Path


def customer_name(customer_id: int) -> str:
    """Return the seeded display name, or a stable name for an unknown id."""
    seed = json.loads(Path(__file__).with_name("dops").joinpath("seed.json").read_text("utf-8"))
    return next(
        (row["name"] for row in seed["customers"] if row["id"] == customer_id),
        f"Customer {customer_id}",
    )
