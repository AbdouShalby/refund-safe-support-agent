"""Select fresh mock state or explicitly configured, reset local DOPS."""

import os
import subprocess
from collections.abc import Iterator
from datetime import UTC, datetime

import pytest

from app.dops.api import DopsApi
from app.dops.http import HttpDopsClient
from app.dops.mock import MockDops


@pytest.fixture
def dops(request: pytest.FixtureRequest) -> Iterator[DopsApi]:
    """Reset before every case; real mode must never silently fall back."""
    if not request.config.getoption("--real-dops"):
        yield MockDops.seeded(now=lambda: datetime(2026, 11, 18, 10, tzinfo=UTC))
        return
    required = ("DOPS_BASE_URL", "DOPS_SERVICE_TOKEN", "DOPS_RESET_CMD")
    missing = [name for name in required if not os.environ.get(name, "").strip()]
    if missing:
        pytest.fail("--real-dops requires environment variables: " + ", ".join(missing))
    # Command comes only from the developer's local test environment, never request data.
    subprocess.run(os.environ["DOPS_RESET_CMD"], shell=True, check=True, timeout=120)  # noqa: S602
    client = HttpDopsClient(os.environ["DOPS_BASE_URL"], os.environ["DOPS_SERVICE_TOKEN"])
    try:
        yield client
    finally:
        client.close()
