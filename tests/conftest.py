"""Register opt-in local DOPS contract verification."""

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    """Expose real mode even when pytest starts from the repository root."""
    parser.addoption("--real-dops", action="store_true", help="Run parity cases against local DOPS")
