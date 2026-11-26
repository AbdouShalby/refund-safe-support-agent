"""Daily reservation accounting delegates atomically to the store."""

from datetime import date

from app.budget import Budget
from app.store import MemoryStore


def test_cap_boundary_and_unsettled_reservation() -> None:
    store = MemoryStore()
    budget = Budget(store, 100, lambda: date(2026, 11, 18))
    assert budget.reserve(60)
    assert budget.remaining() == 40
    assert budget.reserve(40)
    assert budget.remaining() == 0
    assert not budget.reserve(1)
    assert store.remaining("budget:20261118", 100) == 0
    assert Budget(store, 100, lambda: date(2026, 11, 18)).remaining() == 0


def test_settle_lower_and_higher() -> None:
    budget = Budget(MemoryStore(), 100, lambda: date(2026, 11, 18))
    assert budget.reserve(60)
    budget.settle(60, 20)
    assert budget.remaining() == 80
    assert budget.reserve(30)
    budget.settle(30, 50)
    assert budget.remaining() == 30
    assert budget.reserve(30)
    budget.settle(30, 50)
    assert budget.remaining() == 0


def test_day_rollover() -> None:
    day = date(2026, 11, 18)
    store = MemoryStore()
    budget = Budget(store, 100, lambda: day)
    assert budget.reserve(100)
    day = date(2026, 11, 19)
    assert budget.remaining() == 100
    assert budget.reserve(30)
    assert store.remaining("budget:20261119", 100) == 70
    assert store.remaining("budget:20261118", 100) == 0
