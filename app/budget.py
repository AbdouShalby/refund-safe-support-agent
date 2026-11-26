"""Daily model budget backed by atomic store reservations.

On a model error the caller does not settle, so the full reservation stays spent.
"""

from collections.abc import Callable
from datetime import date

from app.store import Store


class Budget:
    """Apply the daily cap to a date-keyed shared store counter."""

    def __init__(self, store: Store, daily_cap_nano: int, today: Callable[[], date]) -> None:
        self._store = store
        self._daily_cap_nano = daily_cap_nano
        self._today = today

    def _key(self) -> str:
        return f"budget:{self._today():%Y%m%d}"

    def reserve(self, worst: int) -> bool:
        """Atomically reserve the worst-case call cost if it fits the cap."""
        return self._store.reserve(self._key(), worst, self._daily_cap_nano)

    def settle(self, worst: int, actual: int) -> None:
        """Replace the reservation with actual cost, even if actual is higher."""
        self._store.settle(self._key(), worst, actual)

    def remaining(self) -> int:
        """Return the current day's unspent allowance."""
        return self._store.remaining(self._key(), self._daily_cap_nano)
