"""Enforce visitor caps using atomic, UTC-date-keyed expiring counters."""

import hashlib
import hmac
from collections.abc import Callable
from datetime import date

from app.store import Store


def ip_hash(ip: str, secret: str) -> str:
    """Return a keyed, truncated visitor identifier without persisting the raw IP."""
    return hmac.new(secret.encode("utf-8"), ip.encode("utf-8"), hashlib.sha256).hexdigest()[:16]


class Limits:
    """Apply per-day session and live-conversation caps before model spend."""

    def __init__(self, store: Store, today: Callable[[], date]) -> None:
        self.store = store
        self.today = today

    def allow_session(self, ip_hash: str) -> bool:
        """Allow at most twenty newly issued sessions per hashed IP per UTC day."""
        return self.store.incr_below(f"limit:{self.today():%Y%m%d}:session:{ip_hash}", 20)

    def start_live(self, sandbox_id: str, ip_hash: str) -> bool:
        """Allow three live starts per sandbox and per hashed IP per UTC day.

        The sandbox counter may be consumed before the IP counter refuses.
        This conservative refusal only affects that visitor and is acceptable.
        """
        prefix = f"limit:{self.today():%Y%m%d}:live"
        return self.store.incr_below(f"{prefix}:sandbox:{sandbox_id}", 3) and self.store.incr_below(
            f"{prefix}:ip:{ip_hash}", 3
        )
