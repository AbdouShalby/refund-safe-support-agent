"""Read an immutable web configuration snapshot without loading credential files."""

import os
from dataclasses import dataclass

from app.pricing import usd_to_nano


@dataclass(frozen=True)
class Settings:
    """Validated process configuration, read once when the web app is imported."""

    session_secret: str
    ip_hash_secret: str
    dops_mode: str
    daily_cap_nano: int
    kv_url: str | None
    kv_token: str | None
    demo_fake_model: bool
    on_vercel: bool

    @classmethod
    def from_env(cls) -> "Settings":
        """Validate environment variables and fail before serving unsafe settings."""
        session_secret = os.environ.get("SESSION_SECRET", "")
        ip_hash_secret = os.environ.get("IP_HASH_SECRET", "")
        for name, value in (("SESSION_SECRET", session_secret), ("IP_HASH_SECRET", ip_hash_secret)):
            if len(value) < 32:
                raise ValueError(f"{name} is required and must contain at least 32 characters")
        mode = os.environ.get("DOPS_MODE", "mock")
        if mode == "http":
            raise ValueError("DOPS_MODE=http is local only, not supported by the web API yet")
        if mode != "mock":
            raise ValueError("DOPS_MODE must be mock for the web API")
        fake = os.environ.get("DEMO_FAKE_MODEL") == "1"
        on_vercel = "VERCEL" in os.environ
        if fake and on_vercel:
            raise ValueError("DEMO_FAKE_MODEL is forbidden when VERCEL is set")
        return cls(
            session_secret,
            ip_hash_secret,
            mode,
            usd_to_nano(os.environ.get("DAILY_CAP_USD", "0.12")),
            os.environ.get("KV_REST_API_URL") or None,
            os.environ.get("KV_REST_API_TOKEN") or None,
            fake,
            on_vercel,
        )
