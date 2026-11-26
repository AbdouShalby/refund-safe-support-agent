"""Exercise signed sessions and fail-closed environment configuration."""

from datetime import UTC, datetime, timedelta

import pytest

from app.config import Settings
from app.session import sign, verify

SECRET = "test-session-secret-with-at-least-32-characters"  # noqa: S105
NOW = datetime(2026, 11, 18, 12, tzinfo=UTC)


def payload() -> dict:
    return {
        "sandbox_id": "s-" + "a" * 24,
        "customer_id": 1,
        "role": "customer",
        "exp": int((NOW + timedelta(days=1)).timestamp()),
    }


def test_round_trip_and_tamper() -> None:
    token = sign(payload(), SECRET)
    assert verify(token, SECRET, NOW) == payload()
    assert verify(token, SECRET + "other", NOW) is None
    body, signature = token.split(".")
    assert verify(body[:-1] + "A." + signature, SECRET, NOW) is None
    assert verify(body + "." + "A" * len(signature), SECRET, NOW) is None


def test_expiry_boundary() -> None:
    token = sign({**payload(), "exp": int(NOW.timestamp())}, SECRET)
    assert verify(token, SECRET, NOW) is None


def test_signature_requires_canonical_base64url() -> None:
    token = sign(payload(), SECRET)
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_"
    # SHA-256 has two unused trailing base64 bits, which must stay zero.
    alternate = token[:-1] + alphabet[alphabet.index(token[-1]) ^ 1]
    assert verify(alternate, SECRET, NOW) is None


@pytest.mark.parametrize("token", ["", "garbage", ".", "a.b.c", "!!!.???", "e30.e30", None, 7])
def test_garbage_never_raises(token: object) -> None:
    assert verify(token, SECRET, NOW) is None


@pytest.mark.parametrize(
    "changes",
    [
        {"customer_id": True},
        {"customer_id": 4},
        {"sandbox_id": "../other"},
        {"role": "admin"},
        {"exp": "tomorrow"},
        {"exp": float("inf")},
        {"extra": "untrusted"},
    ],
)
def test_signed_invalid_payload_is_rejected(changes: dict) -> None:
    assert verify(sign({**payload(), **changes}, SECRET), SECRET, NOW) is None


@pytest.fixture
def environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("VERCEL", "DEMO_FAKE_MODEL", "DOPS_MODE", "DAILY_CAP_USD"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("SESSION_SECRET", SECRET)
    monkeypatch.setenv("IP_HASH_SECRET", "test-ip-secret-with-at-least-32-characters")


def test_settings_defaults_and_snapshot(environment: None, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = Settings.from_env()
    monkeypatch.setenv("DAILY_CAP_USD", "0.50")
    assert settings.daily_cap_nano == 120_000_000
    assert Settings.from_env().daily_cap_nano == 500_000_000
    assert settings.dops_mode == "mock"
    assert not settings.demo_fake_model and not settings.on_vercel


@pytest.mark.parametrize("name", ["SESSION_SECRET", "IP_HASH_SECRET"])
@pytest.mark.parametrize("value", [None, "too-short"])
def test_required_secrets(
    environment: None, monkeypatch: pytest.MonkeyPatch, name: str, value: str | None
) -> None:
    if value is None:
        monkeypatch.delenv(name)
    else:
        monkeypatch.setenv(name, value)
    with pytest.raises(ValueError, match=name):
        Settings.from_env()


def test_fake_model_refused_on_vercel(environment: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VERCEL", "1")
    monkeypatch.setenv("DEMO_FAKE_MODEL", "1")
    with pytest.raises(ValueError, match="DEMO_FAKE_MODEL"):
        Settings.from_env()


def test_http_mode_refused(environment: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DOPS_MODE", "http")
    with pytest.raises(ValueError, match="local only, not supported by the web API yet"):
        Settings.from_env()


@pytest.mark.parametrize("value", ["-1", "NaN", "0.0000000001"])
def test_invalid_budget(environment: None, monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv("DAILY_CAP_USD", value)
    with pytest.raises(ValueError):
        Settings.from_env()
