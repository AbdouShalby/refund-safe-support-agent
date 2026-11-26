"""Exact nano-USD pricing and conservative model-call reservations."""

from decimal import Decimal, InvalidOperation

from app.model import Usage

# Source: Anthropic pricing page https://platform.claude.com/docs/en/about-claude/pricing
# Verified 2026-11-18 for claude-haiku-5-5 (5-minute cache writes).
INPUT_NANO = 100
CACHE_WRITE_NANO = 125
CACHE_READ_NANO = 10
OUTPUT_NANO = 500


def cost_nano(usage: Usage) -> int:
    """Price each billed token category without floating-point arithmetic."""
    return (
        usage.input * INPUT_NANO
        + usage.cache_write * CACHE_WRITE_NANO
        + usage.cache_read * CACHE_READ_NANO
        + usage.output * OUTPUT_NANO
    )


def worst_case_nano(request_bytes: int, max_tokens: int) -> int:
    """Reserve one token per request byte at cache-write price plus all output."""
    return request_bytes * CACHE_WRITE_NANO + max_tokens * OUTPUT_NANO


def usd_to_nano(usd: str) -> int:
    """Convert finite, nonnegative decimal USD exactly; reject sub-nano amounts."""
    if not isinstance(usd, str):
        raise TypeError("USD must be a decimal string")
    try:
        amount = Decimal(usd)
    except InvalidOperation:
        raise ValueError("USD must be a finite nonnegative decimal string") from None
    if not amount.is_finite() or amount < 0:
        raise ValueError("USD must be finite and nonnegative")
    numerator, denominator = amount.as_integer_ratio()
    nano, remainder = divmod(numerator * 1_000_000_000, denominator)
    if remainder:
        raise ValueError("USD must be representable in whole nano-USD")
    return nano


def nano_to_usd(nano: int) -> str:
    """Format USD to six decimals using integer round-half-even arithmetic."""
    sign = "-" if nano < 0 else ""
    micros, remainder = divmod(abs(nano), 1000)
    if remainder > 500 or (remainder == 500 and micros % 2):
        micros += 1
    whole, fraction = divmod(micros, 1_000_000)
    return f"{sign}{whole}.{fraction:06d}"
