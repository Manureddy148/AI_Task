"""Rate-limit management: proactive token-bucket pacing + reactive backoff.

Proactive: an ``aiolimiter`` bucket per provider smooths concurrent bursts
below the provider's published RPM before a 429 can happen.
Reactive: exponential backoff with full jitter, honoring Retry-After.
"""
from __future__ import annotations

import asyncio
import random

from aiolimiter import AsyncLimiter

BACKOFF_BASE_SECONDS = 1.0
BACKOFF_CAP_SECONDS = 60.0


def backoff_delay(attempt: int, base: float = BACKOFF_BASE_SECONDS, cap: float = BACKOFF_CAP_SECONDS) -> float:
    """Full-jitter exponential backoff: uniform(0, min(cap, base * 2^attempt)).

    Full jitter prevents synchronized retry stampedes when hundreds of
    concurrent workers hit a rate limit at the same moment.
    """
    return random.uniform(0, min(cap, base * (2 ** attempt)))


async def sleep_backoff(attempt: int, retry_after: float | None = None) -> float:
    """Sleep for the computed backoff (or the server-mandated Retry-After)."""
    delay = retry_after if retry_after and retry_after > 0 else backoff_delay(attempt)
    delay = min(delay, BACKOFF_CAP_SECONDS)
    await asyncio.sleep(delay)
    return delay


class ProviderLimiter:
    """Requests-per-minute token bucket for one upstream provider."""

    def __init__(self, rpm: int) -> None:
        self._limiter = AsyncLimiter(max_rate=rpm, time_period=60)

    async def __aenter__(self) -> "ProviderLimiter":
        await self._limiter.acquire()
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None
