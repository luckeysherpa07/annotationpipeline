"""Shared retry helpers for synchronous and asynchronous Gemini calls."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from typing import TypeVar


T = TypeVar("T")


def is_quota_or_rate_limit_error(exc: BaseException | str) -> bool:
    text = str(exc).lower()
    return any(
        token in text
        for token in (
            "quota",
            "rate limit",
            "rate_limit",
            "429",
            "resource_exhausted",
        )
    )


def retry_wait_seconds(
    exc: BaseException,
    attempt: int,
    *,
    base_delay_seconds: float = 2.0,
    quota_delay_seconds: float = 30.0,
) -> float:
    delay = (
        quota_delay_seconds
        if is_quota_or_rate_limit_error(exc)
        else base_delay_seconds
    )
    return max(0.0, float(delay)) * max(1, attempt)


def call_with_retry(
    operation: Callable[[], T],
    *,
    max_attempts: int = 3,
    retry_if: Callable[[Exception], bool] | None = None,
    wait_seconds: Callable[[Exception, int], float] | None = None,
    label: str = "Gemini call",
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    attempts = max(1, int(max_attempts))
    should_retry = retry_if or (lambda _exc: True)
    resolve_wait = wait_seconds or (
        lambda exc, attempt: retry_wait_seconds(exc, attempt)
    )
    for attempt in range(1, attempts + 1):
        try:
            return operation()
        except Exception as exc:
            if attempt >= attempts or not should_retry(exc):
                raise
            delay = max(0.0, float(resolve_wait(exc, attempt)))
            print(
                f"{label} failed on attempt {attempt}/{attempts}; "
                f"retrying in {delay:g}s: {exc}",
                flush=True,
            )
            sleep(delay)
    raise RuntimeError(f"{label} failed")


async def call_with_retry_async(
    operation: Callable[[], T],
    *,
    max_attempts: int = 3,
    retry_if: Callable[[Exception], bool] | None = None,
    wait_seconds: Callable[[Exception, int], float] | None = None,
    label: str = "Gemini call",
) -> T:
    attempts = max(1, int(max_attempts))
    should_retry = retry_if or (lambda _exc: True)
    resolve_wait = wait_seconds or (
        lambda exc, attempt: retry_wait_seconds(exc, attempt)
    )
    for attempt in range(1, attempts + 1):
        try:
            return await asyncio.to_thread(operation)
        except Exception as exc:
            if attempt >= attempts or not should_retry(exc):
                raise
            delay = max(0.0, float(resolve_wait(exc, attempt)))
            print(
                f"{label} failed on attempt {attempt}/{attempts}; "
                f"retrying in {delay:g}s: {exc}",
                flush=True,
            )
            await asyncio.sleep(delay)
    raise RuntimeError(f"{label} failed")
