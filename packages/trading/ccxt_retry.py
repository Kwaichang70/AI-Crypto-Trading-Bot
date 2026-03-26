"""
packages/trading/ccxt_retry.py
-------------------------------
Exponential backoff retry wrapper for CCXT async API calls.

Usage
-----
    from trading.ccxt_retry import ccxt_retry

    ticker = await ccxt_retry(
        exchange.fetch_ticker, "BTC/USDT",
        max_retries=2, base_delay=1.0, operation="fetch_ticker(BTC/USDT)",
    )

Retry policy
------------
- Retries on any Exception except KeyboardInterrupt, SystemExit, and
  asyncio.CancelledError (those propagate immediately).
- Delay between attempt n and attempt n+1:
    min(base_delay * 2^n + uniform(0, 1), max_delay)
- After all retries are exhausted the last exception is re-raised so the
  caller's existing error-handling path is always exercised.
"""

from __future__ import annotations

import asyncio
import random
from typing import Any, Callable, TypeVar

import structlog

from trading.ccxt_errors import translate_ccxt_error

__all__ = ["ccxt_retry"]

logger = structlog.get_logger(__name__)

T = TypeVar("T")


async def ccxt_retry(
    fn: Callable[..., Any],
    *args: Any,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    operation: str = "ccxt_call",
    **kwargs: Any,
) -> Any:
    """
    Call an async CCXT function with exponential backoff + jitter.

    Retries on any Exception except KeyboardInterrupt, SystemExit, and
    asyncio.CancelledError. Raises the last exception if all retries are
    exhausted.

    Parameters
    ----------
    fn : callable
        The async CCXT method to call (e.g. ``exchange.fetch_ticker``).
    *args : Any
        Positional arguments forwarded to ``fn``.
    max_retries : int
        Maximum number of retry attempts (default 3 = up to 4 total calls).
    base_delay : float
        Initial delay in seconds before the first retry.
    max_delay : float
        Maximum delay cap in seconds.
    operation : str
        Human-readable name used in log messages.
    **kwargs : Any
        Keyword arguments forwarded to ``fn``.

    Returns
    -------
    Any
        The return value of ``fn`` on success.

    Raises
    ------
    Exception
        The last exception raised by ``fn`` after all retries are exhausted.
    """
    last_exc: Exception | None = None

    for attempt in range(max_retries + 1):
        try:
            return await fn(*args, **kwargs)
        except (KeyboardInterrupt, SystemExit, asyncio.CancelledError):
            raise
        except Exception as exc:
            last_exc = exc
            if attempt >= max_retries:
                break

            # Exponential backoff with full jitter (AWS-style)
            delay = min(
                base_delay * (2**attempt) + random.uniform(0, 1),
                max_delay,
            )

            logger.warning(
                "ccxt_retry.retrying",
                operation=operation,
                attempt=attempt + 1,
                max_retries=max_retries,
                delay=round(delay, 2),
                error=str(exc)[:200],
                user_message=translate_ccxt_error(exc),
            )

            await asyncio.sleep(delay)

    logger.error(
        "ccxt_retry.exhausted",
        operation=operation,
        max_retries=max_retries,
        error=str(last_exc)[:200],
        user_message=translate_ccxt_error(last_exc) if last_exc is not None else "Unknown error",
    )
    raise last_exc  # type: ignore[misc]
