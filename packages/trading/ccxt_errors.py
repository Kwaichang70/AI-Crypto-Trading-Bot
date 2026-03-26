"""
packages/trading/ccxt_errors.py
--------------------------------
Translates raw CCXT exception messages to user-friendly descriptions.

Usage
-----
    from trading.ccxt_errors import translate_ccxt_error

    try:
        await exchange.create_order(...)
    except Exception as exc:
        logger.error("order_failed", user_message=translate_ccxt_error(exc))
"""
from __future__ import annotations

import ccxt

__all__ = ["translate_ccxt_error"]

# Map CCXT exception types to user-readable messages.
# More specific subclasses must appear before their parents so isinstance()
# matches the most precise type first.
_TRANSLATIONS: list[tuple[type, str]] = [
    (ccxt.InsufficientFunds, "Insufficient funds — your account balance is too low for this order."),
    (ccxt.InvalidOrder, "Invalid order — the order parameters were rejected by the exchange."),
    (ccxt.OrderNotFound, "Order not found — the order may have already been filled or cancelled."),
    (ccxt.AuthenticationError, "Authentication failed — check your API key and secret."),
    (ccxt.PermissionDenied, "Permission denied — your API key lacks the required permissions."),
    (ccxt.RateLimitExceeded, "Rate limited — too many requests. The bot will retry automatically."),
    (ccxt.ExchangeNotAvailable, "Exchange unavailable — the exchange may be experiencing issues."),
    (ccxt.RequestTimeout, "Request timeout — the exchange took too long to respond."),
    (ccxt.BadSymbol, "Invalid trading pair — this symbol is not available on the exchange."),
    (ccxt.BadRequest, "Bad request — the exchange rejected the request parameters."),
    # NetworkError is a parent of RequestTimeout — keep it after the more specific entry.
    (ccxt.NetworkError, "Network error — could not connect to the exchange."),
    # ExchangeError is the base for most CCXT exceptions — keep it last.
    (ccxt.ExchangeError, "Exchange error — an unexpected error occurred on the exchange."),
]


def translate_ccxt_error(exc: Exception) -> str:
    """
    Return a user-friendly message for a CCXT exception.

    Iterates the translation table in order so that more specific subclasses
    are matched before their parent classes.  Falls back to the raw exception
    string (truncated to 200 chars) if no entry matches.

    Parameters
    ----------
    exc:
        Any exception, typically raised by a CCXT async method.

    Returns
    -------
    str:
        A human-readable error description.
    """
    for exc_type, message in _TRANSLATIONS:
        if isinstance(exc, exc_type):
            return message
    return f"Exchange error: {str(exc)[:200]}"
