"""CF-97-03 P7b: the per-exchange egress budget (weights, caps, history cooldown, reservation).

Keyed by ``exchange_id`` only (a closed key: an unlisted exchange is ``unknown``); it holds no
key material, no account, no URL and no service handle.  One ``threading.Lock`` guards O(1)
arithmetic on plain dicts and nothing logs inside it (pinned), so a history fetch on another loop
or in a worker thread cannot bypass it and a logging processor can never deadlock on it.  Live and
paper calls never call into this module: their cost is only *registered* (weights) at the spawn
sites, and a live 429 is *observed* by a lock-free structlog processor.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import MutableMapping
from typing import Any, Final

import structlog

from common.types import TimeFrame
from trading.strategy_engine import _TIMEFRAME_SECONDS

CAPS: Final = {"kraken": 20.0, "binance": 100.0, "coinbase": 50.0}  # weight units; code constants
UNKNOWN_CAP: Final = 20.0
# Provisional history rates in req/s (0.5 x the documented limit; a lower sourced figure wins).
HISTORY_RATE: Final = {"kraken": 0.2, "coinbase": 3.0, "binance": 5.0}
UNKNOWN_RATE: Final = 0.5
MIN_SPACING_S: Final = 0.1  # the existing _MIN_REQUEST_INTERVAL: history never goes faster
RATE_FLOOR: Final = 0.05
MAX_WAIT_S: Final = 5.0
COOLDOWN_S: Final = (30.0, 600.0)  # clamp of a parsed Retry-After
DEFAULT_COOLDOWN_S: Final = 60.0
LIVE_SUSPEND_S: Final = 60.0
DDOS_SUSPEND_S: Final = 600.0  # DDoSProtection arrives as a NetworkError retry: a ban outlasts 60 s
RESERVATION_TTL_S: Final = 120.0  # a create that dies before its spawn frees its weight after this
_RATE_LIMIT_TYPES: Final = frozenset({"RateLimitError", "RateLimitExceeded", "DDoSProtection"})
_log = structlog.get_logger("egress_budget")
_lock = threading.Lock()
_weights: dict[str, tuple[str, float]] = {}  # run id -> (exchange key, weight)
_reserved: dict[str, tuple[str, float, float]] = {}  # run id -> (exchange, weight, deadline)
_owner: dict[str, object] = {}  # run id -> the task that registered it (a resume reuses the id)
_next_slot: dict[str, float] = {}
history_suspended_until: dict[str, float] = {}
_LIVE_RATE_LIMITED_UNTIL: dict[str, float] = {}  # written lock-free by the processor


class HistoryBudgetExhausted(Exception):
    """A history fetch is refused (cooldown, or a wait above ``MAX_WAIT_S``); fixed message."""

    def __init__(self) -> None:
        super().__init__("history fetch budget exhausted")


def key(exchange: object) -> str:
    return exchange if isinstance(exchange, str) and exchange in CAPS else "unknown"


def weight(mode: str, n_symbols: int, timeframe: object, exit_check_s: int | None) -> float:
    """Requests per minute of one run: its polls, plus the live intra-bar check if armed."""
    poll = _TIMEFRAME_SECONDS[TimeFrame(str(timeframe))]
    total = n_symbols * 60 / poll
    if mode == "live" and exit_check_s and 0 < exit_check_s < poll:
        total += n_symbols * 60 / exit_check_s
    return total


def _run_weight(mode: str, symbols: list[str], timeframe: object) -> tuple[str, float]:
    from api.config import get_settings

    cfg = get_settings()
    return key(cfg.exchange_id), weight(
        mode, len(symbols), timeframe, cfg.live_exit_check_interval_s
    )


def _used(ex: str, now: float) -> float:
    """Registered weight plus unexpired reservations of one exchange (call under ``_lock``)."""
    live = sum(w for e, w in _weights.values() if e == ex)
    return live + sum(w for e, w, deadline in _reserved.values() if e == ex and deadline > now)


def reserve_new_run(run_id: object, mode: str, symbols: list[str], timeframe: object) -> bool:
    """Create/promote: check the cap and reserve the weight in ONE locked step (concurrent creates
    cannot all pass).  ``register`` at the spawn consumes the reservation.  Fail-closed."""
    try:
        ex, new = _run_weight(mode, symbols, timeframe)
        now = time.monotonic()
        with _lock:
            for rid in [r for r, (_, _, deadline) in _reserved.items() if deadline <= now]:
                del _reserved[rid]  # expired reservations are discarded, not just ignored
            if _used(ex, now) + new > CAPS.get(ex, UNKNOWN_CAP):
                return False
            _reserved[str(run_id)] = (ex, new, now + RESERVATION_TTL_S)
        return True
    except Exception:
        _log.error("egress.cap_check_failed")
        return False


def register(
    run_id: str, mode: str, symbols: list[str], timeframe: object, owner: object = None
) -> None:
    """Add a spawned run's weight; over the cap it only alerts (protection paths never refuse)."""
    ex, new = _run_weight(mode, symbols, timeframe)
    with _lock:
        _reserved.pop(run_id, None)  # consumed: the run is never counted twice
        used = _used(ex, time.monotonic())
        _weights[run_id] = (ex, new)
        _owner[run_id] = owner
    if used + new > CAPS.get(ex, UNKNOWN_CAP):
        _log.warning("egress.cap_exceeded_on_protection", exchange=ex, used=used, new=new)


def unregister(run_id: str, owner: object = None) -> None:
    """Remove the entry only if ``owner`` registered it: an old task must not delete a resume's."""
    with _lock:
        if _owner.get(run_id) is owner:
            _weights.pop(run_id, None)
            _owner.pop(run_id, None)


def _suspended_until(ex: str) -> float:
    return max(
        history_suspended_until.get(ex, 0.0),
        _LIVE_RATE_LIMITED_UNTIL.get(ex, 0.0),
        _LIVE_RATE_LIMITED_UNTIL.get("*", 0.0),
    )


def history_check(exchange: object) -> None:
    """The fetch-start check: refuse while the exchange's history cooldown is in force."""
    with _lock:
        suspended = _suspended_until(key(exchange)) > time.monotonic()
    if suspended:
        raise HistoryBudgetExhausted


def suspend_history(exchange: object, retry_after: object = None) -> None:
    """A history 429/DDoS after the frozen retries: ``Retry-After`` clamped, else the default."""
    secs = DEFAULT_COOLDOWN_S
    if (
        isinstance(retry_after, int | float)
        and not isinstance(retry_after, bool)
        and retry_after >= 0
    ):
        secs = min(max(float(retry_after), COOLDOWN_S[0]), COOLDOWN_S[1])
    with _lock:
        slot = key(exchange)  # never shorten an earlier, longer suspension
        history_suspended_until[slot] = max(
            history_suspended_until.get(slot, 0.0), time.monotonic() + secs
        )
    _log.warning("egress.history_cooldown", seconds=secs)


def reserve_history(exchange: object) -> float:
    """Seconds the caller must wait before one history request (the caller sleeps, outside)."""
    ex, now = key(exchange), time.monotonic()
    with _lock:
        if _suspended_until(ex) > now:
            raise HistoryBudgetExhausted
        live = sum(w for e, w in _weights.values() if e == ex) / 60.0
        rate = min(
            1.0 / MIN_SPACING_S, max(RATE_FLOOR, HISTORY_RATE.get(ex, UNKNOWN_RATE) - 2 * live)
        )
        start = max(now, _next_slot.get(ex, 0.0))
        if start - now > MAX_WAIT_S:
            raise HistoryBudgetExhausted
        _next_slot[ex] = start + 1.0 / rate
    return start - now


def reserve_history_blocking(exchange: object) -> None:
    """For a worker thread (sync ccxt): the same reservation, slept in that thread."""
    if (delay := reserve_history(exchange)) > 0:
        time.sleep(delay)


def live_429_processor(
    _logger: object, _method: str, event_dict: MutableMapping[str, Any]
) -> MutableMapping[str, Any]:
    """structlog processor: a live/paper 429 suspends history for 60 s, a DDoSProtection retry for
    600 s (never shortening a suspension).  No lock, no I/O, no log, never raises."""
    try:
        name, error = event_dict.get("event"), event_dict.get("error_type")
        secs = 0.0
        if name == "rate_limit_hit" or (
            isinstance(name, str)
            and name.startswith(("engine.poll_fetch_failed", "live.", "ccxt_retry."))
            and error in _RATE_LIMIT_TYPES
        ):
            secs = LIVE_SUSPEND_S
        elif name == "network_error_retrying" and error == "DDoSProtection":
            secs = DDOS_SUSPEND_S
        if secs:
            ex = event_dict.get("exchange")  # no (or an unlisted) exchange id: all exchanges
            slot = ex if ex in CAPS else "*"
            _LIVE_RATE_LIMITED_UNTIL[slot] = max(
                _LIVE_RATE_LIMITED_UNTIL.get(slot, 0.0), time.monotonic() + secs
            )
    except Exception:  # never raises into the logging call
        return event_dict
    return event_dict


def install_log_processor() -> None:
    """Register the processor first in BOTH chains: native structlog and the stdlib bridge."""
    chains = [structlog.get_config()["processors"]]
    for handler in logging.getLogger().handlers:
        chains.append(getattr(getattr(handler, "formatter", None), "foreign_pre_chain", None))
    for chain in chains:
        if isinstance(chain, list) and live_429_processor not in chain:
            chain.insert(0, live_429_processor)


def _reset_for_tests() -> None:
    with _lock:
        _weights.clear()
        _reserved.clear()
        _owner.clear()
        _next_slot.clear()
        history_suspended_until.clear()
    _LIVE_RATE_LIMITED_UNTIL.clear()
