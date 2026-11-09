"""Live context cadence policy (CF-24-01 P3; SY-01-08, SY-01-21c, SY-01-22d/h, ER2-05).

Constants and pure helpers only, standard library plus ``data.point_in_time``. Randomness is
injected as a ``unit`` in [0, 1] and time as ``*_ms``; there is no clock, logging or I/O, so
retuning (CF-2401-03) is a reviewed constant change that never touches the quant-reviewed book.
It does not import ``live_context``; the shared 1 h bucket literal is equality-tested instead.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

from data.point_in_time import SERIES_RULES

FLOOR_S: Final = 60  # hard floor between two requests of one provider
CYCLE_BUDGET_S: Final = 30  # per-provider cycle bound (semaphore wait included, N-6)
ATTEMPT_BOUND_S: Final = 10  # one HTTP attempt (ClientTimeout total)
HEARTBEAT_TICK_S: Final = 60  # supervisor tick, independent of the provider intervals
MAX_RETRIES: Final = 2  # 3 attempts per cycle
BACKOFF_CAP_S: Final = 60
FAILED_CYCLES_BEFORE_DOUBLING: Final = 3
RETRY_AFTER_MAX_S: Final = 3_600
AUTH_FAILED_WAIT_S: Final = 21_600
JITTER_PCT: Final = 10
BUCKET_MS: Final = 3_600_000  # scheduling bucket of the snapshot series (SY-01-22d)
BUCKET_JITTER_MAX_S: Final = 300  # first attempt at bucket start + U(0, 300 s)
BUDGET_WINDOW_MS: Final = 3_600_000


@dataclass(frozen=True, slots=True)
class Cadence:
    interval_s: int
    budget_per_hour: int  # requests, retries included
    requests_per_cycle: int = 1  # FRED reads two legs per cycle
    bucketed: bool = False  # bucket-scheduled snapshot series (SY-01-22d)


CADENCE: Final[Mapping[str, Cadence]] = MappingProxyType(
    {
        "alternative_me:fgi": Cadence(3_600, 4),
        "coingecko:btc_dominance": Cadence(600, 12, bucketed=True),
        "fred:T10Y2Y": Cadence(21_600, 6, requests_per_cycle=2),
        "whale:net_flow_1h": Cadence(900, 8, bucketed=True),
    }
)


def max_gap_s(series_id: str) -> int:
    """cap/4 in seconds: the longest own gap between attempts after a failure."""
    return SERIES_RULES[series_id].max_age_ms // 4_000


def next_interval_s(series_id: str, failed_cycles: int) -> int:
    """Table interval; from the 4th consecutive failed cycle it doubles per failure, <= cap/4."""
    extra = min(max(0, failed_cycles - FAILED_CYCLES_BEFORE_DOUBLING), 20)
    return max(FLOOR_S, min(CADENCE[series_id].interval_s << extra, max_gap_s(series_id)))


def retry_after_next_s(retry_after_s: int) -> int:
    """Next poll after a header Retry-After: ``max(floor, min(RA, 3600 s))`` (may exceed cap/4)."""
    return max(FLOOR_S, min(retry_after_s, RETRY_AFTER_MAX_S))


def wait_fits_cycle(wait_s: float, remaining_s: float) -> bool:
    """An in-cycle wait is taken only if one more attempt still fits the cycle (SY-01-21c)."""
    return wait_s + ATTEMPT_BOUND_S <= remaining_s


def backoff_s(retry: int, unit: float) -> float:
    """``min(1 s * 2**retry + U(0, 1), 60 s)`` for the zero-based retry index."""
    return min(float(1 << min(retry, 6)) + unit, float(BACKOFF_CAP_S))
