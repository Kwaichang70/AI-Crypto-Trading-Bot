"""CF-97-03 P7b1: behaviour of the egress budget (C1, reservation, processor, clamp, riders).

Deterministic: no wall time asserted, no Postgres.  The budget is process-global, so every test
starts and ends with ``_reset_for_tests``.  Layer-1 logging tests use ``production_logging``.
"""

from __future__ import annotations

import asyncio
import time
import types
from collections.abc import Iterator
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from structlog.testing import capture_logs

from api.config import get_settings
from api.services import admission, spawn_hooks
from api.services import egress_budget as eb
from api.services.retraining import RetrainingService

_SYMS = ["S0/USDT"]


@pytest.fixture(autouse=True)
def _budget(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://test:test@localhost:5432/test")
    monkeypatch.setenv("EXCHANGE_ID", "kraken")
    get_settings.cache_clear()
    eb._reset_for_tests()
    admission._reset_for_tests()
    yield
    eb._reset_for_tests()
    admission._reset_for_tests()
    get_settings.cache_clear()


def _task() -> asyncio.Future[None]:
    return asyncio.ensure_future(asyncio.Event().wait())  # type: ignore[arg-type]


async def _end(task: asyncio.Future[Any]) -> None:
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    await asyncio.sleep(0)  # the done-callback runs


# --------------------------------------------------------------------------------------------
# pin 3 (C1) and pin 5 (reservation)
# --------------------------------------------------------------------------------------------
async def test_c1_an_old_task_ending_last_never_removes_the_resumed_weight() -> None:
    a, b = _task(), _task()
    spawn_hooks.on_run_spawned("rid", "live", _SYMS * 2, "1m", a)
    a.cancel()  # the stop: its done-callback is still pending
    spawn_hooks.on_run_spawned("rid", "live", _SYMS * 2, "1m", b)  # the quick resume, same id
    await asyncio.gather(a, return_exceptions=True)
    await asyncio.sleep(0)
    assert eb._weights == {"rid": ("kraken", 2.0)} and eb._owner["rid"] is b
    await _end(b)
    assert eb._weights == {} and eb._owner == {}


async def test_reservations_admit_exactly_what_fits_across_a_forced_await() -> None:
    async def create(i: int) -> bool:
        if not eb.reserve_new_run(f"r{i}", "paper", _SYMS * 8, "1m"):
            return False
        await asyncio.sleep(0)  # the commit between the check and the spawn
        spawn_hooks.on_run_spawned(f"r{i}", "paper", _SYMS * 8, "1m", _task())
        return True

    results = await asyncio.gather(*(create(i) for i in range(5)))
    assert sum(results) == 2 and eb._reserved == {}  # 8 + 8 <= 20 < 24; spawns consumed them
    assert sorted(w for _, w in eb._weights.values()) == [8.0, 8.0]


def test_a_reservation_counts_expires_and_is_consumed_without_a_double_count() -> None:
    before = time.monotonic()
    assert eb.reserve_new_run("a", "paper", _SYMS * 20, "1m")
    ex, wt, deadline = eb._reserved["a"]
    assert (ex, wt) == ("kraken", 20.0) and 119.0 < deadline - before < 121.0  # the 120 s TTL
    assert not eb.reserve_new_run("b", "paper", _SYMS, "1m")  # the held weight counts
    eb._reserved["a"] = (ex, wt, time.monotonic() - 1)  # expired
    assert eb.reserve_new_run("c", "paper", _SYMS * 20, "1m")  # an expired reservation frees it
    eb.register("c", "paper", _SYMS * 20, "1m")
    assert "c" not in eb._reserved and eb._weights["c"][1] == 20.0  # consumed, counted once


def test_a_failed_create_keeps_its_share_for_the_ttl_and_never_blocks_a_protection_path() -> None:
    assert eb.reserve_new_run("dead", "paper", _SYMS * 20, "1m")  # the create dies before its spawn
    assert not eb.reserve_new_run("next", "paper", _SYMS, "1m")  # stale share: new runs refused
    with capture_logs() as logs:
        eb.register("resume", "live", _SYMS * 5, "1m")  # protection: never reserves, never refused
    assert "resume" in eb._weights and "resume" not in eb._reserved
    assert [e["event"] for e in logs] == ["egress.cap_exceeded_on_protection"]
    assert set(logs[0]) == {"event", "log_level", "exchange", "used", "new"}


def test_the_cap_comparison_runs_with_the_lock_held(monkeypatch: pytest.MonkeyPatch) -> None:
    class Lock:  # records whether the lock is held; no threads, no timing
        held = False

        def __enter__(self) -> None:
            Lock.held = True

        def __exit__(self, *_: object) -> None:
            Lock.held = False

    seen: list[bool] = []
    real = eb._used
    monkeypatch.setattr(eb, "_lock", Lock())
    monkeypatch.setattr(eb, "_used", lambda ex, now: (seen.append(Lock.held), real(ex, now))[1])
    assert eb.reserve_new_run("a", "paper", _SYMS, "1m") and "a" in eb._reserved
    assert seen == [True]


def test_an_expired_reservation_is_discarded_by_the_next_reserve() -> None:
    for rid in ("a", "b"):
        assert eb.reserve_new_run(rid, "paper", _SYMS * 5, "1m")
        eb._reserved[rid] = ("kraken", 5.0, time.monotonic() - 1)  # both expired
    assert eb._used("kraken", time.monotonic()) == 0.0  # an expired share is not counted
    assert eb.reserve_new_run("c", "paper", _SYMS * 5, "1m")
    assert set(eb._reserved) == {"c"} and eb._used("kraken", time.monotonic()) == 5.0


def test_a_purge_deletes_only_the_expired_reservation_and_the_live_one_still_counts() -> None:
    for rid in ("old", "new"):
        assert eb.reserve_new_run(rid, "paper", _SYMS * 10, "1m")
    eb._reserved["old"] = ("kraken", 10.0, time.monotonic() - 1)  # expired; "new" is not
    assert not eb.reserve_new_run("c", "paper", _SYMS * 11, "1m")  # 10 + 11 > 20: "new" counts
    assert set(eb._reserved) == {"new"}  # the refused call still purged "old"
    assert eb.reserve_new_run("d", "paper", _SYMS * 10, "1m")  # 10 + 10 <= 20


def test_an_unlisted_exchange_gets_the_unknown_cap_not_a_listed_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EXCHANGE_ID", "bitstamp")
    get_settings.cache_clear()
    assert not eb.reserve_new_run("big", "paper", _SYMS * 21, "1m")  # 21 > 20, though < binance 100
    assert eb.reserve_new_run("fits", "paper", _SYMS * 20, "1m")  # exactly the cap
    assert not eb.reserve_new_run("more", "paper", _SYMS, "1h")  # 20 + 1/60 is over it
    assert set(eb._reserved) == {"fits"} and eb._reserved["fits"][0] == "unknown"


def test_a_history_suspension_is_never_shortened_by_a_later_shorter_one() -> None:
    eb.suspend_history("kraken", 600)
    longest = eb.history_suspended_until["kraken"]
    eb.suspend_history("kraken", 30)  # a later, shorter Retry-After
    assert eb.history_suspended_until["kraken"] == longest


def test_the_reservation_fails_closed_and_leaves_nothing() -> None:
    assert not eb.reserve_new_run("x", "paper", _SYMS, "7x")  # an unresolvable timeframe
    assert not eb.reserve_new_run("y", "paper", _SYMS, None)
    assert eb._reserved == {}


async def test_c1_with_a_reservation_for_the_same_run_id() -> None:
    assert eb.reserve_new_run("rid", "paper", _SYMS * 4, "1m")
    a, b = _task(), _task()
    spawn_hooks.on_run_spawned("rid", "paper", _SYMS * 4, "1m", a)
    spawn_hooks.on_run_spawned("rid", "paper", _SYMS * 4, "1m", b)
    assert eb._reserved == {} and eb._weights == {"rid": ("kraken", 4.0)}
    await _end(a)
    assert eb._weights == {"rid": ("kraken", 4.0)}  # the resume's weight survives A
    await _end(b)
    assert eb._weights == {}


# --------------------------------------------------------------------------------------------
# pin 8: the processor
# --------------------------------------------------------------------------------------------
def _emit(event: str, error_type: str | None, exchange: str | None = None) -> dict[str, Any]:
    ed: dict[str, Any] = {"event": event, "error_type": error_type, "x": 1}
    if exchange:
        ed["exchange"] = exchange
    assert eb.live_429_processor(None, "warning", ed) is ed  # the same object, returned
    return ed


_CASES = [
    ("rate_limit_hit", None, 60.0), ("engine.poll_fetch_failed", "RateLimitError", 60.0),
    ("live.order_submission_ambiguous", "RateLimitExceeded", 60.0),
    ("live.x", "DDoSProtection", 60.0), ("ccxt_retry.retrying", "DDoSProtection", 60.0),
    ("network_error_retrying", "DDoSProtection", 600.0),
    ("network_error_retrying", "NetworkError", 0.0),
    ("network_error_retrying", "RequestTimeout", 0.0),
    ("network_error_retrying", "ExchangeNotAvailable", 0.0), ("network_error_retrying", None, 0.0),
    ("engine.poll_fetch_failed", "NetworkError", 0.0), ("live.x", None, 0.0),
    ("other.event", "RateLimitError", 0.0),
]  # fmt: skip


@pytest.mark.parametrize(("event", "error_type", "seconds"), _CASES)
def test_the_closed_event_list_and_the_suspension_seconds(
    event: str, error_type: str | None, seconds: float
) -> None:
    _emit(event, error_type, "kraken")
    got = eb._LIVE_RATE_LIMITED_UNTIL.get("kraken")
    assert (got is not None) == bool(seconds)
    assert got is None or abs(got - time.monotonic() - seconds) < 1.0


def test_no_or_an_unlisted_exchange_id_suspends_all_and_a_suspension_never_shortens() -> None:
    _emit("network_error_retrying", "DDoSProtection")  # no exchange id
    assert sorted(eb._LIVE_RATE_LIMITED_UNTIL) == ["*"]
    longest = eb._LIVE_RATE_LIMITED_UNTIL["*"]
    _emit("rate_limit_hit", None, "unlisted-exchange")  # a later 60 s event, same key
    assert (
        sorted(eb._LIVE_RATE_LIMITED_UNTIL) == ["*"] and eb._LIVE_RATE_LIMITED_UNTIL["*"] >= longest
    )
    eb._LIVE_RATE_LIMITED_UNTIL["*"] = time.monotonic() + 100.0
    _emit("network_error_retrying", "DDoSProtection")  # a DDoS event extends it to 600 s
    assert eb._LIVE_RATE_LIMITED_UNTIL["*"] - time.monotonic() > 590.0


def test_a_hostile_event_dict_never_raises() -> None:
    class Hostile(dict[str, Any]):
        def get(self, *_: Any, **__: Any) -> Any:
            raise RuntimeError("boom")

    hostile = Hostile()
    assert eb.live_429_processor(None, "warning", hostile) is hostile


# --------------------------------------------------------------------------------------------
# pin 9 (clamp) and pin 13 (the kraken consequence)
# --------------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("retry_after", "seconds"),
    [
        (5, 30.0),
        (45, 45.0),
        (9999, 600.0),
        (None, 60.0),
        (-1, 60.0),
        (True, 60.0),
        ("x", 60.0),
        (30.0, 30.0),
        (600, 600.0),
    ],
)
def test_the_retry_after_clamp_and_the_default(retry_after: object, seconds: float) -> None:
    eb.suspend_history("kraken", retry_after)
    assert abs(eb.history_suspended_until["kraken"] - time.monotonic() - seconds) < 1.0


def test_with_any_kraken_weight_only_the_first_history_request_of_a_burst_proceeds() -> None:
    def burst() -> list[object]:
        out: list[object] = []
        for _ in range(3):
            try:
                out.append(round(eb.reserve_history("kraken"), 2))
            except eb.HistoryBudgetExhausted:
                out.append("refused")
        return out

    assert burst() == [0.0, 5.0, "refused"]  # idle: the second waits the 5 s maximum
    for weight in (1.0, 0.5):
        eb._reset_for_tests()
        eb._weights["live"] = ("kraken", weight)
        assert burst() == [0.0, "refused", "refused"]


# --------------------------------------------------------------------------------------------
# pin 14 / 15: the riders (behaviour)
# --------------------------------------------------------------------------------------------
def _scheduler() -> RetrainingService:
    session = MagicMock()
    session.execute = AsyncMock(return_value=MagicMock(scalar_one=MagicMock(return_value=99)))
    ctx = MagicMock()
    ctx.__aenter__.return_value, ctx.__aexit__.return_value = session, False
    return RetrainingService(db_session_factory=lambda: ctx, min_trades_for_retrain=50)


async def _flag_after_a_scheduler_run(svc: RetrainingService) -> bool:
    try:
        await svc._check_and_retrain("BTC/USDT", "1h")
    except (RuntimeError, asyncio.CancelledError):
        pass
    return admission._IN_RESEARCH.get()


async def test_the_scheduler_resets_the_research_flag_in_its_own_context() -> None:
    svc, seen = _scheduler(), []

    async def job(**_: Any) -> None:
        seen.append(admission._IN_RESEARCH.get())

    svc._do_retrain = job  # type: ignore[method-assign]
    assert await asyncio.create_task(_flag_after_a_scheduler_run(svc)) is False
    assert admission._held is None and seen


@pytest.mark.parametrize("outcome", ["raises", "cancelled"])
async def test_the_scheduler_resets_the_flag_when_the_job_raises_or_is_cancelled(
    outcome: str,
) -> None:
    svc, gate = _scheduler(), asyncio.Event()

    async def job(**_: Any) -> None:
        if outcome == "raises":
            raise RuntimeError("job failed")
        await gate.wait()

    svc._do_retrain = job  # type: ignore[method-assign]
    loop_task = asyncio.create_task(_flag_after_a_scheduler_run(svc))
    for _ in range(5):
        await asyncio.sleep(0)
    if outcome == "cancelled":
        loop_task.cancel()  # the polling loop is cancelled while the job runs
    assert await loop_task is False  # released in `finally`: the flag does not stay set
    gate.set()
    for _ in range(3):
        await asyncio.sleep(0)
    assert admission._held is None


async def test_a_task_lease_release_resets_the_flag_where_it_was_set() -> None:
    lease = admission.try_acquire("backtest")
    assert admission._IN_RESEARCH.get() is True
    lease.release()
    assert admission._IN_RESEARCH.get() is False and lease.token is None


async def test_the_checkin_metric_has_its_name_and_a_duration_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    got: dict[str, Any] = {}

    def listens_for(_t: Any, name: str) -> Any:
        return lambda fn: got.setdefault(name, fn)

    rec = MagicMock()
    monkeypatch.setattr(admission, "metrics", rec)
    monkeypatch.setattr(admission, "event", MagicMock(listens_for=listens_for))
    monkeypatch.setattr(
        "api.config.get_settings",
        lambda: types.SimpleNamespace(db_pool_size=10, db_max_overflow=20),
    )
    admission._install_pool_timing()
    got["checkin"](None, types.SimpleNamespace(info={"research_t0": time.monotonic() - 2.0}))
    name, value = rec.observe.call_args.args
    assert name == "admission_checkout_seconds" and 1.5 < value < 5.0  # the S3 and S4 survivors
