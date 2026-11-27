# ruff: noqa: S311  (a seeded stdlib generator builds a deterministic price path)
"""WP24-T-13 / T-14: C10 is closed -- a backtest never reads a provider client or the wall clock.

Binding spec: ``reports/vp2-wp2.4/synthesis-spec.md`` (SY-24-02, SY-24-13 Phase A, G6/G8,
WP24-T-13, T-14, T-18 sanity).

Before WP2.4 the engine injected the process-global FGI / CoinGecko / FRED / Whale clients'
boot-frozen values (and a wall-clock "7 days ago") into every historical bar. Now a BACKTEST
engine defaults to ``UNAVAILABLE``: with the API's boot-warmed clients installed the run is
byte-identical to a run with no clients at all (so the API equals a script, and every golden
digest stays put), and no client attribute, getter or wall clock is touched.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from datetime import UTC, datetime
from decimal import Decimal
from types import SimpleNamespace
from typing import Any

import pytest

from common.types import TimeFrame
from tests.integration.fakes.bar_paths import scripted_path
from trading import clock as clock_mod
from trading.backtest import BacktestRunner
from trading.costs import get_cost_profile
from trading.metrics import BacktestResult
from trading.strategies.dca_rsi_hybrid import DCARSIHybridStrategy
from trading.strategies.rsi_mean_reversion import RSIMeanReversionStrategy
from trading.strategy import BaseStrategy

SYM = "BTC/USDT"
FIRST_OPEN = datetime(2026, 3, 10, 0, 0, tzinfo=UTC)
_MODULES = ("sentiment", "market_signals", "macro_data", "whale_tracker")


def _bars() -> dict[str, Any]:
    rnd = random.Random(5)
    closes = [round(100 + 12 * math.sin(i / 5) + rnd.uniform(-1, 1), 2) for i in range(300)]
    return {SYM: scripted_path(closes, first_open=FIRST_OPEN, symbol=SYM, warmup=50)}


def result_digest(result: BacktestResult) -> str:
    """sha256 over everything deterministic in a result (random trade/run ids excluded)."""
    dump = result.model_dump(
        mode="json",
        exclude={"run_id": True, "trades": {"__all__": {"trade_id", "run_id"}}},
    )
    return hashlib.sha256(json.dumps(dump, sort_keys=True, default=str).encode()).hexdigest()


async def _run(cls: type[BaseStrategy]) -> BacktestResult:
    runner = BacktestRunner(
        [cls("s1", {})],
        [SYM],
        TimeFrame.ONE_HOUR,
        seed=7,
        cost_profile=get_cost_profile("legacy_backtest_v0"),
    )
    return await runner.run(_bars())


class _TouchedClient:
    """A provider client whose every attribute access is recorded and then fails."""

    def __init__(self, name: str, log: list[str]) -> None:
        object.__setattr__(self, "_name", name)
        object.__setattr__(self, "_log", log)

    def __getattribute__(self, attr: str) -> Any:
        if attr in {"_name", "_log"} or attr.startswith("__"):  # dunders: pytest/introspection
            return object.__getattribute__(self, attr)
        object.__getattribute__(self, "_log").append(
            f"{object.__getattribute__(self, '_name')}.{attr}"
        )
        raise AssertionError("a backtest touched a provider client")


def _install_raising_clients(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Raising fakes in all four modules, plus recorders on the four getters."""
    import importlib

    log: list[str] = []
    for name in _MODULES:
        module = importlib.import_module(f"data.{name}")
        monkeypatch.setattr(module, "_global_client", _TouchedClient(name, log))

        def _getter(name: str = name) -> None:
            log.append(f"get_global_client:{name}")
            raise AssertionError("a backtest asked for a provider client")

        monkeypatch.setattr(module, "get_global_client", _getter)
    return log


def _uninstall_clients(monkeypatch: pytest.MonkeyPatch) -> None:
    import importlib

    for name in _MODULES:
        monkeypatch.setattr(importlib.import_module(f"data.{name}"), "_global_client", None)


_CONSUMERS = [RSIMeanReversionStrategy, DCARSIHybridStrategy]


class TestT13C10Kill:
    @pytest.mark.parametrize("cls", _CONSUMERS, ids=lambda c: c.__name__)
    async def test_raising_clients_are_never_touched_and_the_result_equals_no_clients(
        self, cls: type[BaseStrategy], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _uninstall_clients(monkeypatch)
        baseline = await _run(cls)

        touched = _install_raising_clients(monkeypatch)
        with_clients = await _run(cls)

        assert touched == []  # no getter call, no attribute access
        assert result_digest(with_clients) == result_digest(baseline)
        assert len(baseline.trades) > 2  # the consumer really traded

    async def test_control_the_raising_fakes_do_record_a_touch(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        touched = _install_raising_clients(monkeypatch)
        from data.sentiment import get_global_client

        with pytest.raises(AssertionError):
            get_global_client()
        assert touched == ["get_global_client:sentiment"]

    @pytest.mark.parametrize("cls", _CONSUMERS, ids=lambda c: c.__name__)
    async def test_populated_boot_clients_do_not_change_a_backtest(
        self, cls: type[BaseStrategy], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The pre-WP2.4 defect: boot-warmed clients with real values leaked into history."""
        import importlib

        _uninstall_clients(monkeypatch)
        baseline = await _run(cls)

        populated = {
            "sentiment": SimpleNamespace(cached_value=9, value_at_offset_from_cache=lambda **k: 90),
            "market_signals": SimpleNamespace(
                cached_value=SimpleNamespace(
                    btc_dominance=66.0, market_cap_change_24h=2.0, total_volume_change_24h=3.0
                ),
                btc_dominance_at_offset_from_cache=lambda **k: 40.0,
            ),
            "macro_data": SimpleNamespace(
                cached_value=SimpleNamespace(fed_funds_rate=5.0, yield_curve_spread=-1.5)
            ),
            "whale_tracker": SimpleNamespace(cached_value=SimpleNamespace(net_flow=9e7)),
        }
        for name, client in populated.items():
            monkeypatch.setattr(importlib.import_module(f"data.{name}"), "_global_client", client)
        assert result_digest(await _run(cls)) == result_digest(baseline)


class TestT14WallTripwire:
    @pytest.mark.parametrize("cls", _CONSUMERS, ids=lambda c: c.__name__)
    async def test_wall_reads_raise_and_the_result_is_identical(
        self, cls: type[BaseStrategy], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from data import market_signals, sentiment

        _uninstall_clients(monkeypatch)
        baseline = await _run(cls)

        reads: list[str] = []

        def boom(name: str) -> Any:
            def _raise(*_a: Any, **_k: Any) -> Any:
                reads.append(name)
                raise AssertionError(f"wall clock read in the backtest path: {name}")

            return _raise

        class _NoNow(datetime):
            @classmethod
            def now(cls, tz: Any = None) -> Any:  # type: ignore[override]
                reads.append("datetime.now")
                raise AssertionError("wall clock read in the backtest path: datetime.now")

        monkeypatch.setattr(clock_mod, "_wall_now", boom("trading.clock._wall_now"))
        monkeypatch.setattr("trading.portfolio._utc_now", boom("trading.portfolio._utc_now"))
        monkeypatch.setattr(sentiment, "datetime", _NoNow)
        monkeypatch.setattr(market_signals, "datetime", _NoNow)
        # the engine must not even get as far as a provider module with the clients installed
        touched = _install_raising_clients(monkeypatch)

        result = await _run(cls)
        assert reads == [] and touched == []
        assert result_digest(result) == result_digest(baseline)

    async def test_control_the_tripwires_do_fire(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from data import sentiment

        class _NoNow(datetime):
            @classmethod
            def now(cls, tz: Any = None) -> Any:  # type: ignore[override]
                raise AssertionError("wall clock read")

        monkeypatch.setattr(sentiment, "datetime", _NoNow)
        with pytest.raises(AssertionError, match="wall clock read"):
            sentiment.datetime.now(tz=UTC)


class TestT18DefaultPathIsTheNoClientsPath:
    async def test_digest_is_stable_across_runs_and_a_stamp_difference_is_detected(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _uninstall_clients(monkeypatch)
        first = await _run(RSIMeanReversionStrategy)
        second = await _run(RSIMeanReversionStrategy)
        assert result_digest(first) == result_digest(second)  # determinism of the helper itself
        other = first.model_copy(update={"final_equity": first.final_equity + Decimal("1")})
        assert result_digest(other) != result_digest(first)  # and it is sensitive
