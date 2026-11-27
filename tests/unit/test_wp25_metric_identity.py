"""WP2.5 / P1 + P2: metric-identity harness (WP25-T-09, WP25-T-17).

Binding spec: ``reports/vp2-wp2.5/synthesis-spec.md`` (SY-25-26, section 5 rows P1/P2,
WP25-T-09, WP25-T-17).

P2 (the behaviour switch) turns this harness from "identical" into "identical except the
allow-list": ``head_capture.json`` is NOT re-captured and NOT re-pinned.  Every field outside
``ALLOW_LISTED`` + ``NEW_FIELDS`` (equity curve, drawdown, fills, trades, final equity,
exposure, total_bars, dates, duration_days, ``max_drawdown_pct`` ...) must still equal the
01d393d capture; the allow-listed curve-derived metrics must differ in the documented direction
(per-bar n); the 7 new fields must be present.  The new metric VALUES are verified against an
independent oracle in ``tests/integration/test_wp25_runner.py``.

P1 was behaviour-neutral: the runner lifts its metric block into
``trading.metrics.compute_return_metrics`` but still feeds it the equity-curve POINT series, so
every number must equal the one the unmodified tree (``01d393d``) produced.  This module holds

* the scenario generator (``generate_capture`` / ``run_scenario``; not collected, no ``test_``
  prefix) that was run on a clean detached worktree at ``01d393d`` to write
  ``tests/fixtures/wp25/head_capture.json``; and
* the test that regenerates every scenario on the tree under test and requires the serialised
  ``BacktestResult.model_dump(mode="json")`` (minus ``run_id``) and the fills to be IDENTICAL.

The capture also holds the fields WP25-T-17 (P2) compares: ``equity_curve`` (timestamp, equity,
``drawdown_pct``), fills, trades, ``final_equity``, ``total_*``, ``exposure_*``, ``total_bars``,
``start_date`` / ``end_date`` / ``duration_days`` and ``max_drawdown_pct``.  P2 allow-lists only
the metric fields it deliberately changes.

Regenerating the capture on a later tree is a bug: the file is the HEAD reference.  The only
allowed difference to it in P1 was the zero-std edge (``compute_sharpe`` with std / |mean| <= 1e-9),
which none of these scenarios reaches.

Fills carry random UUIDs (``fill_id``, ``order_id``) and trades carry the random run id and a
``trade_id``; those keys are dropped, everything else is kept in engine order.
"""

# ruff: noqa: S311  (seeded stdlib generators build deterministic fixtures, not secrets)

from __future__ import annotations

import asyncio
import json
import math
import random
import sys
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from common.models import OHLCVBar
from common.types import TimeFrame
from data.bar_alignment import AlignmentTolerance
from trading.backtest import BacktestRunner
from trading.costs import get_cost_profile
from trading.strategies.ma_crossover import MACrossoverStrategy

FIXTURE_PATH = Path(__file__).resolve().parents[1] / "fixtures" / "wp25" / "head_capture.json"

T0 = datetime(2025, 1, 1, tzinfo=UTC)
_STEP = {TimeFrame.ONE_HOUR: timedelta(hours=1), TimeFrame.ONE_DAY: timedelta(days=1)}
_LENIENT = AlignmentTolerance(
    max_missing_ppm=1_000_000,
    warn_missing_ppm=1_000_000,
    max_gap_bars=10_000,
    warn_gap_bars=10_000,
    max_head_gap_bars=10_000,
    max_tail_gap_bars=10_000,
)
_BRACKETS: dict[str, object] = {
    "bracket_mode": "fixed",
    "bracket_stop_loss_pct": 0.03,
    "bracket_take_profit_pct": 0.06,
}
# WP25-T-17 (P2): the only fields the switch may change, and the 7 it adds.
ALLOW_LISTED: tuple[str, ...] = (
    "sharpe_ratio", "sortino_ratio", "calmar_ratio", "cagr", "psr", "n_observations",
    "confidence_flag", "max_drawdown_duration_bars", "oos_sharpe", "oos_n_returns",
)  # fmt: skip
NEW_FIELDS: tuple[str, ...] = (
    "metrics_version", "trading_start_at", "trading_end_at", "periods_per_year",
    "metric_sampling", "carried_slots", "oos_start_at",
)  # fmt: skip
_DROPPED_FILL_KEYS = ("fill_id", "order_id")
_DROPPED_TRADE_KEYS = ("run_id", "trade_id")


@dataclass(frozen=True)
class Scenario:
    name: str
    timeframe: TimeFrame
    symbols: tuple[str, ...]
    n_bars: int
    seed: int
    cost_profile: str
    breaker: bool
    oos_start_index: int | None = None
    warmup_override: int | None = None
    holes: tuple[tuple[str, tuple[int, ...]], ...] = ()


H, D = TimeFrame.ONE_HOUR, TimeFrame.ONE_DAY
SCENARIOS: tuple[Scenario, ...] = (
    Scenario("s01_1h_single_legacy_breaker_on", H, ("AAA/EUR",), 600, 11,
             "legacy_backtest_v0", True),
    Scenario("s02_1d_single_coinbase_breaker_on", D, ("AAA/EUR",), 420, 12,
             "coinbase_advanced_1", True),
    Scenario("s03_1h_single_frictionless_breaker_off", H, ("AAA/EUR",), 600, 13,
             "frictionless_diagnostic", False),
    Scenario("s04_1h_single_oos_coinbase", H, ("AAA/EUR",), 600, 14,
             "coinbase_advanced_1", True, oos_start_index=420),
    Scenario("s05_1d_single_oos_legacy_breaker_off", D, ("AAA/EUR",), 420, 15,
             "legacy_backtest_v0", False, oos_start_index=300),
    Scenario("s06_1h_multi3_holes_legacy_breaker_on", H, ("AAA/EUR", "BBB/EUR", "CCC/EUR"), 500, 16,
             "legacy_backtest_v0", True,
             holes=(("BBB/EUR", (70, 71, 200, 333)), ("CCC/EUR", (55, 210, 211, 212)))),
    Scenario("s07_1h_multi3_holes_oos_coinbase_breaker_off", H,
             ("AAA/EUR", "BBB/EUR", "CCC/EUR"), 500, 17,
             "coinbase_advanced_1", False, oos_start_index=380,
             holes=(("AAA/EUR", (90, 300)), ("CCC/EUR", (150, 151, 390)))),
    Scenario("s08_1d_multi2_holes_coinbase_breaker_on", D, ("AAA/EUR", "BBB/EUR"), 400, 18,
             "coinbase_advanced_1", True, holes=(("BBB/EUR", (80, 81, 250)),)),
    Scenario("s09_1h_single_w0_legacy", H, ("AAA/EUR",), 300, 19,
             "legacy_backtest_v0", True, warmup_override=0),
    Scenario("s10_1h_multi2_w0_holes_stress_breaker_off", H, ("AAA/EUR", "BBB/EUR"), 300, 20,
             "coinbase_stress_t80", False, warmup_override=0,
             holes=(("BBB/EUR", (5, 120, 121)),)),
)


# ---------------------------------------------------------------------------
# generator (not collected)
# ---------------------------------------------------------------------------


def _gen_bars(sc: Scenario) -> dict[str, list[OHLCVBar]]:
    """Seeded multi-symbol random walk with a shared factor (so MA crossovers trade)."""
    rng = random.Random(sc.seed)
    step = _STEP[sc.timeframe]
    vol = 0.008 if sc.timeframe == TimeFrame.ONE_HOUR else 0.02
    px = dict.fromkeys(sc.symbols, 100.0)
    out: dict[str, list[OHLCVBar]] = {s: [] for s in sc.symbols}
    for k in range(sc.n_bars):
        factor = rng.gauss(0.0002, vol)
        for s in sc.symbols:
            o = px[s]
            c = max(1.0, o * math.exp(factor + rng.gauss(0.0001, vol * 0.75)))
            hi = max(o, c) * (1 + abs(rng.gauss(0, 0.002)))
            lo = min(o, c) * (1 - abs(rng.gauss(0, 0.002)))
            px[s] = c
            out[s].append(
                OHLCVBar(
                    symbol=s,
                    timeframe=sc.timeframe,
                    timestamp=T0 + k * step,
                    open=Decimal(f"{o:.4f}"),
                    high=Decimal(f"{hi:.4f}"),
                    low=Decimal(f"{lo:.4f}"),
                    close=Decimal(f"{c:.4f}"),
                    volume=Decimal(10),
                )
            )
    missing = {s: set(ix) for s, ix in sc.holes}
    return {
        s: [b for k, b in enumerate(bars) if k not in missing.get(s, set())]
        for s, bars in out.items()
    }


def _runner(sc: Scenario) -> BacktestRunner:
    runner = BacktestRunner(
        strategies=[
            MACrossoverStrategy(strategy_id="ma", params={"fast_period": 5, "slow_period": 20})
        ],
        symbols=list(sc.symbols),
        timeframe=sc.timeframe,
        initial_capital=Decimal("10000"),
        seed=7,
        bracket_config=dict(_BRACKETS),
        trailing_stop_pct=0.04,
        allow_pyramiding=False,
        enable_circuit_breaker=sc.breaker,
        cost_profile=get_cost_profile(sc.cost_profile),
        alignment_tolerance=_LENIENT if sc.holes else None,
    )
    if sc.warmup_override is not None:
        # The only way to reach W = 0 (the runner floors the auto-warm-up at 50).  The
        # engine config is built from this attribute, so engine and runner agree.
        runner._warmup_bars = sc.warmup_override
    return runner


def _jsonable_fill(fill: Any) -> dict[str, Any]:
    data = fill.model_dump(mode="json")
    for key in _DROPPED_FILL_KEYS:
        data.pop(key, None)
    return data  # type: ignore[no-any-return]


async def run_scenario(sc: Scenario) -> dict[str, Any]:
    """Run one scenario; returns ``{"scenario", "result", "fills"}`` (JSON-safe)."""
    runner = _runner(sc)
    result = await runner.run(_gen_bars(sc), oos_start_index=sc.oos_start_index)
    dumped = result.model_dump(mode="json")
    dumped.pop("run_id")
    for trade in dumped["trades"]:  # random per run: the run id and a uuid4 trade id
        for key in _DROPPED_TRADE_KEYS:
            trade.pop(key, None)
    engine = runner._last_execution_engine
    assert engine is not None
    fills = [_jsonable_fill(f) for per_order in engine._fills.values() for f in per_order]
    return {"scenario": sc.name, "result": dumped, "fills": fills}


async def generate_capture() -> dict[str, dict[str, Any]]:
    """Run every scenario (used once on the unmodified ``01d393d`` tree)."""
    return {sc.name: await run_scenario(sc) for sc in SCENARIOS}


def canonical(obj: Any) -> str:
    """Deterministic serialisation (sorted keys; ``inf`` / ``nan`` as JSON constants)."""
    return json.dumps(obj, sort_keys=True, allow_nan=True)


def _roundtrip(obj: Any) -> Any:
    return json.loads(canonical(obj))


if __name__ == "__main__":  # pragma: no cover - manual capture helper
    target = Path(sys.argv[1])
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(canonical(asyncio.run(generate_capture())) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------


def _load_capture() -> dict[str, dict[str, Any]]:
    data: dict[str, dict[str, Any]] = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    return data


def _diff_keys(a: dict[str, Any], b: dict[str, Any]) -> list[str]:
    return sorted(k for k in set(a) | set(b) if canonical(a.get(k)) != canonical(b.get(k)))


def test_capture_matches_scenario_table() -> None:
    capture = _load_capture()
    assert list(capture) == [sc.name for sc in SCENARIOS]
    assert len(SCENARIOS) == 10


def test_capture_holds_the_fields_p2_compares() -> None:
    """WP25-T-17 (P2) reuses this file: the fields must be present and non-trivial."""
    capture = _load_capture()
    needed = {
        "equity_curve", "trades", "final_equity", "total_bars", "start_date", "end_date",
        "duration_days", "max_drawdown_pct", "exposure_pct", "exposure_pct_per_symbol",
        "total_return_pct", "total_trades", "sharpe_ratio", "sortino_ratio", "calmar_ratio",
        "cagr", "psr", "n_observations", "confidence_flag", "max_drawdown_duration_bars",
        "oos_sharpe", "oos_n_returns",
    }
    for name, entry in capture.items():
        assert needed <= set(entry["result"]), name
        assert {"timestamp", "equity", "drawdown_pct"} <= set(entry["result"]["equity_curve"][0])
        assert entry["fills"], f"{name}: no fills captured"
        assert entry["result"]["total_trades"] > 0, f"{name}: no closed trades"
    assert any(e["result"]["oos_sharpe"] is not None for e in capture.values())


def _strip(result: dict[str, Any]) -> dict[str, Any]:
    drop = set(ALLOW_LISTED) | set(NEW_FIELDS)
    return {k: v for k, v in result.items() if k not in drop}


@pytest.mark.parametrize("sc", SCENARIOS, ids=[sc.name for sc in SCENARIOS])
async def test_metric_identity_vs_head_capture(sc: Scenario) -> None:
    """WP25-T-17: everything but the allow-list equals the 01d393d capture; the fills too.

    Strict on the invariant fields (full ``model_dump`` minus ``run_id``, floats compared
    exactly through the sorted-key JSON string).  Only ``ALLOW_LISTED`` may change and only
    ``NEW_FIELDS`` may be added.
    """
    expected = _load_capture()[sc.name]
    actual = _roundtrip(await run_scenario(sc))
    a, e = actual["result"], expected["result"]
    assert set(a) - set(e) == set(NEW_FIELDS), sorted(set(a) ^ set(e))
    assert set(e) <= set(a)
    if canonical(_strip(a)) != canonical(_strip(e)):
        bad = _diff_keys(_strip(a), _strip(e))
        pytest.fail(
            f"{sc.name}: invariant fields differ from the HEAD capture; keys {bad}; "
            f"fills equal: {actual['fills'] == expected['fills']}"
        )
    assert canonical(actual["fills"]) == canonical(expected["fills"])


@pytest.mark.parametrize("sc", SCENARIOS, ids=[sc.name for sc in SCENARIOS])
async def test_allow_listed_fields_moved_as_designed(sc: Scenario) -> None:
    """The switch is real: n is per bar after warm-up and the markers are filled in.

    Guards the harness against going vacuous (an allow-list that never changes would pass
    the strict test above while P2 did nothing).
    """
    e = _load_capture()[sc.name]["result"]
    a = _roundtrip(await run_scenario(sc))["result"]
    assert a["n_observations"] == a["total_bars"] + a["carried_slots"]
    assert a["n_observations"] < e["n_observations"]  # warm-up and fill points are gone
    assert 0 < a["max_drawdown_duration_bars"] <= a["n_observations"] + 1  # bars, not points
    assert a["metrics_version"] == "wp25-1"
    assert a["metric_sampling"] == "calendar_asof"
    assert a["periods_per_year"] == {"1h": 8766.0, "1d": 365.25}[sc.timeframe.value]
    assert a["trading_start_at"] is not None and a["trading_end_at"] is not None
    assert (a["oos_start_at"] is not None) == (sc.oos_start_index is not None)
    if sc.oos_start_index is not None:
        assert a["oos_n_returns"] == a["total_bars"] + (_warmup(sc)) - sc.oos_start_index
    assert a["cagr"] != e["cagr"] and a["sharpe_ratio"] != e["sharpe_ratio"]


def _warmup(sc: Scenario) -> int:
    return 50 if sc.warmup_override is None else sc.warmup_override


def test_scenarios_cover_the_required_axes() -> None:
    assert {sc.timeframe for sc in SCENARIOS} == {TimeFrame.ONE_HOUR, TimeFrame.ONE_DAY}
    assert any(sc.oos_start_index is not None for sc in SCENARIOS)
    assert any(len(sc.symbols) > 1 and sc.holes for sc in SCENARIOS)
    assert {sc.breaker for sc in SCENARIOS} == {True, False}
    assert len({sc.cost_profile for sc in SCENARIOS}) >= 2
    assert any(sc.warmup_override == 0 for sc in SCENARIOS)
