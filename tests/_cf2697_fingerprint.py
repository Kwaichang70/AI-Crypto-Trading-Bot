"""CF-26-97 P0: result-fingerprint harness (append-only from P1, SY-97-22; used by P1..P3).

One sha256 per run over a canonical serialisation of everything a backtest produces:
every ``BacktestResult`` field, the engine summary, the ``engine.stopped`` log, the
per-bar gauge sequence, the raw equity curve, orders, fills and trades.  Floats are
``repr`` + ``float.hex()``, Decimals ``str``.  ``elapsed_seconds``, ``bar_elapsed_ms``
and the timing histograms are excluded.  Run ids, UUIDs and the hex suffix of
``client_order_id`` are replaced by first-appearance ordinals, because HEAD is not
self-identical without that.  An unsupported type raises ``TypeError`` (no repr fallback).
Nothing here reads a clock or the process hash seed.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from enum import Enum
from random import Random
from typing import Any
from uuid import UUID

import structlog
from structlog.testing import capture_logs

from common.metrics import metrics
from common.models import OHLCVBar
from common.types import OrderSide, OrderType, TimeFrame
from trading import strategies as S
from trading.backtest import BacktestRunner
from trading.clock import BarClock
from trading.models import Order
from trading.optimizer import ParameterOptimizer
from trading.risk import RiskParameters
from trading.strategy_engine import StrategyEngine

_EXCLUDED = frozenset({"elapsed_seconds", "bar_elapsed_ms"})
_ID = re.compile(
    r"bt-[0-9a-f]{12}|[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}|(?<=-)[0-9a-f]{12}\b"
)
_STRATS: dict[str, tuple[type, dict[str, Any]]] = {
    "ma_crossover": (S.MACrossoverStrategy, {"fast_period": 5, "slow_period": 20}),
    "rsi": (S.RSIMeanReversionStrategy, {}),
    "breakout": (S.BreakoutStrategy, {}),
    "dca_rsi": (S.DCARSIHybridStrategy, {}),
    "grid": (S.GridTradingStrategy, {"grid_size_pct": 0.003, "position_size": 500.0}),
    "sl_tp": (S.SLTPReversionStrategy, {"entry_threshold": 30.0, "trend_sma_period": 20}),
    "momentum": (S.MomentumBreakoutStrategy, {"lookback": 20, "trend_sma_period": 20}),
    "smoke": (S.SmokeRoundtripStrategy, {}),
}
_FIXED = {"bracket_mode": "fixed", "bracket_stop_loss_pct": 0.02, "bracket_take_profit_pct": 0.04}
_ATR = {
    "bracket_mode": "atr",
    "bracket_atr_sl_multiplier": 2.0,
    "bracket_atr_tp_multiplier": 3.0,
    "bracket_atr_period": 14,
}
_TRAIL = {**_FIXED, "bracket_take_profit_pct": 0.015}  # RK-97P0-03
_GRID = {"fast_period": [5, 8], "slow_period": [20, 30]}
_TIED = {"fast_period": [5], "slow_period": [20], "trailing_stop_pct": [0.4, 0.5]}  # exact ties


def make_bars(symbol: str, n: int, seed: int, drift: int = 0) -> list[OHLCVBar]:
    rnd, price, t0 = Random(seed), 4_000_000, datetime(2024, 1, 1, tzinfo=UTC)  # noqa: S311
    out = []
    for i in range(n):
        o, price = price, max(100, price + rnd.randint(-8000, 8000 + drift))
        out.append(
            OHLCVBar(
                symbol=symbol,
                timeframe=TimeFrame.ONE_MINUTE,
                timestamp=t0 + timedelta(minutes=i),
                open=Decimal(o).scaleb(-2),
                high=Decimal(max(o, price) + rnd.randint(0, 2000)).scaleb(-2),
                low=Decimal(max(1, min(o, price) - rnd.randint(0, 2000))).scaleb(-2),
                close=Decimal(price).scaleb(-2),
                volume=Decimal(rnd.randint(1, 10**7)).scaleb(-3),
            )
        )
    return out


class Canon:
    """Canonical, float-exact, id-normalised JSON-able form of any run artefact."""

    def __init__(self) -> None:
        self.ids: dict[str, str] = {}

    def _s(self, s: str) -> str:
        return _ID.sub(lambda m: self.ids.setdefault(m.group(0), f"#{len(self.ids)}"), s)

    def __call__(self, x: Any) -> Any:
        if x is None or isinstance(x, (bool, int)):
            return x
        if isinstance(x, float):
            return f"F:{x!r}:{x.hex()}"
        if isinstance(x, Decimal):
            return f"D:{x}"
        if isinstance(x, (UUID, str)):
            return self._s(str(x))
        if isinstance(x, datetime):
            return f"T:{x.isoformat()}"
        if isinstance(x, Enum):
            return f"E:{type(x).__name__}.{x.name}"
        if isinstance(x, dict):
            return {self._s(str(self(k))): self(v) for k, v in x.items() if k not in _EXCLUDED}
        if isinstance(x, (list, tuple)):
            return [self(v) for v in x]
        fields = getattr(type(x), "model_fields", None) or getattr(x, "__dataclass_fields__", None)
        if fields is not None:
            return {
                "__t": type(x).__name__,
                **{k: self(getattr(x, k)) for k in fields if k not in _EXCLUDED},
            }
        raise TypeError(f"Canon: unsupported type {type(x).__name__}")


class Capture:
    """Per-run records (keyed by run id) of gauges, engine summaries and stop logs."""

    def __init__(self) -> None:
        self.gauges: list[tuple[str, float, dict[str, str]]] = []
        self.summaries: list[tuple[str, dict[str, Any]]] = []
        self.logs: list[dict[str, Any]] = []


@contextmanager
def capture() -> Iterator[Capture]:
    cap, orig_gauge, orig_rb = Capture(), metrics.gauge, StrategyEngine.run_backtest

    def gauge(
        name: str, value: float, labels: dict[str, str] | None = None, *a: Any, **k: Any
    ) -> Any:
        cap.gauges.append((name, value, dict(labels or {})))
        return orig_gauge(name, value, labels, *a, **k)

    async def run_backtest(self: Any, *a: Any, **k: Any) -> Any:
        summary = await orig_rb(self, *a, **k)
        cap.summaries.append((self._run_id, dict(summary)))
        return summary

    metrics.gauge, StrategyEngine.run_backtest = gauge, run_backtest  # type: ignore[method-assign]
    previous = structlog.get_config()  # every level must reach capture_logs
    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(0), cache_logger_on_first_use=False
    )
    try:
        with capture_logs() as logs:
            cap.logs = logs
            yield cap
    finally:
        structlog.configure(**previous)
        vars(metrics).pop("gauge", None)
        StrategyEngine.run_backtest = orig_rb  # type: ignore[method-assign]


def _case(strat: str, nsym: int, n: int, seed: int, **extra: Any) -> dict[str, Any]:
    return {"strat": strat, "nsym": nsym, "n": n, "seed": seed, **extra}


def matrix() -> dict[str, dict[str, Any]]:
    cases = {}
    for strat in _STRATS:
        bracket = _FIXED if strat in ("sl_tp", "momentum", "smoke") else None
        for nsym in (1, 2):
            for n in (300, 2000):
                cases[f"{strat}-{nsym}s-{n}-s7"] = _case(strat, nsym, n, 7, bracket=bracket)
    for strat in ("ma_crossover", "rsi"):
        cases[f"{strat}-2s-2000-s11"] = _case(strat, 2, 2000, 11)
    cases["ma_crossover-1s-8000-s7"] = _case("ma_crossover", 1, 8000, 7)
    cases["ma_crossover-1s-2000-oos"] = _case("ma_crossover", 1, 2000, 7, oos=1200)
    cases["ma_crossover-1s-2000-atr-trailing"] = _case(
        "ma_crossover", 1, 2000, 7, bracket=_ATR, trailing=0.04
    )
    # P0 r3 (QR-97P0-03/04): rising markets (new peaks after drawdowns) and fresh seed-11 bars.
    for strat, nsym in (("ma_crossover", 1), ("breakout", 1), ("sl_tp", 1), ("ma_crossover", 2)):
        cases[f"{strat}-{nsym}s-2000-up2000"] = _case(
            strat, nsym, 2000, 7, drift=2000, bracket=_FIXED if strat == "sl_tp" else None
        )
    cases["ma_crossover-2s-2000-bars11"] = _case("ma_crossover", 2, 2000, 11, bar_seed=11)
    # P0 r3 (RK-97P0-02): tight limits walk the breaker OK -> DAILY_LIMIT (cleared at the UTC day
    # roll) -> REDUCE -> HALT (gauge 0, 2, 1, 3; the final 0 is the K-20 reset on stop()), with
    # blocked entries and pre-trade rejections.
    cases["ma_crossover-2s-2000-breaker"] = _case(
        "ma_crossover", 2, 2000, 7, risk={"max_drawdown_pct": 0.006, "max_daily_loss_pct": 0.003}
    )
    # P0 r3 (RK-97P0-03): the trailing stop fires (tighter than the fixed bracket).
    cases["ma_crossover-1s-2000-fixed-trailing"] = _case(
        "ma_crossover", 1, 2000, 7, bracket=_TRAIL, trailing=0.005
    )
    # Added in P0 revision r2 (SY-97-20), recorded from the unpatched base c1b45f5.
    cases["paper-resting-limit-orders"] = {"limit": True, "nsym": 2}
    cases["optimizer-in-sample"] = {"opt": False, "n": 600, "nsym": 1}
    cases["optimizer-walk-forward"] = {"opt": True, "n": 1500, "nsym": 2}
    # P0 r3 (CR-97P0-02): exact-tie grids, so the optimizer tie-break is fingerprinted.
    cases["optimizer-tied-in-sample"] = {"opt": False, "n": 600, "nsym": 1, "grid": _TIED}
    cases["optimizer-tied-walk-forward"] = {"opt": True, "n": 600, "nsym": 1, "grid": _TIED}
    return cases


_SYMS = ("BTC/USDT", "ETH/USDT")
_T0 = datetime(2024, 1, 1, tzinfo=UTC)
# per step: A, B.  Step 3 re-crosses every BUY limit after it was filled or cancelled.
_CLOSES = (("40000", "2500"), ("39800", "2480"), ("40200", "2520"), ("39000", "2450"))
# Orders placed per step: "symbol index side limit-price quantity", A and B interleaved ("-" =
# MARKET).  Step 1 makes three BUYs on A and one on B fillable on the same update, step 2 three
# SELLs on A and one on B.  Placement 4 is cancelled; the SELL at 41000 and the BUY at 2000 rest.
_PLAN = {
    0: "0 buy - 1,0 buy 39900 0.3,1 buy 2490 2,0 buy 39850 0.2,1 buy 2495 1,0 buy 39950 0.1,"
    "0 sell 41000 0.5,1 buy 2000 1",
    1: "0 sell 40100 0.4,1 sell 2510 1,0 sell 40150 0.3,0 sell 40050 0.2",
}


def _bar(symbol: str, price: str, at: datetime) -> OHLCVBar:
    p = Decimal(price)
    return OHLCVBar(
        symbol=symbol, timeframe=TimeFrame.ONE_MINUTE, timestamp=at - timedelta(minutes=1),
        open=p, high=p, low=p, close=p, volume=Decimal(1),
    )  # fmt: skip


async def _limit_scenario(cap: Capture, canon: Canon) -> dict[str, Any]:
    """Resting LIMIT orders on a real paper engine; fills routed into a portfolio."""
    run, first_log, steps, ids = "bt-000000000001", len(cap.logs), [], []
    clock = BarClock(start=_T0)
    strategy = S.MACrossoverStrategy(strategy_id="fp-limit", params={})
    runner = BacktestRunner([strategy], list(_SYMS), TimeFrame.ONE_MINUTE, seed=7)
    engine, pf, _, ex = runner._build_engine(run, clock=clock)
    await engine.start(run)
    for k, closes in enumerate(_CLOSES):
        at = _T0 + timedelta(minutes=k + 1)
        clock.advance(at)
        pf.advance_clock(at)
        bars = {sym: _bar(sym, price, at) for sym, price in zip(_SYMS, closes, strict=True)}
        for sym, bar in bars.items():
            ex.set_last_price(sym, bar.close)
        for spec in filter(None, _PLAN.get(k, "").split(",")):
            i, side, limit, qty = spec.split()
            n = len(ids) + 1
            ids.append(UUID(int=n))
            await ex.submit_order(
                Order(
                    order_id=ids[-1], client_order_id=f"{run}-{n:012x}", run_id=run,
                    symbol=_SYMS[int(i)], side=OrderSide(side), quantity=Decimal(qty),
                    order_type=OrderType.MARKET if limit == "-" else OrderType.LIMIT,
                    price=None if limit == "-" else Decimal(limit),
                    created_at=at, updated_at=at,
                )
            )  # fmt: skip
            for fill in await ex.get_fills(ids[-1]):
                pf.update_position(fill, fill.price)
        if k == 0:
            await ex.cancel_order(ids[4])
        pf.update_market_prices({sym: bar.close for sym, bar in bars.items()})
        await engine._check_resting_orders(bars)
        open_ids = [o.order_id for o in ex.get_open_orders()]
        steps.append([open_ids, ex.cash, ex.positions, pf.current_equity])
    await engine.stop()
    events = cap.logs[first_log:]
    assert not [e for e in events if e["event"] == "engine.resting_order_check_error"]
    cancels = [(e["event"], e["order_id"]) for e in events if "canceled" in e["event"]]
    return {
        "steps": canon(steps), "cancels": canon(cancels), "curve": canon(pf.get_equity_curve()),
        "orders": canon(ex.get_all_orders()), "fills": canon(ex.get_all_fills()),
        "trades": canon(pf.get_trade_history()),
    }  # fmt: skip


async def run_case(name: str, cap: Capture) -> str:
    """Run one matrix entry (inside ``capture()``) and return its sha256."""
    cfg, canon = matrix()[name], Canon()
    if "limit" in cfg:
        return hashlib.sha256(
            json.dumps(await _limit_scenario(cap, canon), sort_keys=True).encode()
        ).hexdigest()
    symbols = ["BTC/USDT", "ETH/USDT"][: cfg["nsym"]]
    seed0, drift = cfg.get("bar_seed", 7), cfg.get("drift", 0)
    bars = {s: make_bars(s, cfg["n"], seed0 + 101 * i, drift) for i, s in enumerate(symbols)}
    if "opt" in cfg:
        opt = ParameterOptimizer(
            S.MACrossoverStrategy,
            symbols,
            TimeFrame.ONE_MINUTE,
            cfg.get("grid", _GRID),
            walk_forward=cfg["opt"],
            walk_forward_folds=3,
        )
        payload: Any = {"optimization": canon(await opt.run(bars))}
    else:
        cls, params = _STRATS[cfg["strat"]]
        runner = BacktestRunner(
            strategies=[cls(strategy_id=f"fp-{cfg['strat']}", params=dict(params))],
            symbols=symbols,
            timeframe=TimeFrame.ONE_MINUTE,
            seed=cfg["seed"],
            bracket_config=cfg.get("bracket"),
            trailing_stop_pct=cfg.get("trailing"),
            risk_params=RiskParameters(**cfg["risk"]) if "risk" in cfg else None,
        )
        result = await runner.run(bars, oos_start_index=cfg.get("oos"))
        rid, pf, ex = result.run_id, runner.last_portfolio, runner.last_execution_engine
        assert pf is not None and ex is not None
        payload = {
            "result": canon(result),
            "summaries": canon([s for r, s in cap.summaries if r == rid]),
            "stop_logs": canon(
                [e for e in cap.logs if e.get("run_id") == rid and e["event"] == "engine.stopped"]
            ),
            "gauges": canon([(n, v, lb) for n, v, lb in cap.gauges if lb.get("run_id") == rid]),
            "curve": canon(pf.get_equity_curve()),
            "orders": canon(ex.get_all_orders()),
            "fills": canon(ex.get_all_fills()),
            "trades": canon(pf.get_trade_history()),
        }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
