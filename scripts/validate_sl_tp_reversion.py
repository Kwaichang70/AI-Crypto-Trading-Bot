"""
scripts/validate_sl_tp_reversion.py
------------------------------------
Walk-forward out-of-sample profitability gate for the ``sl_tp_reversion``
strategy + engine bracket exits (Sprint 51 Cycle 3, Task #6).

For each candidate bracket/entry config the script runs a K-fold walk-forward:
every fold trains warm-up on a prefix and trades only its out-of-sample
window, so the reported ``profit_factor`` / ``sharpe`` / ``return`` are OOS.

Promotion gate (both must hold):
    * median OOS Sharpe   > 0
    * median OOS profit_factor > 1.2

Usage::

    python scripts/validate_sl_tp_reversion.py --exchange binance \
        --symbols BTC/USDT,ETH/USDT --timeframe 1h --bars 4000 --folds 5
"""
from __future__ import annotations

import argparse
import asyncio
import statistics
import sys
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from common.models import OHLCVBar
from common.types import TimeFrame
from trading.backtest import BacktestRunner
from trading.strategies.sl_tp_reversion import SLTPReversionStrategy

_PAGE_LIMIT = 300
_TF_DURATION_MS = {
    "1m": 60_000, "5m": 300_000, "15m": 900_000,
    "1h": 3_600_000, "4h": 14_400_000, "1d": 86_400_000,
}
_TF_ENUM = {
    "1m": TimeFrame.ONE_MINUTE, "5m": TimeFrame.FIVE_MINUTES,
    "15m": TimeFrame.FIFTEEN_MINUTES, "1h": TimeFrame.ONE_HOUR,
    "4h": TimeFrame.FOUR_HOURS, "1d": TimeFrame.ONE_DAY,
}


def _fetch_candles(exchange_id: str, symbol: str, timeframe: str, bars: int) -> list[list[Any]]:
    import time as _time

    import ccxt

    exchange = getattr(ccxt, exchange_id)({"enableRateLimit": True})
    exchange.load_markets()
    if symbol not in exchange.markets:
        print(f"ERROR: {symbol} not on {exchange_id}", file=sys.stderr)
        sys.exit(1)
    tf_ms = _TF_DURATION_MS[timeframe]
    since_ms = int(_time.time() * 1000) - (bars + 5) * tf_ms
    out: list[list[Any]] = []
    seen: set[int] = set()
    while len(out) < bars:
        page = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since_ms, limit=min(bars - len(out), _PAGE_LIMIT))
        if not page:
            break
        new = 0
        last = since_ms
        for c in page:
            ts = int(c[0])
            if ts not in seen:
                out.append(c)
                seen.add(ts)
                new += 1
                last = ts
        if new == 0 or len(page) < min(bars - len(out), _PAGE_LIMIT):
            break
        since_ms = last + tf_ms
    out.sort(key=lambda c: c[0])
    return out[:-1]  # drop possibly-incomplete final candle


def _to_bars(symbol: str, tf: TimeFrame, raw: list[list[Any]]) -> list[OHLCVBar]:
    bars: list[OHLCVBar] = []
    for ts_ms, o, h, low, c, v in raw:
        bars.append(OHLCVBar(
            symbol=symbol, timeframe=tf,
            timestamp=datetime.fromtimestamp(int(ts_ms) / 1000, tz=UTC),
            open=Decimal(str(o)), high=Decimal(str(h)), low=Decimal(str(low)),
            close=Decimal(str(c)), volume=Decimal(str(v)),
        ))
    return bars


async def _run_fold(
    bars: list[OHLCVBar], symbol: str, tf: TimeFrame,
    entry: dict[str, Any], bracket: dict[str, object],
) -> dict[str, float]:
    strat = SLTPReversionStrategy(strategy_id=f"sltp-{symbol}", params=entry)
    runner = BacktestRunner(
        strategies=[strat], symbols=[symbol], timeframe=tf,
        initial_capital=Decimal("10000"), bracket_config=bracket, seed=42,
    )
    result = await runner.run({symbol: bars})
    pf = result.profit_factor
    return {
        "profit_factor": float(pf) if pf is not None else (
            999.0 if result.profit_factor_is_infinite else 0.0
        ),
        "sharpe": float(result.sharpe_ratio),
        "return_pct": float(result.total_return_pct),
        "trades": float(result.total_trades),
    }


async def _evaluate(
    bars_by_symbol: dict[str, list[OHLCVBar]], tf: TimeFrame,
    entry: dict[str, Any], bracket: dict[str, object], folds: int,
) -> dict[str, float]:
    warmup = SLTPReversionStrategy(strategy_id="probe", params=entry).min_bars_required
    pfs: list[float] = []
    sharpes: list[float] = []
    rets: list[float] = []
    trades = 0
    for symbol, bars in bars_by_symbol.items():
        n = len(bars)
        oos_total = n - int(n * 0.5)  # second half is OOS, split into folds
        fold_size = oos_total // folds
        if fold_size <= warmup:
            continue
        for i in range(folds):
            test_start = int(n * 0.5) + i * fold_size
            test_end = n if i == folds - 1 else test_start + fold_size
            lo = max(0, test_start - warmup)
            fold_bars = bars[lo:test_end]
            if len(fold_bars) <= warmup:
                continue
            m = await _run_fold(fold_bars, symbol, tf, entry, bracket)
            if m["trades"] > 0:
                pfs.append(m["profit_factor"])
                sharpes.append(m["sharpe"])
                rets.append(m["return_pct"])
                trades += int(m["trades"])
    if not pfs:
        return {"median_pf": 0.0, "median_sharpe": 0.0, "median_return": 0.0, "n_folds": 0, "total_trades": 0}
    return {
        "median_pf": statistics.median(pfs),
        "median_sharpe": statistics.median(sharpes),
        "median_return": statistics.median(rets),
        "n_folds": len(pfs),
        "total_trades": trades,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exchange", default="binance")
    ap.add_argument("--symbols", default="BTC/USDT,ETH/USDT")
    ap.add_argument("--timeframe", default="1h")
    ap.add_argument("--bars", type=int, default=4000)
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    tf = _TF_ENUM[args.timeframe]
    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    bars_by_symbol: dict[str, list[OHLCVBar]] = {}
    for sym in symbols:
        raw = _fetch_candles(args.exchange, sym, args.timeframe, args.bars)
        bars_by_symbol[sym] = _to_bars(sym, tf, raw)
        print(f"  {sym}: {len(bars_by_symbol[sym])} bars", flush=True)

    # Candidate configs: RSI-2 dip entry + fixed bracket (reward:risk grid).
    candidates: list[tuple[dict[str, Any], dict[str, object]]] = []
    for entry_threshold in (5.0, 10.0):
        for sl, tp in ((0.03, 0.05), (0.05, 0.08), (0.05, 0.10), (0.08, 0.12)):
            entry = {
                "rsi_period": 2, "entry_threshold": entry_threshold,
                "trend_sma_period": 200, "position_size": 2000.0,
            }
            bracket = {
                "bracket_mode": "fixed",
                "bracket_stop_loss_pct": sl,
                "bracket_take_profit_pct": tp,
            }
            candidates.append((entry, bracket))

    print("\n=== Walk-forward OOS evaluation ===", flush=True)
    best: dict[str, Any] | None = None
    for entry, bracket in candidates:
        res = asyncio.run(_evaluate(bars_by_symbol, tf, entry, bracket, args.folds))
        label = (
            f"rsi<{entry['entry_threshold']:>4} SL={bracket['bracket_stop_loss_pct']} "
            f"TP={bracket['bracket_take_profit_pct']}"
        )
        print(
            f"  {label}: median_pf={res['median_pf']:.3f} "
            f"median_sharpe={res['median_sharpe']:.3f} "
            f"median_ret={res['median_return']:.2f}% "
            f"folds={res['n_folds']} trades={res['total_trades']}",
            flush=True,
        )
        passes = res["median_sharpe"] > 0 and res["median_pf"] > 1.2 and res["n_folds"] >= 3
        score = res["median_pf"] * (1.0 if res["median_sharpe"] > 0 else 0.0)
        if passes and (best is None or score > best["score"]):
            best = {"entry": entry, "bracket": bracket, "res": res, "score": score}

    print("\n=== GATE RESULT ===", flush=True)
    if best is None:
        print("FAIL — no config cleared median_pf>1.2 AND median_sharpe>0. Stay backtest-only.")
        sys.exit(2)
    print("PASS — promotion gate cleared.")
    print(f"  entry  = {best['entry']}")
    print(f"  bracket= {best['bracket']}")
    print(f"  result = {best['res']}")


if __name__ == "__main__":
    main()
