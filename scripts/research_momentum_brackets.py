"""
scripts/research_momentum_brackets.py
--------------------------------------
Research harness hunting for a ROBUST profitable SL/TP strategy.

Thesis: crypto's most documented edge is time-series momentum / trend, which
pairs with ASYMMETRIC ATR brackets (tight stop, wide target — let winners run).
Unlike RSI-2 mean-reversion (which failed held-out validation), trend has a
fat right tail that a wide TP can capture while the SL caps the losers.

Each (entry archetype x bracket) config is evaluated by warmup-aware
walk-forward on TWO independent symbol sets:
  * SCAN     = BTC/USDT, ETH/USDT       (where we search)
  * HELD-OUT = SOL/USDT, BNB/USDT, XRP/USDT  (never searched — the real test)

A config is ROBUST only if it clears the bar on BOTH sets.  This makes
in-scan p-hacking impossible to pass off as an edge.
"""
from __future__ import annotations

import argparse
import asyncio
import statistics
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from common.models import MultiTimeframeContext, OHLCVBar
from common.types import SignalDirection, TimeFrame
from trading.backtest import BacktestRunner
from trading.models import Signal
from trading.strategy import BaseStrategy, StrategyMetadata

_PAGE_LIMIT = 300
_TF_MS = {"1m": 60_000, "5m": 300_000, "15m": 900_000, "1h": 3_600_000, "4h": 14_400_000, "1d": 86_400_000}
_TF_ENUM = {"1h": TimeFrame.ONE_HOUR, "4h": TimeFrame.FOUR_HOURS, "1d": TimeFrame.ONE_DAY}

SCAN = ["BTC/USDT", "ETH/USDT"]
HELD_OUT = ["SOL/USDT", "BNB/USDT", "XRP/USDT"]


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def _fetch(exchange_id: str, symbol: str, timeframe: str, bars: int) -> list[list[Any]]:
    import time as _t

    import ccxt

    ex = getattr(ccxt, exchange_id)({"enableRateLimit": True})
    ex.load_markets()
    if symbol not in ex.markets:
        return []
    tf_ms = _TF_MS[timeframe]
    since = int(_t.time() * 1000) - (bars + 5) * tf_ms
    out: list[list[Any]] = []
    seen: set[int] = set()
    while len(out) < bars:
        page = ex.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=min(bars - len(out), _PAGE_LIMIT))
        if not page:
            break
        new = 0
        last = since
        for c in page:
            ts = int(c[0])
            if ts not in seen:
                out.append(c)
                seen.add(ts)
                new += 1
                last = ts
        if new == 0 or len(page) < min(bars - len(out), _PAGE_LIMIT):
            break
        since = last + tf_ms
    out.sort(key=lambda c: c[0])
    return out[:-1]


def _to_bars(symbol: str, tf: TimeFrame, raw: list[list[Any]]) -> list[OHLCVBar]:
    return [
        OHLCVBar(
            symbol=symbol, timeframe=tf,
            timestamp=datetime.fromtimestamp(int(ts) / 1000, tz=UTC),
            open=Decimal(str(o)), high=Decimal(str(h)), low=Decimal(str(low)),
            close=Decimal(str(c)), volume=Decimal(str(v)),
        )
        for ts, o, h, low, c, v in raw
    ]


# --------------------------------------------------------------------------- #
# Configurable BUY-only momentum/trend strategy (exits via brackets)
# --------------------------------------------------------------------------- #
class MomentumBracketStrategy(BaseStrategy):
    """BUY-only entries on momentum/trend events; never emits SELL.

    entry_mode:
      * "donchian" — new breakout above the prior `lookback`-bar high.
      * "tsmom"    — N-bar momentum crosses positive, above SMA(trend).
      * "macross"  — fast SMA crosses above slow SMA (golden cross).
    All optionally gated by a long trend filter (close > SMA(trend_sma)).
    """

    metadata = StrategyMetadata(
        name="Momentum Bracket (research)", version="0.1.0",
        description="research-only momentum/trend entry for bracket exits",
        author="quant-research", tags=["research", "momentum", "trend"],
    )

    def _validate_params(self, params: dict[str, Any]) -> dict[str, Any]:
        p = dict(params)
        p.setdefault("entry_mode", "donchian")
        p.setdefault("lookback", 20)
        p.setdefault("trend_sma", 100)
        p.setdefault("fast", 10)
        p.setdefault("slow", 50)
        p.setdefault("position_size", 2000.0)
        return p

    @classmethod
    def parameter_schema(cls) -> dict[str, Any]:
        return {"type": "object", "properties": {}, "additionalProperties": True}

    @property
    def min_bars_required(self) -> int:
        p = self._params
        return max(int(p["lookback"]), int(p["trend_sma"]), int(p["slow"])) + 3

    @staticmethod
    def _sma(vals: Sequence[float], n: int) -> float:
        return sum(vals[-n:]) / n

    def on_bar(self, bars: Sequence[OHLCVBar], *, mtf_context: MultiTimeframeContext | None = None) -> list[Signal]:
        if len(bars) < self.min_bars_required:
            return []
        p = self._params
        mode = p["entry_mode"]
        closes = [float(b.close) for b in bars]
        highs = [float(b.high) for b in bars]
        trend_n = int(p["trend_sma"])
        trend_ok = closes[-1] > self._sma(closes, trend_n) if trend_n > 0 else True

        fire = False
        if mode == "donchian":
            lb = int(p["lookback"])
            prior_high = max(highs[-(lb + 1):-1])
            prior_high_2 = max(highs[-(lb + 2):-2])
            # New breakout event: crossed above the prior-N high this bar.
            fire = closes[-1] > prior_high and closes[-2] <= prior_high_2
        elif mode == "tsmom":
            lb = int(p["lookback"])
            mom = closes[-1] - closes[-1 - lb]
            mom_prev = closes[-2] - closes[-2 - lb]
            fire = mom > 0 and mom_prev <= 0 and trend_ok
        elif mode == "macross":
            fast, slow = int(p["fast"]), int(p["slow"])
            f_now, s_now = self._sma(closes, fast), self._sma(closes, slow)
            f_prev = sum(closes[-fast - 1:-1]) / fast
            s_prev = sum(closes[-slow - 1:-1]) / slow
            fire = f_now > s_now and f_prev <= s_prev

        if mode == "donchian" and not trend_ok:
            fire = False
        if not fire:
            return []
        return [Signal(
            strategy_id=self._strategy_id, symbol=bars[-1].symbol,
            direction=SignalDirection.BUY, target_position=Decimal(str(p["position_size"])),
            confidence=1.0, metadata={"entry_mode": mode},
        )]


# --------------------------------------------------------------------------- #
# Walk-forward evaluation (warmup-aware folds)
# --------------------------------------------------------------------------- #
async def _run(bars: list[OHLCVBar], symbol: str, tf: TimeFrame, entry: dict[str, Any], bracket: dict[str, object]) -> dict[str, float]:
    strat = MomentumBracketStrategy(strategy_id=f"mom-{symbol}", params=entry)
    runner = BacktestRunner(strategies=[strat], symbols=[symbol], timeframe=tf,
                            initial_capital=Decimal("10000"), bracket_config=bracket, seed=42)
    r = await runner.run({symbol: bars})
    pf = r.profit_factor
    return {
        "pf": float(pf) if pf is not None else (999.0 if r.profit_factor_is_infinite else 0.0),
        "sharpe": float(r.sharpe_ratio), "trades": float(r.total_trades),
    }


async def _evaluate(bars_by_symbol: dict[str, list[OHLCVBar]], tf: TimeFrame, entry: dict[str, Any], bracket: dict[str, object], folds: int) -> dict[str, float]:
    min_bars = MomentumBracketStrategy(strategy_id="p", params=entry).min_bars_required
    warmup = max(min_bars * 2, 50)
    pfs: list[float] = []
    sharpes: list[float] = []
    trades = 0
    for _sym, bars in bars_by_symbol.items():
        n = len(bars)
        step = (n - warmup) // folds
        if step <= warmup // 2:
            continue
        for i in range(folds):
            oos_start = warmup + i * step
            oos_end = n if i == folds - 1 else oos_start + step
            fold_bars = bars[oos_start - warmup:oos_end]
            if len(fold_bars) <= warmup:
                continue
            m = await _run(fold_bars, _sym, tf, entry, bracket)
            if m["trades"] > 0:
                pfs.append(m["pf"])
                sharpes.append(m["sharpe"])
                trades += int(m["trades"])
    if not pfs:
        return {"median_pf": 0.0, "median_sharpe": 0.0, "n": 0, "trades": 0}
    return {"median_pf": statistics.median(pfs), "median_sharpe": statistics.median(sharpes), "n": len(pfs), "trades": trades}


def _passes(res: dict[str, float]) -> bool:
    return res["median_pf"] > 1.2 and res["median_sharpe"] > 0 and res["n"] >= 4 and res["trades"] >= 30


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exchange", default="binance")
    ap.add_argument("--timeframe", default="1d")
    ap.add_argument("--bars", type=int, default=1500)
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()
    tf = _TF_ENUM[args.timeframe]

    def load(syms: list[str]) -> dict[str, list[OHLCVBar]]:
        d = {}
        for s in syms:
            raw = _fetch(args.exchange, s, args.timeframe, args.bars)
            if raw:
                d[s] = _to_bars(s, tf, raw)
        return d

    print(f"Loading {args.timeframe} data ({args.bars} bars)...", flush=True)
    scan_bars = load(SCAN)
    held_bars = load(HELD_OUT)
    print(f"  scan={[(s, len(b)) for s, b in scan_bars.items()]}", flush=True)
    print(f"  held={[(s, len(b)) for s, b in held_bars.items()]}", flush=True)

    entries: list[dict[str, Any]] = []
    for tn in (50, 100, 200):
        for lb in (10, 20, 40):
            entries.append({"entry_mode": "donchian", "lookback": lb, "trend_sma": tn, "position_size": 2000.0})
            entries.append({"entry_mode": "tsmom", "lookback": lb, "trend_sma": tn, "position_size": 2000.0})
    for fast, slow in ((10, 50), (20, 100), (10, 30)):
        entries.append({"entry_mode": "macross", "fast": fast, "slow": slow, "trend_sma": 0, "lookback": slow, "position_size": 2000.0})

    # Asymmetric ATR brackets (tight SL, wide TP) — let trend winners run.
    brackets: list[dict[str, object]] = []
    for sl, tp in ((1.5, 3.0), (1.5, 4.5), (2.0, 4.0), (2.0, 6.0), (2.5, 5.0)):
        brackets.append({"bracket_mode": "atr", "bracket_atr_sl_multiplier": sl, "bracket_atr_tp_multiplier": tp, "bracket_atr_period": 14})

    print(f"\n=== Evaluating {len(entries)*len(brackets)} configs on scan + held-out ({args.timeframe}) ===", flush=True)
    robust: list[dict[str, Any]] = []
    for entry in entries:
        for bracket in brackets:
            scan_res = asyncio.run(_evaluate(scan_bars, tf, entry, bracket, args.folds))
            if not _passes(scan_res):
                continue
            held_res = asyncio.run(_evaluate(held_bars, tf, entry, bracket, args.folds))
            em = entry["entry_mode"]
            tag = (f"{em} lb={entry.get('lookback')} sma={entry.get('trend_sma')} "
                   f"f/s={entry.get('fast')}/{entry.get('slow')} "
                   f"SL={bracket['bracket_atr_sl_multiplier']}xATR TP={bracket['bracket_atr_tp_multiplier']}xATR")
            both = _passes(held_res)
            print(f"  [{'ROBUST' if both else 'scan-only'}] {tag}", flush=True)
            print(f"      scan: pf={scan_res['median_pf']:.3f} sh={scan_res['median_sharpe']:.3f} n={scan_res['n']} tr={scan_res['trades']}"
                  f" | held: pf={held_res['median_pf']:.3f} sh={held_res['median_sharpe']:.3f} n={held_res['n']} tr={held_res['trades']}", flush=True)
            if both:
                robust.append({"entry": entry, "bracket": bracket, "scan": scan_res, "held": held_res})

    print("\n=== RESULT ===", flush=True)
    if not robust:
        print("No config robust across BOTH scan and held-out. No edge found this round.")
        sys.exit(2)
    robust.sort(key=lambda r: min(r["scan"]["median_pf"], r["held"]["median_pf"]), reverse=True)
    print(f"{len(robust)} ROBUST config(s). Best:")
    b = robust[0]
    print(f"  entry  = {b['entry']}")
    print(f"  bracket= {b['bracket']}")
    print(f"  scan   = {b['scan']}")
    print(f"  held   = {b['held']}")


if __name__ == "__main__":
    main()
