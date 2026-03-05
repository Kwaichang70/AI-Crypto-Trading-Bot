#!/usr/bin/env python3
"""
scripts/train_model.py
-----------------------
CLI script for training a RandomForestClassifier model for ModelStrategy.

Usage
-----
  python scripts/train_model.py --symbol BTC/USDT --exchange binance \\
      --timeframe 1h --bars 2000 --model-dir models/

  # Or via uv:
  uv run python scripts/train_model.py --symbol BTC/USDT
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_TF_DURATION_MS: dict[str, int] = {
    "1m": 60_000,
    "3m": 180_000,
    "5m": 300_000,
    "15m": 900_000,
    "30m": 1_800_000,
    "1h": 3_600_000,
    "4h": 14_400_000,
    "1d": 86_400_000,
    "1w": 604_800_000,
}

_VALID_TIMEFRAMES: list[str] = sorted(_TF_DURATION_MS.keys())
_PAGE_LIMIT: int = 500


def _fetch_candles(
    exchange_id: str, symbol: str, timeframe: str, bars: int,
) -> list[list[Any]]:
    """Fetch historical OHLCV candles using synchronous ccxt with pagination."""
    try:
        import ccxt
    except ImportError:
        print("ERROR: ccxt is not installed. Run: pip install ccxt", file=sys.stderr)
        sys.exit(1)

    exchange_cls = getattr(ccxt, exchange_id, None)
    if exchange_cls is None:
        print(f"ERROR: Exchange '{exchange_id}' not supported by ccxt.", file=sys.stderr)
        sys.exit(1)

    exchange = exchange_cls({"enableRateLimit": True})

    try:
        exchange.load_markets()
    except Exception as exc:
        print(f"ERROR: Failed to load markets: {exc}", file=sys.stderr)
        sys.exit(1)

    if symbol not in exchange.markets:
        print(f"ERROR: Symbol '{symbol}' not listed on '{exchange_id}'.", file=sys.stderr)
        sys.exit(1)

    print(f"Fetching {bars} x {timeframe} candles for {symbol} from {exchange_id}...", flush=True)

    import time as _time

    all_candles: list[list[Any]] = []
    seen_timestamps: set[int] = set()
    tf_ms = _TF_DURATION_MS.get(timeframe, 3_600_000)
    since_ms: int = int(_time.time() * 1000) - (bars + 5) * tf_ms

    while len(all_candles) < bars:
        needed = min(bars - len(all_candles), _PAGE_LIMIT)
        try:
            page: list[list[Any]] = exchange.fetch_ohlcv(
                symbol=symbol, timeframe=timeframe, since=since_ms, limit=needed,
            )
        except Exception as exc:
            print(f"ERROR: Failed to fetch candles: {exc}", file=sys.stderr)
            sys.exit(1)

        if not page:
            break

        new_in_page = 0
        last_ts_ms = since_ms

        for candle in page:
            ts_ms = int(candle[0])
            if ts_ms not in seen_timestamps:
                all_candles.append(candle)
                seen_timestamps.add(ts_ms)
                new_in_page += 1
                last_ts_ms = ts_ms

        if new_in_page == 0 or len(page) < needed:
            break

        since_ms = last_ts_ms + tf_ms

    all_candles.sort(key=lambda c: c[0])

    # Drop the final potentially-incomplete candle
    if all_candles:
        all_candles = all_candles[:-1]

    print(f"Fetched {len(all_candles)} closed candles.", flush=True)
    return all_candles


def _candles_to_df(candles: list[list[Any]]) -> Any:
    """Convert raw ccxt candle list to an OHLCV DataFrame."""
    import pandas as pd

    rows = []
    for c in candles:
        ts_ms, open_, high, low, close, volume = c[:6]
        rows.append({
            "timestamp": datetime.fromtimestamp(int(ts_ms) / 1000, tz=UTC),
            "open": float(open_),
            "high": float(high),
            "low": float(low),
            "close": float(close),
            "volume": float(volume),
        })

    df = pd.DataFrame(rows)
    df = df.set_index("timestamp")
    return df


def _print_metrics(
    symbol: str, timeframe: str, bars_fetched: int,
    metrics: dict[str, Any], model_path: Path,
) -> None:
    """Print a formatted summary of training metrics."""
    sep = "-" * 62
    print(sep)
    print("  MODEL TRAINING SUMMARY")
    print(sep)
    print(f"  Symbol     : {symbol}")
    print(f"  Timeframe  : {timeframe}")
    print(f"  Candles    : {bars_fetched}")
    print(f"  Model path : {model_path}")
    print(sep)
    print(f"  Accuracy   : {metrics['accuracy']:.4f}")
    print(f"  Train set  : {metrics['train_samples']} samples")
    print(f"  Test set   : {metrics['test_samples']} samples")
    print()
    print("  Feature Importances (top 5):")
    sorted_feats = sorted(
        metrics["feature_importances"].items(), key=lambda kv: kv[1], reverse=True,
    )
    for i, (feat, importance) in enumerate(sorted_feats[:5]):
        print(f"    {i + 1}. {feat:<22}  {importance:.4f}")
    print()

    cr: dict[str, Any] = metrics.get("classification_report", {})
    label_map = {"0": "SELL", "1": "HOLD", "2": "BUY"}
    print("  Per-Class Metrics:")
    print(f"    {'Label':<8} {'Precision':>10} {'Recall':>8} {'F1':>8} {'Support':>9}")
    print("   " + "-" * 52)
    for key, label_name in label_map.items():
        if key in cr:
            cls: dict[str, float] = cr[key]
            print(
                f"    {label_name:<8} "
                f"{cls.get('precision', 0.0):.4f}      "
                f"{cls.get('recall', 0.0):.4f}   "
                f"{cls.get('f1-score', 0.0):.4f}   "
                f"{int(cls.get('support', 0)):>7}"
            )
    print(sep)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="train_model",
        description="Train a RandomForestClassifier for ModelStrategy.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--symbol", required=True, help="Trading pair, e.g. BTC/USDT")
    parser.add_argument("--exchange", default="binance", help="CCXT exchange ID")
    parser.add_argument("--timeframe", default="1h", choices=_VALID_TIMEFRAMES, help="Candle timeframe")
    parser.add_argument("--bars", type=int, default=2000, help="Number of candles to fetch")
    parser.add_argument("--horizon", type=int, default=5, help="Look-ahead bars for labels")
    parser.add_argument("--threshold", type=float, default=0.01, help="Return threshold for BUY/SELL")
    parser.add_argument("--n-estimators", type=int, default=100, dest="n_estimators", help="Number of trees")
    parser.add_argument("--random-state", type=int, default=42, dest="random_state", help="Random seed")
    parser.add_argument("--model-dir", default="models/", dest="model_dir", help="Model output directory")
    return parser


def main() -> None:
    """Main entry point for the training CLI."""
    parser = _build_parser()
    args = parser.parse_args()

    if args.bars < 200:
        print(f"ERROR: --bars must be >= 200. Got {args.bars}.", file=sys.stderr)
        sys.exit(1)

    # Step 1: Fetch candles
    candles = _fetch_candles(args.exchange, args.symbol, args.timeframe, args.bars)

    if len(candles) < 200:
        print(
            f"ERROR: Only {len(candles)} candles fetched. Need at least 200.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Step 2: Convert to DataFrame
    df = _candles_to_df(candles)

    # Step 3–5: Prepare, train, save
    try:
        from data.ml_training import ModelTrainer
    except ImportError as exc:
        print(
            f"ERROR: Cannot import data.ml_training: {exc}.\n"
            "Ensure PYTHONPATH includes packages/:\n"
            "  uv run python scripts/train_model.py --symbol BTC/USDT",
            file=sys.stderr,
        )
        sys.exit(2)

    trainer = ModelTrainer(model_dir=args.model_dir)

    print(f"Building dataset (horizon={args.horizon}, threshold={args.threshold:.3f})...", flush=True)
    X, y = trainer.prepare_dataset(df, horizon=args.horizon, threshold=args.threshold)

    print(f"Training RandomForest (n_estimators={args.n_estimators}, seed={args.random_state})...", flush=True)
    metrics = trainer.train(X, y, n_estimators=args.n_estimators, random_state=args.random_state)

    model_path = trainer.save_model(args.symbol)

    # Step 6: Print summary
    _print_metrics(args.symbol, args.timeframe, len(candles), metrics, model_path)
    print(f"\nModel saved to: {model_path}", flush=True)


if __name__ == "__main__":
    # Ensure workspace packages are importable
    _REPO_ROOT = Path(__file__).resolve().parent.parent
    _PACKAGES_DIR = _REPO_ROOT / "packages"
    if str(_PACKAGES_DIR) not in sys.path:
        sys.path.insert(0, str(_PACKAGES_DIR))
    main()
