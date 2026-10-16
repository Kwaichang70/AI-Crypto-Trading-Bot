r"""
scripts/wp21_rebaseline.py
---------------------------
WP2.1c (SY-21-13, spec section 11): the D13 re-baseline harness.

Runs the attribution matrix of the three SY-21-07 baselines on ONE frozen bar fixture
and writes a markdown table plus a JSON record.  It is a MEASUREMENT tool: it never
touches the database, the API or an exchange account.

Two modes
---------
FETCH (once, needs egress to the public Coinbase Exchange candle endpoint; no API keys
are read, created or sent)::

    python scripts/wp21_rebaseline.py --fetch-fixture fixtures/wp21_bars.csv.gz

  writes the gzip-CSV fixture, ``<fixture>.sha256`` and ``<fixture>.meta.json`` (the
  provenance: exchange id, ccxt version, window, and per series the bar count, first and
  last bar, gap count and dropped-invalid count).  The gzip header carries no timestamp and
  no file name, so the same bars always give the same sha256; the meta file is not part of
  the pinned bytes.  The target must end in ``.csv.gz``; an existing fixture, sidecar or
  meta file (or a symlink) is refused unless ``--force`` is given (a symlink never), and
  every file is written to a temp file in the same directory and ``os.replace``d.  The
  exchange id is limited to ``coinbaseexchange``.

RUN (offline; refuses to run unless the fixture bytes match the pinned sha256)::

    python scripts/wp21_rebaseline.py --fixture fixtures/wp21_bars.csv.gz \
        --fixture-sha256 <hex> [--momentum-params params.json] [--out-dir reports/vp2-wp2.1]

Attribution matrix (spec section 11; the parts are NOT claimed to be additive)
-----------------------------------------------------------------------------
    A  historical DB run, if one exists (``--column-a``)         legacy        off
    B  legacy_backtest_v0                                          legacy        off
    C  frictionless_diagnostic                                     gross edge    off
    D  coinbase_advanced_d6                                        D6 cost       off
    E  coinbase_advanced_1 (DEFAULT)                               tier U-1      off
    F  coinbase_advanced_1                                         CF-26 breaker ON

Columns B-E run with ``enable_circuit_breaker=False`` (the ``--no-circuit-breaker`` path of
the research scripts); only F builds the breaker.  Every backtest uses ``seed=42`` and a
single symbol.  Sharpe is labelled "pre-WP2.5" (the WP2.5 Sharpe fix is not in).

Breakeven taker: for each baseline x symbol a bisection over the taker fee of
``coinbase_advanced_1`` (``CostProfile.with_overrides(taker_fee_bps=...)``, slippage kept at
10 bps, maker clamped to the taker because the profile requires maker <= taker) on [0, 250]
bps, stopping at PF = 1.00 +/- 0.01 within at most 12 bisection steps after 2 bracket
evaluations (at most 14 backtests per breakeven row and breaker state).  SA-21C-01: in spec
section 11 an "iteration" is a bisection step; the two bracket evaluations at 0 and 250 bps
validate the bracket and are not iterations.  This is NOT the WP2.8 go/no-go (that uses the
section 6.1 grid and walk-forward OOS).

momentum_breakout parameters: the SY-21-07 fallback (ATR stop 1.5x / take-profit 3.0x,
registry defaults otherwise).  The live run ``f36fb44e`` params can be supplied with
``--momentum-params`` (a RunORM ``config`` excerpt or a flat strategy-params object).

No secret is read: the script touches no environment variable and builds the exchange
object with ``enableRateLimit`` only.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import gzip
import hashlib
import io
import itertools
import json
import math
import os
import re
import sys
import tempfile
import threading
import time
import zlib
from collections.abc import Awaitable, Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Protocol, cast
from unittest import mock

from common.models import OHLCVBar
from common.types import TimeFrame
from trading.backtest import BacktestRunner
from trading.costs import CostProfile, CostProfileError, get_cost_profile
from trading.exit_config import ExitConfigError
from trading.models import RiskCheckResult, SkippedTrade
from trading.portfolio import RiskSnapshot
from trading.risk_manager import DefaultRiskManager
from trading.safety import CircuitBreaker, CircuitBreakerDecision
from trading.strategies import (
    GridTradingStrategy,
    MomentumBreakoutStrategy,
    RSIMeanReversionStrategy,
)
from trading.strategy import BaseStrategy
from trading.trade_journal import TradeSkipLogger

__all__ = [
    "ALLOWED_EXCHANGES",
    "BASELINES",
    "BREAKEVEN_BASE_PROFILE_ID",
    "BREAKEVEN_HI_BPS",
    "BREAKEVEN_LO_BPS",
    "BREAKEVEN_MAX_BACKTESTS",
    "BREAKEVEN_MAX_ITERATIONS",
    "BREAKEVEN_PF_TOLERANCE",
    "COLUMNS",
    "SEED",
    "BaselineSpec",
    "BreakevenResult",
    "Column",
    "FixtureError",
    "GateProbe",
    "bisect_breakeven",
    "classify_rejection_reason",
    "load_fixture",
    "probe_gates",
    "read_json_object",
    "render_markdown",
    "run_matrix",
    "write_fixture",
]

SEED = 42
INITIAL_CAPITAL = Decimal("10000")
DEFAULT_START = "2023-01-01"
DEFAULT_END = "2026-09-30"  # inclusive (the whole UTC day)

BREAKEVEN_BASE_PROFILE_ID = "coinbase_advanced_1"
BREAKEVEN_LO_BPS = Decimal(0)
BREAKEVEN_HI_BPS = Decimal(250)
BREAKEVEN_PF_TARGET = 1.0
BREAKEVEN_PF_TOLERANCE = 0.01
#: Bisection steps (SA-21C-01).  The 2 bracket evaluations at LO/HI come on top, so a breakeven
#: row costs at most ``BREAKEVEN_MAX_ITERATIONS + 2`` backtests per breaker state.
BREAKEVEN_MAX_ITERATIONS = 12
BREAKEVEN_MAX_BACKTESTS = BREAKEVEN_MAX_ITERATIONS + 2

#: The only exchange FETCH mode talks to (SEC-01): keyless public candles.
ALLOWED_EXCHANGES: tuple[str, ...] = ("coinbaseexchange",)

FIXTURE_HEADER = ("symbol", "timeframe", "ts_ms", "open", "high", "low", "close", "volume")
_MAX_FIXTURE_BYTES = 256 * 1024 * 1024  # decompressed cap: a gzip bomb must not fill memory
_MAX_FIXTURE_FILE_BYTES = 64 * 1024 * 1024  # compressed cap, checked with stat() before a read
_MAX_PARAMS_BYTES = 1024 * 1024
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

_TF_MS = {"1h": 3_600_000, "1d": 86_400_000}
_TF_ENUM = {"1h": TimeFrame.ONE_HOUR, "1d": TimeFrame.ONE_DAY}
_COINBASE_PAGE = 300  # the Coinbase Exchange candle endpoint returns at most 300 rows

#: ``bracket_*`` keys live at engine level (RunORM ``config["bracket_config"]``), not in the
#: strategy parameters -- the same split ``ParameterOptimizer`` and the API perform.
_BRACKET_KEYS = (
    "bracket_mode",
    "bracket_stop_loss_pct",
    "bracket_take_profit_pct",
    "bracket_atr_sl_multiplier",
    "bracket_atr_tp_multiplier",
    "bracket_atr_period",
)

#: SY-21-07 fallback for momentum_breakout when the live-run params are not available.
MOMENTUM_FALLBACK_BRACKET: dict[str, object] = {
    "bracket_mode": "atr",
    "bracket_atr_sl_multiplier": 1.5,
    "bracket_atr_tp_multiplier": 3.0,
    "bracket_atr_period": 14,
}


class FixtureError(Exception):
    """The fixture is missing, malformed, or does not match the pinned sha256."""


# ---------------------------------------------------------------------------
# Matrix definition
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Column:
    """One attribution column: which cost profile, and whether the breaker is on."""

    key: str
    profile_id: str
    breaker: bool
    isolates: str


COLUMNS: tuple[Column, ...] = (
    Column("B", "legacy_backtest_v0", False, "A->B: 1.3a/1.3b/1.5 semantic drift + C25"),
    Column("C", "frictionless_diagnostic", False, "gross edge"),
    Column("D", "coinbase_advanced_d6", False, "B->D: D6 cost effect"),
    Column("E", "coinbase_advanced_1", False, "D->E: tier sensitivity (U-1)"),
    Column("F", "coinbase_advanced_1", True, "E->F: CF-26 breaker effect"),
)

#: The attribution chain of the spec (C is the gross-edge reference, outside the chain).
ATTRIBUTION_CHAIN = ("A", "B", "D", "E", "F")


@dataclass(frozen=True)
class BaselineSpec:
    """One SY-21-07 baseline: a strategy, a timeframe, symbols and its parameters."""

    name: str
    strategy_cls: type[BaseStrategy]
    timeframe: str
    symbols: tuple[str, ...]
    strategy_params: Mapping[str, Any] = field(default_factory=dict)
    bracket: Mapping[str, object] = field(default_factory=dict)
    trailing_stop_pct: float | None = None
    allow_pyramiding: bool | None = None
    params_source: str = "registry-defaults"


BASELINES: tuple[BaselineSpec, ...] = (
    BaselineSpec(
        "momentum_breakout",
        MomentumBreakoutStrategy,
        "1d",
        ("BTC/EUR", "ETH/EUR", "SOL/EUR"),
        bracket=MOMENTUM_FALLBACK_BRACKET,
        params_source="sy-21-07-fallback",
    ),
    BaselineSpec("grid_trading", GridTradingStrategy, "1h", ("BTC/EUR", "ETH/EUR")),
    BaselineSpec("rsi_mean_reversion", RSIMeanReversionStrategy, "1h", ("BTC/EUR", "ETH/EUR")),
)


def split_params(
    raw: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, object], float | None, bool | None]:
    """Split one flat params object into (strategy params, bracket config, trailing, pyramiding)."""
    strategy_params = dict(raw)
    bracket: dict[str, object] = {}
    for key in _BRACKET_KEYS:
        if key in strategy_params:
            value = strategy_params.pop(key)
            if value is not None and value != "":
                bracket[key] = value
    trailing_raw = strategy_params.get("trailing_stop_pct")
    trailing = (
        float(trailing_raw)
        if isinstance(trailing_raw, int | float) and not isinstance(trailing_raw, bool)
        else None
    )
    pyr_raw = strategy_params.pop("allow_pyramiding", None)
    pyramiding = pyr_raw if isinstance(pyr_raw, bool) else None
    return strategy_params, bracket, trailing, pyramiding


def read_json_object(path: Path) -> tuple[dict[str, Any], str]:
    """Read a JSON object file ONCE: ``(object, sha256 of the exact bytes that were parsed)``.

    ``stat`` cap first, then ONE bounded ``read(cap + 1)``, then ``json.loads`` of those same
    bytes -- so the recorded provenance hash describes what was parsed and a file that grows
    (or lies about its size) is still never read past the cap (SEC-03).  A missing or
    unreadable file is an error, never an empty input.
    """
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise FixtureError(f"cannot read JSON file: {type(exc).__name__}") from exc
    if size > _MAX_PARAMS_BYTES:
        raise FixtureError("JSON file is larger than 1 MiB")
    try:
        with path.open("rb") as handle:
            data = handle.read(_MAX_PARAMS_BYTES + 1)
    except OSError as exc:
        raise FixtureError(f"cannot read JSON file: {type(exc).__name__}") from exc
    if len(data) > _MAX_PARAMS_BYTES:
        raise FixtureError("JSON file is larger than 1 MiB")
    try:
        doc = json.loads(data.decode("utf-8"))
    except (ValueError, RecursionError) as exc:  # JSONDecodeError and UnicodeDecodeError
        raise FixtureError(f"JSON file is not valid: {type(exc).__name__}") from exc
    if not isinstance(doc, dict):
        raise FixtureError("JSON file must contain an object")
    return doc, sha256_hex(data)


def load_json_object(path: Path) -> dict[str, Any]:
    """Read a JSON file (bounded size, must be an object); see :func:`read_json_object`."""
    return read_json_object(path)[0]


def momentum_spec_from_params(doc: Mapping[str, Any], *, source: str) -> BaselineSpec:
    """The momentum_breakout baseline for caller-supplied (live-run) parameters.

    Accepts a ``RunORM.config`` excerpt (``strategy_params`` / ``bracket_config`` /
    ``allow_pyramiding`` / ``symbols``) or a flat strategy-params object that may carry
    ``bracket_*`` keys.  The file REPLACES the fallback (no merge: a fixed-percent bracket
    must not inherit the fallback's ATR multipliers).
    """
    base = BASELINES[0]
    symbols = base.symbols
    if "strategy_params" in doc:
        sp = doc["strategy_params"]
        if not isinstance(sp, dict):
            raise FixtureError("strategy_params must be a JSON object")
        flat: dict[str, Any] = dict(sp)
        bc = doc.get("bracket_config")
        if isinstance(bc, dict):
            flat.update(bc)
        if "allow_pyramiding" in doc:
            flat["allow_pyramiding"] = doc["allow_pyramiding"]
        raw_symbols = doc.get("symbols")
        if (
            isinstance(raw_symbols, list)
            and raw_symbols
            and all(isinstance(s, str) for s in raw_symbols)
        ):
            symbols = tuple(raw_symbols)
    else:
        flat = dict(doc)
    strategy_params, bracket, trailing, pyramiding = split_params(flat)
    return BaselineSpec(
        base.name,
        base.strategy_cls,
        base.timeframe,
        symbols,
        strategy_params=strategy_params,
        bracket=bracket,
        trailing_stop_pct=trailing,
        allow_pyramiding=pyramiding,
        params_source=source,
    )


# ---------------------------------------------------------------------------
# Fixture: gzip CSV + sha256
# ---------------------------------------------------------------------------


Candle = tuple[int, str, str, str, str, str]  # ts_ms, open, high, low, close, volume (as text)


class OhlcvExchange(Protocol):
    """The one ccxt method the fetch uses (public candles; no credentials)."""

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = ...,
        since: int | None = ...,
        limit: int | None = ...,
        params: dict[str, Any] = ...,
    ) -> list[list[float]]: ...


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


_FIXTURE_SUFFIX = ".csv.gz"
_META_SCHEMA = "wp21-fixture-meta-1"


def sidecar_path(path: Path) -> Path:
    """``<fixture>.sha256``."""
    return path.with_name(path.name + ".sha256")


def meta_path(path: Path) -> Path:
    """``<fixture>.meta.json`` (the provenance of a FETCH)."""
    return path.with_name(path.name + ".meta.json")


def check_write_targets(path: Path, *, with_meta: bool, force: bool) -> None:
    """Refuse a bad fixture target BEFORE anything is fetched or written (SEC-04).

    The target must end in ``.csv.gz`` (a typo such as ``docs/risk-model.md`` must never be
    replaced by gzip).  A symlink -- the fixture, its sidecar or its meta file -- is refused
    even with ``force``; an existing file is refused unless ``force``.
    """
    if not path.name.endswith(_FIXTURE_SUFFIX) or path.name == _FIXTURE_SUFFIX:
        raise FixtureError(f"the fixture target must be a *{_FIXTURE_SUFFIX} file")
    targets = [path, sidecar_path(path)]
    if with_meta:
        targets.append(meta_path(path))
    for target in targets:
        if target.is_symlink():
            raise FixtureError(f"refusing to write through a symlink: {target.name}")
        if target.is_dir():
            raise FixtureError(f"{target.name} is a directory")
        if target.exists() and not force:
            raise FixtureError(f"{target.name} already exists; pass --force to overwrite it")


def _fixture_bytes(series: Mapping[tuple[str, str], Sequence[Candle]]) -> bytes:
    text = io.StringIO(newline="")
    writer = csv.writer(text, lineterminator="\n")
    writer.writerow(FIXTURE_HEADER)
    for symbol, timeframe in sorted(series):
        for ts, o, h, low, c, v in sorted(series[(symbol, timeframe)]):
            writer.writerow((symbol, timeframe, ts, o, h, low, c, v))
    buffer = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=buffer, mtime=0, compresslevel=9) as gz:
        gz.write(text.getvalue().encode("utf-8"))
    return buffer.getvalue()


def _stage_bytes(directory: Path, data: bytes) -> Path:
    """Write ``data`` to a fresh (O_EXCL) temp file in ``directory``; the caller replaces it."""
    fd, name = tempfile.mkstemp(dir=directory, prefix=".wp21-", suffix=".tmp")
    staged = Path(name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
        staged.chmod(0o644)
    except BaseException:
        staged.unlink(missing_ok=True)
        raise
    return staged


def write_fixture(
    path: Path,
    series: Mapping[tuple[str, str], Sequence[Candle]],
    *,
    provenance: Mapping[str, Any] | None = None,
    force: bool = False,
) -> str:
    """Write the gzip-CSV fixture, ``<path>.sha256`` and (FETCH) ``<path>.meta.json``.

    Returns the sha256 of the gz bytes.  Rows are sorted by (symbol, timeframe, ts) and the
    gzip header is fixed (mtime 0, no file name), so identical bars always produce identical
    bytes; the meta file never changes them.  Every file is staged in a temp file in the same
    directory and only then ``os.replace``d: a failure while staging touches no target, and a
    failure while replacing removes the targets this call created.  (Overwriting with
    ``force`` cannot be rolled back after the first replace; the sidecar and the meta file
    carry the sha256, so a mixed set is detectable.)
    """
    check_write_targets(path, with_meta=provenance is not None, force=force)
    payload = _fixture_bytes(series)
    digest = sha256_hex(payload)
    files: list[tuple[Path, bytes]] = [
        (path, payload),
        (sidecar_path(path), f"{digest}  {path.name}\n".encode()),
    ]
    if provenance is not None:
        meta = {"schema": _META_SCHEMA, "fixture_sha256": digest, **provenance}
        files.append(
            (meta_path(path), (json.dumps(meta, indent=2, sort_keys=True) + "\n").encode())
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    existed = {target: target.exists() for target, _data in files}
    staged: list[tuple[Path, Path]] = []
    replaced: list[Path] = []
    try:
        for target, data in files:
            staged.append((target, _stage_bytes(target.parent, data)))
        for target, temp in staged:
            os.replace(temp, target)
            replaced.append(target)
    except BaseException:
        for _target, temp in staged:
            temp.unlink(missing_ok=True)
        for target in replaced:
            if not existed[target]:
                target.unlink(missing_ok=True)
        raise
    return digest


def verify_fixture_sha256(path: Path, expected: str) -> bytes:
    """Return the fixture bytes iff their sha256 equals ``expected``; else raise.

    The compressed size is capped (``stat`` first, then a bounded read) before anything is
    hashed.
    """
    expected = expected.strip().lower()
    if not _SHA256_RE.fullmatch(expected):
        raise FixtureError("--fixture-sha256 must be 64 lowercase hex characters")
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise FixtureError(f"cannot read fixture: {type(exc).__name__}") from exc
    if size > _MAX_FIXTURE_FILE_BYTES:
        raise FixtureError("fixture file is larger than the compressed size cap")
    try:
        with path.open("rb") as handle:
            payload = handle.read(_MAX_FIXTURE_FILE_BYTES + 1)
    except OSError as exc:
        raise FixtureError(f"cannot read fixture: {type(exc).__name__}") from exc
    if len(payload) > _MAX_FIXTURE_FILE_BYTES:
        raise FixtureError("fixture file is larger than the compressed size cap")
    actual = sha256_hex(payload)
    if actual != expected:
        raise FixtureError(
            f"fixture sha256 mismatch: expected {expected}, file has {actual}; refusing to run"
        )
    return payload


def _parse_fixture_bytes(payload: bytes) -> dict[tuple[str, str], list[OHLCVBar]]:
    try:
        with gzip.GzipFile(fileobj=io.BytesIO(payload), mode="rb") as gz:
            raw = gz.read(_MAX_FIXTURE_BYTES + 1)
    except (OSError, EOFError, zlib.error) as exc:  # BadGzipFile is an OSError
        raise FixtureError(f"fixture is not a valid gzip file: {type(exc).__name__}") from exc
    if len(raw) > _MAX_FIXTURE_BYTES:
        raise FixtureError("fixture is larger than the decompressed size cap")
    try:
        # the pin only proves the bytes are the pinned ones, not that they are well-formed
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise FixtureError(f"fixture is not valid UTF-8 text: {type(exc).__name__}") from exc
    try:
        return _parse_fixture_rows(csv.reader(io.StringIO(text, newline="")))
    except csv.Error as exc:
        raise FixtureError(f"fixture is not valid CSV: {type(exc).__name__}") from exc


def _parse_fixture_rows(reader: Iterator[list[str]]) -> dict[tuple[str, str], list[OHLCVBar]]:
    header = next(reader, None)
    if tuple(header or ()) != FIXTURE_HEADER:
        raise FixtureError("fixture header does not match the expected columns")
    out: dict[tuple[str, str], list[OHLCVBar]] = {}
    for lineno, row in enumerate(reader, start=2):
        if len(row) != len(FIXTURE_HEADER):
            raise FixtureError(f"fixture line {lineno}: wrong column count")
        symbol, timeframe, ts_raw, o, h, low, c, v = row
        if timeframe not in _TF_ENUM:
            raise FixtureError(f"fixture line {lineno}: unsupported timeframe")
        try:
            bar = OHLCVBar(
                symbol=symbol,
                timeframe=_TF_ENUM[timeframe],
                timestamp=datetime.fromtimestamp(int(ts_raw) / 1000, tz=UTC),
                open=Decimal(o),
                high=Decimal(h),
                low=Decimal(low),
                close=Decimal(c),
                volume=Decimal(v),
            )
        except (ValueError, ArithmeticError, OSError, OverflowError) as exc:
            # OSError / OverflowError: ``fromtimestamp`` on an out-of-range timestamp
            raise FixtureError(
                f"fixture line {lineno}: invalid bar ({type(exc).__name__})"
            ) from exc
        out.setdefault((symbol, timeframe), []).append(bar)
    for key, bars in out.items():
        stamps = [b.timestamp for b in bars]
        if any(b <= a for a, b in itertools.pairwise(stamps)):
            raise FixtureError(f"fixture series {key[0]} {key[1]} is not strictly increasing")
    return out


def load_fixture(
    path: Path, expected_sha256: str
) -> tuple[dict[tuple[str, str], list[OHLCVBar]], str]:
    """Verify the sha256, then parse; returns (series by (symbol, timeframe), sha256)."""
    payload = verify_fixture_sha256(path, expected_sha256)
    return _parse_fixture_bytes(payload), sha256_hex(payload)


# ---------------------------------------------------------------------------
# Fetch (public, read-only OHLCV; no API keys)
# ---------------------------------------------------------------------------


def _day_start_ms(day: str) -> int:
    d = date.fromisoformat(day)
    return int(datetime(d.year, d.month, d.day, tzinfo=UTC).timestamp() * 1000)


def _valid_candle(o: float, h: float, low: float, c: float, v: float) -> bool:
    values = (o, h, low, c, v)
    if any(not isinstance(x, int | float) or not math.isfinite(x) for x in values):
        return False
    return min(o, h, low, c) > 0 and v >= 0 and low <= o <= h and low <= c <= h


def fetch_candles(
    exchange: OhlcvExchange,
    symbol: str,
    timeframe: str,
    start_ms: int,
    end_ms: int,
    *,
    sleep: Callable[[float], None] = time.sleep,
    max_attempts: int = 5,
) -> tuple[list[Candle], int]:
    """Page ``fetch_ohlcv`` over [start_ms, end_ms); returns (candles, dropped invalid count).

    The cursor advances by one page window whether or not a page had rows, so a market gap
    never stalls the fetch.  A transient failure is retried with exponential backoff (1s, 2s,
    4s, ...); anything else propagates.  Candles are NOT back-filled: a gap stays a gap.
    """
    import ccxt  # local import: only fetch mode needs it

    tf_ms = _TF_MS[timeframe]
    seen: dict[int, Candle] = {}
    dropped = 0
    cursor = start_ms
    while cursor < end_ms:
        page_end = min(cursor + _COINBASE_PAGE * tf_ms, end_ms)
        for attempt in range(max_attempts):
            try:
                page = exchange.fetch_ohlcv(
                    symbol,
                    timeframe=timeframe,
                    since=cursor,
                    limit=_COINBASE_PAGE,
                    params={"until": page_end - 1},
                )
                break
            except (ccxt.NetworkError, ccxt.RateLimitExceeded) as exc:
                if attempt == max_attempts - 1:
                    raise
                print(f"  retry {attempt + 1} after {type(exc).__name__}", file=sys.stderr)
                sleep(2.0**attempt)
        for ts, o, h, low, c, v in ((int(r[0]), r[1], r[2], r[3], r[4], r[5]) for r in page):
            if not (start_ms <= ts < end_ms):
                continue
            if not _valid_candle(o, h, low, c, v):
                dropped += 1
                continue
            seen[ts] = (ts, str(o), str(h), str(low), str(c), str(v))
        cursor = page_end
    return [seen[k] for k in sorted(seen)], dropped


def series_provenance(
    candles: Sequence[Candle], timeframe: str, *, dropped: int, start_ms: int, end_ms: int
) -> dict[str, Any]:
    """Provenance of one fetched series (all integers / ISO strings: deterministic).

    ``gap_count`` counts the expected intervals missing BETWEEN the first and the last bar;
    ``missing_vs_window`` counts every expected interval of the requested window that has no
    bar (a late listing or an early end shows up here, not in ``gap_count``).
    """
    step = _TF_MS[timeframe]
    stamps = [c[0] for c in candles]
    gaps = sum(-(-(b - a) // step) - 1 for a, b in itertools.pairwise(stamps) if b - a > step)
    expected = -(-(end_ms - start_ms) // step)

    def iso(ms: int) -> str:
        return datetime.fromtimestamp(ms / 1000, tz=UTC).isoformat()

    return {
        "bars": len(stamps),
        "first_ts": iso(stamps[0]) if stamps else None,
        "last_ts": iso(stamps[-1]) if stamps else None,
        "gap_count": gaps,
        "missing_vs_window": max(0, expected - len(stamps)),
        "dropped_invalid": dropped,
    }


def fetch_fixture(out: Path, *, exchange_id: str, start: str, end: str, force: bool = False) -> str:
    """Fetch every baseline series once and write the fixture; returns its sha256.

    Everything that can be refused is refused BEFORE the first request: an exchange id outside
    :data:`ALLOWED_EXCHANGES` (so ``getattr(ccxt, ...)`` never sees an arbitrary name), and a
    target that is not a fresh ``.csv.gz`` path (see :func:`check_write_targets`).
    """
    if exchange_id not in ALLOWED_EXCHANGES:
        raise FixtureError(f"unsupported exchange id; allowed: {', '.join(ALLOWED_EXCHANGES)}")
    check_write_targets(out, with_meta=True, force=force)
    start_ms = _day_start_ms(start)
    end_ms = _day_start_ms(end) + 86_400_000  # inclusive end day

    import ccxt

    # Keyless by construction: the only option is the rate limiter, and nothing here reads an
    # environment variable or a file.  The check below is defence in depth (a ccxt build that
    # injected credentials would be refused), not the control.
    exchange = getattr(ccxt, exchange_id)({"enableRateLimit": True})
    if getattr(exchange, "apiKey", None) or getattr(exchange, "secret", None):
        raise FixtureError("refusing to fetch with credentials configured")
    wanted = sorted({(s, b.timeframe) for b in BASELINES for s in b.symbols})
    series: dict[tuple[str, str], list[Candle]] = {}
    provenance_series: dict[str, Any] = {}
    for symbol, timeframe in wanted:
        candles, dropped = fetch_candles(exchange, symbol, timeframe, start_ms, end_ms)
        series[(symbol, timeframe)] = candles
        print(
            f"  {symbol} {timeframe}: {len(candles)} bars ({dropped} invalid dropped)",
            file=sys.stderr,
        )
        if not candles:
            raise FixtureError(f"no candles returned for {symbol} {timeframe}")
        provenance_series[f"{symbol}|{timeframe}"] = series_provenance(
            candles, timeframe, dropped=dropped, start_ms=start_ms, end_ms=end_ms
        )
    provenance = {
        "exchange": exchange_id,
        "ccxt_version": str(getattr(ccxt, "__version__", "unknown")),
        "window": {"start": start, "end": end},
        "series": provenance_series,
    }
    return write_fixture(out, series, provenance=provenance, force=force)


# ---------------------------------------------------------------------------
# Instrumentation: gate rejections and circuit-breaker transitions
# ---------------------------------------------------------------------------


#: Rejection classes of ``DefaultRiskManager.pre_trade_check``.  ``RiskCheckResult`` carries only
#: ``rejection_reasons: list[str]`` (built from ``RiskViolation.message``), and both daily-loss
#: messages share ``rule="max_daily_loss"``, so the classification is by message prefix.  The
#: contract tests in ``tests/unit/test_wp21c_rebaseline.py`` drive the REAL risk manager into
#: each state, so a wording change fails a test instead of silently yielding zero (CF-21C-05
#: would replace this with engine-exposed counters).  Exposure-reducing bypass messages start
#: with ``bypassed`` and are warnings, never rejection reasons.
_DAILY_BASIS_PREFIX = "Daily loss basis unavailable"
_DAILY_PREFIX = "Daily loss "
_DRAWDOWN_PREFIX = "Drawdown "


def classify_rejection_reason(reason: str) -> str:
    """``daily_basis_unavailable`` | ``daily`` | ``drawdown`` | ``other`` for one reason text."""
    if reason.startswith(_DAILY_BASIS_PREFIX):  # before ``daily``: it shares the prefix
        return "daily_basis_unavailable"
    if reason.startswith(_DAILY_PREFIX):
        return "daily"
    if reason.startswith(_DRAWDOWN_PREFIX):
        return "drawdown"
    return "other"


@dataclass
class GateProbe:
    """Counts what the engine does not expose: gate rejections and breaker activity.

    The gate counters are per rejected ``pre_trade_check``: a check counts once in every class
    one of its reasons belongs to (so the classes are not additive), and a rejected check
    always lands in at least one class -- ``other_rejections`` catches every reason that is
    not a daily-loss or drawdown message, and a rejection without any reason.
    """

    daily_rejections: int = 0  # daily-loss threshold breached
    daily_basis_unavailable: int = 0  # fail-closed: the day-start basis was <= 0
    drawdown_rejections: int = 0
    other_rejections: int = 0
    cb_entry_drops: dict[str, int] = field(default_factory=dict)
    cb_transitions: int = 0
    cb_transitions_to: dict[str, int] = field(default_factory=dict)

    @property
    def cb_rejections(self) -> int:
        return sum(self.cb_entry_drops.values())

    def record_rejection(self, reasons: Sequence[str]) -> None:
        classes = {classify_rejection_reason(r) for r in reasons} or {"other"}
        if "daily" in classes:
            self.daily_rejections += 1
        if "daily_basis_unavailable" in classes:
            self.daily_basis_unavailable += 1
        if "drawdown" in classes:
            self.drawdown_rejections += 1
        if "other" in classes:
            self.other_rejections += 1


#: ``probe_gates`` patches CLASS attributes, which every thread sees: one probe at a time.
_PROBE_LOCK = threading.Lock()


@contextmanager
def probe_gates() -> Iterator[GateProbe]:
    """Wrap three engine entry points for the duration of ONE backtest (restored on exit).

    * ``DefaultRiskManager.pre_trade_check``: rejected checks, classified by
      :func:`classify_rejection_reason` -- the WP1.5 gates; exits bypass them.
    * ``TradeSkipLogger.log_skip``: BUYs the breaker dropped (``circuit_breaker_*``).
    * ``CircuitBreaker.evaluate``: level changes (``decision.transition``).

    The probe only observes; the wrapped calls return exactly what they returned before.

    **Never call this from a process that runs paper or live engines**: the patches are
    process-global (every thread sees the wrappers) while the probe is active.  It is
    therefore not re-entrant either: a second ``probe_gates()`` while one is active raises
    ``RuntimeError`` instead of stacking wrappers.
    """
    if not _PROBE_LOCK.acquire(blocking=False):
        raise RuntimeError("probe_gates() is already active; it is not re-entrant")
    try:
        probe = GateProbe()
        orig_check = cast("Callable[..., RiskCheckResult]", DefaultRiskManager.pre_trade_check)
        orig_skip = cast("Callable[..., SkippedTrade]", TradeSkipLogger.log_skip)
        orig_eval = cast("Callable[..., CircuitBreakerDecision]", CircuitBreaker.evaluate)

        def check(self: DefaultRiskManager, *args: object, **kwargs: object) -> RiskCheckResult:
            result = orig_check(self, *args, **kwargs)
            if not result.approved:
                probe.record_rejection(result.rejection_reasons)
            return result

        def skip(
            self: TradeSkipLogger, symbol: str, skip_reason: str, *args: object, **kwargs: object
        ) -> SkippedTrade:
            if skip_reason.startswith("circuit_breaker_"):
                level = skip_reason.removeprefix("circuit_breaker_")
                probe.cb_entry_drops[level] = probe.cb_entry_drops.get(level, 0) + 1
            return orig_skip(self, symbol, skip_reason, *args, **kwargs)

        def evaluate(self: CircuitBreaker, snapshot: RiskSnapshot) -> CircuitBreakerDecision:
            decision = orig_eval(self, snapshot)
            if decision.transition is not None:
                probe.cb_transitions += 1
                to = str(decision.transition.to_level.value)
                probe.cb_transitions_to[to] = probe.cb_transitions_to.get(to, 0) + 1
            return decision

        # ``mock.patch.object`` on purpose: it restores the original on every exit path
        # (including an exception inside the run) and leaves no other state behind.  This is
        # a measurement SCRIPT: nothing under ``apps/`` or ``packages/`` imports it or
        # ``unittest.mock`` (CF-21C-02 will pin that); CF-21C-05 removes the need altogether.
        with (
            mock.patch.object(DefaultRiskManager, "pre_trade_check", check),
            mock.patch.object(TradeSkipLogger, "log_skip", skip),
            mock.patch.object(CircuitBreaker, "evaluate", evaluate),
        ):
            yield probe
    finally:
        _PROBE_LOCK.release()


# ---------------------------------------------------------------------------
# One cell
# ---------------------------------------------------------------------------

_EXIT_REASONS = ("stop_loss", "take_profit", "trailing_stop", "signal_exit")


def _num(value: float) -> float:
    """Stable float for the JSON record (the runs are deterministic; this only trims noise)."""
    return round(value, 10)


def _pf_value(profit_factor: float | None, infinite: bool) -> float | None:
    if infinite:
        return math.inf
    return profit_factor


def _pf_json(pf: float | None) -> float | str | None:
    if pf is None:
        return None
    if math.isinf(pf):
        return "inf"
    return _num(pf)


async def run_cell(
    spec: BaselineSpec,
    symbol: str,
    bars: list[OHLCVBar],
    profile: CostProfile,
    *,
    breaker: bool,
) -> dict[str, Any]:
    """Run ONE backtest (single symbol, seed 42) and return its metrics record."""
    strategy = spec.strategy_cls(
        strategy_id=f"{spec.name}-{symbol}", params=dict(spec.strategy_params)
    )
    runner = BacktestRunner(
        strategies=[strategy],
        symbols=[symbol],
        timeframe=_TF_ENUM[spec.timeframe],
        initial_capital=INITIAL_CAPITAL,
        trailing_stop_pct=spec.trailing_stop_pct,
        bracket_config=dict(spec.bracket),
        allow_pyramiding=spec.allow_pyramiding,
        seed=SEED,
        enable_circuit_breaker=breaker,
        cost_profile=profile,
    )
    with probe_gates() as probe:
        result = await runner.run({symbol: bars})
    engine = runner.last_execution_engine
    fills = engine.get_all_fills() if engine is not None else []
    traded = sum((f.price * f.quantity for f in fills), Decimal(0))
    entry_notional = sum((t.entry_price * t.quantity for t in result.trades), Decimal(0))
    total_cost = result.total_fees_paid + result.total_slippage_cost
    exit_counts = {r: sum(1 for t in result.trades if t.exit_reason == r) for r in _EXIT_REASONS}
    n_trades = result.total_trades
    sl = exit_counts["stop_loss"]
    pf = _pf_value(result.profit_factor, result.profit_factor_is_infinite)
    return {
        "cost_profile_id": profile.id,
        "cost_profile_fingerprint": profile.fingerprint(),
        "taker_fee_bps": str(profile.taker_fee_bps),
        "breaker": breaker,
        "trades": n_trades,
        "stop_loss_exits": sl,
        "stop_loss_share": _num(sl / n_trades) if n_trades else None,
        "exit_reasons": exit_counts,
        "win_rate": _num(result.win_rate),
        "profit_factor": _pf_json(pf),
        "cagr": _num(result.cagr),
        "max_drawdown_pct": _num(result.max_drawdown_pct),
        "total_return_pct": _num(result.total_return_pct),
        "final_equity": str(result.final_equity),
        "total_fees": str(result.total_fees_paid),
        "total_slippage_cost": str(result.total_slippage_cost),
        # traded notional (both sides) over the initial capital
        "turnover_x": _num(float(traded / INITIAL_CAPITAL)),
        # (fees + slippage) over the entry notional of the closed trades, in bps: the all-in
        # cost of one round trip
        "cost_per_trade_bps": _num(float(total_cost / entry_notional * 10_000))
        if entry_notional > 0
        else None,
        "gate_rejections": {
            "daily": probe.daily_rejections,
            "daily_basis_unavailable": probe.daily_basis_unavailable,
            "drawdown": probe.drawdown_rejections,
            "other": probe.other_rejections,
            "circuit_breaker": probe.cb_rejections,
            "circuit_breaker_by_level": dict(sorted(probe.cb_entry_drops.items())),
        },
        "cb_transitions": probe.cb_transitions,
        "sharpe_pre_wp25": _num(result.sharpe_ratio),
        "open_positions_at_end": len(result.open_positions_mtm),
    }


# ---------------------------------------------------------------------------
# Breakeven bisection
# ---------------------------------------------------------------------------


class UnevaluableError(Exception):
    """The evaluator cannot price this taker (e.g. the exit config is rejected at it)."""


@dataclass
class BreakevenResult:
    # converged | max_iterations | below_target_at_lo | above_target_at_hi | no_trades | unevaluable
    status: str
    taker_bps: Decimal | None
    pf: float | None
    iterations: int
    evaluations: list[tuple[Decimal, float | None]]

    def to_json(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "breakeven_taker_bps": None if self.taker_bps is None else str(self.taker_bps),
            "pf_at_breakeven": _pf_json(self.pf),
            "iterations": self.iterations,
            "evaluations": [[str(t), _pf_json(p)] for t, p in self.evaluations],
        }


def _q2(value: Decimal) -> Decimal:
    return value.quantize(Decimal("0.01"))


async def bisect_breakeven(
    pf_at: Callable[[Decimal], Awaitable[float | None]],
    *,
    lo: Decimal = BREAKEVEN_LO_BPS,
    hi: Decimal = BREAKEVEN_HI_BPS,
    target: float = BREAKEVEN_PF_TARGET,
    tolerance: float = BREAKEVEN_PF_TOLERANCE,
    max_iterations: int = BREAKEVEN_MAX_ITERATIONS,
) -> BreakevenResult:
    """Find the taker (bps) where PF = ``target`` +/- ``tolerance``.

    PF is assumed to fall as the taker rises (the bracket basis moves stops closer too, X-15,
    so this is not guaranteed -- every evaluation is recorded).

    Budget (SA-21C-01): at most 12 bisection steps after 2 bracket evaluations (at most 14
    backtests per breakeven row and breaker state), where 12 is ``max_iterations``.  In spec
    section 11 an "iteration" is a bisection step; the two end points (lo, hi) are evaluated
    first, validate the bracket and decide the status (``below_target_at_lo`` /
    ``above_target_at_hi``), and are not iterations.  ``BreakevenResult.iterations`` counts
    bisection steps only; ``evaluations`` lists every backtest.  Midpoints are rounded to
    0.01 bps, the ``CostProfile`` resolution.  ``pf_at`` returns ``None`` when PF is undefined
    (no trades), may return ``math.inf``, and raises :class:`UnevaluableError` when a taker
    cannot be priced.
    """
    evals: list[tuple[Decimal, float | None]] = []

    async def probe(taker: Decimal) -> float | None:
        pf = await pf_at(taker)
        evals.append((taker, pf))
        return pf

    def distance(pf: float | None) -> float:
        return math.inf if pf is None or math.isinf(pf) else abs(pf - target)

    def result(status: str, iterations: int) -> BreakevenResult:
        finite = [e for e in evals if not math.isinf(distance(e[1]))]
        if status in ("converged", "max_iterations") and finite:
            taker, pf = min(finite, key=lambda e: distance(e[1]))
            return BreakevenResult(status, taker, pf, iterations, evals)
        last_pf = evals[-1][1] if evals else None
        return BreakevenResult(status, None, last_pf, iterations, evals)

    iterations = 0
    try:
        pf_lo = await probe(lo)
        if distance(pf_lo) <= tolerance:
            return result("converged", 0)
        pf_hi = await probe(hi)
        if distance(pf_hi) <= tolerance:
            return result("converged", 0)
        if pf_lo is None or pf_hi is None:
            return result("no_trades", 0)
        if pf_lo < target:
            return result("below_target_at_lo", 0)
        if pf_hi > target:
            return result("above_target_at_hi", 0)
        low, high = lo, hi
        while iterations < max_iterations:
            mid = _q2((low + high) / 2)
            if mid in (low, high):  # the 0.01 bps grid is exhausted
                break
            iterations += 1
            pf_mid = await probe(mid)
            if pf_mid is None:
                return result("no_trades", iterations)
            if distance(pf_mid) <= tolerance:
                return result("converged", iterations)
            if pf_mid > target:  # still profitable: the breakeven is at a higher taker
                low = mid
            else:
                high = mid
        return result("max_iterations", iterations)
    except UnevaluableError:
        return result("unevaluable", iterations)


def breakeven_profile(base: CostProfile, taker_bps: Decimal) -> CostProfile:
    """``base`` with the taker replaced (maker clamped to it: the profile needs maker <= taker)."""
    return base.with_overrides(
        taker_fee_bps=taker_bps, maker_fee_bps=min(base.maker_fee_bps, taker_bps)
    )


# ---------------------------------------------------------------------------
# The matrix
# ---------------------------------------------------------------------------


def _window(bars: list[OHLCVBar], start: str, end: str) -> list[OHLCVBar]:
    lo = datetime.fromtimestamp(_day_start_ms(start) / 1000, tz=UTC)
    hi = datetime.fromtimestamp((_day_start_ms(end) + 86_400_000) / 1000, tz=UTC)
    return [b for b in bars if lo <= b.timestamp < hi]


async def run_matrix(
    series: Mapping[tuple[str, str], list[OHLCVBar]],
    *,
    fixture_sha256: str,
    baselines: Sequence[BaselineSpec] = BASELINES,
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
    breakeven: bool = True,
    breakeven_breaker: str = "on",
    column_a: Mapping[str, Mapping[str, Any]] | None = None,
    column_a_source: str | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Run every (baseline, symbol) row across the columns; the record is deterministic."""
    say = progress or (lambda _msg: None)
    columns = list(COLUMNS)
    rows: list[dict[str, Any]] = []
    for spec in baselines:
        for symbol in spec.symbols:
            key = (symbol, spec.timeframe)
            if key not in series:
                raise FixtureError(f"fixture has no series for {symbol} {spec.timeframe}")
            bars = _window(series[key], start, end)
            if not bars:
                raise FixtureError(f"no bars of {symbol} {spec.timeframe} inside {start}..{end}")
            row: dict[str, Any] = {
                "baseline": spec.name,
                "symbol": symbol,
                "timeframe": spec.timeframe,
                "bars": len(bars),
                "first_bar": bars[0].timestamp.isoformat(),
                "last_bar": bars[-1].timestamp.isoformat(),
                "params_source": spec.params_source,
                "strategy_params": dict(spec.strategy_params),
                "bracket": dict(spec.bracket),
                "cells": {},
            }
            a_cell = (column_a or {}).get(f"{spec.name}|{symbol}")
            if a_cell is not None:
                row["cells"]["A"] = {**a_cell, "source": "historical_db_run"}
            for col in columns:
                say(f"{spec.name} {symbol} column {col.key}")
                profile = get_cost_profile(col.profile_id)
                row["cells"][col.key] = await run_cell(
                    spec, symbol, bars, profile, breaker=col.breaker
                )
            if breakeven:
                row["breakeven"] = {}
                states = ("on", "off") if breakeven_breaker == "both" else (breakeven_breaker,)
                base = get_cost_profile(BREAKEVEN_BASE_PROFILE_ID)
                for state in states:
                    say(f"{spec.name} {symbol} breakeven (breaker {state})")

                    async def pf_at(
                        taker: Decimal,
                        _spec: BaselineSpec = spec,
                        _symbol: str = symbol,
                        _bars: list[OHLCVBar] = bars,
                        _on: bool = state == "on",
                        _base: CostProfile = base,
                    ) -> float | None:
                        try:
                            cell = await run_cell(
                                _spec, _symbol, _bars, breakeven_profile(_base, taker), breaker=_on
                            )
                        except (ExitConfigError, CostProfileError) as exc:
                            raise UnevaluableError from exc
                        pf = cell["profit_factor"]
                        if pf == "inf":
                            return math.inf
                        return None if pf is None else float(pf)

                    result = await bisect_breakeven(pf_at)
                    row["breakeven"][state] = result.to_json()
            rows.append(row)

    has_a = any("A" in r["cells"] for r in rows)
    provenance: dict[str, Any] = {}
    if column_a_source is not None and has_a:
        provenance["column_a_source"] = column_a_source
    return {
        **provenance,
        "schema": "wp21-rebaseline-1",
        "fixture_sha256": fixture_sha256,
        "window": {"start": start, "end": end},
        "seed": SEED,
        "initial_capital": str(INITIAL_CAPITAL),
        "matrix_order": (["A"] if has_a else []) + [c.key for c in columns],
        "attribution_chain": [k for k in ATTRIBUTION_CHAIN if k != "A" or has_a],
        "columns": {
            c.key: {
                "cost_profile_id": c.profile_id,
                "cost_profile_fingerprint": get_cost_profile(c.profile_id).fingerprint(),
                "circuit_breaker": c.breaker,
                "isolates": c.isolates,
            }
            for c in columns
        },
        "breakeven": {
            "base_profile": BREAKEVEN_BASE_PROFILE_ID,
            "range_bps": [str(BREAKEVEN_LO_BPS), str(BREAKEVEN_HI_BPS)],
            "target_pf": BREAKEVEN_PF_TARGET,
            "tolerance": BREAKEVEN_PF_TOLERANCE,
            "max_iterations": BREAKEVEN_MAX_ITERATIONS,
            "max_backtests_per_row": BREAKEVEN_MAX_BACKTESTS,
            "breaker": breakeven_breaker if breakeven else None,
        },
        "sharpe_label": "pre-WP2.5",
        "rows": rows,
    }


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _fmt_pf(v: float | str | None) -> str:
    if v is None:
        return "n/a"
    if v == "inf":
        return "inf"
    return f"{float(v):.3f}"


def _fmt_pct(v: float | None) -> str:
    return "n/a" if v is None else f"{v * 100:.2f}%"


def _fmt_money(v: str | None) -> str:
    return "n/a" if v is None else f"{Decimal(v):.2f}"


def _fmt_float(v: float | None, nd: int = 2) -> str:
    return "n/a" if v is None else f"{v:.{nd}f}"


_METRIC_ROWS: tuple[tuple[str, Callable[[dict[str, Any]], str]], ...] = (
    ("trades", lambda c: str(c["trades"])),
    ("stop-loss exits", lambda c: str(c["stop_loss_exits"])),
    ("stop-loss share", lambda c: _fmt_pct(c["stop_loss_share"])),
    ("win rate", lambda c: _fmt_pct(c["win_rate"])),
    ("profit factor", lambda c: _fmt_pf(c["profit_factor"])),
    ("CAGR", lambda c: _fmt_pct(c["cagr"])),
    ("max drawdown", lambda c: _fmt_pct(c["max_drawdown_pct"])),
    ("total fees", lambda c: _fmt_money(c["total_fees"])),
    ("total slippage cost", lambda c: _fmt_money(c["total_slippage_cost"])),
    ("turnover (x capital)", lambda c: _fmt_float(c["turnover_x"])),
    ("cost per trade (bps)", lambda c: _fmt_float(c["cost_per_trade_bps"], 1)),
    ("rejections: daily", lambda c: str(c["gate_rejections"]["daily"])),
    (
        "rejections: daily basis unavailable",
        lambda c: str(c["gate_rejections"]["daily_basis_unavailable"]),
    ),
    ("rejections: drawdown", lambda c: str(c["gate_rejections"]["drawdown"])),
    ("rejections: other", lambda c: str(c["gate_rejections"]["other"])),
    ("rejections: breaker", lambda c: str(c["gate_rejections"]["circuit_breaker"])),
    ("breaker transitions", lambda c: str(c["cb_transitions"])),
    ("Sharpe (pre-WP2.5)", lambda c: _fmt_float(c["sharpe_pre_wp25"], 3)),
)


def _render_cell(fmt: Callable[[dict[str, Any]], str], cell: dict[str, Any] | None) -> str:
    """One table cell; a metric the cell does not carry (a partial column A) is ``n/a``."""
    if cell is None:
        return "n/a"
    try:
        return fmt(cell)
    except (KeyError, TypeError, ValueError, ArithmeticError):
        return "n/a"


#: Static caveat of every rendered table (spec X-15; ASCII on purpose).
X15_NOTE = (
    "- X-15: bracket levels sit on the all-in (fee-inclusive) entry basis, so a higher taker "
    "moves every stop closer to the market; the B -> D -> E deltas include earlier stop-outs, "
    "not only fee drag."
)


def render_markdown(record: Mapping[str, Any]) -> str:
    """The markdown table of one matrix record (no timestamps: same record, same text)."""
    order: list[str] = list(record["matrix_order"])
    out: list[str] = ["# WP2.1c D13 re-baseline matrix", ""]
    out.append(f"- fixture sha256: `{record['fixture_sha256']}`")
    out.append(
        f"- window: {record['window']['start']} .. {record['window']['end']} (UTC, inclusive)"
    )
    out.append(
        f"- seed: {record['seed']}; single-symbol runs; initial capital {record['initial_capital']}"
    )
    out.append(
        f"- matrix order: {' -> '.join(order)}; "
        f"attribution chain: {' -> '.join(record['attribution_chain'])}"
        " (the parts are not claimed to be additive)"
    )
    out.append("- Sharpe is labelled pre-WP2.5 (not comparable with later numbers)")
    if "column_a_source" in record:
        out.append(f"- column A source: `{record['column_a_source']}`")
    out.append(X15_NOTE)
    be_meta = record.get("breakeven", {})
    if be_meta.get("breaker") is not None:
        out.append(
            f"- breakeven: taker {be_meta['range_bps'][0]}..{be_meta['range_bps'][1]} bps on "
            f"{be_meta['base_profile']}, PF {be_meta['target_pf']:.2f} +/- "
            f"{be_meta['tolerance']:.2f}, at most {be_meta['max_iterations']} bisection steps "
            f"after 2 bracket evaluations (at most {be_meta['max_backtests_per_row']} backtests "
            "per breakeven row and breaker state)"
        )
    out.append("")
    out.append("| Col | Cost profile | Fingerprint | Breaker | Isolates |")
    out.append("|---|---|---|---|---|")
    for key in order:
        meta = record["columns"].get(key)
        if meta is None:
            out.append(f"| {key} | historical DB run | - | off | - |")
            continue
        out.append(
            f"| {key} | {meta['cost_profile_id']} | {meta['cost_profile_fingerprint']} | "
            f"{'on' if meta['circuit_breaker'] else 'off'} | {meta['isolates']} |"
        )
    for row in record["rows"]:
        out.append("")
        out.append(f"## {row['baseline']} {row['symbol']} ({row['timeframe']}, {row['bars']} bars)")
        out.append("")
        out.append(
            f"Parameters: {row['params_source']}; "
            f"strategy {json.dumps(row['strategy_params'], sort_keys=True)}; "
            f"bracket {json.dumps(row['bracket'], sort_keys=True)}"
        )
        out.append("")
        out.append("| Metric | " + " | ".join(order) + " |")
        out.append("|---|" + "---|" * len(order))
        for label, fmt in _METRIC_ROWS:
            cells = [_render_cell(fmt, row["cells"].get(k)) for k in order]
            out.append(f"| {label} | " + " | ".join(cells) + " |")
        for state, be in sorted(row.get("breakeven", {}).items()):
            taker = be["breakeven_taker_bps"]
            out.append("")
            out.append(
                f"Breakeven taker (breaker {state}): "
                f"{'n/a' if taker is None else taker + ' bps'} "
                f"[{be['status']}, PF {_fmt_pf(be['pf_at_breakeven'])}, "
                f"{be['iterations']} bisection steps, "
                f"{len(be['evaluations'])} backtests]"
            )
    out.append("")
    return "\n".join(out)


def render_json(record: Mapping[str, Any]) -> str:
    return json.dumps(record, indent=2, sort_keys=True, default=str) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--fetch-fixture",
        metavar="OUT.csv.gz",
        type=Path,
        help="FETCH mode: write the gzip-CSV fixture + .sha256, then exit",
    )
    ap.add_argument(
        "--exchange",
        default=ALLOWED_EXCHANGES[0],
        choices=ALLOWED_EXCHANGES,
        help="ccxt exchange id for FETCH mode (public candles only; the only choice and the "
        "default: coinbaseexchange)",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="FETCH mode: overwrite an existing fixture, .sha256 and .meta.json "
        "(a symlink target is always refused)",
    )
    ap.add_argument("--fixture", type=Path, help="RUN mode: the gzip-CSV fixture")
    ap.add_argument(
        "--fixture-sha256",
        metavar="HEX",
        help="RUN mode: the pinned sha256 of the fixture bytes (a mismatch refuses to run)",
    )
    ap.add_argument("--start", default=DEFAULT_START, help="window start, UTC date (inclusive)")
    ap.add_argument("--end", default=DEFAULT_END, help="window end, UTC date (inclusive)")
    ap.add_argument(
        "--momentum-params",
        type=Path,
        metavar="PARAMS.json",
        help="momentum_breakout parameters (replaces the SY-21-07 fallback)",
    )
    ap.add_argument(
        "--column-a",
        type=Path,
        metavar="A.json",
        help="optional column A: JSON {'<baseline>|<symbol>': <cell metrics>} (historical run)",
    )
    ap.add_argument("--skip-breakeven", action="store_true", help="skip the breakeven bisection")
    ap.add_argument(
        "--breakeven-breaker",
        choices=("on", "off", "both"),
        default="on",
        help="circuit-breaker state(s) of the breakeven runs (default: on, the platform default)",
    )
    ap.add_argument(
        "--only",
        metavar="BASELINE[:SYMBOL]",
        action="append",
        default=[],
        help="restrict the rows (repeatable), e.g. grid_trading or grid_trading:BTC/EUR",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("reports/vp2-wp2.1"),
        help="where rebaseline-matrix.md / .json are written",
    )
    return ap


def _select(baselines: Sequence[BaselineSpec], only: Sequence[str]) -> list[BaselineSpec]:
    if not only:
        return list(baselines)
    out: list[BaselineSpec] = []
    for spec in baselines:
        symbols = tuple(
            s for s in spec.symbols if any(o == spec.name or o == f"{spec.name}:{s}" for o in only)
        )
        if symbols:
            out.append(
                BaselineSpec(
                    spec.name,
                    spec.strategy_cls,
                    spec.timeframe,
                    symbols,
                    spec.strategy_params,
                    spec.bracket,
                    spec.trailing_stop_pct,
                    spec.allow_pyramiding,
                    spec.params_source,
                )
            )
    if not out:
        raise FixtureError("--only matched no baseline row")
    return out


def main(argv: Sequence[str] | None = None) -> int:
    ap = build_parser()
    args = ap.parse_args(argv)

    if args.fetch_fixture is not None:
        if args.fixture is not None:
            ap.error("--fetch-fixture and --fixture are mutually exclusive")
        import ccxt  # lazy: only FETCH mode needs it

        try:
            digest = fetch_fixture(
                args.fetch_fixture,
                exchange_id=args.exchange,
                start=args.start,
                end=args.end,
                force=args.force,
            )
        except (FixtureError, ValueError) as exc:
            print(f"wp21-rebaseline: {exc}", file=sys.stderr)
            return 2
        except ccxt.BaseError as exc:
            # The exception type only (CF-36): a ccxt message embeds the request URL and the
            # response body.
            print(f"wp21-rebaseline: fetch failed ({type(exc).__name__})", file=sys.stderr)
            return 2
        print(f"fixture {args.fetch_fixture} sha256={digest}")
        return 0
    if args.force:
        ap.error("--force only applies to --fetch-fixture")

    if args.fixture is None or args.fixture_sha256 is None:
        ap.error("RUN mode needs both --fixture and --fixture-sha256 (or use --fetch-fixture)")
    try:
        series, digest = load_fixture(args.fixture, args.fixture_sha256)
        baselines = list(BASELINES)
        if args.momentum_params is not None:
            params_doc, params_sha = read_json_object(args.momentum_params)
            baselines[0] = momentum_spec_from_params(params_doc, source=f"file:sha256={params_sha}")
        column_a: dict[str, Mapping[str, Any]] | None = None
        column_a_source: str | None = None
        if args.column_a is not None:
            raw_a, a_sha = read_json_object(args.column_a)
            if not all(isinstance(v, dict) for v in raw_a.values()):
                raise FixtureError("--column-a must map '<baseline>|<symbol>' to metric objects")
            column_a = dict(raw_a)
            column_a_source = f"file:sha256={a_sha}"
        selected = _select(baselines, args.only)
        record = asyncio.run(
            run_matrix(
                series,
                fixture_sha256=digest,
                baselines=selected,
                start=args.start,
                end=args.end,
                breakeven=not args.skip_breakeven,
                breakeven_breaker=args.breakeven_breaker,
                column_a=column_a,
                column_a_source=column_a_source,
                progress=lambda m: print(f"  running {m}", file=sys.stderr, flush=True),
            )
        )
    except FixtureError as exc:
        print(f"wp21-rebaseline: {exc}", file=sys.stderr)
        return 2
    markdown = render_markdown(record)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "rebaseline-matrix.md").write_text(markdown, encoding="utf-8")
    (args.out_dir / "rebaseline-matrix.json").write_text(render_json(record), encoding="utf-8")
    print(markdown)
    return 0


def _quiet_engine_logs() -> None:
    """Engine logs go to stderr at WARNING+, so stdout carries only the table.

    Only the command line does this (a process-global change); ``main()`` called from a test
    or another script leaves the logging configuration alone.
    """
    import logging

    import structlog

    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(logging.WARNING),
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
    )


if __name__ == "__main__":
    _quiet_engine_logs()
    sys.exit(main())
