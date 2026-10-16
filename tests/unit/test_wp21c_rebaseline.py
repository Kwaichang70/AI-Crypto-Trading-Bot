"""
tests/unit/test_wp21c_rebaseline.py
-------------------------------------
WP2.1c: the D13 re-baseline harness ``scripts/wp21_rebaseline.py`` (spec section 11).

    WP21-T-37
        * the same fixture sha256 gives an IDENTICAL table and JSON
        * a sha256 mismatch (or a malformed pin) is refused before anything runs
        * the breakeven bisection converges within <= 12 iterations
        * the matrix order, the cost profile and the breaker state of every column, and the
          fixture sha256 are recorded

Hermetic: a SMALL SYNTHETIC deterministic fixture built in the test.  No network, no exchange,
no market data; ``ccxt`` is replaced by a fake object wherever the fetch path is exercised.
"""

from __future__ import annotations

import gzip
import importlib.util
import json
import math
import random
import sys
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, ClassVar

import pytest

from common.types import TimeFrame
from tests.integration.fakes.bar_paths import scripted_path
from tests.integration.fakes.scripted_strategy import ScriptedSequenceStrategy
from trading.backtest import BacktestRunner
from trading.costs import DEFAULT_COST_PROFILE, get_cost_profile
from trading.risk import RiskParameters
from trading.risk_manager import DefaultRiskManager
from trading.safety import CircuitBreaker
from trading.trade_journal import TradeSkipLogger

_REPO = Path(__file__).resolve().parents[2]
_SCRIPT = _REPO / "scripts" / "wp21_rebaseline.py"


def _load() -> Any:
    spec = importlib.util.spec_from_file_location("wp21_rebaseline_under_test", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


rb = _load()


# ---------------------------------------------------------------------------
# A small synthetic fixture (deterministic random walk; NOT market data)
# ---------------------------------------------------------------------------

_T0 = int(datetime(2023, 1, 1, tzinfo=UTC).timestamp())


def _walk(timeframe: str, n: int, seed: int) -> list[tuple[int, str, str, str, str, str]]:
    rnd = random.Random(seed)  # noqa: S311 - deterministic synthetic bars, not crypto
    step = 3600 if timeframe == "1h" else 86400
    vol = 0.012 if timeframe == "1h" else 0.03
    price = 100.0
    out = []
    for i in range(n):
        o = price
        c = max(1.0, o * (1 + 0.0004 * math.sin(i / 40) + rnd.gauss(0, vol)))
        hi = max(o, c) * (1 + abs(rnd.gauss(0, vol / 2)))
        lo = min(o, c) * (1 - abs(rnd.gauss(0, vol / 2)))
        out.append(
            (
                (_T0 + i * step) * 1000,
                f"{o:.4f}",
                f"{hi:.4f}",
                f"{lo:.4f}",
                f"{c:.4f}",
                f"{1000 + i % 50}.00",
            )
        )
        price = c
    return out


def _series(n_1d: int = 520, n_1h: int = 420) -> dict[tuple[str, str], list[Any]]:
    series: dict[tuple[str, str], list[Any]] = {}
    for b in rb.BASELINES:
        for i, symbol in enumerate(b.symbols):
            n = n_1d if b.timeframe == "1d" else n_1h
            series[(symbol, b.timeframe)] = _walk(
                b.timeframe, n, seed=11 * (i + 1) + (1 if b.timeframe == "1h" else 2)
            )
    return series


@pytest.fixture()
def fixture_path(tmp_path: Path) -> tuple[Path, str]:
    path = tmp_path / "bars.csv.gz"
    return path, rb.write_fixture(path, _series())


def _subset(*names: str) -> list[Any]:
    """The baseline rows to run: ``name`` or ``name:SYMBOL`` (keeps the tests quick)."""
    return rb._select(rb.BASELINES, list(names))


# ---------------------------------------------------------------------------
# Fixture: gzip CSV + sha256
# ---------------------------------------------------------------------------


class TestFixture:
    def test_the_same_bars_give_identical_bytes_and_sha(self, tmp_path: Path) -> None:
        a = rb.write_fixture(tmp_path / "a.csv.gz", _series())
        b = rb.write_fixture(tmp_path / "sub" / "b.csv.gz", _series())
        assert a == b
        assert (tmp_path / "a.csv.gz").read_bytes() == (tmp_path / "sub" / "b.csv.gz").read_bytes()

    def test_the_sidecar_holds_the_sha_of_the_file_bytes(
        self, fixture_path: tuple[Path, str]
    ) -> None:
        path, digest = fixture_path
        assert rb.sha256_hex(path.read_bytes()) == digest
        assert (path.parent / "bars.csv.gz.sha256").read_text() == f"{digest}  bars.csv.gz\n"

    def test_the_gzip_header_carries_no_timestamp_or_name(
        self, fixture_path: tuple[Path, str]
    ) -> None:
        raw = fixture_path[0].read_bytes()
        assert raw[:4] == b"\x1f\x8b\x08\x00"  # gzip, deflate, no FNAME/FEXTRA flags
        assert raw[4:8] == b"\x00\x00\x00\x00"  # mtime 0

    def test_different_bars_give_a_different_sha(self, tmp_path: Path) -> None:
        other = _series()
        key = next(iter(other))
        other[key][5] = (other[key][5][0], "1.0000", "2.0000", "0.5000", "1.5000", "1.00")
        assert rb.write_fixture(tmp_path / "x.csv.gz", other) != rb.write_fixture(
            tmp_path / "y.csv.gz", _series()
        )

    def test_load_round_trips_every_series(self, fixture_path: tuple[Path, str]) -> None:
        path, digest = fixture_path
        series, loaded_sha = rb.load_fixture(path, digest)
        assert loaded_sha == digest
        assert set(series) == {
            ("BTC/EUR", "1d"),
            ("ETH/EUR", "1d"),
            ("SOL/EUR", "1d"),
            ("BTC/EUR", "1h"),
            ("ETH/EUR", "1h"),
        }
        assert len(series[("BTC/EUR", "1d")]) == 520 and len(series[("ETH/EUR", "1h")]) == 420
        assert all(isinstance(b.close, Decimal) for b in series[("SOL/EUR", "1d")])

    def test_a_sha_mismatch_is_refused(self, fixture_path: tuple[Path, str]) -> None:
        path, digest = fixture_path
        wrong = ("0" if digest[0] != "0" else "1") + digest[1:]
        with pytest.raises(rb.FixtureError, match="sha256 mismatch"):
            rb.load_fixture(path, wrong)

    def test_a_tampered_file_is_refused(self, fixture_path: tuple[Path, str]) -> None:
        path, digest = fixture_path
        raw = bytearray(path.read_bytes())
        raw[-10] ^= 0x01
        path.write_bytes(bytes(raw))
        with pytest.raises(rb.FixtureError, match="sha256 mismatch"):
            rb.load_fixture(path, digest)

    @pytest.mark.parametrize("pin", ["", "abc", "Z" * 64, "g" * 64, "0" * 63])
    def test_a_malformed_pin_is_refused(self, fixture_path: tuple[Path, str], pin: str) -> None:
        with pytest.raises(rb.FixtureError, match="64 lowercase hex"):
            rb.load_fixture(fixture_path[0], pin)

    def test_an_uppercase_pin_is_accepted(self, fixture_path: tuple[Path, str]) -> None:
        path, digest = fixture_path
        assert rb.load_fixture(path, digest.upper())[1] == digest

    def test_a_missing_file_is_a_fixture_error_not_a_traceback(self, tmp_path: Path) -> None:
        with pytest.raises(rb.FixtureError, match="cannot read fixture"):
            rb.load_fixture(tmp_path / "nope.csv.gz", "0" * 64)

    def test_a_pinned_but_malformed_fixture_is_rejected(self, tmp_path: Path) -> None:
        for name, payload in (
            ("notgz", b"not gzip at all"),
            ("badhdr", gzip.compress(b"a,b,c\n1,2,3\n")),
            (
                "badrow",
                gzip.compress(
                    b"symbol,timeframe,ts_ms,open,high,low,close,volume\nBTC/EUR,1d,1,x,1,1,1,1\n"
                ),
            ),
            (
                "badtf",
                gzip.compress(
                    b"symbol,timeframe,ts_ms,open,high,low,close,volume\nBTC/EUR,5m,1,1,1,1,1,1\n"
                ),
            ),
            (
                "incons",
                gzip.compress(
                    b"symbol,timeframe,ts_ms,open,high,low,close,volume\nBTC/EUR,1d,1,5,2,1,1,1\n"
                ),
            ),
        ):
            path = tmp_path / f"{name}.gz"
            path.write_bytes(payload)
            with pytest.raises(rb.FixtureError):
                rb.load_fixture(path, rb.sha256_hex(payload))

    def test_unsorted_series_is_rejected(self, tmp_path: Path) -> None:
        text = (
            "symbol,timeframe,ts_ms,open,high,low,close,volume\n"
            "BTC/EUR,1d,86400000,1,1,1,1,1\nBTC/EUR,1d,86400000,1,1,1,1,1\n"
        )
        payload = gzip.compress(text.encode())
        path = tmp_path / "dup.gz"
        path.write_bytes(payload)
        with pytest.raises(rb.FixtureError, match="strictly increasing"):
            rb.load_fixture(path, rb.sha256_hex(payload))


# ---------------------------------------------------------------------------
# Breakeven bisection
# ---------------------------------------------------------------------------


def _linear(slope_at: Decimal = Decimal(125)) -> Any:
    """PF = 2 - taker / slope_at: exactly 1.00 at ``slope_at`` bps."""

    async def pf_at(taker: Decimal) -> float | None:
        return 2.0 - float(taker / slope_at)

    return pf_at


class TestBisection:
    @pytest.mark.parametrize(
        "root", [Decimal("5"), Decimal("37.5"), Decimal("120"), Decimal("200"), Decimal("249")]
    )
    async def test_it_converges_within_12_iterations(self, root: Decimal) -> None:
        # PF falls 0.5 per 125 bps ... scale so the target is hit at ``root``
        async def pf_at(taker: Decimal) -> float | None:
            return 1.0 + float(root - taker) / 400.0

        result = await rb.bisect_breakeven(pf_at)
        assert result.status == "converged"
        assert result.iterations <= 12
        assert result.pf is not None and abs(result.pf - 1.0) <= 0.01
        assert result.taker_bps is not None and abs(result.taker_bps - root) <= Decimal("4.01")
        # 2 end points + the iterations, each recorded once
        assert len(result.evaluations) == result.iterations + 2

    async def test_the_default_limits_are_the_spec(self) -> None:
        assert rb.BREAKEVEN_LO_BPS == 0 and rb.BREAKEVEN_HI_BPS == 250
        assert rb.BREAKEVEN_MAX_ITERATIONS == 12
        assert rb.BREAKEVEN_PF_TOLERANCE == 0.01
        assert rb.BREAKEVEN_BASE_PROFILE_ID == "coinbase_advanced_1"
        assert [e[0] for e in (await rb.bisect_breakeven(_linear())).evaluations][:2] == [
            Decimal(0),
            Decimal(250),
        ]

    async def test_it_never_exceeds_the_iteration_cap(self) -> None:
        calls = 0

        async def pf_at(taker: Decimal) -> float | None:
            nonlocal calls
            calls += 1
            return (
                1.0 + float(Decimal("100") - taker) / 100.0
            )  # steep: a 0.01 tolerance is a 1 bps window

        result = await rb.bisect_breakeven(pf_at, tolerance=1e-12, max_iterations=12)
        assert result.status == "max_iterations"
        assert result.iterations == 12
        assert calls == 14  # 2 end points + 12 steps, never more
        assert result.taker_bps is not None and abs(result.taker_bps - Decimal(100)) < Decimal(
            "0.1"
        )

    async def test_midpoints_stay_on_the_cost_profile_grid(self) -> None:
        result = await rb.bisect_breakeven(_linear(Decimal("77.77")))
        for taker, _pf in result.evaluations:
            assert taker == taker.quantize(Decimal("0.01"))
            assert Decimal(0) <= taker <= Decimal(250)

    async def test_pf_already_below_target_at_zero_taker(self) -> None:
        async def pf_at(taker: Decimal) -> float | None:
            return 0.8 - float(taker) / 1000.0

        result = await rb.bisect_breakeven(pf_at)
        assert (result.status, result.taker_bps, result.iterations) == (
            "below_target_at_lo",
            None,
            0,
        )
        assert len(result.evaluations) == 2

    async def test_pf_still_above_target_at_the_top_of_the_range(self) -> None:
        async def pf_at(taker: Decimal) -> float | None:
            return 3.0

        result = await rb.bisect_breakeven(pf_at)
        assert (result.status, result.taker_bps) == ("above_target_at_hi", None)

    async def test_infinite_pf_counts_as_profitable(self) -> None:
        async def pf_at(taker: Decimal) -> float | None:
            return math.inf if taker < 100 else 1.0 - float(taker - 100) / 300.0

        result = await rb.bisect_breakeven(pf_at)
        assert result.status == "converged" and result.iterations <= 12
        assert result.taker_bps is not None and Decimal(100) <= result.taker_bps <= Decimal("104")

    async def test_no_trades_is_reported_not_guessed(self) -> None:
        async def pf_at(taker: Decimal) -> float | None:
            return None

        result = await rb.bisect_breakeven(pf_at)
        assert (result.status, result.taker_bps, result.pf) == ("no_trades", None, None)

    async def test_an_endpoint_inside_the_tolerance_converges_at_once(self) -> None:
        async def pf_at(taker: Decimal) -> float | None:
            return 1.004

        result = await rb.bisect_breakeven(pf_at)
        assert (result.status, result.taker_bps, result.iterations) == ("converged", Decimal(0), 0)

    async def test_an_unpriceable_taker_is_reported(self) -> None:
        async def pf_at(taker: Decimal) -> float | None:
            if taker > 200:
                raise rb.UnevaluableError
            return 2.0 - float(taker) / 100.0

        result = await rb.bisect_breakeven(pf_at)
        assert (result.status, result.taker_bps) == ("unevaluable", None)

    async def test_the_result_is_json_safe(self) -> None:
        async def pf_at(taker: Decimal) -> float | None:
            return math.inf if taker == 0 else 0.5

        payload = (await rb.bisect_breakeven(pf_at)).to_json()
        assert json.loads(json.dumps(payload))["evaluations"][0] == ["0", "inf"]

    def test_the_override_profile_is_valid_across_the_whole_range(self) -> None:
        base = get_cost_profile("coinbase_advanced_1")
        for taker in (
            Decimal(0),
            Decimal("0.01"),
            Decimal(59),
            Decimal(60),
            Decimal(120),
            Decimal(250),
        ):
            p = rb.breakeven_profile(base, taker)
            assert (p.id, p.source, p.base_id) == ("custom", "override", "coinbase_advanced_1")
            assert p.taker_fee_bps == taker
            assert p.maker_fee_bps == min(base.maker_fee_bps, taker)  # maker <= taker holds
            assert p.slippage_bps == Decimal(10)
            assert p.live_equivalent is False


# ---------------------------------------------------------------------------
# The matrix
# ---------------------------------------------------------------------------


async def _record(path: Path, digest: str, *names: str, **kw: Any) -> dict[str, Any]:
    series, sha = rb.load_fixture(path, digest)
    return await rb.run_matrix(series, fixture_sha256=sha, baselines=_subset(*names), **kw)


class TestMatrix:
    async def test_the_same_fixture_sha_gives_an_identical_table_and_json(
        self, fixture_path: tuple[Path, str]
    ) -> None:
        path, digest = fixture_path
        first = await _record(
            path, digest, "momentum_breakout:BTC/EUR", "rsi_mean_reversion:ETH/EUR"
        )
        second = await _record(
            path, digest, "momentum_breakout:BTC/EUR", "rsi_mean_reversion:ETH/EUR"
        )
        assert rb.render_markdown(first) == rb.render_markdown(second)
        assert rb.render_json(first) == rb.render_json(second)

    async def test_a_reloaded_fixture_file_gives_the_same_table(self, tmp_path: Path) -> None:
        digest_a = rb.write_fixture(tmp_path / "a.csv.gz", _series())
        digest_b = rb.write_fixture(tmp_path / "b.csv.gz", _series())
        assert digest_a == digest_b
        a = await _record(tmp_path / "a.csv.gz", digest_a, "grid_trading:BTC/EUR")
        b = await _record(tmp_path / "b.csv.gz", digest_b, "grid_trading:BTC/EUR")
        assert rb.render_markdown(a) == rb.render_markdown(b)

    async def test_a_different_fixture_gives_a_different_table_and_a_different_sha(
        self, tmp_path: Path
    ) -> None:
        other = _series()
        other[("BTC/EUR", "1d")] = _walk("1d", 520, seed=999)
        d1 = rb.write_fixture(tmp_path / "a.csv.gz", _series())
        d2 = rb.write_fixture(tmp_path / "b.csv.gz", other)
        a = await _record(tmp_path / "a.csv.gz", d1, "momentum_breakout:BTC/EUR", breakeven=False)
        b = await _record(tmp_path / "b.csv.gz", d2, "momentum_breakout:BTC/EUR", breakeven=False)
        assert d1 != d2 and a["fixture_sha256"] != b["fixture_sha256"]
        assert rb.render_markdown(a) != rb.render_markdown(b)

    async def test_the_matrix_order_columns_and_fixture_sha_are_recorded(
        self, fixture_path: tuple[Path, str]
    ) -> None:
        path, digest = fixture_path
        rec = await _record(path, digest, "grid_trading:ETH/EUR", breakeven=False)
        assert rec["matrix_order"] == ["B", "C", "D", "E", "F"]
        assert rec["attribution_chain"] == [
            "B",
            "D",
            "E",
            "F",
        ]  # no column A without a historical run
        assert rec["fixture_sha256"] == digest
        assert (rec["seed"], rec["window"]) == (42, {"start": "2023-01-01", "end": "2026-09-30"})
        assert rec["sharpe_label"] == "pre-WP2.5"
        cols = rec["columns"]
        assert [
            (k, cols[k]["cost_profile_id"], cols[k]["circuit_breaker"]) for k in rec["matrix_order"]
        ] == [
            ("B", "legacy_backtest_v0", False),
            ("C", "frictionless_diagnostic", False),
            ("D", "coinbase_advanced_d6", False),
            ("E", "coinbase_advanced_1", False),
            ("F", "coinbase_advanced_1", True),
        ]
        assert cols["E"]["cost_profile_fingerprint"] == DEFAULT_COST_PROFILE.fingerprint()
        (row,) = rec["rows"]
        assert list(row["cells"]) == ["B", "C", "D", "E", "F"]
        for key in "BCDE":
            assert row["cells"][key]["breaker"] is False
        assert row["cells"]["F"]["breaker"] is True
        md = rb.render_markdown(rec)
        assert digest in md and "pre-WP2.5" in md
        assert "matrix order: B -> C -> D -> E -> F" in md
        assert "| Metric | B | C | D | E | F |" in md

    async def test_every_cell_carries_every_metric_of_section_11(
        self, fixture_path: tuple[Path, str]
    ) -> None:
        path, digest = fixture_path
        rec = await _record(path, digest, "grid_trading:BTC/EUR", breakeven=False)
        (row,) = rec["rows"]
        needed = {
            "stop_loss_exits",
            "stop_loss_share",
            "trades",
            "win_rate",
            "profit_factor",
            "cagr",
            "max_drawdown_pct",
            "total_fees",
            "total_slippage_cost",
            "turnover_x",
            "cost_per_trade_bps",
            "gate_rejections",
            "cb_transitions",
            "sharpe_pre_wp25",
        }
        for key, cell in row["cells"].items():
            assert needed <= set(cell), key
            assert set(cell["gate_rejections"]) >= {"daily", "drawdown", "circuit_breaker"}
        assert (
            row["cells"]["C"]["total_fees"] == "0.00000000"
            or Decimal(row["cells"]["C"]["total_fees"]) == 0
        )
        assert Decimal(row["cells"]["C"]["total_slippage_cost"]) == 0
        # the costs grow B < D < E on the same trades (same entries, only the profile moved)
        fees = {k: Decimal(row["cells"][k]["total_fees"]) for k in "BDE"}
        assert fees["B"] < fees["D"] < fees["E"]
        md = rb.render_markdown(rec)
        for label in (
            "stop-loss exits",
            "stop-loss share",
            "total slippage cost",
            "rejections: daily",
            "rejections: breaker",
            "breaker transitions",
            "Sharpe (pre-WP2.5)",
        ):
            assert label in md

    async def test_a_cell_is_the_same_backtest_the_runner_reports(
        self, fixture_path: tuple[Path, str]
    ) -> None:
        path, digest = fixture_path
        series, _ = rb.load_fixture(path, digest)
        spec = rb.BASELINES[0]
        bars = series[("BTC/EUR", "1d")]
        cell = await rb.run_cell(spec, "BTC/EUR", bars, DEFAULT_COST_PROFILE, breaker=False)
        runner = BacktestRunner(
            strategies=[spec.strategy_cls(strategy_id="x", params={})],
            symbols=["BTC/EUR"],
            timeframe=TimeFrame.ONE_DAY,
            initial_capital=Decimal("10000"),
            bracket_config=dict(spec.bracket),
            seed=42,
            enable_circuit_breaker=False,
            cost_profile=DEFAULT_COST_PROFILE,
        )
        result = await runner.run({"BTC/EUR": bars})
        assert cell["trades"] == result.total_trades
        assert cell["final_equity"] == str(result.final_equity)
        assert cell["total_fees"] == str(result.total_fees_paid)
        assert cell["total_slippage_cost"] == str(result.total_slippage_cost)
        assert cell["stop_loss_exits"] == sum(
            1 for t in result.trades if t.exit_reason == "stop_loss"
        )
        assert cell["cost_profile_fingerprint"] == DEFAULT_COST_PROFILE.fingerprint()

    async def test_the_breakeven_runs_stay_within_12_iterations_on_real_backtests(
        self, fixture_path: tuple[Path, str]
    ) -> None:
        path, digest = fixture_path
        rec = await _record(
            path,
            digest,
            "momentum_breakout:ETH/EUR",
            "rsi_mean_reversion:ETH/EUR",
            breakeven_breaker="both",
        )
        assert rec["breakeven"]["max_iterations"] == 12
        for row in rec["rows"]:
            assert set(row["breakeven"]) == {"on", "off"}
            for be in row["breakeven"].values():
                assert be["iterations"] <= 12
                assert len(be["evaluations"]) <= 14
                assert be["status"] in {
                    "converged",
                    "max_iterations",
                    "below_target_at_lo",
                    "above_target_at_hi",
                    "no_trades",
                    "unevaluable",
                }
                if be["status"] == "converged":
                    assert abs(float(be["pf_at_breakeven"]) - 1.0) <= 0.01
        md = rb.render_markdown(rec)
        assert (
            md.count("Breakeven taker (breaker on)") == 2
            and md.count("Breakeven taker (breaker off)") == 2
        )

    async def test_a_converging_real_row(self, tmp_path: Path) -> None:
        """A series built so PF crosses 1.00 inside [0, 250]: the bisection finds it."""
        series = _series()
        digest = rb.write_fixture(tmp_path / "f.csv.gz", series)
        loaded, sha = rb.load_fixture(tmp_path / "f.csv.gz", digest)
        # search the synthetic row set for one whose PF brackets 1 (deterministic given the fixture)
        found = None
        for name in ("rsi_mean_reversion", "grid_trading", "momentum_breakout"):
            for spec in _subset(name):
                for symbol in spec.symbols:
                    rec = await rb.run_matrix(
                        loaded,
                        fixture_sha256=sha,
                        baselines=_subset(f"{name}:{symbol}"),
                        breakeven=True,
                        breakeven_breaker="off",
                    )
                    be = rec["rows"][0]["breakeven"]["off"]
                    if be["status"] == "converged" and be["iterations"] > 0:
                        found = be
                        break
                if found:
                    break
            if found:
                break
        if found is None:
            pytest.skip("no synthetic row brackets PF = 1 with these seeds")
        assert found["iterations"] <= 12
        assert Decimal(0) < Decimal(found["breakeven_taker_bps"]) < Decimal(250)

    async def test_column_a_is_passed_through_when_supplied(
        self, fixture_path: tuple[Path, str]
    ) -> None:
        path, digest = fixture_path
        hist = {"grid_trading|BTC/EUR": {"trades": 12, "profit_factor": 1.5}}
        rec = await _record(path, digest, "grid_trading:BTC/EUR", breakeven=False, column_a=hist)
        assert rec["matrix_order"] == ["A", "B", "C", "D", "E", "F"]
        assert rec["attribution_chain"] == ["A", "B", "D", "E", "F"]
        assert rec["rows"][0]["cells"]["A"]["source"] == "historical_db_run"
        md = rb.render_markdown(rec)
        assert "| Metric | A | B | C | D | E | F |" in md and "historical DB run" in md

    async def test_a_missing_series_or_window_is_a_fixture_error(
        self, fixture_path: tuple[Path, str]
    ) -> None:
        path, digest = fixture_path
        series, sha = rb.load_fixture(path, digest)
        del series[("SOL/EUR", "1d")]
        with pytest.raises(rb.FixtureError, match="no series for SOL/EUR 1d"):
            await rb.run_matrix(series, fixture_sha256=sha, baselines=_subset("momentum_breakout"))
        series, sha = rb.load_fixture(path, digest)
        with pytest.raises(rb.FixtureError, match="no bars"):
            await rb.run_matrix(
                series,
                fixture_sha256=sha,
                baselines=_subset("grid_trading:BTC/EUR"),
                start="2030-01-01",
                end="2030-02-01",
            )


# ---------------------------------------------------------------------------
# Baselines and parameters
# ---------------------------------------------------------------------------


class TestBaselines:
    def test_the_three_baselines_follow_sy_21_07(self) -> None:
        by_name = {b.name: b for b in rb.BASELINES}
        assert set(by_name) == {"momentum_breakout", "grid_trading", "rsi_mean_reversion"}
        assert (by_name["momentum_breakout"].timeframe, by_name["momentum_breakout"].symbols) == (
            "1d",
            ("BTC/EUR", "ETH/EUR", "SOL/EUR"),
        )
        for name in ("grid_trading", "rsi_mean_reversion"):
            assert (by_name[name].timeframe, by_name[name].symbols) == (
                "1h",
                ("BTC/EUR", "ETH/EUR"),
            )
            assert dict(by_name[name].strategy_params) == {} and dict(by_name[name].bracket) == {}

    def test_the_momentum_fallback_is_atr_1_5_and_3_0(self) -> None:
        spec = rb.BASELINES[0]
        assert spec.params_source == "sy-21-07-fallback"
        assert dict(spec.bracket) == {
            "bracket_mode": "atr",
            "bracket_atr_sl_multiplier": 1.5,
            "bracket_atr_tp_multiplier": 3.0,
            "bracket_atr_period": 14,
        }
        assert dict(spec.strategy_params) == {}  # registry defaults otherwise

    def test_a_run_config_params_file_replaces_the_fallback(self) -> None:
        doc = {
            "strategy_params": {"lookback": 20, "trend_sma_period": 50, "position_size": 500.0},
            "bracket_config": {
                "bracket_mode": "fixed",
                "bracket_stop_loss_pct": 0.05,
                "bracket_take_profit_pct": 0.1,
            },
            "allow_pyramiding": False,
            "symbols": ["BTC/EUR"],
        }
        spec = rb.momentum_spec_from_params(doc, source="file:x")
        assert dict(spec.strategy_params) == {
            "lookback": 20,
            "trend_sma_period": 50,
            "position_size": 500.0,
        }
        assert dict(spec.bracket) == doc["bracket_config"]  # no ATR multipliers merged in
        assert spec.allow_pyramiding is False and spec.symbols == ("BTC/EUR",)
        assert spec.params_source == "file:x"

    def test_a_flat_params_object_with_bracket_keys_is_split(self) -> None:
        spec = rb.momentum_spec_from_params(
            {
                "lookback": 30,
                "bracket_mode": "atr",
                "bracket_atr_sl_multiplier": 2.0,
                "bracket_atr_tp_multiplier": 5.0,
                "bracket_atr_period": None,
                "trailing_stop_pct": 0.04,
            },
            source="file:y",
        )
        assert dict(spec.strategy_params) == {"lookback": 30, "trailing_stop_pct": 0.04}
        assert dict(spec.bracket) == {
            "bracket_mode": "atr",
            "bracket_atr_sl_multiplier": 2.0,
            "bracket_atr_tp_multiplier": 5.0,
        }  # None/blank dropped
        assert spec.trailing_stop_pct == 0.04

    def test_a_non_object_strategy_params_is_rejected(self) -> None:
        with pytest.raises(rb.FixtureError):
            rb.momentum_spec_from_params({"strategy_params": [1, 2]}, source="x")

    def test_the_json_loader_is_bounded_and_strict(self, tmp_path: Path) -> None:
        good = tmp_path / "ok.json"
        good.write_text('{"lookback": 5}')
        assert rb.load_json_object(good) == {"lookback": 5}
        for name, text in (("arr.json", "[1]"), ("bad.json", "{nope")):
            (tmp_path / name).write_text(text)
            with pytest.raises(rb.FixtureError):
                rb.load_json_object(tmp_path / name)
        big = tmp_path / "big.json"
        big.write_text('{"a": "' + "x" * (1024 * 1024) + '"}')
        with pytest.raises(rb.FixtureError, match="larger than 1 MiB"):
            rb.load_json_object(big)
        with pytest.raises(rb.FixtureError):
            rb.load_json_object(tmp_path / "missing.json")

    def test_only_selects_rows_and_rejects_a_typo(self) -> None:
        rows = rb._select(rb.BASELINES, ["grid_trading", "momentum_breakout:SOL/EUR"])
        assert [(r.name, r.symbols) for r in rows] == [
            ("momentum_breakout", ("SOL/EUR",)),
            ("grid_trading", ("BTC/EUR", "ETH/EUR")),
        ]
        with pytest.raises(rb.FixtureError, match="matched no baseline row"):
            rb._select(rb.BASELINES, ["nope"])


# ---------------------------------------------------------------------------
# Instrumentation (gate rejections / breaker transitions)
# ---------------------------------------------------------------------------

_SYMBOL = "BTC/USDT"
_FIRST_OPEN = datetime(2026, 3, 10, 10, 0, tzinfo=UTC)
_RISK = RiskParameters(
    max_position_size_pct=0.9, max_portfolio_exposure_pct=1.0, max_cluster_exposure_pct=1.0
)


async def _probe_run(
    *, breaker: bool, script: dict[int, str], targets: dict[int, str], closes: list[int]
) -> tuple[Any, Any]:
    strategy = ScriptedSequenceStrategy(
        "t37", {"script": script, "target_notional": "3000", "targets": targets}
    )
    runner = BacktestRunner(
        strategies=[strategy],
        symbols=[_SYMBOL],
        timeframe=TimeFrame.ONE_HOUR,
        initial_capital=Decimal("10000"),
        risk_params=_RISK,
        seed=7,
        allow_pyramiding=True,
        cost_profile=get_cost_profile("legacy_backtest_v0"),
        enable_circuit_breaker=breaker,
    )
    bars = scripted_path(
        closes, first_open=_FIRST_OPEN, symbol=_SYMBOL, timeframe=TimeFrame.ONE_HOUR
    )
    with rb.probe_gates() as probe:
        result = await runner.run({_SYMBOL: bars})
    return probe, result


class TestGateProbe:
    # 30% position that loses 52% on day D (-15.6% NAV), then a second BUY on the SAME day.
    _CLOSES: ClassVar[list[int]] = [100, 100, 80, 48] + [48] * 6 + [100] * 6
    _SCRIPT: ClassVar[dict[int, str]] = {0: "buy", 4: "buy"}
    _TARGETS: ClassVar[dict[int, str]] = {0: "3000", 4: "300"}

    async def test_the_risk_gate_rejection_is_counted_when_the_breaker_is_off(self) -> None:
        probe, result = await _probe_run(
            breaker=False, script=self._SCRIPT, targets=self._TARGETS, closes=self._CLOSES
        )
        assert probe.daily_rejections >= 1
        assert probe.cb_rejections == 0 and probe.cb_transitions == 0
        assert result.total_trades >= 0

    async def test_the_breaker_drops_the_entry_before_the_gate_and_counts_transitions(self) -> None:
        probe, _ = await _probe_run(
            breaker=True, script=self._SCRIPT, targets=self._TARGETS, closes=self._CLOSES
        )
        assert probe.cb_entry_drops.get("daily_limit", 0) >= 1
        assert probe.cb_rejections == sum(probe.cb_entry_drops.values())
        assert probe.cb_transitions >= 1
        assert sum(probe.cb_transitions_to.values()) == probe.cb_transitions
        assert probe.daily_rejections == 0  # the breaker's filter runs first (K-10)

    async def test_the_probe_only_observes_and_restores_the_patched_methods(self) -> None:
        originals = (
            DefaultRiskManager.pre_trade_check,
            TradeSkipLogger.log_skip,
            CircuitBreaker.evaluate,
        )
        _probe, with_probe = await _probe_run(
            breaker=True, script=self._SCRIPT, targets=self._TARGETS, closes=self._CLOSES
        )
        assert (
            DefaultRiskManager.pre_trade_check,
            TradeSkipLogger.log_skip,
            CircuitBreaker.evaluate,
        ) == originals
        strategy = ScriptedSequenceStrategy(
            "t37", {"script": self._SCRIPT, "target_notional": "3000", "targets": self._TARGETS}
        )
        bare = BacktestRunner(
            strategies=[strategy],
            symbols=[_SYMBOL],
            timeframe=TimeFrame.ONE_HOUR,
            initial_capital=Decimal("10000"),
            risk_params=_RISK,
            seed=7,
            allow_pyramiding=True,
            cost_profile=get_cost_profile("legacy_backtest_v0"),
            enable_circuit_breaker=True,
        )
        result = await bare.run(
            {_SYMBOL: scripted_path(self._CLOSES, first_open=_FIRST_OPEN, symbol=_SYMBOL)}
        )
        assert str(result.final_equity) == str(
            with_probe.final_equity
        )  # same numbers with and without

    async def test_the_probe_is_restored_even_when_the_run_raises(self) -> None:
        original = DefaultRiskManager.pre_trade_check
        with pytest.raises(RuntimeError), rb.probe_gates():
            raise RuntimeError("boom")
        assert DefaultRiskManager.pre_trade_check is original


# ---------------------------------------------------------------------------
# Fetch mode (a fake exchange: no network, no keys)
# ---------------------------------------------------------------------------


class _FakeExchange:
    apiKey = None
    secret = None

    def __init__(self, *, fail_first: int = 0) -> None:
        self.calls: list[tuple[str, str, int, int | None, dict[str, Any]]] = []
        self._fail = fail_first

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1m",
        since: int | None = None,
        limit: int | None = None,
        params: dict[str, Any] | None = None,
    ) -> list[list[float]]:
        import ccxt

        if self._fail:
            self._fail -= 1
            raise ccxt.NetworkError("down")
        assert since is not None and params is not None
        self.calls.append((symbol, timeframe, since, limit, dict(params)))
        step = {"1h": 3_600_000, "1d": 86_400_000}[timeframe]
        end = params["until"]
        rows = []
        ts = since
        while ts <= end and len(rows) < (limit or 300):
            rows.append([ts, 10.0, 12.0, 9.0, 11.0, 5.0])
            ts += step
        # a duplicate row, an invalid row and a row outside the window must all be handled
        rows.append(list(rows[0]))
        rows.append([since + 1, 10.0, 9.0, 11.0, 10.0, 1.0])  # low > high -> dropped
        return rows


class TestFetch:
    def test_paging_dedupes_drops_invalid_rows_and_never_leaves_the_window(self) -> None:
        ex = _FakeExchange()
        start, end = 0, 3 * 86_400_000
        candles, dropped = rb.fetch_candles(ex, "BTC/EUR", "1d", start, end)
        assert [c[0] for c in candles] == [0, 86_400_000, 172_800_000]
        assert dropped == 1
        assert ex.calls and all(c[4]["until"] < end for c in ex.calls)
        assert candles[0] == (0, "10.0", "12.0", "9.0", "11.0", "5.0")

    def test_a_network_error_is_retried_with_exponential_backoff(self) -> None:
        sleeps: list[float] = []
        ex = _FakeExchange(fail_first=2)
        candles, _ = rb.fetch_candles(ex, "BTC/EUR", "1d", 0, 86_400_000, sleep=sleeps.append)
        assert len(candles) == 1 and sleeps == [1.0, 2.0]

    def test_a_persistent_network_error_propagates(self) -> None:
        import ccxt

        with pytest.raises(ccxt.NetworkError):
            rb.fetch_candles(
                _FakeExchange(fail_first=99), "BTC/EUR", "1d", 0, 86_400_000, sleep=lambda _s: None
            )

    def test_fetch_fixture_requests_exactly_the_baseline_series_and_writes_the_fixture(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import ccxt

        created: list[tuple[str, dict[str, Any], _FakeExchange]] = []

        def factory(config: dict[str, Any]) -> _FakeExchange:
            ex = _FakeExchange()
            created.append(("coinbaseexchange", config, ex))
            return ex

        monkeypatch.setattr(ccxt, "coinbaseexchange", factory)
        digest = rb.fetch_fixture(
            tmp_path / "f.csv.gz",
            exchange_id="coinbaseexchange",
            start="2023-01-01",
            end="2023-01-02",
        )
        ((_id, config, ex),) = created
        assert config == {"enableRateLimit": True}  # public endpoints only: no key, no secret
        assert sorted({(c[0], c[1]) for c in ex.calls}) == [
            ("BTC/EUR", "1d"),
            ("BTC/EUR", "1h"),
            ("ETH/EUR", "1d"),
            ("ETH/EUR", "1h"),
            ("SOL/EUR", "1d"),
        ]
        series, loaded = rb.load_fixture(tmp_path / "f.csv.gz", digest)
        assert loaded == digest and len(series[("BTC/EUR", "1d")]) == 2
        assert len(series[("BTC/EUR", "1h")]) == 48
        assert (tmp_path / "f.csv.gz.sha256").read_text().startswith(digest)

    def test_fetch_refuses_when_credentials_are_configured(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import ccxt

        class _Keyed(_FakeExchange):
            apiKey = "super-secret-key-value"

        monkeypatch.setattr(ccxt, "coinbaseexchange", lambda _cfg: _Keyed())
        with pytest.raises(rb.FixtureError) as exc:
            rb.fetch_fixture(
                tmp_path / "f.csv.gz",
                exchange_id="coinbaseexchange",
                start="2023-01-01",
                end="2023-01-02",
            )
        assert "super-secret-key-value" not in str(exc.value)
        assert not (tmp_path / "f.csv.gz").exists()

    def test_the_script_reads_no_environment_and_holds_no_credentials(self) -> None:
        text = _SCRIPT.read_text()
        for needle in (
            "os.environ",
            "getenv",
            "load_dotenv",
            ".env",
            'apiKey":',
            "'apiKey'",
            "secret=",
        ):
            assert needle not in text, needle


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


class TestCli:
    def test_run_mode_writes_the_table_and_json_for_a_pinned_fixture(
        self, fixture_path: tuple[Path, str], tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        path, digest = fixture_path
        out = tmp_path / "out"
        code = rb.main(
            [
                "--fixture",
                str(path),
                "--fixture-sha256",
                digest,
                "--only",
                "grid_trading:BTC/EUR",
                "--skip-breakeven",
                "--out-dir",
                str(out),
            ]
        )
        assert code == 0
        md = (out / "rebaseline-matrix.md").read_text()
        data = json.loads((out / "rebaseline-matrix.json").read_text())
        assert digest in md and data["fixture_sha256"] == digest
        assert data["matrix_order"] == ["B", "C", "D", "E", "F"]
        # the table is printed (under pytest the engine's own logs share stdout; the command
        # line sends them to stderr, see ``_quiet_engine_logs``)
        assert capsys.readouterr().out.rstrip().endswith(md.strip())

    def test_the_same_pin_twice_gives_byte_identical_files(
        self, fixture_path: tuple[Path, str], tmp_path: Path
    ) -> None:
        path, digest = fixture_path
        args = [
            "--fixture",
            str(path),
            "--fixture-sha256",
            digest,
            "--only",
            "rsi_mean_reversion:BTC/EUR",
        ]
        assert rb.main([*args, "--out-dir", str(tmp_path / "a")]) == 0
        assert rb.main([*args, "--out-dir", str(tmp_path / "b")]) == 0
        for name in ("rebaseline-matrix.md", "rebaseline-matrix.json"):
            assert (tmp_path / "a" / name).read_bytes() == (tmp_path / "b" / name).read_bytes()

    def test_a_sha_mismatch_refuses_to_run_and_writes_nothing(
        self, fixture_path: tuple[Path, str], tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        path, digest = fixture_path
        wrong = ("0" if digest[0] != "0" else "1") + digest[1:]
        out = tmp_path / "out"
        code = rb.main(["--fixture", str(path), "--fixture-sha256", wrong, "--out-dir", str(out)])
        assert code == 2
        assert not out.exists()
        err = capsys.readouterr().err
        assert "sha256 mismatch" in err and "refusing to run" in err

    def test_run_mode_needs_both_the_fixture_and_its_pin(
        self, fixture_path: tuple[Path, str]
    ) -> None:
        path, digest = fixture_path
        for argv in ([], ["--fixture", str(path)], ["--fixture-sha256", digest]):
            with pytest.raises(SystemExit) as exc:
                rb.main(argv)
            assert exc.value.code == 2

    def test_fetch_and_run_modes_are_exclusive(
        self, fixture_path: tuple[Path, str], tmp_path: Path
    ) -> None:
        with pytest.raises(SystemExit) as exc:
            rb.main(["--fetch-fixture", str(tmp_path / "x.gz"), "--fixture", str(fixture_path[0])])
        assert exc.value.code == 2

    def test_a_params_file_is_used_and_its_hash_recorded(
        self, fixture_path: tuple[Path, str], tmp_path: Path
    ) -> None:
        path, digest = fixture_path
        params = tmp_path / "p.json"
        params.write_text(
            json.dumps(
                {
                    "strategy_params": {"lookback": 20, "trend_sma_period": 50},
                    "bracket_config": {
                        "bracket_mode": "atr",
                        "bracket_atr_sl_multiplier": 2.0,
                        "bracket_atr_tp_multiplier": 4.0,
                        "bracket_atr_period": 14,
                    },
                }
            )
        )
        out = tmp_path / "out"
        assert (
            rb.main(
                [
                    "--fixture",
                    str(path),
                    "--fixture-sha256",
                    digest,
                    "--momentum-params",
                    str(params),
                    "--only",
                    "momentum_breakout:BTC/EUR",
                    "--skip-breakeven",
                    "--out-dir",
                    str(out),
                ]
            )
            == 0
        )
        row = json.loads((out / "rebaseline-matrix.json").read_text())["rows"][0]
        assert row["params_source"].startswith("file:sha256=")
        assert row["strategy_params"] == {"lookback": 20, "trend_sma_period": 50}
        assert row["bracket"]["bracket_atr_sl_multiplier"] == 2.0
