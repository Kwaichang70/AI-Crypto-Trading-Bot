"""WP2.5 / P3: the ``scripts/backfill_metrics_v2.py`` marker guard (SY-25-18, WP25-T-31).

A run whose ``config.engine_semantics.metrics`` is a known version already carries the per-bar
numbers the runner wrote.  The backfill's PSR recompute (sample-stdev moments, one return per
snapshot) would overwrite them, so such a row is skipped BEFORE any recompute and without any
write -- not even the ``metrics_v2_backfilled`` sentinel -- and is counted as ``skipped_wp25``.
Legacy rows are processed exactly as before (CR-08: they also keep the 1h annualisation fallback).

Hermetic: the session factory is a mock (the pattern of
``tests/integration/test_backfill_metrics_v2.py``); no Postgres.
"""

from __future__ import annotations

import ast
import copy
import sys
import uuid
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from structlog.testing import capture_logs

_REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (_REPO_ROOT / "packages", _REPO_ROOT / "apps", _REPO_ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import backfill_metrics_v2 as bm  # type: ignore[import]  # noqa: E402

_NOW = datetime(2026, 3, 10, 12, 0, tzinfo=UTC)
_SENTINEL = "WP25SENTINEL-BACKFILL-MARKER"

_ID_WP25 = uuid.UUID("aaaaaaaa-0000-0000-0000-000000000025")
_ID_LEGACY = uuid.UUID("bbbbbbbb-0000-0000-0000-000000000025")
_ID_WP25_B = uuid.UUID("cccccccc-0000-0000-0000-000000000025")


def _metrics_block() -> dict[str, Any]:
    return {
        "total_trades": 5,
        "winning_trades": 3,
        "losing_trades": 2,
        "win_rate": 0.6,
        "profit_factor": 1.7,
        "profit_factor_is_infinite": False,
        # the per-bar numbers the runner wrote; the backfill must never touch them
        "psr": 0.9731,
        "confidence_flag": "high",
        "n_observations": 120,
        "quote_currency": "USD",
        "exposure_pct_per_symbol": {"BTC/USDT": 0.4},
        "open_positions_mtm": [],
    }


def _config(marker: Any = "wp25-1", *, with_semantics: bool = True) -> dict[str, Any]:
    cfg: dict[str, Any] = {
        "strategy_name": "ma_crossover",
        "symbols": ["BTC/USDT"],
        "timeframe": "1h",
        "mode": "backtest",
        "initial_capital": "10000.00",
        "backtest_metrics": _metrics_block(),
    }
    if with_semantics:
        cfg["engine_semantics"] = {"schema": 1, "metrics": marker}
    return cfg


def _snapshots(n: int = 50) -> list[Any]:
    equity = Decimal("10000")
    out = []
    for i in range(n):
        equity = equity * (Decimal("1.001") if i % 2 == 0 else Decimal("0.9995"))
        out.append(
            SimpleNamespace(
                bar_index=i,
                timestamp=_NOW,
                equity=equity,
                cash=equity,
                unrealised_pnl=Decimal(0),
                realised_pnl=Decimal(0),
                drawdown_pct=Decimal(0),
            )
        )
    return out


class _Row(SimpleNamespace):
    """A RunORM stand-in that records every attribute write made after construction."""

    def __init__(self, **kw: Any) -> None:
        object.__setattr__(self, "_writes", [])
        super().__init__(**kw)
        object.__setattr__(self, "_armed", True)

    def __setattr__(self, name: str, value: Any) -> None:
        if getattr(self, "_armed", False):
            self._writes.append(name)
        super().__setattr__(name, value)


def _row(run_id: uuid.UUID, config: dict[str, Any], *, n_closed: int | None = 5) -> _Row:
    return _Row(
        id=run_id,
        run_mode="backtest",
        status="stopped",
        config=config,
        created_at=_NOW,
        n_closed_trades=n_closed,
        metrics_v2_backfilled=False,
        equity_snapshots=_snapshots(),
    )


def _factory(candidate_ids: list[uuid.UUID], runs: list[Any]) -> MagicMock:
    state = {"n": 0}

    def _new_cm() -> AsyncMock:
        cm = AsyncMock()
        n = state["n"]
        state["n"] += 1
        session = AsyncMock()
        result = MagicMock()
        result.scalars.return_value.all.return_value = candidate_ids if n == 0 else runs
        session.execute = AsyncMock(return_value=result)
        session.commit = AsyncMock()
        session.rollback = AsyncMock()
        cm.__aenter__ = AsyncMock(return_value=session)
        cm.__aexit__ = AsyncMock(return_value=False)
        return cm

    factory = MagicMock()
    factory.side_effect = _new_cm
    return factory


async def _run(runs: list[Any], *, apply: bool = True) -> tuple[int, list[dict[str, Any]]]:
    factory = _factory([r.id for r in runs], runs)
    with patch.object(bm, "get_session_factory", return_value=factory), capture_logs() as logs:
        code = await bm._run(apply=apply, run_id_filter=None, batch_size=100, limit=None)
    return code, logs


# ---------------------------------------------------------------------------
# T-31
# ---------------------------------------------------------------------------


class TestT31Guard:
    @pytest.mark.parametrize("apply", [True, False])
    async def test_a_wp25_row_is_skipped_with_zero_writes_and_counted(self, apply: bool) -> None:
        wp25 = _row(_ID_WP25, _config())
        before = copy.deepcopy(wp25.config)
        code, logs = await _run([wp25], apply=apply)
        assert code == 0
        assert wp25._writes == []  # not even the sentinel
        assert wp25.metrics_v2_backfilled is False
        assert wp25.config == before
        assert wp25.config["backtest_metrics"]["psr"] == 0.9731
        assert wp25.config["backtest_metrics"]["confidence_flag"] == "high"
        assert wp25.n_closed_trades == 5
        (summary,) = [e for e in logs if e["event"] == "backfill.summary_line"]
        assert summary["skipped_wp25"] == 1
        assert summary["runs_processable"] == 0
        assert summary["runs_skipped"] == 0  # a separate counter, not the generic one

    async def test_the_guard_runs_before_any_recompute(self) -> None:
        wp25 = _row(_ID_WP25, _config())
        with patch.object(
            bm, "_recompute_psr_from_snapshots", side_effect=AssertionError("recompute")
        ):
            diff = bm._compute_diff(wp25, wp25.equity_snapshots)
        assert diff.skipped_wp25 is True and diff.is_skip is True
        assert diff.skip_reason == bm._WP25_SKIP_REASON
        assert (diff.psr, diff.confidence_flag, diff.n_closed_trades, diff.quote_currency) == (
            None,
            None,
            None,
            None,
        )

    async def test_a_legacy_row_is_processed_as_before(self) -> None:
        legacy = _row(_ID_LEGACY, _config(with_semantics=False), n_closed=None)
        code, logs = await _run([legacy])
        assert code == 0
        assert legacy.metrics_v2_backfilled is True
        assert legacy.n_closed_trades == 5
        assert legacy.config["backtest_metrics"]["quote_currency"] == "USDT"
        (summary,) = [e for e in logs if e["event"] == "backfill.summary_line"]
        assert summary["skipped_wp25"] == 0 and summary["runs_processable"] == 1

    async def test_a_mixed_batch_writes_only_the_legacy_row(self) -> None:
        wp25_a = _row(_ID_WP25, _config())
        legacy = _row(_ID_LEGACY, _config(with_semantics=False), n_closed=None)
        wp25_b = _row(_ID_WP25_B, _config())
        before = [copy.deepcopy(r.config) for r in (wp25_a, wp25_b)]
        code, logs = await _run([wp25_a, legacy, wp25_b])
        assert code == 0
        assert wp25_a._writes == [] and wp25_b._writes == []
        assert [wp25_a.config, wp25_b.config] == before
        assert legacy.metrics_v2_backfilled is True and legacy.n_closed_trades == 5
        (summary,) = [e for e in logs if e["event"] == "backfill.summary_line"]
        assert (summary["skipped_wp25"], summary["runs_processable"]) == (2, 1)

    @pytest.mark.parametrize(
        "marker", [None, "", "wp24-9", 1, True, ["wp25-1"], {"v": "wp25-1"}], ids=repr
    )
    async def test_an_unknown_or_malformed_marker_is_a_legacy_row(self, marker: Any) -> None:
        """Only a known string skips; everything else keeps the pre-WP2.5 behaviour."""
        row = _row(_ID_LEGACY, _config(marker), n_closed=None)
        diff = bm._compute_diff(row, row.equity_snapshots)
        assert diff.skipped_wp25 is False and diff.n_closed_trades == 5

    @pytest.mark.parametrize(
        "config",
        [{}, {"engine_semantics": None}, {"engine_semantics": "wp25-1"}, {"engine_semantics": []}],
    )
    def test_the_marker_reader_never_raises(self, config: Any) -> None:
        assert bm._has_known_metrics_marker(config) is False
        assert bm._has_known_metrics_marker(None) is False
        assert bm._has_known_metrics_marker({"engine_semantics": {"metrics": "wp25-1"}}) is True

    async def test_the_skip_logs_carry_numbers_and_fixed_text_only(self) -> None:
        cfg = _config()
        cfg["engine_semantics"]["metrics"] = "wp25-1"
        row = _row(_ID_WP25, cfg)
        _, logs = await _run([row], apply=False)
        dumped = repr(logs)
        assert "wp25-1" not in dumped
        assert "engine_semantics" not in dumped
        assert "0.9731" not in dumped  # no metric payload
        skip = [e for e in logs if e["event"] == "backfill.skip"]
        assert skip and skip[0]["reason"] == bm._WP25_SKIP_REASON

    async def test_an_unknown_marker_value_is_never_logged(self) -> None:
        row = _row(_ID_LEGACY, _config(_SENTINEL), n_closed=None)
        _, logs = await _run([row])
        assert _SENTINEL not in repr(logs)


class TestLegacyPathIsUnchanged:
    """CR-08: the 1h fallback in ``_recompute_psr_from_snapshots`` is for legacy rows only."""

    def test_the_legacy_fallback_is_still_there_and_commented(self) -> None:
        src = (_REPO_ROOT / "scripts/backfill_metrics_v2.py").read_text()
        assert "TIMEFRAME_PERIODS_PER_YEAR.get(tf_enum or TimeFrame.ONE_HOUR, 8_766.0)" in src
        assert "LEGACY rows only" in src

    def test_the_guard_precedes_every_recompute_in_compute_diff(self) -> None:
        tree = ast.parse((_REPO_ROOT / "scripts/backfill_metrics_v2.py").read_text())
        func = next(
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == "_compute_diff"
        )
        calls = [
            (n.lineno, n.func.id)
            for n in ast.walk(func)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
        ]
        guard = min(line for line, name in calls if name == "_has_known_metrics_marker")
        recompute = [
            line
            for line, name in calls
            if name
            in {
                "_recompute_psr_from_snapshots",
                "_recompute_profit_factor",
                "_derive_confidence_flag",
            }
        ]
        assert recompute and all(guard < line for line in recompute)

    def test_the_apply_loop_skips_wp25_before_the_generic_skip_sentinel(self) -> None:
        src = (_REPO_ROOT / "scripts/backfill_metrics_v2.py").read_text()
        wp25 = src.index("if diff.skipped_wp25:\n                            # WP2.5")
        generic = src.index("if diff.is_skip:\n                            # Even")
        assert wp25 < generic

    async def test_legacy_output_is_identical_to_the_pre_wp25_recompute(self) -> None:
        """A legacy row's PSR/flag come from the unchanged recompute."""
        row = _row(_ID_LEGACY, _config(with_semantics=False), n_closed=None)
        diff = bm._compute_diff(row, row.equity_snapshots)
        expected_psr = bm._recompute_psr_from_snapshots(row.equity_snapshots, "1h")
        assert diff.psr == expected_psr
        assert diff.skipped_wp25 is False
