"""
tests/unit/test_sprint43_quick_wins.py
---------------------------------------
Sprint 43 quick-wins coverage.  Bundles tests for items that touch
disparate modules so each tiny improvement has at least one focused
assertion verifying the contract.

Items covered
-------------
SEC-001  SecretStr for FRED + Whale Alert API keys (model_dump scrub).
SEC-007  CORS wildcard rejection under production auth posture.
QT-011   Exposure bar-count fix — overlapping trade spans no longer
         double-count.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

import pytest
from pydantic import SecretStr

from api.config import Settings


# ---------------------------------------------------------------------------
# SEC-001: SecretStr scrubs FRED + Whale Alert keys
# ---------------------------------------------------------------------------


class TestSEC001SecretStr:
    def test_fred_api_key_is_secretstr(self) -> None:
        s = Settings(
            database_url="postgresql+asyncpg://u:p@h:5432/d",
            fred_api_key="super-secret-fred-key",
        )
        assert isinstance(s.fred_api_key, SecretStr)
        assert s.fred_api_key.get_secret_value() == "super-secret-fred-key"

    def test_whale_alert_api_key_is_secretstr(self) -> None:
        s = Settings(
            database_url="postgresql+asyncpg://u:p@h:5432/d",
            whale_alert_api_key="super-secret-whale-key",
        )
        assert isinstance(s.whale_alert_api_key, SecretStr)
        assert s.whale_alert_api_key.get_secret_value() == "super-secret-whale-key"

    def test_keys_redacted_in_model_dump(self) -> None:
        """model_dump() must NOT expose raw secret values — the whole point
        of SecretStr is that ``str(settings)``-style serialisation cannot
        leak the credential into logs / debug payloads."""
        s = Settings(
            database_url="postgresql+asyncpg://u:p@h:5432/d",
            fred_api_key="leak-this-and-die",
            whale_alert_api_key="leak-this-too",
        )
        dumped = s.model_dump()
        # Pydantic's SecretStr renders as "**********" in dumps by default.
        assert "leak-this-and-die" not in str(dumped)
        assert "leak-this-too" not in str(dumped)


# ---------------------------------------------------------------------------
# SEC-007: CORS wildcard validator
# ---------------------------------------------------------------------------


class TestSEC007CORSValidator:
    def test_wildcard_rejected_when_auth_enabled(self) -> None:
        with pytest.raises(ValueError, match="wildcards are forbidden"):
            Settings(
                database_url="postgresql+asyncpg://u:p@h:5432/d",
                require_api_auth=True,
                allowed_origins=["*"],
                api_key_hash="a" * 64,
            )

    def test_wildcard_prefix_rejected_when_auth_enabled(self) -> None:
        with pytest.raises(ValueError, match="wildcards are forbidden"):
            Settings(
                database_url="postgresql+asyncpg://u:p@h:5432/d",
                require_api_auth=True,
                allowed_origins=["*.example.com"],
                api_key_hash="a" * 64,
            )

    def test_wildcard_suffix_rejected_when_auth_enabled(self) -> None:
        with pytest.raises(ValueError, match="wildcards are forbidden"):
            Settings(
                database_url="postgresql+asyncpg://u:p@h:5432/d",
                require_api_auth=True,
                allowed_origins=["https://example.*"],
                api_key_hash="a" * 64,
            )

    def test_wildcard_allowed_in_dev_mode(self) -> None:
        """Wildcards stay legal when require_api_auth=False so local dev
        workflows that rely on permissive CORS keep working."""
        s = Settings(
            database_url="postgresql+asyncpg://u:p@h:5432/d",
            require_api_auth=False,
            allowed_origins=["*"],
        )
        assert s.allowed_origins == ["*"]

    def test_explicit_origins_accepted_with_auth(self) -> None:
        s = Settings(
            database_url="postgresql+asyncpg://u:p@h:5432/d",
            require_api_auth=True,
            allowed_origins=["https://trading.example.com"],
            api_key_hash="a" * 64,
        )
        assert s.allowed_origins == ["https://trading.example.com"]


# ---------------------------------------------------------------------------
# QT-011: exposure bar-count fix
# ---------------------------------------------------------------------------


def _make_equity_curve(num_bars: int, start: datetime) -> list[Any]:
    from trading.metrics import EquityCurvePoint

    return [
        EquityCurvePoint(
            timestamp=start + timedelta(hours=i),
            equity=Decimal("10000"),
            drawdown_pct=0.0,
        )
        for i in range(num_bars)
    ]


def _make_trade_stub(*, entry_at: datetime, exit_at: datetime) -> Any:
    """Minimal stub object compatible with the trade iteration in
    ``_estimate_bars_in_market``.  Only entry_at / exit_at are read."""
    from types import SimpleNamespace

    return SimpleNamespace(entry_at=entry_at, exit_at=exit_at)


class TestQT011ExposureBarCount:
    """QT-011 (Sprint 43) / M2 (Sprint 49): overlapping trades on different symbols
    must NOT inflate the bars-in-market count.

    As of M2 the authoritative source is ``StrategyEngine.run_backtest()``'s
    per-bar tracker (live portfolio positions), not this post-hoc estimate.
    ``_estimate_bars_in_market`` is retained for backward compatibility and
    these tests continue to verify its standalone contract."""

    def test_single_trade_counts_each_bar_once(self) -> None:
        from trading.backtest import BacktestRunner

        start = datetime(2026, 1, 1, tzinfo=UTC)
        curve = _make_equity_curve(num_bars=10, start=start)
        # Trade spans bars 2..6 inclusive (5 bars)
        trade = _make_trade_stub(
            entry_at=start + timedelta(hours=2),
            exit_at=start + timedelta(hours=6),
        )
        portfolio = type("P", (), {"get_trade_history": lambda self: [trade]})()
        runner = BacktestRunner.__new__(BacktestRunner)  # bypass __init__

        bars = runner._estimate_bars_in_market(curve, portfolio)
        assert bars == 5

    def test_overlapping_trades_do_not_double_count(self) -> None:
        from trading.backtest import BacktestRunner

        start = datetime(2026, 1, 1, tzinfo=UTC)
        curve = _make_equity_curve(num_bars=10, start=start)
        # Two trades on different symbols — fully overlapping spans
        trade_a = _make_trade_stub(
            entry_at=start + timedelta(hours=2),
            exit_at=start + timedelta(hours=6),
        )
        trade_b = _make_trade_stub(
            entry_at=start + timedelta(hours=2),
            exit_at=start + timedelta(hours=6),
        )
        portfolio = type(
            "P", (), {"get_trade_history": lambda self: [trade_a, trade_b]}
        )()
        runner = BacktestRunner.__new__(BacktestRunner)

        bars = runner._estimate_bars_in_market(curve, portfolio)
        # Was 10 (5+5) clamped to 10 in the legacy heuristic — corrected
        # path returns 5 because each bar is counted at most once even if
        # multiple trades are concurrently open.
        assert bars == 5

    def test_disjoint_trades_sum_to_total_unique_bars(self) -> None:
        from trading.backtest import BacktestRunner

        start = datetime(2026, 1, 1, tzinfo=UTC)
        curve = _make_equity_curve(num_bars=10, start=start)
        # Disjoint: bars 1..3 and bars 7..9 — 3+3 = 6 bars total
        trade_a = _make_trade_stub(
            entry_at=start + timedelta(hours=1),
            exit_at=start + timedelta(hours=3),
        )
        trade_b = _make_trade_stub(
            entry_at=start + timedelta(hours=7),
            exit_at=start + timedelta(hours=9),
        )
        portfolio = type(
            "P", (), {"get_trade_history": lambda self: [trade_a, trade_b]}
        )()
        runner = BacktestRunner.__new__(BacktestRunner)

        bars = runner._estimate_bars_in_market(curve, portfolio)
        assert bars == 6

    def test_no_trades_yields_zero(self) -> None:
        from trading.backtest import BacktestRunner

        start = datetime(2026, 1, 1, tzinfo=UTC)
        curve = _make_equity_curve(num_bars=10, start=start)
        portfolio = type("P", (), {"get_trade_history": lambda self: []})()
        runner = BacktestRunner.__new__(BacktestRunner)
        assert runner._estimate_bars_in_market(curve, portfolio) == 0

    def test_empty_curve_yields_zero(self) -> None:
        from trading.backtest import BacktestRunner

        trade = _make_trade_stub(
            entry_at=datetime(2026, 1, 1, tzinfo=UTC),
            exit_at=datetime(2026, 1, 2, tzinfo=UTC),
        )
        portfolio = type("P", (), {"get_trade_history": lambda self: [trade]})()
        runner = BacktestRunner.__new__(BacktestRunner)
        assert runner._estimate_bars_in_market([], portfolio) == 0
