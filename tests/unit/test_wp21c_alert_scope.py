"""
tests/unit/test_wp21c_alert_scope.py
-------------------------------------
WP2.1c-delta D-14 (CR-01 / ruling C-01): keep backtest runs out of the
per-run Grafana safety alerts.

Since WP2.1c the circuit breaker is on by default in backtests, so a backtest
emits the ``trading_circuit_breaker_state`` gauge (2 / 3 on a DAILY_LIMIT /
HALT) under ``run_id="bt-<12 hex>"``. Without a scope, ``trading-cb-halt``
(``max(...) > 2.5``) would fire as a false live HALT. The three per-run rules
therefore carry the matcher ``{run_id!~"bt-.*"}``.

Checks:
  (1) static: the provisioned YAML (parsed, not grepped) carries the matcher on
      exactly the three per-run rules, with thresholds and structure unchanged;
  (2) behavioural: a tiny default backtest writes its breaker / kill-switch /
      drawdown gauges under a ``run_id`` that fully matches ``bt-.*``;
  (3) fail direction: a ``str(uuid4())`` (the paper / live run id format) never
      fully matches ``bt-.*``, so the matcher cannot exclude a live run.

Prometheus label regexes are fully anchored, so ``re.fullmatch`` is the faithful
model of ``run_id=~"bt-.*"`` / ``run_id!~"bt-.*"``.
"""

from __future__ import annotations

import math
import re
import uuid
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
import yaml

from api.prometheus import _parse_label_key
from common.metrics import metrics
from common.models import OHLCVBar
from common.types import TimeFrame
from trading.backtest import BacktestRunner
from trading.strategies.ma_crossover import MACrossoverStrategy

_REPO_ROOT = Path(__file__).resolve().parents[2]
_ALERTS = _REPO_ROOT / "infra" / "grafana" / "provisioning" / "alerting" / "trading-alerts.yml"

_MATCHER = '{run_id!~"bt-.*"}'
_BACKTEST_RUN_ID = re.compile(r"bt-.*")

# uid -> (expected expr, expected threshold evaluator type, expected params)
_SCOPED_RULES: dict[str, tuple[str, str, list[float]]] = {
    "trading-cb-halt": ('max(trading_circuit_breaker_state{run_id!~"bt-.*"})', "gt", [2.5]),
    "trading-kill-switch": ('max(trading_kill_switch_active{run_id!~"bt-.*"})', "gt", [0.5]),
    "trading-high-drawdown": ('max(trading_drawdown_pct{run_id!~"bt-.*"})', "gt", [0.10]),
}
_UNSCOPED_RULES: dict[str, tuple[str, str, list[float]]] = {
    "trading-api-down": ('min(up{job="trading-bot-api"})', "lt", [0.5]),
}


# ---------------------------------------------------------------------------
# YAML helpers
# ---------------------------------------------------------------------------


def _load_rules() -> dict[str, dict[str, Any]]:
    doc = yaml.safe_load(_ALERTS.read_text(encoding="utf-8"))
    assert isinstance(doc, dict)
    groups = doc["groups"]
    assert len(groups) == 1
    return {rule["uid"]: rule for rule in groups[0]["rules"]}


def _query(rule: dict[str, Any]) -> dict[str, Any]:
    (query,) = [d for d in rule["data"] if d["refId"] == "A"]
    return query


def _threshold(rule: dict[str, Any]) -> dict[str, Any]:
    (cond,) = [d for d in rule["data"] if d["refId"] == "C"]
    (evaluator,) = [c["evaluator"] for c in cond["model"]["conditions"]]
    return evaluator


# ---------------------------------------------------------------------------
# (1) static: the provisioned expressions carry the matcher
# ---------------------------------------------------------------------------


class TestAlertExpressionsAreScoped:
    @pytest.mark.parametrize("uid", sorted(_SCOPED_RULES))
    def test_expr_contains_the_backtest_exclusion_matcher(self, uid: str) -> None:
        expr = _query(_load_rules()[uid])["model"]["expr"]
        assert _MATCHER in expr

    @pytest.mark.parametrize("uid", sorted(_SCOPED_RULES))
    def test_expr_is_exactly_the_scoped_selector_and_threshold_is_unchanged(self, uid: str) -> None:
        expected_expr, op, params = _SCOPED_RULES[uid]
        rule = _load_rules()[uid]
        assert _query(rule)["model"]["expr"] == expected_expr
        evaluator = _threshold(rule)
        assert evaluator["type"] == op
        assert evaluator["params"] == params

    def test_the_api_down_rule_is_not_scoped_and_unchanged(self) -> None:
        for uid, (expr, op, params) in _UNSCOPED_RULES.items():
            rule = _load_rules()[uid]
            assert _query(rule)["model"]["expr"] == expr
            assert _MATCHER not in expr
            assert _threshold(rule) == {"type": op, "params": params}

    def test_provisioning_structure_is_unchanged(self) -> None:
        doc = yaml.safe_load(_ALERTS.read_text(encoding="utf-8"))
        assert doc["apiVersion"] == 1
        (group,) = doc["groups"]
        assert (group["orgId"], group["name"], group["folder"], group["interval"]) == (
            1,
            "trading-safety",
            "Trading",
            "1m",
        )
        uids = [r["uid"] for r in group["rules"]]
        assert uids == [
            "trading-cb-halt",
            "trading-kill-switch",
            "trading-high-drawdown",
            "trading-api-down",
        ]
        for rule in group["rules"]:
            assert rule["condition"] == "C"
            assert [d["refId"] for d in rule["data"]] == ["A", "C"]
            assert _query(rule)["datasourceUid"] == "prometheus"
            assert _query(rule)["model"]["instant"] is True
        by_uid = {r["uid"]: r for r in group["rules"]}
        assert by_uid["trading-cb-halt"]["labels"] == {"severity": "critical"}
        assert by_uid["trading-kill-switch"]["labels"] == {"severity": "critical"}
        assert by_uid["trading-high-drawdown"]["labels"] == {"severity": "warning"}
        assert by_uid["trading-api-down"]["noDataState"] == "Alerting"


# ---------------------------------------------------------------------------
# (2) behavioural: a default backtest emits its gauges under run_id=bt-...
# ---------------------------------------------------------------------------

_ALERTED_GAUGES = ("circuit_breaker_state", "kill_switch_active", "portfolio_drawdown_pct")


def _make_bars(n: int = 60) -> list[OHLCVBar]:
    """Deterministic oscillating bars (no RNG): high >= close >= low always."""
    bars: list[OHLCVBar] = []
    for i in range(n):
        close = 30_000.0 + 400.0 * math.sin(i / 4.0)
        open_ = 30_000.0 + 400.0 * math.sin((i - 1) / 4.0)
        bars.append(
            OHLCVBar(
                symbol="BTC/USDT",
                timestamp=datetime(2024, 1, 1 + i // 24, i % 24, 0, 0, tzinfo=UTC),
                open=Decimal(str(round(open_, 2))),
                high=Decimal(str(round(max(open_, close) + 50.0, 2))),
                low=Decimal(str(round(min(open_, close) - 50.0, 2))),
                close=Decimal(str(round(close, 2))),
                volume=Decimal("5"),
                timeframe=TimeFrame.ONE_HOUR,
            )
        )
    return bars


def _gauge_run_ids(gauge_name: str) -> set[str | None]:
    """Return the ``run_id`` label of every gauge key of ``gauge_name``."""
    run_ids: set[str | None] = set()
    for key in metrics.get_all()["gauges"]:
        name, labels = _parse_label_key(key)
        if name == gauge_name:
            run_ids.add(labels.get("run_id"))
    return run_ids


class TestBacktestGaugesCarryTheBacktestRunId:
    @pytest.mark.asyncio
    async def test_default_backtest_breaker_gauge_run_id_fullmatches_bt(self) -> None:
        before = {name: _gauge_run_ids(name) for name in _ALERTED_GAUGES}

        # Default constructor: enable_circuit_breaker is the WP2.1c default (on).
        runner = BacktestRunner(
            strategies=[
                MACrossoverStrategy(
                    strategy_id="ma-alert-scope",
                    params={"fast_period": 5, "slow_period": 10},
                )
            ],
            symbols=["BTC/USDT"],
            timeframe=TimeFrame.ONE_HOUR,
            initial_capital=Decimal("10000"),
            seed=7,
        )
        # The scenario under test is the WP2.1c default: breaker on.
        assert runner.circuit_breaker_enabled is True
        result = await runner.run({"BTC/USDT": _make_bars()})

        assert _BACKTEST_RUN_ID.fullmatch(result.run_id)
        for name in _ALERTED_GAUGES:
            new_ids = _gauge_run_ids(name) - before[name]
            # The engine writes all three gauges every bar, under the run's id.
            assert result.run_id in new_ids, name
            for run_id in new_ids:
                assert run_id is not None, name
                assert _BACKTEST_RUN_ID.fullmatch(run_id), (name, run_id)


# ---------------------------------------------------------------------------
# (3) fail direction: a paper / live run id is never excluded
# ---------------------------------------------------------------------------


class TestLiveRunIdsAreNeverExcluded:
    def test_str_uuid4_never_fullmatches_the_backtest_pattern(self) -> None:
        for _ in range(2000):
            assert _BACKTEST_RUN_ID.fullmatch(str(uuid.uuid4())) is None

    def test_a_uuid_cannot_even_contain_the_prefix_letter_t(self) -> None:
        # Hex digits are 0-9a-f, so "bt-" (which needs a "t") cannot occur.
        for _ in range(2000):
            assert "t" not in str(uuid.uuid4())

    def test_the_pattern_is_anchored_like_a_prometheus_matcher(self) -> None:
        assert _BACKTEST_RUN_ID.fullmatch("bt-0123456789ab")
        assert _BACKTEST_RUN_ID.fullmatch("xbt-0123456789ab") is None
        assert _BACKTEST_RUN_ID.fullmatch("unknown") is None
        assert _BACKTEST_RUN_ID.fullmatch("") is None
