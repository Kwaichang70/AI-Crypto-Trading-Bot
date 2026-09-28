"""
tests/integration/test_wp13a_exit_config_api.py
--------------------------------------------------
WP1.3a (reports/vp2-wp1.3a/synthesis-spec.md) -- API-level coverage for the
exit-config validator wired into ``POST /api/v1/runs`` (create) in BACKTEST
mode.  Backtest mode is used throughout because it runs synchronously with
no background task / live-gate scaffolding, which keeps these tests fast
and hermetic while still exercising the real ``create_run`` handler,
``trading.exit_config`` and (where noted) the real ``BacktestRunner``.

Paper/live-mode 422 parity (create/promote/resume), the pyramiding held
gate, and the "two breakouts" harness are covered in:
  - tests/unit/test_wp13a_exit_config.py (validator unit coverage)
  - tests/integration/test_live_protective_paths.py (AC1 harness)
  - tests/integration/test_live_nav_sizing_harness.py (H2-H4 backward-compat)
"""

from __future__ import annotations

from unittest.mock import patch
from uuid import uuid4

from fastapi.testclient import TestClient

from tests.conftest import make_bars

_URL = "/api/v1/runs"
_SYMBOL = "BTC/USD"

# Enough bars for a small-lookback momentum_breakout backtest (min_bars = 6,
# BacktestRunner warmup = max(6*2, 50) = 50) and comfortable margin.
_BARS = make_bars(120, symbol=_SYMBOL)
_BARS_BY_SYMBOL = {_SYMBOL: _BARS}


def _base_payload(**overrides: object) -> dict:
    body: dict = {
        "strategyName": "momentum_breakout",
        "strategyParams": {
            "lookback": 5,
            "trend_sma_period": 5,
            "position_size": 100.0,
        },
        "symbols": [_SYMBOL],
        "timeframe": "1h",
        "mode": "backtest",
        "initialCapital": "10000.00",
        "backtestStart": "2024-01-01T00:00:00Z",
        "backtestEnd": "2024-06-01T00:00:00Z",
    }
    body.update(overrides)
    return body


def _post(client: TestClient, payload: dict):
    # WP7.0 (SY-70-01/§4c): Idempotency-Key is required on every
    # POST /api/v1/runs call, including backtest. A fresh UUID per call
    # avoids any accidental idempotency_key_reused/replay across the
    # distinct payloads this module posts.
    with patch(
        "api.routers.runs._fetch_bars_for_backtest",
        return_value=_BARS_BY_SYMBOL,
    ):
        return client.post(
            _URL, json=payload, headers={"Idempotency-Key": str(uuid4())}
        )


def _post_never_fetches(client: TestClient, payload: dict):
    """Assert the bar fetch is never reached (used for pre-422 cases)."""
    with patch(
        "api.routers.runs._fetch_bars_for_backtest",
        side_effect=AssertionError("bar fetch must not be reached"),
    ):
        return client.post(
            _URL, json=payload, headers={"Idempotency-Key": str(uuid4())}
        )


# ---------------------------------------------------------------------------
# ST-01/02: malformed values -> 422, never 500
# ---------------------------------------------------------------------------


class TestMalformedValues:
    def test_non_numeric_string_gives_422_not_500(
        self, client_dev_with_db: TestClient, mock_db_session
    ) -> None:
        payload = _base_payload(
            strategyParams={
                **_base_payload()["strategyParams"],
                "bracket_stop_loss_pct": "abc",
            }
        )
        resp = _post_never_fetches(client_dev_with_db, payload)
        assert resp.status_code == 422, resp.text
        detail = resp.json()["detail"]
        assert detail["code"] == "invalid_exit_config"
        assert any(e["reason"] == "not_a_number" for e in detail["errors"])
        mock_db_session.add.assert_not_called()

    def test_nan_gives_not_finite(
        self, client_dev_with_db: TestClient, mock_db_session
    ) -> None:
        payload = _base_payload(
            strategyParams={
                **_base_payload()["strategyParams"],
                "bracket_stop_loss_pct": "nan",
            }
        )
        resp = _post_never_fetches(client_dev_with_db, payload)
        assert resp.status_code == 422
        detail = resp.json()["detail"]
        assert any(e["reason"] == "not_finite" for e in detail["errors"])

    def test_bool_gives_invalid_type(
        self, client_dev_with_db: TestClient, mock_db_session
    ) -> None:
        payload = _base_payload(
            strategyParams={
                **_base_payload()["strategyParams"],
                "bracket_stop_loss_pct": True,
            }
        )
        resp = _post_never_fetches(client_dev_with_db, payload)
        assert resp.status_code == 422
        detail = resp.json()["detail"]
        assert any(e["reason"] == "invalid_type" for e in detail["errors"])

    def test_invalid_bracket_mode(
        self, client_dev_with_db: TestClient, mock_db_session
    ) -> None:
        payload = _base_payload(
            strategyParams={
                **_base_payload()["strategyParams"],
                "bracket_mode": "trailing",
            }
        )
        resp = _post_never_fetches(client_dev_with_db, payload)
        assert resp.status_code == 422
        assert resp.json()["detail"]["code"] == "invalid_exit_config"

    def test_huge_int_gives_not_finite_not_500(
        self, client_dev_with_db: TestClient, mock_db_session
    ) -> None:
        """WP13a-S-R5-03 (security round 5, AC3): a Python int far outside
        float range (e.g. a 4000-digit integer -- well under Python
        3.11+'s ~4300-digit JSON-parse ceiling, so it parses fine and
        reaches ``float(raw)``) must classify as a clean 422 ``not_finite``
        -- NOT an uncaught ``OverflowError`` surfacing as a 500."""
        huge_int = 10**3999
        payload = _base_payload(
            strategyParams={
                **_base_payload()["strategyParams"],
                "bracket_stop_loss_pct": huge_int,
            }
        )
        resp = _post_never_fetches(client_dev_with_db, payload)
        assert resp.status_code == 422, resp.text
        detail = resp.json()["detail"]
        assert detail["code"] == "invalid_exit_config"
        assert any(e["reason"] == "not_finite" for e in detail["errors"])
        mock_db_session.add.assert_not_called()


# ---------------------------------------------------------------------------
# ST-06/07: bounds
# ---------------------------------------------------------------------------


class TestBounds:
    def test_sl_above_policy_max_out_of_range(
        self, client_dev_with_db: TestClient, mock_db_session
    ) -> None:
        payload = _base_payload(
            strategyParams={
                **_base_payload()["strategyParams"],
                "bracket_mode": "fixed",
                "bracket_stop_loss_pct": 0.51,
            }
        )
        resp = _post_never_fetches(client_dev_with_db, payload)
        assert resp.status_code == 422
        detail = resp.json()["detail"]
        assert any(e["reason"] == "out_of_range" for e in detail["errors"])

    def test_sl_at_entry_cost_is_stop_inside_entry_cost(
        self, client_dev_with_db: TestClient, mock_db_session
    ) -> None:
        payload = _base_payload(
            strategyParams={
                **_base_payload()["strategyParams"],
                "bracket_mode": "fixed",
                "bracket_stop_loss_pct": 0.005,
            }
        )
        resp = _post_never_fetches(client_dev_with_db, payload)
        assert resp.status_code == 422
        detail = resp.json()["detail"]
        assert any(e["reason"] == "stop_inside_entry_cost" for e in detail["errors"])

    def test_sl_just_above_cost_succeeds_with_warning(
        self, client_dev_with_db: TestClient, mock_db_session
    ) -> None:
        payload = _base_payload(
            strategyParams={
                **_base_payload()["strategyParams"],
                "bracket_mode": "fixed",
                "bracket_stop_loss_pct": 0.0066,
                "bracket_take_profit_pct": 0.2,
            }
        )
        resp = _post(client_dev_with_db, payload)
        assert resp.status_code == 201, resp.text
        body = resp.json()
        assert any(w["code"] == "sl_below_round_trip_cost" for w in body["configWarnings"])


# ---------------------------------------------------------------------------
# ST-09/10: exit_manager_required, TP-only, trailing-only
# ---------------------------------------------------------------------------


class TestExitManagerRequired:
    def test_momentum_breakout_no_exits_gives_exit_manager_required(
        self, client_dev_with_db: TestClient, mock_db_session
    ) -> None:
        payload = _base_payload(
            strategyParams={
                "lookback": 5,
                "trend_sma_period": 5,
                "position_size": 100.0,
                "bracket_mode": "atr",
                "bracket_atr_sl_multiplier": None,
                "bracket_atr_tp_multiplier": None,
            }
        )
        resp = _post_never_fetches(client_dev_with_db, payload)
        assert resp.status_code == 422, resp.text
        detail = resp.json()["detail"]
        assert detail["code"] == "exit_manager_required"
        assert detail["strategy"]
        assert detail["requires_one_of"]
        mock_db_session.add.assert_not_called()

    def test_tp_only_gives_exit_manager_required(
        self, client_dev_with_db: TestClient, mock_db_session
    ) -> None:
        payload = _base_payload(
            strategyParams={
                **_base_payload()["strategyParams"],
                "bracket_mode": "fixed",
                "bracket_take_profit_pct": 0.1,
            }
        )
        resp = _post_never_fetches(client_dev_with_db, payload)
        assert resp.status_code == 422
        assert resp.json()["detail"]["code"] == "exit_manager_required"

    def test_trailing_only_succeeds(
        self, client_dev_with_db: TestClient, mock_db_session
    ) -> None:
        payload = _base_payload(
            strategyParams={
                **_base_payload()["strategyParams"],
                "trailing_stop_pct": 0.02,
            }
        )
        resp = _post(client_dev_with_db, payload)
        assert resp.status_code == 201, resp.text


# ---------------------------------------------------------------------------
# ST-04/G-1: zero from the UI means unset
# ---------------------------------------------------------------------------


class TestZeroMeansUnsetAtApi:
    def test_ui_shaped_momentum_payload_succeeds(
        self, client_dev_with_db: TestClient, mock_db_session
    ) -> None:
        """The UI's initDefaults sends 0 for every blank nullable numeric
        field: ATR 1.5/3.0 active plus fixed SL/TP 0/0 (inactive)."""
        payload = _base_payload(
            strategyParams={
                "lookback": 5,
                "trend_sma_period": 5,
                "position_size": 100.0,
                "bracket_mode": "atr",
                "bracket_atr_sl_multiplier": 1.5,
                "bracket_atr_tp_multiplier": 3.0,
                "bracket_atr_period": 14,
                "bracket_stop_loss_pct": 0,
                "bracket_take_profit_pct": 0,
            }
        )
        resp = _post(client_dev_with_db, payload)
        assert resp.status_code == 201, resp.text
        config = mock_db_session.add.call_args[0][0].config
        assert config["bracket_config"] == {
            "bracket_mode": "atr",
            "bracket_atr_period": 14,
            "bracket_atr_sl_multiplier": 1.5,
            "bracket_atr_tp_multiplier": 3.0,
        }

    def test_trailing_stop_pct_zero_removed_not_400(
        self, client_dev_with_db: TestClient, mock_db_session
    ) -> None:
        """G-1 regression: rsi_mean_reversion's schema has trailing_stop_pct
        ge=0.005 -- a raw 0 used to reach the schema and 400.  It must now
        be dropped before the schema ever sees it."""
        payload = {
            "strategyName": "rsi_mean_reversion",
            "strategyParams": {"trailing_stop_pct": 0},
            "symbols": [_SYMBOL],
            "timeframe": "1h",
            "mode": "backtest",
            "initialCapital": "10000.00",
            "backtestStart": "2024-01-01T00:00:00Z",
            "backtestEnd": "2024-06-01T00:00:00Z",
        }
        resp = _post(client_dev_with_db, payload)
        assert resp.status_code == 201, resp.text
        config = mock_db_session.add.call_args[0][0].config
        assert "trailing_stop_pct" not in config["strategy_params"]


# ---------------------------------------------------------------------------
# ST-13: allowPyramiding
# ---------------------------------------------------------------------------


class TestAllowPyramiding:
    def test_non_bool_gives_pydantic_422(
        self, client_dev_with_db: TestClient, mock_db_session
    ) -> None:
        payload = _base_payload(allowPyramiding="yes")
        resp = _post_never_fetches(client_dev_with_db, payload)
        assert resp.status_code == 422
        mock_db_session.add.assert_not_called()

    def test_absent_persists_resolved_false_for_momentum(
        self, client_dev_with_db: TestClient, mock_db_session
    ) -> None:
        payload = _base_payload(
            strategyParams={
                **_base_payload()["strategyParams"],
                "bracket_mode": "fixed",
                "bracket_stop_loss_pct": 0.05,
            }
        )
        resp = _post(client_dev_with_db, payload)
        assert resp.status_code == 201, resp.text
        config = mock_db_session.add.call_args[0][0].config
        assert config["allow_pyramiding"] is False

    def test_dca_defaults_true_in_backtest(
        self, client_dev_with_db: TestClient, mock_db_session
    ) -> None:
        payload = {
            "strategyName": "dca_rsi_hybrid",
            "strategyParams": {},
            "symbols": [_SYMBOL],
            "timeframe": "1h",
            "mode": "backtest",
            "initialCapital": "10000.00",
            "backtestStart": "2024-01-01T00:00:00Z",
            "backtestEnd": "2024-06-01T00:00:00Z",
        }
        resp = _post(client_dev_with_db, payload)
        assert resp.status_code == 201, resp.text
        config = mock_db_session.add.call_args[0][0].config
        assert config["allow_pyramiding"] is True
