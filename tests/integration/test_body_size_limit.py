"""
tests/integration/test_body_size_limit.py
-------------------------------------------
WP1.3a round 6 (user decision) -- ``apps.api.body_size_limit.BodySizeLimitMiddleware``.

A blanket, endpoint-agnostic request-body size cap enforced at the ASGI
transport layer, closing the whole DoS class WP1.3a rounds 1-5 chased
field-by-field (WP13a-S-R2-01, S-R3-02, S-R4-03, S-R5-02, S-R5-03): every
one of those stemmed from a single request body carrying an arbitrarily
large value into ``trading.exit_config``. This middleware bounds the
transport itself, independent of endpoint or field.

Covers, per the round-6 spec:
  - Content-Length 2 MiB -> 413, before any body is read.
  - a chunked 2 MiB body (no Content-Length) -> 413, enforced while
    streaming.
  - a normal create/optimize body -> unaffected.
  - the optimize 1M-list amplification case -> now 413 before any parsing
    (the exit_config-level fix from round 5 is never even reached).
  - the limit is configurable via settings.
"""

from __future__ import annotations

from collections.abc import Generator
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

_OPTIMIZE_URL = "/api/v1/optimize"


def _chunked_body(total_bytes: int, chunk_size: int = 65536):
    """A generator body -- httpx sends this as a chunked-transfer request
    with NO ``Content-Length`` header, exercising the streaming guard."""
    sent = 0
    chunk = b"x" * chunk_size
    while sent < total_bytes:
        remaining = total_bytes - sent
        piece = chunk if remaining >= chunk_size else chunk[:remaining]
        sent += len(piece)
        yield piece


class TestBodySizeLimitWP13aRound6:
    def test_content_length_2mib_returns_413_before_any_read(
        self, client_dev: TestClient
    ) -> None:
        """A declared Content-Length over the cap must be rejected via the
        fast path -- the fetch spy proves the body was never even
        forwarded to routing/parsing."""
        fetch_spy = AsyncMock(side_effect=AssertionError("must not be reached"))
        big_body = b"x" * (2 * 1024 * 1024)
        with patch("api.routers.optimize._fetch_bars_for_backtest", new=fetch_spy):
            resp = client_dev.post(
                _OPTIMIZE_URL,
                content=big_body,
                headers={"Content-Type": "application/json"},
            )
        assert resp.status_code == 413, resp.text
        detail = resp.json()["detail"]
        assert detail["code"] == "request_body_too_large"
        assert detail["max_bytes"] == 1_048_576
        fetch_spy.assert_not_called()

    def test_chunked_2mib_body_returns_413_via_streaming_guard(
        self, client_dev: TestClient
    ) -> None:
        """A chunked body (no Content-Length header at all) must still be
        rejected -- enforced while streaming, one chunk at a time."""
        fetch_spy = AsyncMock(side_effect=AssertionError("must not be reached"))
        with patch("api.routers.optimize._fetch_bars_for_backtest", new=fetch_spy):
            resp = client_dev.post(
                _OPTIMIZE_URL, content=_chunked_body(2 * 1024 * 1024)
            )
        assert resp.status_code == 413, resp.text
        detail = resp.json()["detail"]
        assert detail["code"] == "request_body_too_large"
        fetch_spy.assert_not_called()

    def test_normal_optimize_body_unaffected(self, client_dev: TestClient) -> None:
        """A normal, small request body must reach the application's own
        logic untouched -- NOT get a spurious 413."""
        resp = client_dev.post(_OPTIMIZE_URL, json={"strategyName": "nope"})
        assert resp.status_code != 413
        assert resp.status_code == 422  # pydantic: missing required fields

    def test_normal_create_run_body_unaffected(self, client_dev: TestClient) -> None:
        resp = client_dev.post("/api/v1/runs", json={"strategyName": "nope"})
        assert resp.status_code != 413

    def test_get_health_unaffected(self, client_dev: TestClient) -> None:
        resp = client_dev.get("/health")
        assert resp.status_code == 200

    def test_optimize_1m_list_amplification_now_413_before_any_parsing(
        self, client_dev: TestClient
    ) -> None:
        """The exact WP13a-S-R4-03/S-R5-xx amplification body (a
        1,000,000-element list, ~7.9MB serialised) must now be rejected by
        the transport-layer cap BEFORE it ever reaches
        ``validate_param_grid_exit_config`` or the bar fetch -- the
        per-field fix from earlier rounds is correct, but this body is
        now stopped even earlier."""
        import trading.exit_config as exit_config_module

        validate_spy = AsyncMock(side_effect=AssertionError("must not be reached"))
        fetch_spy = AsyncMock(side_effect=AssertionError("must not be reached"))
        body = {
            "strategyName": "momentum_breakout",
            "paramGrid": {
                "lookback": list(range(1000)),
                "bracket_stop_loss_pct": [list(range(1_000_000))],
            },
            "symbols": ["BTC/USDT"],
            "timeframe": "1h",
            "backtestStart": "2024-01-01T00:00:00Z",
            "backtestEnd": "2024-06-01T00:00:00Z",
            "rankBy": "sharpe_ratio",
            "topN": 10,
            "maxCombinations": 1000,
        }
        with (
            patch.object(
                exit_config_module,
                "validate_param_grid_exit_config",
                new=validate_spy,
            ),
            patch("api.routers.optimize._fetch_bars_for_backtest", new=fetch_spy),
        ):
            resp = client_dev.post(_OPTIMIZE_URL, json=body)

        assert resp.status_code == 413, resp.text
        assert resp.json()["detail"]["code"] == "request_body_too_large"
        validate_spy.assert_not_called()
        fetch_spy.assert_not_called()


class TestBodySizeLimitConfigurableWP13aRound6:
    """The cap is read from ``settings.max_request_body_bytes`` -- not
    hardcoded -- proven by overriding it to something far smaller than
    the default and observing a normally-tiny body get rejected too."""

    @pytest.fixture()
    def client_with_tiny_limit(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> Generator[TestClient, None, None]:
        monkeypatch.setenv("REQUIRE_API_AUTH", "false")
        monkeypatch.setenv("RATE_LIMIT_ENABLED", "false")
        monkeypatch.setenv("PROMETHEUS_ENABLED", "false")
        monkeypatch.setenv(
            "DATABASE_URL", "postgresql+asyncpg://test:test@localhost:5432/test"
        )
        monkeypatch.setenv("DEBUG", "true")
        monkeypatch.setenv("MAX_REQUEST_BODY_BYTES", "1024")  # 1 KiB

        from api.config import get_settings

        get_settings.cache_clear()
        from api.main import create_app

        app: Any = create_app()
        with TestClient(app, raise_server_exceptions=False) as c:
            from api.services import kill_switch as _kill_switch

            _kill_switch.reset_state_for_tests()
            yield c
        get_settings.cache_clear()

    def test_configured_tiny_limit_rejects_a_body_well_under_the_default(
        self, client_with_tiny_limit: TestClient
    ) -> None:
        """2 KiB is far under the 1 MiB DEFAULT, but over the 1 KiB
        limit configured for this test -- proving the value is actually
        read from settings, not hardcoded to 1 MiB."""
        body = b"x" * 2048
        resp = client_with_tiny_limit.post(
            _OPTIMIZE_URL, content=body, headers={"Content-Type": "application/json"}
        )
        assert resp.status_code == 413, resp.text
        assert resp.json()["detail"]["max_bytes"] == 1024
