"""
tests/integration/test_live_confirm_token_header.py
-----------------------------------------------------
Integration coverage for SEC-004 (Sprint 41) — the live-trading
confirm-token must be accepted via the ``X-Live-Confirm-Token`` header
in addition to the legacy body-field.  Body-field remains as a
deprecated fallback until all clients migrate.

The test layer 3 of the LiveTradingGate in isolation by configuring a
known token via ``LIVE_TRADING_CONFIRM_TOKEN`` env var and asserting
which failure codes the gate returns.  Layers 1 and 2 intentionally
fail (no ENABLE_LIVE_TRADING / API keys) so we never exercise the
downstream CCXT path — the test stays fast and deterministic.
"""

from __future__ import annotations

from typing import Any, Generator

import pytest
from fastapi.testclient import TestClient

from api.config import get_settings


# LiveTradingGate._check_confirmation_layer compares the raw tokens via
# ``hmac.compare_digest`` — so the env var stores the token *verbatim*, not a
# hash.  The API key hash (SEC-003) uses SHA-256; do not conflate the two.
_CONFIRM_TOKEN = "sec-004-header-test-token"


@pytest.fixture()
def client_with_token(monkeypatch: pytest.MonkeyPatch) -> Generator[TestClient, None, None]:
    """App with a known LIVE_TRADING_CONFIRM_TOKEN hash so Layer 3 is
    deterministic; Layers 1 and 2 stay failing (env flag off, no keys)."""
    monkeypatch.setenv("REQUIRE_API_AUTH", "false")
    monkeypatch.setenv("RATE_LIMIT_ENABLED", "false")
    monkeypatch.setenv("PROMETHEUS_ENABLED", "false")
    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://test:test@localhost:5432/test")
    monkeypatch.setenv("DEBUG", "true")
    monkeypatch.setenv("LIVE_TRADING_CONFIRM_TOKEN", _CONFIRM_TOKEN)
    get_settings.cache_clear()
    from api.main import create_app

    app = create_app()
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c
    get_settings.cache_clear()


def _live_payload(*, confirm_token: str | None = None) -> dict[str, Any]:
    # Sprint 51 C2 lockdown: use the ACTIVE, live-eligible ``grid_trading`` so
    # the request reaches the LiveTradingGate (the focus of these SEC-004
    # confirm-token tests).  ma_crossover is now DEMOTED -> backtest-only and
    # would be rejected with 422 BEFORE the gate, which is unrelated to the
    # confirm-token behaviour under test here.
    body: dict[str, Any] = {
        "strategyName": "grid_trading",
        "strategyParams": {},
        "symbols": ["BTC/USDT"],
        "timeframe": "1h",
        "mode": "live",
        "initialCapital": "10000.00",
    }
    if confirm_token is not None:
        body["confirmToken"] = confirm_token
    return body


def _gate_failure_layers(response_body: dict[str, Any]) -> list[str]:
    """Extract the list of failed layer names from a 403 gate-failure body."""
    detail: str = response_body.get("detail", "")
    # Router uses a single-string detail including the failed layer names.
    # Accept either the structured form or the joined form.
    if isinstance(detail, list):
        return detail
    return [detail]


class TestConfirmTokenViaHeader:
    def test_missing_token_everywhere_fails_confirmation_layer(
        self, client_with_token: TestClient
    ) -> None:
        """With no header and no body-field the gate must fail at Layer 3."""
        response = client_with_token.post("/api/v1/runs", json=_live_payload())
        assert response.status_code == 403
        detail = response.json()["detail"]
        # The detail text identifies the failed layers; Confirmation Token
        # must be among them along with the env/keys layers.
        assert "confirmation" in detail

    def test_header_token_clears_confirmation_layer(
        self, client_with_token: TestClient
    ) -> None:
        """When the header carries a correct token the Confirmation layer
        must pass — other layers still fail but Layer 3 is clean."""
        response = client_with_token.post(
            "/api/v1/runs",
            json=_live_payload(),
            headers={"X-Live-Confirm-Token": _CONFIRM_TOKEN},
        )
        assert response.status_code == 403
        detail = response.json()["detail"]
        # The token layer is satisfied; only env/keys layers are failing.
        assert "confirmation" not in detail

    def test_body_token_still_accepted_as_deprecated_fallback(
        self, client_with_token: TestClient
    ) -> None:
        """Body-field confirmToken is retained for backwards-compat until
        all clients migrate to the header."""
        response = client_with_token.post(
            "/api/v1/runs", json=_live_payload(confirm_token=_CONFIRM_TOKEN)
        )
        assert response.status_code == 403
        detail = response.json()["detail"]
        # Fallback path clears the Confirmation layer too
        assert "confirmation" not in detail

    def test_header_takes_precedence_over_body_on_conflict(
        self, client_with_token: TestClient
    ) -> None:
        """If a client sends a valid header AND a wrong body-field the gate
        must accept the request — the header is authoritative."""
        response = client_with_token.post(
            "/api/v1/runs",
            json=_live_payload(confirm_token="definitely-wrong"),
            headers={"X-Live-Confirm-Token": _CONFIRM_TOKEN},
        )
        assert response.status_code == 403
        detail = response.json()["detail"]
        assert "confirmation" not in detail

    def test_wrong_header_rejects_even_if_body_correct(
        self, client_with_token: TestClient
    ) -> None:
        """Inverse of precedence — a present-but-wrong header must NOT fall
        back to the body-field; security posture is header-authoritative."""
        response = client_with_token.post(
            "/api/v1/runs",
            json=_live_payload(confirm_token=_CONFIRM_TOKEN),
            headers={"X-Live-Confirm-Token": "wrong-token"},
        )
        assert response.status_code == 403
        detail = response.json()["detail"]
        # Confirmation Token layer failed because the header was wrong and
        # the fallback to body-field only triggers when header is ABSENT.
        assert "confirmation" in detail
