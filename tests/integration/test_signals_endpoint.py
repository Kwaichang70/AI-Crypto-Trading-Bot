"""
tests/integration/test_signals_endpoint.py
--------------------------------------------
Integration coverage for apps/api/routers/signals.py (Sprint 41 TO-001).

Endpoint under test
-------------------
- GET /api/v1/signals/current — cached values for every market signal
  source.  Public (no auth), never queries upstream APIs directly — only
  reads the in-process cache populated by the 4 data clients.

The endpoint prefers ``request.app.state.container.services.<client>`` per
Sprint 40 Stap 1d, falling back to ``data.*.get_global_client()`` when the
container is unavailable.  These tests exercise both paths and the
graceful-degradation behaviour when individual clients fail.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Generator

import pytest
from fastapi.testclient import TestClient


_EXPECTED_KEYS = {
    "fearGreedIndex",
    "fearGreedClassification",
    "btcDominance",
    "marketCapChange24h",
    "totalVolumeChange24h",
    "fedFundsRate",
    "yieldCurveSpread",
    "whaleNetFlow",
    "whaleTxCount",
}


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_global_clients() -> Generator[None, None, None]:
    """Reset the four data/* module-level ``_global_client`` singletons so
    fallback assertions are deterministic regardless of earlier test state."""
    from data import macro_data, market_signals, sentiment, whale_tracker

    sentiment._global_client = None
    market_signals._global_client = None
    macro_data._global_client = None
    whale_tracker._global_client = None
    yield
    sentiment._global_client = None
    market_signals._global_client = None
    macro_data._global_client = None
    whale_tracker._global_client = None


@pytest.fixture()
def clean_signal_container(
    client_dev: TestClient, app_dev_mode: Any
) -> Generator[Any, None, None]:
    """Null every signal-client slot AFTER the TestClient lifespan has run.

    The lifespan instantiates the four data clients unconditionally and also
    writes them into ``data.*._global_client`` via ``set_global_client``.
    Depending on network availability the warmup step may populate caches,
    which would otherwise leak through the router's fallback path.  This
    fixture clears BOTH the container.services slots AND the module globals
    so the test body observes an actually-empty signal surface.
    """
    from data import macro_data, market_signals, sentiment, whale_tracker

    container = app_dev_mode.state.container
    container.services.fgi_client = None
    container.services.coingecko_client = None
    container.services.fred_client = None
    container.services.whale_alert_client = None
    sentiment._global_client = None
    market_signals._global_client = None
    macro_data._global_client = None
    whale_tracker._global_client = None
    yield container


def _make_fgi_client(
    *, value: int = 55, classification: str = "Greed"
) -> Any:
    snapshot = SimpleNamespace(classification=classification)
    client = SimpleNamespace(
        cached_value=value,
        _latest_cache=(snapshot, 0.0),
    )
    return client


def _make_coingecko_client(
    *,
    btc_dominance: float = 51.234,
    market_cap_change_24h: float = 1.23,
    total_volume_change_24h: float = -0.45,
) -> Any:
    snap = SimpleNamespace(
        btc_dominance=btc_dominance,
        market_cap_change_24h=market_cap_change_24h,
        total_volume_change_24h=total_volume_change_24h,
    )
    return SimpleNamespace(cached_value=snap)


def _make_fred_client(
    *, fed_funds_rate: float = 5.25, yield_curve_spread: float = -0.42
) -> Any:
    snap = SimpleNamespace(
        fed_funds_rate=fed_funds_rate,
        yield_curve_spread=yield_curve_spread,
    )
    return SimpleNamespace(cached_value=snap)


def _make_whale_client(
    *, net_flow: float = 12345.678, large_tx_count: int = 9
) -> Any:
    snap = SimpleNamespace(
        net_flow=net_flow,
        large_tx_count=large_tx_count,
    )
    return SimpleNamespace(cached_value=snap)


# ---------------------------------------------------------------------------
# GET /api/v1/signals/current
# ---------------------------------------------------------------------------


class TestGetCurrentSignalsContract:
    """Every response must carry all 9 keys, independent of client state."""

    def test_returns_200_with_all_fields_null_when_no_clients(
        self, client_dev: TestClient, clean_signal_container: Any
    ) -> None:
        response = client_dev.get("/api/v1/signals/current")
        assert response.status_code == 200
        body = response.json()
        assert set(body.keys()) == _EXPECTED_KEYS
        non_null = {k: v for k, v in body.items() if v is not None}
        assert non_null == {}, f"expected all fields None, got non-null: {non_null}"


class TestGetCurrentSignalsContainerPath:
    """When ``app.state.container.services.<client>`` is populated the router
    MUST prefer the container lookup over the module-global fallback."""

    def test_returns_fgi_values_from_container(
        self, client_dev: TestClient, app_dev_mode: Any
    ) -> None:
        container = app_dev_mode.state.container
        container.services.fgi_client = _make_fgi_client(
            value=73, classification="Greed"
        )

        response = client_dev.get("/api/v1/signals/current")
        body = response.json()
        assert body["fearGreedIndex"] == 73
        assert body["fearGreedClassification"] == "Greed"

    def test_returns_coingecko_values_from_container(
        self, client_dev: TestClient, app_dev_mode: Any
    ) -> None:
        container = app_dev_mode.state.container
        container.services.coingecko_client = _make_coingecko_client(
            btc_dominance=52.789, market_cap_change_24h=2.5
        )

        response = client_dev.get("/api/v1/signals/current")
        body = response.json()
        assert body["btcDominance"] == 52.79
        assert body["marketCapChange24h"] == 2.5

    def test_returns_fred_values_from_container(
        self, client_dev: TestClient, app_dev_mode: Any
    ) -> None:
        container = app_dev_mode.state.container
        container.services.fred_client = _make_fred_client(
            fed_funds_rate=5.5, yield_curve_spread=-0.37
        )

        response = client_dev.get("/api/v1/signals/current")
        body = response.json()
        assert body["fedFundsRate"] == 5.5
        assert body["yieldCurveSpread"] == -0.37

    def test_returns_whale_values_from_container(
        self, client_dev: TestClient, app_dev_mode: Any
    ) -> None:
        container = app_dev_mode.state.container
        container.services.whale_alert_client = _make_whale_client(
            net_flow=98765.4321, large_tx_count=17
        )

        response = client_dev.get("/api/v1/signals/current")
        body = response.json()
        assert body["whaleNetFlow"] == 98765.0
        assert body["whaleTxCount"] == 17


class TestGetCurrentSignalsFallbackToGlobalClient:
    """When the container has no client set the router falls back to the
    legacy ``data.*.get_global_client()`` shim.  The shim is still used by
    startup warmup and by strategies running outside a request context."""

    def test_falls_back_to_fgi_global_client(
        self, client_dev: TestClient, clean_signal_container: Any
    ) -> None:
        from data import sentiment

        sentiment._global_client = _make_fgi_client(
            value=18, classification="Extreme Fear"
        )

        response = client_dev.get("/api/v1/signals/current")
        body = response.json()
        assert body["fearGreedIndex"] == 18
        assert body["fearGreedClassification"] == "Extreme Fear"

    def test_falls_back_to_whale_global_client(
        self, client_dev: TestClient, clean_signal_container: Any
    ) -> None:
        from data import whale_tracker

        whale_tracker._global_client = _make_whale_client(
            net_flow=1000.0, large_tx_count=3
        )

        response = client_dev.get("/api/v1/signals/current")
        body = response.json()
        assert body["whaleNetFlow"] == 1000.0
        assert body["whaleTxCount"] == 3


class TestGetCurrentSignalsResilience:
    """A failing client must NEVER prevent the endpoint from returning 200."""

    def test_one_failing_client_does_not_affect_others(
        self, client_dev: TestClient, app_dev_mode: Any
    ) -> None:
        container = app_dev_mode.state.container

        # Healthy coingecko + whale, broken FGI (raises on attr access)
        class _BrokenFgi:
            @property
            def cached_value(self) -> int:
                raise RuntimeError("simulated upstream failure")

        container.services.fgi_client = _BrokenFgi()
        container.services.coingecko_client = _make_coingecko_client()
        container.services.whale_alert_client = _make_whale_client()

        response = client_dev.get("/api/v1/signals/current")
        assert response.status_code == 200
        body = response.json()
        assert body["fearGreedIndex"] is None  # FGI swallowed
        assert body["btcDominance"] is not None  # coingecko survived
        assert body["whaleNetFlow"] is not None  # whale survived

    def test_returns_200_when_client_has_no_cache_populated(
        self, client_dev: TestClient, app_dev_mode: Any
    ) -> None:
        """A client that exists but has not yet warmed up must yield None,
        not raise."""
        container = app_dev_mode.state.container
        # cached_value=None simulates a not-yet-warmed-up client
        container.services.fgi_client = SimpleNamespace(
            cached_value=None, _latest_cache=None
        )

        response = client_dev.get("/api/v1/signals/current")
        assert response.status_code == 200
        assert response.json()["fearGreedIndex"] is None
