"""
tests/unit/test_whale_tracker.py
----------------------------------
Unit tests for the Whale Alert on-chain transaction monitoring client.

Covers:
- WhaleFlowSnapshot Pydantic validation (frozen, non-negative count)
- WhaleAlertClient cache hit / miss / stale fallback / total failure
- WhaleAlertClient _aggregate_flow: inflow, outflow, mixed, empty
- WhaleAlertClient _is_exchange_address: owner_type, owner name
- Module-level singleton set/get helpers
"""

from __future__ import annotations

import time
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from data.whale_tracker import (
    WhaleAlertClient,
    WhaleFlowSnapshot,
    get_global_client,
    set_global_client,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_snapshot(
    net_flow: float = 0.0,
    large_tx_count: int = 0,
) -> WhaleFlowSnapshot:
    return WhaleFlowSnapshot(
        net_flow=net_flow,
        large_tx_count=large_tx_count,
        timestamp=datetime.now(UTC),
    )


def _make_tx(
    amount_usd: float,
    to_owner_type: str | None = None,
    to_owner: str | None = None,
    from_owner_type: str | None = None,
    from_owner: str | None = None,
) -> dict:
    tx: dict = {"amount_usd": amount_usd}
    if to_owner_type is not None or to_owner is not None:
        tx["to"] = {}
        if to_owner_type is not None:
            tx["to"]["owner_type"] = to_owner_type
        if to_owner is not None:
            tx["to"]["owner"] = to_owner
    if from_owner_type is not None or from_owner is not None:
        tx["from"] = {}
        if from_owner_type is not None:
            tx["from"]["owner_type"] = from_owner_type
        if from_owner is not None:
            tx["from"]["owner"] = from_owner
    return tx


# ---------------------------------------------------------------------------
# TestWhaleFlowSnapshot
# ---------------------------------------------------------------------------

class TestWhaleFlowSnapshot:
    def test_valid_snapshot_positive_flow(self) -> None:
        snap = _make_snapshot(net_flow=5_000_000.0, large_tx_count=3)
        assert snap.net_flow == 5_000_000.0
        assert snap.large_tx_count == 3

    def test_valid_snapshot_negative_flow(self) -> None:
        snap = _make_snapshot(net_flow=-8_000_000.0, large_tx_count=5)
        assert snap.net_flow == -8_000_000.0

    def test_zero_flow(self) -> None:
        snap = _make_snapshot(net_flow=0.0, large_tx_count=0)
        assert snap.net_flow == 0.0
        assert snap.large_tx_count == 0

    def test_large_tx_count_cannot_be_negative(self) -> None:
        with pytest.raises(Exception):
            _make_snapshot(large_tx_count=-1)

    def test_frozen(self) -> None:
        snap = _make_snapshot()
        with pytest.raises(Exception):
            snap.net_flow = 999.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# TestWhaleAlertClientCache
# ---------------------------------------------------------------------------

class TestWhaleAlertClientCache:
    @pytest.mark.asyncio
    async def test_cache_hit_skips_api(self) -> None:
        client = WhaleAlertClient(api_key="test_key", cache_ttl_seconds=3600)
        cached_snap = _make_snapshot(net_flow=-2_000_000.0)
        client._latest_cache = (cached_snap, time.monotonic())

        snap = await client.get_latest()
        assert snap is cached_snap

    @pytest.mark.asyncio
    async def test_stale_cache_fallback_on_failure(self) -> None:
        client = WhaleAlertClient(api_key="test_key", cache_ttl_seconds=0.001)
        stale_snap = _make_snapshot(net_flow=10_000_000.0)
        client._latest_cache = (stale_snap, time.monotonic() - 10)

        mock_session = MagicMock()
        mock_session.closed = False
        mock_session.get = MagicMock(side_effect=RuntimeError("network error"))
        client._session = mock_session

        snap = await client.get_latest()
        assert snap is stale_snap

    @pytest.mark.asyncio
    async def test_total_failure_returns_none(self) -> None:
        client = WhaleAlertClient(api_key="test_key", cache_ttl_seconds=60)
        mock_session = MagicMock()
        mock_session.closed = False
        mock_session.get = MagicMock(side_effect=RuntimeError("network error"))
        client._session = mock_session

        snap = await client.get_latest()
        assert snap is None

    def test_cached_value_property_none_initially(self) -> None:
        client = WhaleAlertClient(api_key="test_key")
        assert client.cached_value is None

    def test_cached_value_property_returns_snapshot(self) -> None:
        client = WhaleAlertClient(api_key="test_key")
        snap = _make_snapshot(net_flow=-3_000_000.0)
        client._latest_cache = (snap, time.monotonic())
        assert client.cached_value is snap


# ---------------------------------------------------------------------------
# TestWhaleAlertClientAggregation
# ---------------------------------------------------------------------------

class TestWhaleAlertClientAggregation:
    def setup_method(self) -> None:
        self.client = WhaleAlertClient(api_key="test_key", min_value_usd=1_000_000)

    def test_empty_transactions(self) -> None:
        snap = self.client._aggregate_flow({"transactions": []})
        assert snap.net_flow == 0.0
        assert snap.large_tx_count == 0

    def test_inflow_to_exchange(self) -> None:
        tx = _make_tx(amount_usd=5_000_000.0, to_owner_type="exchange")
        snap = self.client._aggregate_flow({"transactions": [tx]})
        assert snap.net_flow > 0  # positive = to exchange
        assert snap.large_tx_count == 1

    def test_outflow_from_exchange(self) -> None:
        tx = _make_tx(amount_usd=5_000_000.0, from_owner_type="exchange")
        snap = self.client._aggregate_flow({"transactions": [tx]})
        assert snap.net_flow < 0  # negative = from exchange
        assert snap.large_tx_count == 1

    def test_mixed_flows_net_calculation(self) -> None:
        # 8M inflow, 3M outflow → net = +5M
        tx_in = _make_tx(amount_usd=8_000_000.0, to_owner_type="exchange")
        tx_out = _make_tx(amount_usd=3_000_000.0, from_owner_type="exchange")
        snap = self.client._aggregate_flow({"transactions": [tx_in, tx_out]})
        assert abs(snap.net_flow - 5_000_000.0) < 1.0

    def test_below_min_value_excluded(self) -> None:
        tx = _make_tx(amount_usd=500_000.0, to_owner_type="exchange")
        snap = self.client._aggregate_flow({"transactions": [tx]})
        assert snap.net_flow == 0.0
        assert snap.large_tx_count == 0

    def test_non_exchange_tx_not_counted_in_flow(self) -> None:
        # Wallet-to-wallet: neither address is an exchange
        tx = _make_tx(amount_usd=10_000_000.0)
        snap = self.client._aggregate_flow({"transactions": [tx]})
        # No exchange address detected — flow stays 0 but count increments
        assert snap.net_flow == 0.0

    def test_is_exchange_by_owner_type(self) -> None:
        assert self.client._is_exchange_address({"owner_type": "exchange"}) is True

    def test_is_exchange_by_owner_name(self) -> None:
        assert self.client._is_exchange_address({"owner": "Binance"}) is True

    def test_not_exchange(self) -> None:
        assert self.client._is_exchange_address({"owner_type": "unknown", "owner": "satoshi"}) is False

    def test_none_address_not_exchange(self) -> None:
        assert self.client._is_exchange_address(None) is False


# ---------------------------------------------------------------------------
# TestWhaleAlertClientSingleton
# ---------------------------------------------------------------------------

class TestWhaleAlertClientSingleton:
    def test_get_returns_none_before_set(self) -> None:
        import data.whale_tracker as wt
        original = wt._global_client
        wt._global_client = None
        try:
            assert get_global_client() is None
        finally:
            wt._global_client = original

    def test_set_and_get_round_trip(self) -> None:
        import data.whale_tracker as wt
        original = wt._global_client
        client = WhaleAlertClient(api_key="test_key")
        try:
            set_global_client(client)
            assert get_global_client() is client
        finally:
            wt._global_client = original
