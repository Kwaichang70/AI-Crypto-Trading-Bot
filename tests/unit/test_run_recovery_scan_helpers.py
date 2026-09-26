"""
tests/unit/test_run_recovery_scan_helpers.py
------------------------------------------------
Unit tests for ``apps.api.services.run_recovery._fetch_prefixed_orders``
(WP1.8b round 2 security findings WP18b-S-05 and WP18b-C-01).

Pure/fast: a minimal stub exchange (only ``fetch_orders`` is called by
this helper) instead of the full ``FakeCCXTExchange`` fixture.
"""

from __future__ import annotations

from typing import Any
from uuid import uuid4

import pytest
import structlog

from api.services.run_recovery import _fetch_prefixed_orders
from trading.recovery import ResumeRejected

SYMBOL = "BTC/USD"
_log = structlog.get_logger(__name__)


class _StubExchange:
    def __init__(self, orders: list[dict[str, Any]]) -> None:
        self._orders = orders
        self.calls: list[dict[str, Any]] = []

    async def fetch_orders(
        self, symbol: str, since: int | None = None, params: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        self.calls.append({"symbol": symbol, "since": since, "params": params})
        return list(self._orders)


class _RaisingExchange:
    async def fetch_orders(self, *args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        raise RuntimeError("boom")


def _order(client_order_id: str | None, **overrides: Any) -> dict[str, Any]:
    base = {
        "id": f"ex-{uuid4().hex[:8]}",
        "clientOrderId": client_order_id,
        "symbol": SYMBOL,
        "side": "buy",
        "type": "market",
        "amount": 0.01,
        "price": 50000.0,
        "average": 50000.0,
        "status": "closed",
        "filled": 0.01,
        "timestamp": 1_767_225_600_000,
    }
    base.update(overrides)
    return base


class TestPrefixExactShapeMatch:
    """WP18b-S-05: only ``f"{prefix}{12 hex chars}"`` counts as this run's
    own order -- a prefix-only match is foreign (never cancelled/imported)."""

    async def test_real_shape_matches(self) -> None:
        run_id = uuid4()
        prefix = f"{run_id}-"
        cid = f"{prefix}{uuid4().hex[:12]}"
        exchange = _StubExchange([_order(cid)])

        matched, foreign = await _fetch_prefixed_orders(
            exchange, SYMBOL, since_ms=0, prefix=prefix, log=_log
        )

        assert set(matched.keys()) == {cid}
        assert foreign == 0

    async def test_prefix_only_spoofed_id_is_foreign(self) -> None:
        """The PA/S-05 evidence case: a manually crafted clientOrderId that
        starts with the run's prefix but does not match the engine's own
        ``{12 hex chars}`` suffix shape must be treated as foreign."""
        run_id = uuid4()
        prefix = f"{run_id}-"
        spoofed_cid = f"{prefix}SPOOFED-by-manual-api-call"
        exchange = _StubExchange([_order(spoofed_cid)])

        matched, foreign = await _fetch_prefixed_orders(
            exchange, SYMBOL, since_ms=0, prefix=prefix, log=_log
        )

        assert matched == {}
        assert foreign == 1

    async def test_uppercase_suffix_is_foreign(self) -> None:
        run_id = uuid4()
        prefix = f"{run_id}-"
        cid = f"{prefix}{uuid4().hex[:12].upper()}"
        exchange = _StubExchange([_order(cid)])

        matched, foreign = await _fetch_prefixed_orders(
            exchange, SYMBOL, since_ms=0, prefix=prefix, log=_log
        )

        assert matched == {}
        assert foreign == 1

    async def test_wrong_length_suffix_is_foreign(self) -> None:
        run_id = uuid4()
        prefix = f"{run_id}-"
        cid = f"{prefix}{uuid4().hex[:11]}"  # 11 hex chars, not 12
        exchange = _StubExchange([_order(cid)])

        matched, foreign = await _fetch_prefixed_orders(
            exchange, SYMBOL, since_ms=0, prefix=prefix, log=_log
        )

        assert matched == {}
        assert foreign == 1

    async def test_no_client_order_id_is_foreign(self) -> None:
        exchange = _StubExchange([_order(None)])
        matched, foreign = await _fetch_prefixed_orders(
            exchange, SYMBOL, since_ms=0, prefix=f"{uuid4()}-", log=_log
        )
        assert matched == {}
        assert foreign == 1

    async def test_different_runs_prefix_is_foreign(self) -> None:
        other_run_id = uuid4()
        cid = f"{other_run_id}-{uuid4().hex[:12]}"
        exchange = _StubExchange([_order(cid)])
        matched, foreign = await _fetch_prefixed_orders(
            exchange, SYMBOL, since_ms=0, prefix=f"{uuid4()}-", log=_log
        )
        assert matched == {}
        assert foreign == 1


class TestDuplicateClientOrderIdDedup:
    """WP18b-C-01: the same order appearing twice in a paginated
    ``fetch_orders`` response (a stale snapshot followed by an updated
    one) must collapse to a single matched entry."""

    async def test_duplicate_entries_collapse_to_one(self) -> None:
        run_id = uuid4()
        prefix = f"{run_id}-"
        cid = f"{prefix}{uuid4().hex[:12]}"
        stale = _order(cid, status="open", filled=0.0)
        updated = _order(cid, status="closed", filled=0.01)
        # Same id, different snapshot -- simulates ccxt's paginate
        # returning the same order across two overlapping pages.
        updated["id"] = stale["id"]
        exchange = _StubExchange([stale, updated])

        matched, foreign = await _fetch_prefixed_orders(
            exchange, SYMBOL, since_ms=0, prefix=prefix, log=_log
        )

        assert len(matched) == 1
        assert foreign == 0
        # dict insertion order means the LAST entry for a given key wins --
        # the more up-to-date "closed" snapshot is what scan_and_import
        # will actually process.
        assert matched[cid]["status"] == "closed"

    async def test_three_duplicates_still_collapse_to_one(self) -> None:
        run_id = uuid4()
        prefix = f"{run_id}-"
        cid = f"{prefix}{uuid4().hex[:12]}"
        exchange = _StubExchange([_order(cid), _order(cid), _order(cid)])

        matched, _foreign = await _fetch_prefixed_orders(
            exchange, SYMBOL, since_ms=0, prefix=prefix, log=_log
        )

        assert len(matched) == 1


class TestFetchOrdersFailure:
    async def test_fetch_orders_exception_rejected(self) -> None:
        with pytest.raises(ResumeRejected) as exc_info:
            await _fetch_prefixed_orders(
                _RaisingExchange(), SYMBOL, since_ms=0, prefix=f"{uuid4()}-", log=_log
            )
        assert exc_info.value.reason == "exchange_scan_failed"
