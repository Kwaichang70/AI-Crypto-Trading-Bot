"""
tests/unit/test_run_recovery_scan_helpers.py
------------------------------------------------
Unit tests for ``apps.api.services.run_recovery._fetch_prefixed_orders``
(WP1.8b round 2 security findings WP18b-S-05 and WP18b-C-01).

Pure/fast: a minimal stub exchange (only ``fetch_orders`` is called by
this helper) instead of the full ``FakeCCXTExchange`` fixture.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, ClassVar
from urllib.parse import parse_qs, urlsplit
from uuid import uuid4

import ccxt.async_support as ccxt_async
import pytest
import structlog

from api.services.run_recovery import (
    _fetch_prefixed_orders,
    _raise_if_scan_truncated,
    _targeted_cid_lookup,
)
from trading.recovery import ResumeRejected

SYMBOL = "BTC/USD"
_log = structlog.get_logger(__name__)


class _StubExchange:
    def __init__(self, orders: list[dict[str, Any]]) -> None:
        self._orders = orders
        self.calls: list[dict[str, Any]] = []

    async def fetch_orders(
        self,
        symbol: str,
        since: int | None = None,
        limit: int | None = None,
        params: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        # WP1.4b round 3 (S-R2-01): production now always passes
        # limit=None explicitly (the real ccxt Coinbase adapter defaults
        # it to 100 and truncates oldest-first) -- accept (and record) it
        # here so this stub's signature matches every real call site.
        self.calls.append({"symbol": symbol, "since": since, "limit": limit, "params": params})
        # WP1.4b round 3 (S-R2-02): honour since/until so the structural
        # truncation probe (_raise_if_scan_truncated's own follow-up
        # fetch_orders call, params["until"] = min_ts - 1) sees ground
        # truth instead of the unfiltered fixture -- every OTHER test in
        # this module seeds orders at/after ``since`` with no ``until``,
        # so this is a no-op for them.
        until = (params or {}).get("until") if params else None
        return [
            o for o in self._orders
            if (since is None or o["timestamp"] >= since)
            and (until is None or o["timestamp"] <= until)
        ]


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


# ---------------------------------------------------------------------------
# WP1.4b round 3 (S-R2-02): the structural truncation probe -- a scan
# result BELOW the count-based cap can still be truncated if the real
# exchange's pagination silently drops entries older than what it returned
# (a cursor bug, or a page smaller than the assumed maxEntriesPerRequest).
# ---------------------------------------------------------------------------


class _StructuralProbeExchange:
    """Controls exactly two things a real exchange call site distinguishes:
    the PRIMARY listing (no ``params["until"]``) and the structural probe
    (``params["until"]`` set) -- returns ``probe_result`` for the latter
    regardless of its own ``since``/``limit`` arguments, so a test can
    assert the exact "something older exists" / "nothing older" outcome
    without needing a full timestamp-filtering fixture.
    """

    def __init__(self, probe_result: list[dict[str, Any]]) -> None:
        self.options: dict[str, Any] = {"paginationCalls": 10, "maxEntriesPerRequest": 1000}
        self._probe_result = probe_result
        self.probe_calls: list[dict[str, Any]] = []

    async def fetch_orders(
        self,
        symbol: str,
        since: int | None = None,
        limit: int | None = None,
        params: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        until = (params or {}).get("until")
        assert until is not None, "only the structural probe calls fetch_orders here"
        self.probe_calls.append({"symbol": symbol, "since": since, "limit": limit, "until": until})
        return list(self._probe_result)


class _RaisingStructuralProbeExchange:
    options: ClassVar[dict[str, Any]] = {"paginationCalls": 10, "maxEntriesPerRequest": 1000}

    async def fetch_orders(self, *args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        raise RuntimeError("probe transport error")


class TestStructuralTruncationProbe:
    """WP1.4b round 3 (S-R2-02): below the count cap, a non-empty probe
    (something strictly older than our oldest entry, still inside the
    window) means the paginated fetch never reached it -- definite
    truncation regardless of the assumed page size."""

    async def test_nonempty_probe_raises_truncated(self) -> None:
        raw_orders = [_order("cid-a", timestamp=1_767_225_600_000)]
        older = _order("cid-older", timestamp=1_767_225_500_000)
        exchange = _StructuralProbeExchange(probe_result=[older])

        with pytest.raises(ResumeRejected) as exc_info:
            await _raise_if_scan_truncated(
                raw_orders, exchange, log=_log, symbol=SYMBOL, since_ms=0,
            )
        assert exc_info.value.reason == "exchange_scan_truncated"

        # The probe itself must be single-page (no re-pagination) and
        # scoped to strictly before the oldest entry we already have.
        assert len(exchange.probe_calls) == 1
        assert exchange.probe_calls[0]["until"] == 1_767_225_600_000 - 1
        assert exchange.probe_calls[0]["limit"] == 1

    async def test_empty_probe_does_not_raise(self) -> None:
        raw_orders = [_order("cid-a", timestamp=1_767_225_600_000)]
        exchange = _StructuralProbeExchange(probe_result=[])

        await _raise_if_scan_truncated(
            raw_orders, exchange, log=_log, symbol=SYMBOL, since_ms=0,
        )  # must not raise

    async def test_empty_raw_orders_skips_probe_entirely(self) -> None:
        """An empty (not merely below-cap) result has no "oldest entry" to
        probe below -- the structural check is a no-op, never a spurious
        AssertionError from ``min()`` on an empty sequence."""
        exchange = _StructuralProbeExchange(probe_result=[_order("cid-x")])

        await _raise_if_scan_truncated(
            [], exchange, log=_log, symbol=SYMBOL, since_ms=0,
        )
        assert exchange.probe_calls == []

    async def test_probe_failure_is_fail_closed(self) -> None:
        raw_orders = [_order("cid-a", timestamp=1_767_225_600_000)]
        with pytest.raises(ResumeRejected) as exc_info:
            await _raise_if_scan_truncated(
                raw_orders, _RaisingStructuralProbeExchange(), log=_log, symbol=SYMBOL, since_ms=0,
            )
        assert exc_info.value.reason == "exchange_scan_failed"

    async def test_count_at_cap_still_raises_before_structural_check(self) -> None:
        """The count-based check (round 2) still fires first/independently
        of the structural check -- a probe is never even attempted once
        the cap itself is reached."""
        exchange = _StructuralProbeExchange(probe_result=[])
        exchange.options = {"paginationCalls": 1, "maxEntriesPerRequest": 1}
        raw_orders = [_order("cid-a", timestamp=1_767_225_600_000)]  # count(1) >= cap(1)

        with pytest.raises(ResumeRejected) as exc_info:
            await _raise_if_scan_truncated(
                raw_orders, exchange, log=_log, symbol=SYMBOL, since_ms=0,
            )
        assert exc_info.value.reason == "exchange_scan_truncated"
        assert exchange.probe_calls == []


# ---------------------------------------------------------------------------
# WP1.4b round 3 (S-R2-03): a FAILED targeted cid lookup must fail closed
# (ResumeRejected) instead of silently returning None (which the caller
# would otherwise treat as definitive "not found" evidence).
# ---------------------------------------------------------------------------


class TestTargetedCidLookupFailsClosed:
    async def test_fetch_orders_exception_raises_instead_of_none(self) -> None:
        from datetime import UTC, datetime

        with pytest.raises(ResumeRejected) as exc_info:
            await _targeted_cid_lookup(
                _RaisingExchange(),
                SYMBOL,
                "some-cid",
                created_at=datetime.now(tz=UTC),
                log=_log,
            )
        assert exc_info.value.reason == "exchange_scan_failed"

    async def test_genuinely_absent_still_returns_none(self) -> None:
        """Negative control: a REACHABLE exchange that simply has no
        matching order still returns ``None`` (only a failure fails
        closed, not a clean empty answer)."""
        from datetime import UTC, datetime

        exchange = _StubExchange([_order("some-other-cid")])
        result = await _targeted_cid_lookup(
            exchange, SYMBOL, "the-cid-we-want", created_at=datetime.now(tz=UTC), log=_log,
        )
        assert result is None


# ---------------------------------------------------------------------------
# WP1.4b round 3 (S-R2-01, resume side): the real ccxt Coinbase adapter's
# ``fetch_orders`` defaults ``limit`` to 100 and its paginated path keeps
# the OLDEST ``limit`` entries -- proven end-to-end against the real ccxt
# pagination/filtering code (not a re-implementation of it), the same way
# ``tests/unit/test_wp14b_coinbase_cid.py`` proves the LIVE ENGINE side.
# This is the RESUME side: ``_fetch_prefixed_orders`` (the primary scan
# helper ``scan_and_import`` calls per symbol) must still find this run's
# own order among 150 total orders in-window when it is the single NEWEST
# entry.
# ---------------------------------------------------------------------------

_RESUME_MARKET: dict[str, Any] = {
    "id": "BTC-USD",
    "symbol": SYMBOL,
    "base": "BTC",
    "quote": "USD",
    "type": "spot",
    "spot": True,
    "precision": {"amount": 8, "price": 2},
    "limits": {"amount": {"min": 0.0001}, "cost": {"min": 1}},
}


def _iso(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _ms_from_iso(s: str) -> int:
    return int(datetime.strptime(s, "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=UTC).timestamp() * 1000)


def _raw_coinbase_order(
    *, exchange_id: str, ms: int, cid: str, status: str = "FILLED",
) -> dict[str, Any]:
    return {
        "order_id": exchange_id,
        "product_id": _RESUME_MARKET["id"],
        "side": "BUY",
        "client_order_id": cid,
        "status": status,
        "order_configuration": {"market_market_ioc": {"base_size": "0.01"}},
        "created_time": _iso(ms),
        "filled_size": "0.01",
        "average_filled_price": "50000",
        "total_fees": "0.5",
    }


def _make_real_coinbase_exchange(orders_response: list[dict[str, Any]]) -> Any:
    exchange = ccxt_async.coinbase({
        "enableRateLimit": False,
        "apiKey": "wp14b-test-key",
        "secret": "wp14b-test-secret",
    })
    exchange.markets = {SYMBOL: dict(_RESUME_MARKET)}
    exchange.options["brokerId"] = "ccxt"

    async def _fake_fetch(
        url: str, method: str = "GET", headers: Any = None, body: Any = None,
    ) -> dict[str, Any]:
        # WP1.4b round 3 (S-R2-01/S-R2-02): the real (unmocked)
        # ``fetch_orders`` builds ``start_date``/``end_date``/``limit``
        # into the request's query string -- ``_raise_if_scan_truncated``'s
        # own structural-truncation probe (a SECOND, narrower GET) depends
        # on this stub actually honouring them like a real server would,
        # not just echoing back every order regardless of the window
        # asked for.
        assert method == "GET"  # this helper never places orders
        query = parse_qs(urlsplit(url).query)
        start_ms = _ms_from_iso(query["start_date"][0]) if "start_date" in query else None
        end_ms = _ms_from_iso(query["end_date"][0]) if "end_date" in query else None
        server_limit = int(query["limit"][0]) if "limit" in query else None

        filtered = [
            o for o in orders_response
            if (start_ms is None or _ms_from_iso(o["created_time"]) >= start_ms)
            and (end_ms is None or _ms_from_iso(o["created_time"]) <= end_ms)
        ]
        filtered.sort(key=lambda o: o["created_time"])
        if server_limit is not None:
            filtered = filtered[:server_limit]

        return {
            "orders": filtered,
            "sequence": "0",
            "has_next": False,
            "cursor": "",
        }

    exchange.fetch = _fake_fetch
    return exchange


class TestRealCoinbaseFetchOrdersBeyondLimit100:
    async def test_fetch_prefixed_orders_finds_newest_order_beyond_limit_100(self) -> None:
        run_id = uuid4()
        prefix = f"{run_id}-"
        cid = f"{prefix}{uuid4().hex[:12]}"

        now_ms = int(datetime.now(tz=UTC).timestamp() * 1000)
        since_ms = now_ms - 3_600_000  # matches _fetch_prefixed_orders' own margin usage

        # 150 foreign filler orders (a different run's prefix), oldest
        # first -- under the buggy limit=100 default these are exactly the
        # 100 that WOULD be returned, burying our own (newest) order.
        fillers = [
            _raw_coinbase_order(
                exchange_id=f"filler-{i:04d}",
                ms=since_ms + 1_000 + i * 1_000,
                cid=f"{uuid4()}-{uuid4().hex[:12]}",
            )
            for i in range(150)
        ]
        target = _raw_coinbase_order(
            exchange_id="target-exch-id", ms=now_ms, cid=cid, status="OPEN",
        )

        exchange = _make_real_coinbase_exchange([*fillers, target])

        matched, foreign = await _fetch_prefixed_orders(
            exchange, SYMBOL, since_ms=since_ms, prefix=prefix, log=_log,
        )

        assert set(matched.keys()) == {cid}
        assert matched[cid]["id"] == "target-exch-id"
        assert foreign == 150

        await exchange.close()
