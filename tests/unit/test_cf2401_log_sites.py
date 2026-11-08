"""
tests/unit/test_cf2401_log_sites.py
-----------------------------------
CF-24-01 commit P1 / T-03 (CF-24-02, CF01-SEC-04): the 11 provider and
retraining ``except`` sites log ``**exc_log_fields(exc)`` and never
``error=str(exc)``.

Layout (so the retraining half can be dropped on its own):

* ``TestP1aProviderSites``     4 sites: sentiment.py x2, market_signals.py x2.
* ``TestP1bRetrainingSites``   7 sites: apps/api/services/retraining.py.

Every site test runs twice through the production logging chain
(``tests/_cf36_logging.production_logging``), as in ``test_cf36_sites.py``:

``layer1_off``  the redactor is removed, so a pass proves the SITE FIX itself.
``layer1_on``   the full production chain.

The exception text embeds a signed URL, a host and a response body.  At INFO
none of it may reach the sink, and each leak assertion is paired with a
presence assertion (event name and ``error_type``).  A separate test per class
runs at DEBUG with layer 1 on and checks that the ``<event>.trace`` event is
emitted and the signature never appears (layer 1 is the backstop for traces).
"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import aiohttp
import pytest

from tests._cf36_logging import SENTINEL as S
from tests._cf36_logging import Capture, ConfigureFn, disable_layer1, json_lines

SIG = f"{S}SIG0123456789abcdef"
HOST = "api.cf2401-provider.example"
URL = f"https://{HOST}/api/v3/data?limit=1&signature={SIG}"
BODY = f"{S}-RESPONSE-BODY-0123"
LEAK_TEXT = f'GET {URL} 502 Bad Gateway {{"msg":"{BODY}"}}'
LEAK_MARKERS = (SIG, HOST, BODY, S, "502 Bad Gateway", LEAK_TEXT)
_ERROR_AT = re.compile(r"^[\w.\-]+\.py:\d+:[\w<>]+$")


class _SiteLog:
    def __init__(self, mode: str, production_logging: ConfigureFn) -> None:
        self.mode = mode
        self._production_logging = production_logging

    def __call__(self, *, level: str = "INFO") -> Capture:
        cap = self._production_logging(json_output=True, log_level=level)
        if self.mode == "layer1_off":
            disable_layer1()
        return cap


@pytest.fixture(params=["layer1_off", "layer1_on"])
def site_log(request: pytest.FixtureRequest, production_logging: ConfigureFn) -> _SiteLog:
    return _SiteLog(request.param, production_logging)


def _events(cap: Capture, name: str) -> list[dict[str, Any]]:
    return [e for e in json_lines(cap.out) if e.get("event") == name]


def _assert_site(
    cap: Capture, name: str, error_type: str, *, status: int | None = None
) -> dict[str, Any]:
    evs = _events(cap, name)
    assert len(evs) == 1, f"event {name!r} not emitted exactly once; got {cap.out[:500]}"
    text = cap.text
    assert [m for m in LEAK_MARKERS if m in text] == [], "exception text reached the sink"
    ev = evs[0]
    assert ev["error_type"] == error_type
    assert "error" not in ev and "exception" not in ev and "exc_info" not in ev
    if status is not None:
        assert ev["http_status"] == status
    assert _ERROR_AT.match(ev["error_at"]), ev.get("error_at")
    # INFO drops the DEBUG trace before any processor
    assert _events(cap, f"{name}.trace") == []
    return ev


def _assert_trace_redacted(cap: Capture, name: str) -> None:
    """DEBUG + layer 1 on: the trace event exists, the signed-URL secret does not."""
    assert len(_events(cap, f"{name}.trace")) == 1, f"{name}.trace not emitted: {cap.out[:500]}"
    assert SIG not in cap.text


def _client_response_error() -> aiohttp.ClientResponseError:
    return aiohttp.ClientResponseError(
        request_info=MagicMock(), history=(), status=502, message=LEAK_TEXT
    )


class _Leaky(Exception):
    """Non-I/O exception whose text carries the leak (KeyError/ValueError shape)."""


# ---------------------------------------------------------------------------
# P1a: provider sites (mandatory)
# ---------------------------------------------------------------------------


class TestP1aProviderSites:
    async def test_sentiment_latest(self, site_log: _SiteLog) -> None:
        from data.sentiment import FearGreedClient

        cap = site_log()
        client = FearGreedClient()
        client._get_session = AsyncMock(side_effect=_client_response_error())
        assert await client.get_latest() is None
        ev = _assert_site(cap, "fear_greed_client.fetch_failed", "ClientResponseError", status=502)
        assert ev["type"] == "latest"

    async def test_sentiment_history(self, site_log: _SiteLog) -> None:
        from data.sentiment import FearGreedClient

        cap = site_log()
        client = FearGreedClient()
        client._get_session = AsyncMock(side_effect=_client_response_error())
        assert await client.get_history() == []
        ev = _assert_site(cap, "fear_greed_client.fetch_failed", "ClientResponseError", status=502)
        assert ev["type"] == "history"

    async def test_market_signals_latest(self, site_log: _SiteLog) -> None:
        from data.market_signals import CoinGeckoClient

        cap = site_log()
        client = CoinGeckoClient()
        client._get_session = AsyncMock(side_effect=_client_response_error())
        assert await client.get_latest() is None
        _assert_site(cap, "coingecko_client.fetch_failed", "ClientResponseError", status=502)

    async def test_market_signals_btc_dom_history(self, site_log: _SiteLog) -> None:
        from data.market_signals import CoinGeckoClient

        cap = site_log()
        client = CoinGeckoClient()
        client._get_session = AsyncMock(side_effect=_Leaky(LEAK_TEXT))
        await client.fetch_btc_dominance_history(days=30)
        _assert_site(cap, "coingecko_client.btc_dom_history_fetch_failed", "_Leaky")

    @pytest.mark.parametrize(
        ("event", "call"),
        [
            ("fear_greed_client.fetch_failed", "fgi_latest"),
            ("fear_greed_client.fetch_failed", "fgi_history"),
            ("coingecko_client.fetch_failed", "cg_latest"),
            ("coingecko_client.btc_dom_history_fetch_failed", "cg_hist"),
        ],
    )
    async def test_debug_trace_is_emitted_and_redacted(
        self, production_logging: ConfigureFn, event: str, call: str
    ) -> None:
        from data.market_signals import CoinGeckoClient
        from data.sentiment import FearGreedClient

        cap = production_logging(json_output=True, log_level="DEBUG")
        exc = _client_response_error()
        if call.startswith("fgi"):
            fgi = FearGreedClient()
            fgi._get_session = AsyncMock(side_effect=exc)
            await (fgi.get_latest() if call == "fgi_latest" else fgi.get_history())
        else:
            cg = CoinGeckoClient()
            cg._get_session = AsyncMock(side_effect=exc)
            if call == "cg_latest":
                await cg.get_latest()
            else:
                await cg.fetch_btc_dominance_history(days=30)
        _assert_trace_redacted(cap, event)


# ---------------------------------------------------------------------------
# P1b: retraining sites (droppable together with this class)
# ---------------------------------------------------------------------------


def _service(tmp_path: Path, **kw: Any) -> Any:
    from api.services.retraining import RetrainingService

    return RetrainingService(
        db_session_factory=MagicMock(),
        model_dir=str(tmp_path),
        min_trades_for_retrain=1,
        **kw,
    )


def _boom(exc: BaseException) -> Any:
    def _raise(*_a: Any, **_k: Any) -> Any:
        raise exc

    return _raise


def _patch_prelude(svc: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """Get ``_do_retrain`` past the OHLCV fetch and the trade query."""
    monkeypatch.setattr(svc, "_fetch_ohlcv_sync", lambda **_k: object())
    monkeypatch.setattr(svc, "_fetch_trade_dicts", AsyncMock(return_value=[{"x": 1}]))


class TestP1bRetrainingSites:
    async def test_poll_loop_check_failed(
        self, site_log: _SiteLog, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cap = site_log()
        svc = _service(tmp_path)
        pairs = AsyncMock(return_value=[("BTC/USDT", "1h")])
        monkeypatch.setattr(svc, "_get_active_symbol_timeframe_pairs", pairs)
        monkeypatch.setattr(svc, "_check_and_retrain", AsyncMock(side_effect=_Leaky(LEAK_TEXT)))
        calls = 0

        async def _sleep(_d: float) -> None:
            nonlocal calls
            calls += 1
            if calls > 1:
                raise asyncio.CancelledError

        monkeypatch.setattr(asyncio, "sleep", _sleep)
        await svc._poll_loop()
        ev = _assert_site(cap, "retraining_service.check_failed", "_Leaky")
        assert ev["symbol"] == "BTC/USDT" and ev["timeframe"] == "1h"

    async def test_ohlcv_fetch_failed(
        self, site_log: _SiteLog, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cap = site_log()
        svc = _service(tmp_path)
        monkeypatch.setattr(svc, "_fetch_ohlcv_sync", _boom(_Leaky(LEAK_TEXT)))
        await svc._do_retrain(symbol="BTC/USDT", timeframe="1h", trigger="auto")
        ev = _assert_site(cap, "retraining_service.ohlcv_fetch_failed", "_Leaky")
        assert ev["symbol"] == "BTC/USDT"

    async def test_fgi_history_fetch_failed(
        self, site_log: _SiteLog, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import data.market_signals as ms
        import data.sentiment as sent

        cap = site_log()
        svc = _service(tmp_path)
        _patch_prelude(svc, monkeypatch)
        monkeypatch.setattr(svc, "_train_sync", _boom(ValueError("fixed")))
        fgi = SimpleNamespace(get_history_as_series=AsyncMock(side_effect=_Leaky(LEAK_TEXT)))
        monkeypatch.setattr(sent, "get_global_client", lambda: fgi)
        monkeypatch.setattr(ms, "get_global_client", lambda: None)
        await svc._do_retrain(symbol="BTC/USDT", timeframe="1h", trigger="auto")
        _assert_site(cap, "retraining_service.fgi_history_fetch_failed", "_Leaky")

    async def test_btc_dom_history_fetch_failed(
        self, site_log: _SiteLog, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import data.market_signals as ms
        import data.sentiment as sent

        cap = site_log()
        svc = _service(tmp_path)
        _patch_prelude(svc, monkeypatch)
        monkeypatch.setattr(svc, "_train_sync", _boom(ValueError("fixed")))
        cg = SimpleNamespace(fetch_btc_dominance_history=AsyncMock(side_effect=_Leaky(LEAK_TEXT)))
        monkeypatch.setattr(sent, "get_global_client", lambda: None)
        monkeypatch.setattr(ms, "get_global_client", lambda: cg)
        await svc._do_retrain(symbol="BTC/USDT", timeframe="1h", trigger="auto")
        _assert_site(cap, "retraining_service.btc_dom_history_fetch_failed", "_Leaky")

    async def test_dataset_error(
        self, site_log: _SiteLog, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cap = site_log()
        svc = _service(tmp_path, feature_schema_version=1)
        _patch_prelude(svc, monkeypatch)
        monkeypatch.setattr(svc, "_train_sync", _boom(ValueError(LEAK_TEXT)))
        await svc._do_retrain(symbol="BTC/USDT", timeframe="1h", trigger="auto")
        _assert_site(cap, "retraining_service.dataset_error", "ValueError")

    async def test_train_failed(
        self, site_log: _SiteLog, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cap = site_log()
        svc = _service(tmp_path, feature_schema_version=1)
        _patch_prelude(svc, monkeypatch)
        monkeypatch.setattr(svc, "_train_sync", _boom(RuntimeError(LEAK_TEXT)))
        await svc._do_retrain(symbol="BTC/USDT", timeframe="1h", trigger="auto")
        _assert_site(cap, "retraining_service.train_failed", "RuntimeError")

    async def test_model_file_delete_failed(
        self, site_log: _SiteLog, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cap = site_log()
        svc = _service(tmp_path, max_model_versions=0)
        model_file = tmp_path / "old.joblib"
        model_file.write_bytes(b"x")
        mv = SimpleNamespace(is_active=False, model_path=str(model_file), id="v1")

        class _Sess:
            async def __aenter__(self) -> _Sess:
                return self

            async def __aexit__(self, *_a: object) -> None:
                return None

            def begin(self) -> _Sess:
                return self

            async def execute(self, *_a: Any, **_k: Any) -> Any:
                scalars = SimpleNamespace(all=lambda: [mv])
                return SimpleNamespace(scalars=lambda: scalars)

        svc._db_session_factory = _Sess
        monkeypatch.setattr(Path, "unlink", _boom(OSError(LEAK_TEXT)))
        await svc._prune_old_versions(symbol="BTC/USDT", timeframe="1h")
        ev = _assert_site(cap, "retraining_service.model_file_delete_failed", "OSError")
        # the path is logged on purpose (pre-existing field), the exception text is not
        assert ev["path"] == str(model_file)

    async def test_debug_trace_is_emitted_and_redacted(
        self, production_logging: ConfigureFn, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cap = production_logging(json_output=True, log_level="DEBUG")
        svc = _service(tmp_path)
        monkeypatch.setattr(svc, "_fetch_ohlcv_sync", _boom(_client_response_error()))
        await svc._do_retrain(symbol="BTC/USDT", timeframe="1h", trigger="auto")
        _assert_trace_redacted(cap, "retraining_service.ohlcv_fetch_failed")
