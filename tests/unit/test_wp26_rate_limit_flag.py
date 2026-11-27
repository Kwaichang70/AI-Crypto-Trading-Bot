"""WP2.6 P1 (SY-26-15): ``enable_rate_limit`` sets ccxt's ``enableRateLimit`` (default True);
the service's own ``_throttle`` / ``_MIN_REQUEST_INTERVAL`` / semaphore seam stays. No network."""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import data.services.ccxt_market_data as cmd
from common.types import TimeFrame
from data.services.ccxt_market_data import CCXTMarketDataService

_ROW = [1_700_000_000_000, 1.0, 2.0, 0.5, 1.5, 10.0]


def _captured_config(**kwargs: Any) -> dict[str, Any]:
    """Config dict handed to the (patched) ccxt exchange class."""
    with patch.object(cmd, "ccxt_async") as patched:
        patched.binance = MagicMock()
        CCXTMarketDataService(exchange_id="binance", **kwargs)
    config: dict[str, Any] = patched.binance.call_args.args[0]
    return config


def test_default_enables_ccxt_rate_limit() -> None:
    assert _captured_config()["enableRateLimit"] is True


@pytest.mark.parametrize("flag", [True, False])
def test_flag_is_forwarded_verbatim(flag: bool) -> None:
    assert _captured_config(enable_rate_limit=flag)["enableRateLimit"] is flag


def test_extra_exchange_config_still_wins() -> None:
    config = _captured_config(extra_exchange_config={"enableRateLimit": False})
    assert config["enableRateLimit"] is False


@pytest.mark.parametrize("exchange_id", ["coinbase", "binance"])
def test_real_ccxt_instance_default_on_and_opt_out(exchange_id: str) -> None:
    assert CCXTMarketDataService(exchange_id=exchange_id)._exchange.enableRateLimit is True
    off = CCXTMarketDataService(exchange_id=exchange_id, enable_rate_limit=False)
    assert off._exchange.enableRateLimit is False


def test_signature_default_true_and_keyword_only() -> None:
    param = inspect.signature(CCXTMarketDataService.__init__).parameters["enable_rate_limit"]
    assert param.default is True
    assert param.kind is inspect.Parameter.KEYWORD_ONLY


async def test_own_throttle_seam_stays_on_both_fetch_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert cmd._MIN_REQUEST_INTERVAL == 0.1
    monkeypatch.setattr(cmd, "_MIN_REQUEST_INTERVAL", 0.0)
    svc = CCXTMarketDataService(exchange_id="binance", cache_ttl_seconds=0, max_concurrency=3)
    fake = MagicMock()
    fake.fetch_ohlcv = AsyncMock(return_value=[_ROW])
    fake.fetch_ticker = AsyncMock(
        return_value={"bid": 1.0, "ask": 2.0, "last": 1.5, "timestamp": _ROW[0]}
    )
    svc._exchange = fake
    original = svc._throttle
    calls: list[str] = []

    async def counting() -> None:
        calls.append("t")
        assert svc._semaphore._value == 2  # the caller holds one of the 3 slots (CF-26-24)
        await original()

    monkeypatch.setattr(svc, "_throttle", counting)
    await svc.fetch_ohlcv("BTC/USDT", TimeFrame.ONE_HOUR, limit=1)
    assert calls == ["t"]
    await svc.fetch_ticker_quote("BTC/USDT")
    assert calls == ["t", "t"]
    assert svc._semaphore._value == 3


def test_stale_rate_limit_comments_are_gone() -> None:
    text = Path(cmd.__file__).read_text(encoding="utf-8")
    assert "off on this handle" not in text
    assert "We handle rate-limiting ourselves" not in text
