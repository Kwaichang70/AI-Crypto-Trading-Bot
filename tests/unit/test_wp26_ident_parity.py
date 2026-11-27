"""WP2.6 P4: identifier parity between the leaf and the DDL CHECK (SY-26-23, spec-addendum-p3).

No database. The two regexes are taken from the ORM CHECK text and from the migration source
and must be identical; the shared vectors and a character sweep then run through the leaf and
through ``re.fullmatch`` on the extracted DDL pattern. The same vectors run on a real
PostgreSQL in ``tests/integration/test_wp26_ohlcv_store_pg.py``.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

import data.ohlcv_integrity as leaf
from api.db.models import OhlcvBarORM, OhlcvCoverageORM
from data.ohlcv_integrity import OhlcvDataError, check_identifiers

_MIGRATION = Path(__file__).resolve().parents[2] / "infra/alembic/versions/022_add_ohlcv_cache.py"
_PATTERN = re.compile(r"~ '\^(.+?)\$'")
_LONG = "A" * 30 + "/B"  # 32 chars: the longest accepted
SYMBOL_ACCEPT = ["BTC/EUR", "A/B", "1INCH/EUR", "BTC/USD:USD", "ETH-PERP/USD", _LONG]
SYMBOL_REJECT = ["", "/", "/AB", "AB/", "AB", "A//B", "A/B/C", "BTC|X/EUR", "BTC /EUR"]
SYMBOL_REJECT += ["BTC/EUR\n", "BTC/EÜR", "A" * 31 + "/B"]  # the last one has 33 characters
EXCHANGE_ACCEPT = ["coinbase", "binance", "a_1", "a" * 32]
EXCHANGE_REJECT = ["", "Coinbase", "coin-base", "coin base", "coinbase\n", "a" * 33]


def _ddl_patterns() -> tuple[str, str]:
    """(exchange, symbol) pattern text from the ORM CHECK, asserted equal to the migration."""
    texts = {
        str(c.sqltext)
        for table in (OhlcvBarORM.__table__, OhlcvCoverageORM.__table__)
        for c in table.constraints
        if c.name in ("ck_ohlcv_bars_ident", "ck_ohlcv_cov_ident")
    }
    assert len(texts) == 1  # the same text on both tables
    orm = _PATTERN.findall(texts.pop())
    assert set(_PATTERN.findall(_MIGRATION.read_text())) == set(orm)  # migration source
    exchange, symbol = orm
    return exchange, symbol


def _leaf_ok(exchange: str, symbol: str) -> bool:
    try:
        check_identifiers(exchange, symbol, 3600)
    except OhlcvDataError:
        return False
    return True


def test_leaf_patterns_are_the_ddl_patterns() -> None:
    exchange, symbol = _ddl_patterns()
    assert (exchange, symbol) == ("[a-z0-9_]+", "[!-.0-{}~]+/[!-.0-{}~]+")
    assert leaf._EXCHANGE_RE.pattern == exchange
    assert leaf._SYMBOL_RE.pattern == symbol


@pytest.mark.parametrize("symbol", SYMBOL_ACCEPT)
def test_symbol_accept_vectors(symbol: str) -> None:
    assert _leaf_ok("coinbase", symbol)
    assert re.fullmatch(_ddl_patterns()[1], symbol)


@pytest.mark.parametrize("symbol", SYMBOL_REJECT)
def test_symbol_reject_vectors(symbol: str) -> None:
    assert not _leaf_ok("coinbase", symbol)
    if len(symbol) <= 32:  # a 33-character value passes the pattern; VARCHAR(32) rejects it
        assert not re.fullmatch(_ddl_patterns()[1], symbol)


@pytest.mark.parametrize("exchange", EXCHANGE_ACCEPT)
def test_exchange_accept_vectors(exchange: str) -> None:
    assert _leaf_ok(exchange, "BTC/EUR")
    assert re.fullmatch(_ddl_patterns()[0], exchange)


@pytest.mark.parametrize("exchange", EXCHANGE_REJECT)
def test_exchange_reject_vectors(exchange: str) -> None:
    assert not _leaf_ok(exchange, "BTC/EUR")
    if len(exchange) <= 32:
        assert not re.fullmatch(_ddl_patterns()[0], exchange)


def test_character_sweep_agrees_with_the_ddl_pattern() -> None:
    exchange, symbol = _ddl_patterns()
    for code in range(0x300):
        ch = chr(code)
        assert _leaf_ok("coinbase", f"A{ch}/B") == bool(re.fullmatch(symbol, f"A{ch}/B")), code
        assert _leaf_ok(f"a{ch}", "A/B") == bool(re.fullmatch(exchange, f"a{ch}")), code
