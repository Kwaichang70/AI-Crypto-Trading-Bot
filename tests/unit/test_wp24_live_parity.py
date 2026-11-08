# ruff: noqa: S110  (the frozen HEAD oracle keeps its swallow-all blocks)
"""WP24-T-15 -- live parity: PAPER and LIVE keep the pre-WP2.4 provider reads, byte for byte.

Binding spec: ``reports/vp2-wp2.4/synthesis-spec.md`` (SY-24-02, SY-24-16, WP24-T-15).

The ORACLE below is a frozen copy of the signal block of ``StrategyEngine._build_mtf_context``
at HEAD ``bbe7142`` (``packages/trading/strategy_engine.py:2129-2216``). It must never be
edited to follow the implementation: the tests compare the new engine against it for the same
fake clients, including both 7d paths and every exception path. A second test compares the
AST of the moved statements in ``LiveSignals.values_at`` with the oracle's, so the move is
verbatim, not merely equivalent.
"""

from __future__ import annotations

import ast
import hashlib
import inspect
import textwrap
from collections.abc import Callable, Iterator
from contextlib import ExitStack, contextmanager
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from common.models import MultiTimeframeContext
from common.types import RunMode, TimeFrame
from trading import mtf_signals as mtf_mod
from trading.mtf_signals import LIVE_SIGNALS, LiveSignals
from trading.strategy_engine import StrategyEngine

SYM = "BTC/USDT"
TS = datetime(2026, 3, 10, 12, 0, tzinfo=UTC)

# CR-04: sha256 of ``inspect.getsource(_head_signal_block)`` -- the frozen oracle text.
_HEAD_ORACLE_SHA256 = "d83f38862f8656908e5001c9804fc919684b1c387cdd54abc648ac92d46067bd"
# E5b: sha256 of the 65 moved lines of ``LiveSignals.values_at`` (HEAD ``:2129-2193``), verbatim.
_MOVED_BLOCK_SHA256 = "8c31b7d57a9f271585189633e815ff732d531deb1e002b3504d6bf05e79c5447"
_MOVED_MARKER = "strategy_engine.py:2129-2193): the four best-effort singleton reads"
_MOVED_LINES = 65

# ---------------------------------------------------------------------------
# the frozen HEAD bbe7142 oracle (strategy_engine.py:2129-2216, verbatim)
# ---------------------------------------------------------------------------


def _head_signal_block() -> dict[str, Any]:
    # Read cached FGI value (best-effort; never crash bar processing
    # if the client is unavailable or has not populated its cache yet).
    fgi_value: int | None = None
    fgi_value_7d_ago: int | None = None
    try:
        from data.sentiment import get_global_client as _get_fgi_client
        _fgi_client = _get_fgi_client()
        if _fgi_client is not None:
            fgi_value = _fgi_client.cached_value
            # QT-007 v2 feature pipeline: cache-only sync read (no I/O).
            # Returns None when history has not been populated yet by the
            # background refresh task; v2 feature builder treats None as
            # "no observed change".
            fgi_value_7d_ago = _fgi_client.value_at_offset_from_cache(
                days_ago=7,
            )
    except Exception:
        pass  # FGI is best-effort; never crash bar processing

    # CoinGecko market structure signals (best-effort)
    btc_dominance: float | None = None
    btc_dominance_7d_ago: float | None = None
    market_cap_change_24h: float | None = None
    total_volume_change_24h: float | None = None
    try:
        from data.market_signals import get_global_client as _get_cg_client
        _cg_client = _get_cg_client()
        if _cg_client is not None:
            _cg_snap = _cg_client.cached_value
            if _cg_snap is not None:
                btc_dominance = _cg_snap.btc_dominance
                market_cap_change_24h = _cg_snap.market_cap_change_24h
                total_volume_change_24h = _cg_snap.total_volume_change_24h
            # QT-007 v2 feature pipeline: cache-only sync read (no I/O).
            btc_dominance_7d_ago = (
                _cg_client.btc_dominance_at_offset_from_cache(days_ago=7)
            )
    except Exception:
        pass  # CoinGecko is best-effort; never crash bar processing

    # FRED macro-economic signals (best-effort)
    fed_funds_rate: float | None = None
    yield_curve_spread: float | None = None
    try:
        from data.macro_data import get_global_client as _get_fred_client
        _fred_client = _get_fred_client()
        if _fred_client is not None:
            _fred_snap = _fred_client.cached_value
            if _fred_snap is not None:
                fed_funds_rate = _fred_snap.fed_funds_rate
                yield_curve_spread = _fred_snap.yield_curve_spread
    except Exception:
        pass  # FRED is best-effort; never crash bar processing

    # Whale Alert on-chain flow signals (best-effort)
    whale_net_flow: float | None = None
    try:
        from data.whale_tracker import get_global_client as _get_whale_client
        _whale_client = _get_whale_client()
        if _whale_client is not None:
            _whale_snap = _whale_client.cached_value
            if _whale_snap is not None:
                whale_net_flow = _whale_snap.net_flow
    except Exception:
        pass  # Whale Alert is best-effort; never crash bar processing

    # Build kwargs for MultiTimeframeContext — only include non-None signal
    # fields so the frozen dataclass default values are used for all absent
    # signals, preserving backward compatibility with existing tests.
    ctx_kwargs: dict[str, Any] = {}
    if fgi_value is not None:
        ctx_kwargs["fear_greed_index"] = fgi_value
    if btc_dominance is not None:
        ctx_kwargs["btc_dominance"] = btc_dominance
    if market_cap_change_24h is not None:
        ctx_kwargs["market_cap_change_24h"] = market_cap_change_24h
    if total_volume_change_24h is not None:
        ctx_kwargs["total_volume_change_24h"] = total_volume_change_24h
    if fed_funds_rate is not None:
        ctx_kwargs["fed_funds_rate"] = fed_funds_rate
    if yield_curve_spread is not None:
        ctx_kwargs["yield_curve_spread"] = yield_curve_spread
    if whale_net_flow is not None:
        ctx_kwargs["whale_net_flow"] = whale_net_flow
    if fgi_value_7d_ago is not None:
        ctx_kwargs["fear_greed_index_7d_ago"] = fgi_value_7d_ago
    if btc_dominance_7d_ago is not None:
        ctx_kwargs["btc_dominance_7d_ago"] = btc_dominance_7d_ago
    return ctx_kwargs


def _head_context(htf_bars: dict[str, Any] | None) -> MultiTimeframeContext | None:
    """The HEAD tail for an engine with no real HTF bars (``:2218-2224`` / ``:2250``)."""
    ctx_kwargs = _head_signal_block()
    if htf_bars is None:
        if ctx_kwargs:
            return MultiTimeframeContext(**ctx_kwargs)
        return None
    assert htf_bars == {}  # the only non-None shape this oracle models
    return MultiTimeframeContext(htf_bars={}, **ctx_kwargs)


# ---------------------------------------------------------------------------
# fake clients and the scenario matrix
# ---------------------------------------------------------------------------


class _Boom(RuntimeError):
    pass


def _fgi(cached: Any = 55, offset: Any = 40, *, cached_raises: bool = False,
         offset_raises: bool = False) -> Any:
    class _Client:
        @property
        def cached_value(self) -> Any:
            if cached_raises:
                raise _Boom
            return cached

        def value_at_offset_from_cache(self, days_ago: int) -> Any:
            assert days_ago == 7
            if offset_raises:
                raise _Boom
            return offset

    return _Client()


def _cg(snap: Any = "full", offset: Any = 50.0, *, offset_raises: bool = False,
        snap_raises: bool = False) -> Any:
    if snap == "full":
        snap = SimpleNamespace(
            btc_dominance=52.5, market_cap_change_24h=1.5, total_volume_change_24h=-3.25
        )

    class _Client:
        @property
        def cached_value(self) -> Any:
            if snap_raises:
                raise _Boom
            return snap

        def btc_dominance_at_offset_from_cache(self, days_ago: int) -> Any:
            assert days_ago == 7
            if offset_raises:
                raise _Boom
            return offset

    return _Client()


def _fred(snap: Any = "full", *, snap_raises: bool = False) -> Any:
    if snap == "full":
        snap = SimpleNamespace(fed_funds_rate=5.25, yield_curve_spread=-0.4)

    class _Client:
        @property
        def cached_value(self) -> Any:
            if snap_raises:
                raise _Boom
            return snap

    return _Client()


def _whale(snap: Any = "full", *, snap_raises: bool = False) -> Any:
    if snap == "full":
        snap = SimpleNamespace(net_flow=1234.5)

    class _Client:
        @property
        def cached_value(self) -> Any:
            if snap_raises:
                raise _Boom
            return snap

    return _Client()


_RAISE = "getter-raises"
_MODULES = ("sentiment", "market_signals", "macro_data", "whale_tracker")

#: name -> (fgi, coingecko, fred, whale); ``None`` means "no client installed".
_SCENARIOS: dict[str, tuple[Any, Any, Any, Any]] = {
    "no-clients": (None, None, None, None),
    "all-populated": (_fgi(), _cg(), _fred(), _whale()),
    "fgi-7d-none": (_fgi(offset=None), _cg(), _fred(), _whale()),
    "fgi-cached-none-7d-set": (_fgi(cached=None, offset=33), None, None, None),
    "cg-7d-none": (None, _cg(offset=None), None, None),
    "cg-snapshot-none-7d-set": (None, _cg(snap=None, offset=48.5), None, None),
    "fred-snapshot-none": (None, None, _fred(snap=None), None),
    "whale-snapshot-none": (None, None, None, _whale(snap=None)),
    "fgi-zero-values": (_fgi(cached=0, offset=0), _cg(snap=SimpleNamespace(
        btc_dominance=0.0, market_cap_change_24h=0.0, total_volume_change_24h=0.0), offset=0.0),
        _fred(snap=SimpleNamespace(fed_funds_rate=0.0, yield_curve_spread=0.0)),
        _whale(snap=SimpleNamespace(net_flow=0.0))),
    "fgi-getter-raises": (_RAISE, _cg(), _fred(), _whale()),
    "cg-getter-raises": (_fgi(), _RAISE, _fred(), _whale()),
    "fred-getter-raises": (_fgi(), _cg(), _RAISE, _whale()),
    "whale-getter-raises": (_fgi(), _cg(), _fred(), _RAISE),
    "all-getters-raise": (_RAISE, _RAISE, _RAISE, _RAISE),
    "fgi-cached-raises": (_fgi(cached_raises=True), _cg(), _fred(), _whale()),
    "fgi-offset-raises-keeps-cached": (_fgi(offset_raises=True), _cg(), _fred(), _whale()),
    "cg-snapshot-raises": (_fgi(), _cg(snap_raises=True), _fred(), _whale()),
    "cg-offset-raises-keeps-snapshot": (_fgi(), _cg(offset_raises=True), _fred(), _whale()),
    "fred-snapshot-raises": (_fgi(), _cg(), _fred(snap_raises=True), _whale()),
    "whale-snapshot-raises": (_fgi(), _cg(), _fred(), _whale(snap_raises=True)),
}


@contextmanager
def _installed(scenario: tuple[Any, Any, Any, Any]) -> Iterator[None]:
    """Patch the four ``get_global_client`` getters (the lazy import sees the patch)."""
    with ExitStack() as stack:
        for module, client in zip(_MODULES, scenario, strict=True):
            kwargs: dict[str, Any]
            if client is _RAISE:
                kwargs = {"side_effect": _Boom}
            else:
                kwargs = {"return_value": client}
            stack.enter_context(patch(f"data.{module}.get_global_client", **kwargs))
        yield


def _engine(mode: RunMode, **kw: Any) -> StrategyEngine:
    strategy = MagicMock()
    strategy.strategy_id = "s"
    strategy.min_bars_required = 20
    strategy.htf_timeframes = None
    return StrategyEngine(
        strategies=[strategy],
        execution_engine=AsyncMock(),
        risk_manager=MagicMock(),
        market_data=AsyncMock(),
        portfolio=MagicMock(),
        symbols=[SYM],
        timeframe=TimeFrame.ONE_HOUR,
        run_mode=mode,
        **kw,
    )


def _typed(ctx: MultiTimeframeContext | None) -> Any:
    if ctx is None:
        return None
    return {
        name: (type(getattr(ctx, name)), getattr(ctx, name))
        for name in MultiTimeframeContext.__dataclass_fields__
        if name != "htf_bars"
    }


# ---------------------------------------------------------------------------
# T-15
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", [RunMode.PAPER, RunMode.LIVE])
@pytest.mark.parametrize("name", sorted(_SCENARIOS))
class TestT15LiveParity:
    def test_engine_context_equals_the_frozen_head_oracle(self, mode: RunMode, name: str) -> None:
        engine = _engine(mode)
        assert engine._mtf_signals is LIVE_SIGNALS
        with _installed(_SCENARIOS[name]):
            expected = _head_context(None)
            actual = engine._build_mtf_context(TS)
        assert actual == expected
        assert _typed(actual) == _typed(expected)  # same value AND same type, field by field

    def test_with_an_htf_dict_present_the_signal_fields_are_the_oracles(
        self, mode: RunMode, name: str
    ) -> None:
        engine = _engine(mode)
        engine._htf_bars = {}
        with _installed(_SCENARIOS[name]):
            expected = _head_context({})
            actual = engine._build_mtf_context(TS)
        assert actual == expected
        assert actual is not None  # an HTF dict always yields a context

    def test_decision_at_is_ignored_on_the_live_path(self, mode: RunMode, name: str) -> None:
        engine = _engine(mode)
        with _installed(_SCENARIOS[name]):
            without = engine._build_mtf_context(TS)
            with_stamp = engine._build_mtf_context(TS, decision_at=datetime(2001, 1, 1, tzinfo=UTC))
        assert with_stamp == without


class TestT15SourceMoveIsVerbatim:
    """The moved statements are AST-identical to the oracle's (comments aside)."""

    @staticmethod
    def _body(fn: Callable[..., Any]) -> list[ast.stmt]:
        tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
        node = tree.body[0]
        assert isinstance(node, ast.FunctionDef)
        body = node.body
        if isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
            body = body[1:]  # docstring
        return body

    def test_signal_reads_are_the_same_statements_in_the_same_order(self) -> None:
        oracle = self._body(_head_signal_block)
        moved = self._body(LiveSignals.values_at)
        # oracle: 13 read statements, then the ``ctx_kwargs`` rule (AnnAssign + 9 ifs + return)
        reads_oracle = oracle[:13]
        reads_moved = moved[:13]
        assert [ast.dump(s) for s in reads_moved] == [ast.dump(s) for s in reads_oracle]
        assert sum(isinstance(s, ast.Try) for s in reads_moved) == 4
        # what follows in the moved code is exactly one ``return SignalValues(...)``
        assert len(moved) == 14 and isinstance(moved[13], ast.Return)

    def test_the_non_none_rule_is_the_oracles_in_the_head_order(self) -> None:
        oracle_ifs = [s for s in self._body(_head_signal_block) if isinstance(s, ast.If)]
        values = mtf_mod.SignalValues
        to_kwargs = self._body(values.to_context_kwargs)
        moved_ifs = [s for s in to_kwargs if isinstance(s, ast.If)]
        assert len(oracle_ifs) == len(moved_ifs) == 9

        def key_of(stmt: ast.If) -> str:
            sub = stmt.body[0]
            assert isinstance(sub, ast.Assign)
            target = sub.targets[0]
            assert isinstance(target, ast.Subscript) and isinstance(target.slice, ast.Constant)
            return str(target.slice.value)

        assert [key_of(s) for s in moved_ifs] == [key_of(s) for s in oracle_ifs]

    def test_the_moved_block_references_only_the_four_lazy_imports(self) -> None:
        imports = [
            n for n in ast.walk(ast.parse(inspect.getsource(LiveSignals)))
            if isinstance(n, ast.ImportFrom)
        ]
        assert sorted(n.module or "" for n in imports) == [
            "data.macro_data", "data.market_signals", "data.sentiment", "data.whale_tracker",
        ]
        assert all(n.col_offset > 0 for n in imports)  # function-local (lazy), as at HEAD


class TestT15SwallowAllIsPreserved:
    def test_a_base_exception_is_not_swallowed_by_either_implementation(self) -> None:
        engine = _engine(RunMode.PAPER)
        with patch("data.sentiment.get_global_client", side_effect=KeyboardInterrupt):
            with pytest.raises(KeyboardInterrupt):
                _head_signal_block()
            with pytest.raises(KeyboardInterrupt):
                engine._build_mtf_context(TS)

    def test_the_moved_code_logs_nothing_new(self, caplog: pytest.LogCaptureFixture) -> None:
        engine = _engine(RunMode.LIVE)
        with caplog.at_level("DEBUG"), _installed(_SCENARIOS["all-getters-raise"]):
            assert engine._build_mtf_context(TS) is None
        assert caplog.records == []
        src = inspect.getsource(mtf_mod)
        for banned in ("structlog", "getLogger", "logging", "print(", "exc_log_fields"):
            assert banned not in src


class TestT15FrozenText:
    def test_the_head_oracle_is_frozen(self) -> None:  # CR-04
        digest = hashlib.sha256(inspect.getsource(_head_signal_block).encode()).hexdigest()
        assert digest == _HEAD_ORACLE_SHA256, "the frozen HEAD oracle was edited"

    def test_the_moved_block_is_byte_identical_to_head(self) -> None:  # E5b
        lines = Path(mtf_mod.__file__).read_bytes().decode().splitlines(keepends=True)
        marks = [i for i, line in enumerate(lines) if _MOVED_MARKER in line]
        assert len(marks) == 1, marks
        block = lines[marks[0] + 1 : marks[0] + 1 + _MOVED_LINES]
        assert len(block) == _MOVED_LINES
        assert block[-1].strip() == "pass  # Whale Alert is best-effort; never crash bar processing"
        digest = hashlib.sha256("".join(block).encode()).hexdigest()
        assert digest == _MOVED_BLOCK_SHA256, "the moved live block is no longer verbatim"
