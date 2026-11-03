"""WP22-T-01 -- clock unit semantics (``trading.clock``, WP2.2 commit P1)."""

from __future__ import annotations

import ast
import asyncio
import logging
import sys
import threading
from collections.abc import Generator, Iterator
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import pytest
import structlog
from structlog.testing import capture_logs

from tests._wp15_logging import _structlog_unfiltered  # noqa: F401
from trading import clock as clock_mod
from trading.clock import (
    TIMESTAMP_CLOCK_VERSION,
    WALL_CLOCK,
    BarClock,
    Clock,
    WallClock,
    ambient_clock,
    use_clock,
    utc_now,
)

_REPO = Path(__file__).resolve().parents[2]
_T0 = datetime(2024, 1, 1, 1, 0, tzinfo=UTC)
_FIXED = datetime(2001, 2, 3, 4, 5, 6, tzinfo=UTC)
_NAIVE = datetime(2024, 1, 1, 1, 0)
_WALL_TOLERANCE = timedelta(seconds=60)  # wall-vs-wall comparison; immune to small NTP steps


def _assert_near_wall_now(value: datetime) -> None:
    assert abs(value - datetime.now(tz=UTC)) < _WALL_TOLERANCE, value


class _Fixed:
    def __init__(self, at: datetime) -> None:
        self._at = at

    def now(self) -> datetime:
        return self._at


@pytest.fixture(scope="module", autouse=True)
def _polluted_global_structlog() -> Iterator[dict[str, object]]:
    """WP22-P1V2-01 (CR-01 regression): run the WHOLE module under a hostile
    global structlog config -- INFO-filtering wrapper with
    ``cache_logger_on_first_use=True`` (what ``configure_structlog`` installs).

    Module scope instantiates before the function-scoped autouse
    ``_structlog_unfiltered``, so that fixture is what must override it.  Drop
    the ``_structlog_unfiltered`` import above and the debug-log tests fail.
    """
    previous = structlog.get_config()
    wrapper = structlog.make_filtering_bound_logger(logging.INFO)
    structlog.configure(wrapper_class=wrapper, cache_logger_on_first_use=True)
    try:
        yield {"wrapper_class": wrapper}
    finally:
        structlog.configure(**previous)


@pytest.fixture
def frozen_wall(monkeypatch: pytest.MonkeyPatch) -> datetime:
    monkeypatch.setattr(clock_mod, "_wall_now", lambda: _FIXED)
    return _FIXED


class TestVersionTag:
    def test_timestamp_clock_version_is_pinned(self) -> None:
        assert TIMESTAMP_CLOCK_VERSION == "wp22-1"


class TestBarClock:
    def test_start_is_now(self) -> None:
        assert BarClock(_T0).now() == _T0

    def test_naive_start_rejected(self) -> None:
        with pytest.raises(ValueError, match="timezone-aware"):
            BarClock(_NAIVE)

    def test_naive_advance_rejected_and_state_unchanged(self) -> None:
        c = BarClock(_T0)
        with pytest.raises(ValueError, match="timezone-aware"):
            c.advance(_NAIVE + timedelta(hours=1))
        assert c.now() == _T0

    def test_start_normalised_to_utc(self) -> None:
        plus2 = timezone(timedelta(hours=2))
        c = BarClock(datetime(2024, 1, 1, 3, 0, tzinfo=plus2))
        assert c.now() == _T0
        assert c.now().utcoffset() == timedelta(0)
        assert c.now().tzinfo is UTC

    def test_advance_normalised_to_utc(self) -> None:
        plus2 = timezone(timedelta(hours=2))
        c = BarClock(_T0)
        c.advance(datetime(2024, 1, 1, 5, 0, tzinfo=plus2))
        assert c.now() == datetime(2024, 1, 1, 3, 0, tzinfo=UTC)
        assert c.now().tzinfo is UTC

    def test_advance_forward(self) -> None:
        c = BarClock(_T0)
        later = _T0 + timedelta(hours=1)
        c.advance(later)
        assert c.now() == later

    def test_equal_advance_is_noop_with_debug_log(self) -> None:
        c = BarClock(_T0)
        with capture_logs() as logs:
            c.advance(_T0)
        assert c.now() == _T0
        assert [e["log_level"] for e in logs] == ["debug"]
        assert logs[0]["event"] == "bar_clock.advance_ignored"

    def test_earlier_advance_is_noop_with_debug_log(self) -> None:
        c = BarClock(_T0)
        c.advance(_T0 + timedelta(hours=2))
        kept = c.now()
        with capture_logs() as logs:
            c.advance(_T0 + timedelta(hours=1))
        assert c.now() == kept
        assert [e["log_level"] for e in logs] == ["debug"]
        assert logs[0]["event"] == "bar_clock.advance_ignored"

    def test_forward_advance_does_not_log(self) -> None:
        c = BarClock(_T0)
        with capture_logs() as logs:
            c.advance(_T0 + timedelta(hours=1))
        assert logs == []

    def test_same_instant_other_offset_counts_as_equal(self) -> None:
        c = BarClock(_T0)
        c.advance(datetime(2024, 1, 1, 3, 0, tzinfo=timezone(timedelta(hours=2))))
        assert c.now() == _T0

    def test_satisfies_clock_protocol_and_is_final_for_isinstance_guards(self) -> None:
        c = BarClock(_T0)
        assert isinstance(c, Clock)
        assert isinstance(c, BarClock)
        assert not isinstance(WALL_CLOCK, BarClock)

    def test_subclassing_is_rejected_at_runtime(self) -> None:
        # ``@final`` is static only; __init_subclass__ enforces it (SEC-04).
        with pytest.raises(TypeError, match="BarClock is final"):
            type("EvilBar", (BarClock,), {})

    def test_barclock_is_slotted_instance_cannot_be_patched(self) -> None:
        c = BarClock(_T0)
        assert not hasattr(c, "__dict__")
        with pytest.raises(AttributeError):
            c.now = lambda: _FIXED  # type: ignore[method-assign]

    def test_instances_are_independent(self) -> None:
        a, b = BarClock(_T0), BarClock(_T0)
        a.advance(_T0 + timedelta(hours=1))
        assert b.now() == _T0


class TestWallClock:
    def test_now_is_aware_utc_and_near_wall_time(self) -> None:
        got = WALL_CLOCK.now()
        assert got.tzinfo is not None and got.utcoffset() == timedelta(0)
        _assert_near_wall_now(got)

    def test_now_resolves_wall_now_at_call_time(self, frozen_wall: datetime) -> None:
        assert WALL_CLOCK.now() == frozen_wall
        assert WallClock().now() == frozen_wall

    def test_instance_created_before_patch_sees_patch(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        wc = WallClock()
        monkeypatch.setattr(clock_mod, "_wall_now", lambda: _FIXED)
        assert wc.now() == _FIXED

    def test_satisfies_clock_protocol(self) -> None:
        assert isinstance(WALL_CLOCK, Clock)

    def test_subclassing_is_rejected_at_runtime(self) -> None:
        # A WallClock subclass with a faked now() must not be creatable, so
        # an ``isinstance(c, WallClock)`` allowlist stays sound (SEC-04).
        with pytest.raises(TypeError, match="WallClock is final"):
            type("EvilWall", (WallClock,), {"now": lambda self: _FIXED})


class TestAmbientClock:
    def test_default_is_wall_time(self, frozen_wall: datetime) -> None:
        assert utc_now() == frozen_wall

    def test_default_ambient_is_wall_clock(self) -> None:
        assert clock_mod._AMBIENT.get() is WALL_CLOCK

    def test_use_clock_overrides_and_restores(self, frozen_wall: datetime) -> None:
        bar = BarClock(_T0)
        with use_clock(bar):
            assert utc_now() == _T0
            bar.advance(_T0 + timedelta(hours=1))
            assert utc_now() == _T0 + timedelta(hours=1)
        assert utc_now() == frozen_wall

    def test_use_clock_restores_on_exception(self, frozen_wall: datetime) -> None:
        with pytest.raises(RuntimeError, match="boom"):
            with use_clock(BarClock(_T0)):
                assert utc_now() == _T0
                raise RuntimeError("boom")
        assert utc_now() == frozen_wall
        assert clock_mod._AMBIENT.get() is WALL_CLOCK

    def test_nested_use_clock_restores_previous_not_default(self) -> None:
        outer, inner = BarClock(_T0), _Fixed(_FIXED)
        with use_clock(outer):
            with use_clock(inner):
                assert utc_now() == _FIXED
            assert utc_now() == _T0
            assert clock_mod._AMBIENT.get() is outer
        assert clock_mod._AMBIENT.get() is WALL_CLOCK

    def test_nested_inner_exception_restores_outer(self) -> None:
        outer = BarClock(_T0)
        with use_clock(outer):
            with pytest.raises(KeyError):
                with use_clock(_Fixed(_FIXED)):
                    raise KeyError("x")
            assert utc_now() == _T0

    def test_generator_close_restores(self) -> None:
        def gen() -> Generator[datetime, None, None]:
            with use_clock(BarClock(_T0)):
                yield utc_now()

        g = gen()
        assert next(g) == _T0
        g.close()
        assert clock_mod._AMBIENT.get() is WALL_CLOCK

    def test_ambient_clock_accessor(self) -> None:
        assert ambient_clock() is WALL_CLOCK
        outer, inner = BarClock(_T0), _Fixed(_FIXED)
        with use_clock(outer):
            assert ambient_clock() is outer
            with use_clock(inner):
                assert ambient_clock() is inner
            assert ambient_clock() is outer
        assert ambient_clock() is WALL_CLOCK

    def test_ambient_clock_restored_after_exception(self) -> None:
        with pytest.raises(RuntimeError):
            with use_clock(BarClock(_T0)):
                raise RuntimeError("boom")
        assert ambient_clock() is WALL_CLOCK

    @pytest.mark.parametrize("bad", [None, 42, "clock", object(), BarClock])
    def test_use_clock_rejects_non_clock_on_entry(self, bad: object) -> None:
        before = ambient_clock()
        with pytest.raises(TypeError, match="use_clock requires a Clock"):
            with use_clock(bad):  # type: ignore[arg-type]
                pytest.fail("body must not run")  # pragma: no cover
        assert ambient_clock() is before

    def test_use_clock_type_error_names_type_only(self) -> None:
        with pytest.raises(TypeError) as ei:
            with use_clock("secret-value"):  # type: ignore[arg-type]
                pass  # pragma: no cover
        assert "str" in str(ei.value)
        assert "secret-value" not in str(ei.value)

    def test_use_clock_accepts_duck_typed_clock(self) -> None:
        fake = _Fixed(_FIXED)
        with use_clock(fake):
            assert ambient_clock() is fake
            assert utc_now() == _FIXED

    def test_concurrent_asyncio_tasks_are_isolated(self) -> None:
        a_start, b_start = asyncio.Event(), asyncio.Event()
        seen: dict[str, list[datetime]] = {"a": [], "b": []}

        async def run_a() -> None:
            with use_clock(BarClock(_T0)):
                a_start.set()
                await b_start.wait()
                seen["a"].append(utc_now())

        async def run_b() -> None:
            await a_start.wait()
            seen["b"].append(utc_now())  # A is inside use_clock; B must see wall time
            with use_clock(_Fixed(_FIXED)):
                b_start.set()
                await asyncio.sleep(0)
                seen["b"].append(utc_now())

        async def main() -> None:
            await asyncio.gather(run_a(), run_b())

        asyncio.run(main())
        assert seen["a"] == [_T0]
        _assert_near_wall_now(seen["b"][0])
        assert seen["b"][1] == _FIXED

    def test_other_thread_sees_wall_time(self, frozen_wall: datetime) -> None:
        """A plain ``threading.Thread`` starts with an EMPTY context.

        Python-version note (WP22-P1-SEC-08): this holds on CPython 3.11-3.13.
        CPython 3.14 adds ``sys.flags.thread_inherit_context`` (on by default
        in free-threaded builds), under which a thread started inside
        ``use_clock`` inherits the BarClock and this test would fail.
        ``requires-python`` has no upper bound: re-pin this expectation (and
        the P2b no-spawn-in-scope AST guard) before adopting 3.14.
        ``asyncio.to_thread`` always copies the context, on every version.
        """
        out: list[datetime] = []
        with use_clock(BarClock(_T0)):
            t = threading.Thread(target=lambda: out.append(utc_now()))
            t.start()
            t.join()
        assert out == [frozen_wall]


class TestDebugLogUnderPollutedStructlog:
    """WP22-P1V2-01 (CR-01): debug-log assertions must not depend on global
    structlog state.  The module-scoped ``_polluted_global_structlog`` fixture
    installs an INFO-filtering, logger-caching config before every test; the
    autouse ``_structlog_unfiltered`` import is what keeps them deterministic.
    """

    def test_pollution_is_active_and_overridden_by_unfiltered_fixture(
        self, _polluted_global_structlog: dict[str, object]
    ) -> None:
        polluted = _polluted_global_structlog["wrapper_class"]
        current = structlog.get_config()["wrapper_class"]
        assert current is not polluted
        assert current(None, [], {}).is_enabled_for(logging.DEBUG)
        assert not polluted(None, [], {}).is_enabled_for(logging.DEBUG)  # type: ignore[operator]
        assert structlog.get_config()["cache_logger_on_first_use"] is False

    def test_debug_event_is_captured_despite_polluted_config(self) -> None:
        c = BarClock(_T0)
        with capture_logs() as logs:
            c.advance(_T0)
        assert [(e["log_level"], e["event"]) for e in logs] == [
            ("debug", "bar_clock.advance_ignored")
        ]

    def test_forward_advance_logs_nothing_not_vacuously(self) -> None:
        # Guards the vacuous-pass risk: prove debug is capturable here, then
        # that a forward advance emits nothing.
        c = BarClock(_T0)
        with capture_logs() as logs:
            structlog.get_logger("probe").debug("probe.visible")
            c.advance(_T0 + timedelta(hours=1))
        assert [e["event"] for e in logs] == ["probe.visible"]


class TestImportHygiene:
    """Leaf module: stdlib + structlog only, no other ``trading.*`` import."""

    _ALLOWED_THIRD_PARTY = frozenset({"structlog"})
    _PATH = _REPO / "packages/trading/clock.py"

    def test_imports_only_stdlib_and_structlog(self) -> None:
        tree = ast.parse(self._PATH.read_text())
        stdlib = sys.stdlib_module_names
        found: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                found += [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                assert node.level == 0, "relative import in leaf module"
                found.append(node.module or "")
        assert found, "no imports parsed"
        for name in found:
            top = name.split(".")[0]
            assert top in stdlib or top in self._ALLOWED_THIRD_PARTY, name
            assert top not in {"trading", "common", "apps", "api"}, name

    def test_no_dynamic_imports_or_environment_reads(self) -> None:
        tree = ast.parse(self._PATH.read_text())
        used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | {
            n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)
        }
        forbidden = {"environ", "getenv", "putenv", "os", "sys", "dotenv", "argv"}
        forbidden |= {"import_module", "__import__", "spec_from_file_location"}
        assert used.isdisjoint(forbidden), used & forbidden

    def test_wall_now_is_the_only_wall_clock_read(self) -> None:
        tree = ast.parse(self._PATH.read_text())
        offenders: list[tuple[str, int]] = []
        for fn in ast.walk(tree):
            if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for node in ast.walk(fn):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                    if node.func.attr in {"now", "utcnow", "today", "time", "time_ns"} and (
                        isinstance(node.func.value, ast.Name)
                        and node.func.value.id in {"datetime", "time"}
                    ):
                        offenders.append((fn.name, node.lineno))
        assert {name for name, _ in offenders} == {"_wall_now"}, offenders


class TestFinalityMessageNamesTheClass:
    """WP2.2 P2b / R15 (SEC-03b): the finality ``TypeError`` names the class literally.

    It used to print ``cls.__mro__[1].__name__``: with a mixin first in the bases
    that was the mixin ("Mixin is final"), pointing the reader at the wrong class.
    """

    class _Mixin:
        pass

    def test_bar_clock_message_is_literal(self) -> None:
        with pytest.raises(TypeError) as direct:
            type("EvilBar", (BarClock,), {})
        with pytest.raises(TypeError) as mixed:
            type("EvilMixedBar", (self._Mixin, BarClock), {})
        for err in (direct, mixed):
            assert str(err.value) == "BarClock is final and cannot be subclassed"
        assert "_Mixin" not in str(mixed.value)

    def test_wall_clock_message_is_literal(self) -> None:
        with pytest.raises(TypeError) as direct:
            type("EvilWall", (WallClock,), {})
        with pytest.raises(TypeError) as mixed:
            type("EvilMixedWall", (self._Mixin, WallClock), {})
        for err in (direct, mixed):
            assert str(err.value) == "WallClock is final and cannot be subclassed"
        assert "_Mixin" not in str(mixed.value)
