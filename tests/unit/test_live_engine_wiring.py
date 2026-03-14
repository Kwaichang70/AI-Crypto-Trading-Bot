"""
tests/unit/test_live_engine_wiring.py
--------------------------------------
Unit tests for the _run_live_engine background coroutine in
apps/api/routers/runs.py.

Module under test
-----------------
    apps/api/routers/runs.py  --  _run_live_engine()

Coverage groups
---------------
1. TestRunLiveEngine  -- 6 tests verifying component construction, run_mode,
                         persistence, error status, exchange close, and task
                         registry cleanup.

Design notes
------------
- _run_live_engine uses lazy imports inside the function body for all trading
  components. Because Python resolves lazy ``import`` and ``from ... import``
  statements against the canonical module in sys.modules, the correct patch
  targets are the *source* module attributes:
      api.config.get_settings
      api.db.session.get_session_factory
      ccxt.async_support                  (module-level, so getattr() works)
      data.services.ccxt_market_data.CCXTMarketDataService
      trading.engines.live.LiveExecutionEngine
      trading.portfolio.PortfolioAccounting
      trading.risk_manager.DefaultRiskManager
      trading.strategy_engine.StrategyEngine
      api.routers.runs._flush_incremental        (intra-module reference)
      api.routers.runs._incremental_flush_loop   (intra-module reference)

- The coroutine always executes the ``finally`` block, which:
  1. Cancels the periodic flush task (if running).
  2. Removes the run_id from _RUN_TASKS.
  3. Calls _flush_incremental (final flush) when portfolio is not None.
  4. Updates the run status in the DB.
  5. Calls exchange.close().

- StrategyEngine.run_live_loop() is the natural "park" point for the coroutine.
  Tests configure it to either return normally (normal stop) or raise an
  Exception (error path). CancelledError is re-raised by the coroutine so it
  is not tested via this helper — it propagates to the task runner.

- The DB session mock must satisfy the async context-manager protocol and
  return an execute result whose scalar_one_or_none() yields a mock RunORM
  with status="running" (so the finally block updates it).

- @pytest.mark.asyncio is explicit on all tests; asyncio_mode = "auto" in
  pyproject.toml means it is not strictly required but is included for clarity.

- _RUN_TASKS is imported directly from api.routers.runs so its state can be
  inspected and seeded within each test.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from api.routers.runs import _RUN_TASKS, _run_live_engine
from common.types import RunMode, TimeFrame

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_RUN_ID = "bbbb0000-0000-0000-0000-000000000001"
_STRATEGY_NAME = "ma_crossover"
_SYMBOL = "BTC/USDT"
_INITIAL_CAPITAL = "10000"

# ---------------------------------------------------------------------------
# Patch targets
# ---------------------------------------------------------------------------

_PT_SETTINGS = "api.config.get_settings"
_PT_SESSION_FACTORY = "api.db.session.get_session_factory"
_PT_CCXT = "ccxt.async_support"
_PT_MDS = "data.services.ccxt_market_data.CCXTMarketDataService"
_PT_LIVE_ENGINE = "trading.engines.live.LiveExecutionEngine"
_PT_PORTFOLIO = "trading.portfolio.PortfolioAccounting"
_PT_RISK_MGR = "trading.risk_manager.DefaultRiskManager"
_PT_STRAT_ENGINE = "trading.strategy_engine.StrategyEngine"
_PT_FLUSH = "api.routers.runs._flush_incremental"
_PT_FLUSH_LOOP = "api.routers.runs._incremental_flush_loop"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_settings_mock() -> MagicMock:
    """
    Build a mock Settings object satisfying _run_live_engine's attribute access.

    Attributes accessed:
        settings.exchange_id           -> str ("binance")
        settings.exchange_api_key      -> object with .get_secret_value()
        settings.exchange_api_secret   -> object with .get_secret_value()
    """
    secret_key = MagicMock()
    secret_key.get_secret_value.return_value = "test-api-key"

    secret_secret = MagicMock()
    secret_secret.get_secret_value.return_value = "test-api-secret"

    secret_passphrase = MagicMock()
    secret_passphrase.get_secret_value.return_value = "test-passphrase"

    settings = MagicMock()
    settings.exchange_id = "binance"
    settings.exchange_api_key = secret_key
    settings.exchange_api_secret = secret_secret
    settings.exchange_api_passphrase = secret_passphrase
    return settings


def _make_db_session_factory(run_status: str = "running") -> tuple[MagicMock, MagicMock]:
    """
    Build a (get_session_factory mock, db_session mock) pair.

    The coroutine accesses the DB in the finally block:
        factory = get_session_factory()
        async with factory() as db:
            result = await db.execute(select(...))
            run = result.scalar_one_or_none()
            if run is not None and run.status == "running":
                ...
                await db.commit()

    Returns
    -------
    (get_session_factory_mock, db_session_mock)
    """
    # Mock RunORM row
    mock_run = MagicMock()
    mock_run.status = run_status

    # Mock execute result
    mock_execute_result = MagicMock()
    mock_execute_result.scalar_one_or_none.return_value = mock_run

    # Mock DB session
    db_mock = MagicMock()
    db_mock.execute = AsyncMock(return_value=mock_execute_result)
    db_mock.commit = AsyncMock()
    db_mock.rollback = AsyncMock()

    # Async context manager: async with factory() as db
    async_ctx = MagicMock()

    async def _aenter(self: Any = None) -> MagicMock:
        return db_mock

    async def _aexit(self: Any = None, *args: Any) -> bool:
        return False

    async_ctx.__aenter__ = _aenter
    async_ctx.__aexit__ = _aexit

    factory_fn = MagicMock(return_value=async_ctx)
    get_session_factory_mock = MagicMock(return_value=factory_fn)

    return get_session_factory_mock, db_mock


def _make_ccxt_module_mock() -> MagicMock:
    """
    Build a mock for ccxt.async_support module.

    _run_live_engine calls: exchange_cls = getattr(ccxt_async, settings.exchange_id, None)
    The returned class is then called with a config dict to build the exchange instance.
    """
    mock_exchange = MagicMock()
    mock_exchange.close = AsyncMock()

    mock_exchange_cls = MagicMock(return_value=mock_exchange)

    ccxt_module = MagicMock()
    # getattr(ccxt_module, "binance", None) -> mock_exchange_cls
    ccxt_module.binance = mock_exchange_cls

    return ccxt_module, mock_exchange


def _make_strategy_engine_mock(raise_on_run_loop: Exception | None = None) -> MagicMock:
    """
    Build a mock StrategyEngine.

    start() and run_live_loop() are async. run_live_loop() can be configured
    to raise an exception to simulate error / normal-stop paths.

    Parameters
    ----------
    raise_on_run_loop:
        If provided, run_live_loop() raises this exception. When None,
        run_live_loop() returns normally (simulating a clean stop).
    """
    mock_cls = MagicMock()
    mock_instance = MagicMock()

    if raise_on_run_loop is not None:
        mock_instance.run_live_loop = AsyncMock(side_effect=raise_on_run_loop)
    else:
        mock_instance.run_live_loop = AsyncMock(return_value=None)

    mock_instance.start = AsyncMock(return_value=None)
    mock_instance.stop = AsyncMock(return_value=None)

    mock_cls.return_value = mock_instance
    return mock_cls, mock_instance


def _make_strategy_cls_mock() -> MagicMock:
    """Build a minimal strategy class mock that satisfies the constructor call."""
    mock_instance = MagicMock()
    mock_cls = MagicMock(return_value=mock_instance)
    return mock_cls


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------


class TestRunLiveEngine:
    """
    Verify _run_live_engine wires components correctly and handles the
    finally-block contract (DB update, persist, exchange close, task cleanup).

    All tests use the same patch stack with variations in StrategyEngine
    behaviour to exercise different paths through the coroutine.
    """

    def _base_run(
        self,
        *,
        se_raise: Exception | None = None,
        run_id: str = _RUN_ID,
    ) -> tuple[Any, Any, Any, Any, Any, Any, Any, Any]:
        """
        Return a tuple of (patch-target, mock) context managers for use as a
        composite patch stack.  Not used directly — each test assembles its own
        patch context for clarity and isolation.
        """
        raise NotImplementedError  # Tests inline their own patch stacks

    @pytest.mark.asyncio
    async def test_run_live_engine_creates_live_execution_engine(self) -> None:
        """
        LiveExecutionEngine must be constructed with enable_live_trading=True.

        The gate is always True in _run_live_engine because the 3-layer safety
        check is performed by the POST handler before the background task is
        created. We verify the constructor call keyword argument directly.
        """
        settings_mock = _make_settings_mock()
        get_sf_mock, _ = _make_db_session_factory()
        ccxt_module_mock, _ = _make_ccxt_module_mock()
        se_cls_mock, _ = _make_strategy_engine_mock(raise_on_run_loop=None)
        strategy_cls_mock = _make_strategy_cls_mock()

        live_engine_cls = MagicMock()
        live_engine_instance = MagicMock()
        live_engine_cls.return_value = live_engine_instance

        portfolio_cls = MagicMock()
        portfolio_cls.return_value = MagicMock()

        risk_mgr_cls = MagicMock()
        risk_mgr_cls.return_value = MagicMock()

        mds_cls = MagicMock()
        mds_cls.return_value = MagicMock()

        _RUN_TASKS.pop(_RUN_ID, None)

        with (
            patch(_PT_SETTINGS, return_value=settings_mock),
            patch(_PT_SESSION_FACTORY, get_sf_mock),
            patch(_PT_CCXT, ccxt_module_mock),
            patch(_PT_MDS, mds_cls),
            patch(_PT_LIVE_ENGINE, live_engine_cls),
            patch(_PT_PORTFOLIO, portfolio_cls),
            patch(_PT_RISK_MGR, risk_mgr_cls),
            patch(_PT_STRAT_ENGINE, se_cls_mock),
            patch(_PT_FLUSH, new_callable=AsyncMock),
            patch(_PT_FLUSH_LOOP, new_callable=AsyncMock),
        ):
            await _run_live_engine(
                run_id_str=_RUN_ID,
                strategy_cls=strategy_cls_mock,
                strategy_name=_STRATEGY_NAME,
                strategy_params={},
                symbols=[_SYMBOL],
                timeframe=TimeFrame.ONE_HOUR,
                initial_capital=_INITIAL_CAPITAL,
            )

        # Verify LiveExecutionEngine was constructed with enable_live_trading=True
        live_engine_cls.assert_called_once()
        call_kwargs = live_engine_cls.call_args.kwargs
        assert call_kwargs.get("enable_live_trading") is True, (
            f"LiveExecutionEngine must be created with enable_live_trading=True, "
            f"got: {call_kwargs!r}"
        )

    @pytest.mark.asyncio
    async def test_run_live_engine_uses_run_mode_live(self) -> None:
        """
        StrategyEngine must be constructed with run_mode=RunMode.LIVE.

        This ensures the strategy engine routes market events through the live
        execution path, not the paper or backtest path.
        """
        settings_mock = _make_settings_mock()
        get_sf_mock, _ = _make_db_session_factory()
        ccxt_module_mock, _ = _make_ccxt_module_mock()
        se_cls_mock, _ = _make_strategy_engine_mock(raise_on_run_loop=None)
        strategy_cls_mock = _make_strategy_cls_mock()

        _RUN_TASKS.pop(_RUN_ID, None)

        with (
            patch(_PT_SETTINGS, return_value=settings_mock),
            patch(_PT_SESSION_FACTORY, get_sf_mock),
            patch(_PT_CCXT, ccxt_module_mock),
            patch(_PT_MDS, MagicMock()),
            patch(_PT_LIVE_ENGINE, MagicMock()),
            patch(_PT_PORTFOLIO, MagicMock()),
            patch(_PT_RISK_MGR, MagicMock()),
            patch(_PT_STRAT_ENGINE, se_cls_mock),
            patch(_PT_FLUSH, new_callable=AsyncMock),
            patch(_PT_FLUSH_LOOP, new_callable=AsyncMock),
        ):
            await _run_live_engine(
                run_id_str=_RUN_ID,
                strategy_cls=strategy_cls_mock,
                strategy_name=_STRATEGY_NAME,
                strategy_params={},
                symbols=[_SYMBOL],
                timeframe=TimeFrame.ONE_HOUR,
                initial_capital=_INITIAL_CAPITAL,
            )

        se_cls_mock.assert_called_once()
        call_kwargs = se_cls_mock.call_args.kwargs
        assert call_kwargs.get("run_mode") == RunMode.LIVE, (
            f"StrategyEngine must be created with run_mode=RunMode.LIVE, "
            f"got: {call_kwargs.get('run_mode')!r}"
        )

    @pytest.mark.asyncio
    async def test_run_live_engine_flushes_on_normal_stop(self) -> None:
        """
        When run_live_loop() completes normally, _flush_incremental must
        be called at least once in the finally block (the final flush).

        This verifies the persistence contract for the clean-stop path.
        Sprint 25 replaced the one-time _persist_paper_results with
        incremental flushing — the final _flush_incremental in the finally
        block captures any remaining data.
        """
        settings_mock = _make_settings_mock()
        get_sf_mock, _ = _make_db_session_factory()
        ccxt_module_mock, _ = _make_ccxt_module_mock()
        se_cls_mock, _ = _make_strategy_engine_mock(raise_on_run_loop=None)
        strategy_cls_mock = _make_strategy_cls_mock()
        flush_mock = AsyncMock()

        _RUN_TASKS.pop(_RUN_ID, None)

        with (
            patch(_PT_SETTINGS, return_value=settings_mock),
            patch(_PT_SESSION_FACTORY, get_sf_mock),
            patch(_PT_CCXT, ccxt_module_mock),
            patch(_PT_MDS, MagicMock()),
            patch(_PT_LIVE_ENGINE, MagicMock()),
            patch(_PT_PORTFOLIO, MagicMock()),
            patch(_PT_RISK_MGR, MagicMock()),
            patch(_PT_STRAT_ENGINE, se_cls_mock),
            patch(_PT_FLUSH, flush_mock),
            patch(_PT_FLUSH_LOOP, new_callable=AsyncMock),
        ):
            await _run_live_engine(
                run_id_str=_RUN_ID,
                strategy_cls=strategy_cls_mock,
                strategy_name=_STRATEGY_NAME,
                strategy_params={},
                symbols=[_SYMBOL],
                timeframe=TimeFrame.ONE_HOUR,
                initial_capital=_INITIAL_CAPITAL,
            )

        flush_mock.assert_called_once()
        call_kwargs = flush_mock.call_args.kwargs
        assert call_kwargs.get("run_id_str") == _RUN_ID, (
            f"_flush_incremental called with wrong run_id_str: "
            f"{call_kwargs.get('run_id_str')!r}"
        )
        assert "state" in call_kwargs, (
            "_flush_incremental must receive 'state' kwarg (flush_state)"
        )

    @pytest.mark.asyncio
    async def test_run_live_engine_sets_error_status(self) -> None:
        """
        When run_live_loop() raises a non-CancelledError exception, the
        finally block must update final_status to "error" in the DB.

        Verification: the mock run object's status field is set to "error"
        before db.commit() is called. We inspect the mock_run's status
        attribute after the coroutine completes.
        """
        settings_mock = _make_settings_mock()
        get_sf_mock, db_mock = _make_db_session_factory(run_status="running")
        ccxt_module_mock, _ = _make_ccxt_module_mock()
        se_cls_mock, _ = _make_strategy_engine_mock(
            raise_on_run_loop=RuntimeError("exchange feed died")
        )
        strategy_cls_mock = _make_strategy_cls_mock()

        # Capture the mock run object to inspect its status after the call
        mock_run = MagicMock()
        mock_run.status = "running"
        execute_result = MagicMock()
        execute_result.scalar_one_or_none.return_value = mock_run
        db_mock.execute = AsyncMock(return_value=execute_result)

        _RUN_TASKS.pop(_RUN_ID, None)

        with (
            patch(_PT_SETTINGS, return_value=settings_mock),
            patch(_PT_SESSION_FACTORY, get_sf_mock),
            patch(_PT_CCXT, ccxt_module_mock),
            patch(_PT_MDS, MagicMock()),
            patch(_PT_LIVE_ENGINE, MagicMock()),
            patch(_PT_PORTFOLIO, MagicMock()),
            patch(_PT_RISK_MGR, MagicMock()),
            patch(_PT_STRAT_ENGINE, se_cls_mock),
            patch(_PT_FLUSH, new_callable=AsyncMock),
            patch(_PT_FLUSH_LOOP, new_callable=AsyncMock),
        ):
            await _run_live_engine(
                run_id_str=_RUN_ID,
                strategy_cls=strategy_cls_mock,
                strategy_name=_STRATEGY_NAME,
                strategy_params={},
                symbols=[_SYMBOL],
                timeframe=TimeFrame.ONE_HOUR,
                initial_capital=_INITIAL_CAPITAL,
            )

        # The mock run's status must have been set to "error" by the finally block
        assert mock_run.status == "error", (
            f"Run status must be 'error' after run_live_loop raises, "
            f"got: {mock_run.status!r}"
        )

    @pytest.mark.asyncio
    async def test_run_live_engine_closes_exchange(self) -> None:
        """
        exchange.close() must be called in the finally block regardless of
        whether run_live_loop() succeeded or raised an exception.

        This is a belt-and-suspenders close — LiveExecutionEngine.on_stop()
        also closes the exchange, but _run_live_engine closes it again to
        cover cases where on_stop() was never reached.
        """
        settings_mock = _make_settings_mock()
        get_sf_mock, _ = _make_db_session_factory()
        ccxt_module_mock, mock_exchange = _make_ccxt_module_mock()
        # Normal stop: run_live_loop completes without error
        se_cls_mock, _ = _make_strategy_engine_mock(raise_on_run_loop=None)
        strategy_cls_mock = _make_strategy_cls_mock()

        _RUN_TASKS.pop(_RUN_ID, None)

        with (
            patch(_PT_SETTINGS, return_value=settings_mock),
            patch(_PT_SESSION_FACTORY, get_sf_mock),
            patch(_PT_CCXT, ccxt_module_mock),
            patch(_PT_MDS, MagicMock()),
            patch(_PT_LIVE_ENGINE, MagicMock()),
            patch(_PT_PORTFOLIO, MagicMock()),
            patch(_PT_RISK_MGR, MagicMock()),
            patch(_PT_STRAT_ENGINE, se_cls_mock),
            patch(_PT_FLUSH, new_callable=AsyncMock),
            patch(_PT_FLUSH_LOOP, new_callable=AsyncMock),
        ):
            await _run_live_engine(
                run_id_str=_RUN_ID,
                strategy_cls=strategy_cls_mock,
                strategy_name=_STRATEGY_NAME,
                strategy_params={},
                symbols=[_SYMBOL],
                timeframe=TimeFrame.ONE_HOUR,
                initial_capital=_INITIAL_CAPITAL,
            )

        mock_exchange.close.assert_called_once(), (
            "exchange.close() must be called exactly once in the finally block"
        )

    @pytest.mark.asyncio
    async def test_run_live_engine_removes_from_run_tasks(self) -> None:
        """
        The run_id must be removed from _RUN_TASKS in the finally block,
        regardless of whether run_live_loop raised or completed normally.

        Setup: pre-seed _RUN_TASKS[_RUN_ID] with a dummy Task object to
        simulate the state that the POST handler creates when it calls
        asyncio.create_task(_run_live_engine(...)).

        Verification: after the coroutine returns, _RUN_TASKS must not
        contain _RUN_ID.
        """
        settings_mock = _make_settings_mock()
        get_sf_mock, _ = _make_db_session_factory()
        ccxt_module_mock, _ = _make_ccxt_module_mock()
        se_cls_mock, _ = _make_strategy_engine_mock(raise_on_run_loop=None)
        strategy_cls_mock = _make_strategy_cls_mock()

        # Pre-seed the task registry as the POST handler would
        _RUN_TASKS[_RUN_ID] = MagicMock()  # type: ignore[assignment]

        assert _RUN_ID in _RUN_TASKS, "Pre-condition: _RUN_TASKS must contain the run_id"

        with (
            patch(_PT_SETTINGS, return_value=settings_mock),
            patch(_PT_SESSION_FACTORY, get_sf_mock),
            patch(_PT_CCXT, ccxt_module_mock),
            patch(_PT_MDS, MagicMock()),
            patch(_PT_LIVE_ENGINE, MagicMock()),
            patch(_PT_PORTFOLIO, MagicMock()),
            patch(_PT_RISK_MGR, MagicMock()),
            patch(_PT_STRAT_ENGINE, se_cls_mock),
            patch(_PT_FLUSH, new_callable=AsyncMock),
            patch(_PT_FLUSH_LOOP, new_callable=AsyncMock),
        ):
            await _run_live_engine(
                run_id_str=_RUN_ID,
                strategy_cls=strategy_cls_mock,
                strategy_name=_STRATEGY_NAME,
                strategy_params={},
                symbols=[_SYMBOL],
                timeframe=TimeFrame.ONE_HOUR,
                initial_capital=_INITIAL_CAPITAL,
            )

        assert _RUN_ID not in _RUN_TASKS, (
            f"_RUN_TASKS must not contain run_id after coroutine exits, "
            f"but found: {_RUN_TASKS.get(_RUN_ID)!r}"
        )

    @pytest.mark.asyncio
    async def test_run_live_engine_passes_passphrase_to_exchange(self) -> None:
        """
        When settings.exchange_api_passphrase is configured, _run_live_engine()
        must pass it as the "password" key in the CCXT exchange config dict.

        This is the Coinbase compatibility requirement: Coinbase Advanced Trade
        requires a passphrase which CCXT expects under the "password" config key.
        Exchanges that do not need a passphrase (e.g. Binance) leave
        settings.exchange_api_passphrase as None, so the key is omitted entirely.

        Verification: inspect the positional argument passed to the exchange class
        constructor via ccxt_module_mock.binance.call_args[0][0].
        """
        settings_mock = _make_settings_mock()
        get_sf_mock, _ = _make_db_session_factory()
        ccxt_module_mock, _ = _make_ccxt_module_mock()
        se_cls_mock, _ = _make_strategy_engine_mock(raise_on_run_loop=None)
        strategy_cls_mock = _make_strategy_cls_mock()

        _RUN_TASKS.pop(_RUN_ID, None)

        with (
            patch(_PT_SETTINGS, return_value=settings_mock),
            patch(_PT_SESSION_FACTORY, get_sf_mock),
            patch(_PT_CCXT, ccxt_module_mock),
            patch(_PT_MDS, MagicMock()),
            patch(_PT_LIVE_ENGINE, MagicMock()),
            patch(_PT_PORTFOLIO, MagicMock()),
            patch(_PT_RISK_MGR, MagicMock()),
            patch(_PT_STRAT_ENGINE, se_cls_mock),
            patch(_PT_FLUSH, new_callable=AsyncMock),
            patch(_PT_FLUSH_LOOP, new_callable=AsyncMock),
        ):
            await _run_live_engine(
                run_id_str=_RUN_ID,
                strategy_cls=strategy_cls_mock,
                strategy_name=_STRATEGY_NAME,
                strategy_params={},
                symbols=[_SYMBOL],
                timeframe=TimeFrame.ONE_HOUR,
                initial_capital=_INITIAL_CAPITAL,
            )

        # The CCXT exchange class is called with a positional config dict.
        # Assert the passphrase is wired as 'password' in that dict.
        ccxt_module_mock.binance.assert_called_once()
        call_args = ccxt_module_mock.binance.call_args
        # _run_live_engine calls exchange_cls(exchange_config) with positional arg
        exchange_config = call_args[0][0]
        assert "password" in exchange_config, (
            f"Expected password key in CCXT exchange config, got: {exchange_config!r}"
        )
        assert exchange_config["password"] == "test-passphrase", (
            f"Expected passphrase value test-passphrase, got: {exchange_config!r}"
        )
