"""
packages/trading/strategy.py
-----------------------------
Abstract Strategy base class and supporting types.

Every concrete strategy MUST:
1. Inherit from ``BaseStrategy``
2. Implement ``on_bar`` — the single mandatory hook called each candle
3. Optionally override ``on_start`` / ``on_stop`` for lifecycle events
4. Declare its parameter schema by overriding ``parameter_schema``

Design contract
---------------
- ``on_bar`` must be synchronous for backtesting vectorisation.
- Strategies are stateless with respect to order placement;
  they only produce Signals. The ExecutionEngine and RiskManager consume them.
- Strategy IDs must be globally unique within a run.
"""

from __future__ import annotations

import abc
from collections.abc import Sequence
from typing import Any, ClassVar

import structlog

from common.models import MultiTimeframeContext, OHLCVBar
from trading.models import Signal

__all__ = ["BaseStrategy", "StrategyMetadata"]

logger = structlog.get_logger(__name__)


class StrategyMetadata:
    """
    Declarative metadata attached to each Strategy class.

    Set as a ClassVar to enable introspection without instantiation.
    """

    __slots__ = ("name", "version", "description", "author", "tags")

    def __init__(
        self,
        *,
        name: str,
        version: str = "0.1.0",
        description: str = "",
        author: str = "",
        tags: list[str] | None = None,
    ) -> None:
        self.name = name
        self.version = version
        self.description = description
        self.author = author
        self.tags: list[str] = tags or []

    def __repr__(self) -> str:
        return (
            f"StrategyMetadata(name={self.name!r}, version={self.version!r})"
        )


class BaseStrategy(abc.ABC):
    """
    Abstract base class for all trading strategies.

    Lifecycle
    ---------
    1. ``__init__`` — inject strategy parameters (validated by subclass)
    2. ``on_start(run_id)`` — called once before the first bar
    3. ``on_bar(bars)`` — called on every new candle batch; returns Signal list
    4. ``on_stop()`` — called once after the last bar or on graceful shutdown

    Parameters
    ----------
    strategy_id:
        Unique string identifying this strategy instance within a run.
        Used as the ``strategy_id`` field on emitted Signals.
    params:
        Arbitrary key-value parameters validated by the concrete subclass.
    """

    #: Subclasses SHOULD override this with a StrategyMetadata instance.
    metadata: ClassVar[StrategyMetadata] = StrategyMetadata(name="BaseStrategy")

    def __init__(self, strategy_id: str, params: dict[str, Any] | None = None) -> None:
        self._strategy_id = strategy_id
        self._params: dict[str, Any] = self._validate_params(params or {})
        self._run_id: str | None = None
        self._log = structlog.get_logger(__name__).bind(strategy_id=strategy_id)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def strategy_id(self) -> str:
        return self._strategy_id

    @property
    def run_id(self) -> str | None:
        return self._run_id

    @property
    def params(self) -> dict[str, Any]:
        return dict(self._params)

    @property
    def min_bars_required(self) -> int:
        """
        Minimum number of historical bars this strategy needs before it
        can produce meaningful signals.

        Subclasses MUST override this property. The backtesting engine
        uses it to skip the warm-up period in metrics calculations, and
        the data pipeline uses it to pre-fetch the correct history length.

        Returns
        -------
        int
            Minimum bar count. Default 0 (subclasses override).
        """
        return 0

    @property
    def htf_timeframes(self) -> list[str]:
        """
        Higher timeframes this strategy wants to consume.

        Override to declare which higher timeframes the strategy needs.
        The engine will fetch and provide these via mtf_context.

        Returns
        -------
        list[str]
            Timeframe strings (e.g. ["4h", "1d"]). Default empty.
        """
        return []

    # ------------------------------------------------------------------
    # Lifecycle hooks
    # ------------------------------------------------------------------

    def on_start(self, run_id: str) -> None:
        """
        Called once before the first bar is processed.

        Use this hook to initialise any stateful accumulators, load
        model weights, or warm up indicator buffers.

        Parameters
        ----------
        run_id:
            The run identifier for the current trading session.
        """
        self._run_id = run_id
        self._log.info("strategy.started", run_id=run_id)

    @abc.abstractmethod
    def on_bar(
        self,
        bars: Sequence[OHLCVBar],
        *,
        mtf_context: MultiTimeframeContext | None = None,
    ) -> list[Signal]:
        """
        Process a batch of OHLCV bars and return zero or more signals.

        This method is called on every completed candle. In backtest mode
        ``bars`` contains the full history up to and including the current
        bar. In paper/live mode it contains only recent history sufficient
        for indicator warm-up.

        Parameters
        ----------
        bars:
            Sequence of OHLCVBar objects, oldest first, most recent last.
            The strategy must NOT look ahead (i.e. must not access bars
            beyond the last element).
        mtf_context:
            Optional higher-timeframe bar context. Contains bars from
            timeframes above the primary, filtered to prevent look-ahead
            bias. None if no higher timeframes are configured.

        Returns
        -------
        list[Signal]:
            Zero or more Signal objects. An empty list is equivalent to
            HOLD for all symbols.
        """
        ...

    def on_stop(self) -> None:
        """
        Called once after the last bar or on graceful shutdown.

        Override to flush buffers, persist state, or release resources.
        Default implementation is a no-op.
        """
        self._log.info("strategy.stopped", run_id=self._run_id)

    # ------------------------------------------------------------------
    # Parameter validation
    # ------------------------------------------------------------------

    def _validate_params(self, params: dict[str, Any]) -> dict[str, Any]:
        """
        Validate strategy parameters against ``parameter_schema``.

        The default implementation performs no validation. Subclasses
        should override this with Pydantic model validation.

        Parameters
        ----------
        params:
            Raw parameter dictionary from the API request or config file.

        Returns
        -------
        dict[str, Any]:
            Validated (and possibly coerced) parameter dictionary.
        """
        return params

    @classmethod
    def parameter_schema(cls) -> dict[str, Any]:
        """
        Return a JSON Schema dict describing accepted parameters.

        Used by the API to validate strategy config payloads before
        instantiation. Subclasses should return a valid JSON Schema object.

        Returns
        -------
        dict[str, Any]:
            JSON Schema object. Default returns an empty schema
            (accepts anything).
        """
        return {"type": "object", "properties": {}, "additionalProperties": True}

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"strategy_id={self._strategy_id!r}, "
            f"run_id={self._run_id!r})"
        )
