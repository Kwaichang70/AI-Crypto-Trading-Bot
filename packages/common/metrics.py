"""
packages/common/metrics.py
--------------------------
Lightweight in-memory metrics collection for MVP observability.

This module provides a ``MetricsCollector`` that tracks counters, gauges,
and histograms entirely in-memory with zero external dependencies.  The
collected data is exposed via the ``/api/v1/metrics`` FastAPI endpoint
(see ``apps/api/main.py``).

Sprint 2 migration path
-----------------------
Replace or wrap this module with ``prometheus_client`` so that existing
call sites need no changes::

    # Sprint 2: swap this implementation; call sites stay identical
    metrics.increment("fills_executed_total")
    metrics.gauge("portfolio_equity", 10_500.25)
    metrics.observe("bar_processing_duration_seconds", 0.003)

Usage
-----
Import the process-wide singleton::

    from common.metrics import metrics

    metrics.increment("bars_processed_total")
    metrics.gauge("portfolio_equity", 10_500.25)
    metrics.observe("bar_processing_duration_seconds", 0.003)

    # Time a code block automatically
    with metrics.timer("bar_processing_duration_seconds"):
        await process_bar(...)

    # Snapshot for the /api/v1/metrics endpoint
    snapshot = metrics.get_all()

Labelled metrics::

    metrics.increment(
        "signals_generated_total",
        labels={"strategy": "ma_crossover"},
    )
    metrics.increment(
        "orders_submitted_total",
        labels={"side": "buy"},
    )

Thread safety
-------------
All mutation methods acquire a ``threading.Lock``.  This is intentional:
SQLAlchemy's async bridge runs sync operations in a ``ThreadPoolExecutor``,
and Uvicorn may interact with application state from multiple threads.  The
lock is uncontended in the common case (single-threaded asyncio event loop)
so the overhead is negligible.

Pre-defined metrics (always present in ``get_all()`` output)
-------------------------------------------------------------
Counters:
    bars_processed_total
    signals_generated_total
    orders_submitted_total
    fills_executed_total

Gauges:
    portfolio_equity
    portfolio_drawdown_pct
    active_positions

Histograms:
    bar_processing_duration_seconds
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict
from contextlib import contextmanager
from typing import Any, Generator

__all__ = ["MetricsCollector", "metrics"]

# ---------------------------------------------------------------------------
# Label encoding
# ---------------------------------------------------------------------------


def _label_key(name: str, labels: dict[str, str] | None) -> str:
    """
    Encode a metric name and optional label pairs as a single storage key.

    Labels are sorted alphabetically for deterministic output so that callers
    need not pass them in any specific order.

    Parameters
    ----------
    name:
        Base metric name, e.g. ``"orders_submitted_total"``.
    labels:
        Optional label mapping, e.g. ``{"side": "buy", "symbol": "BTC/USDT"}``.

    Returns
    -------
    str
        Encoded key, e.g. ``'orders_submitted_total{side="buy"}'`` or
        ``"bars_processed_total"`` when labels is None/empty.

    Examples
    --------
    >>> _label_key("orders_total", {"side": "buy"})
    'orders_total{side="buy"}'
    >>> _label_key("orders_total", {"b": "2", "a": "1"})
    'orders_total{a="1",b="2"}'
    >>> _label_key("bars_total", None)
    'bars_total'
    """
    if not labels:
        return name
    pairs = ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()))
    return f"{name}{{{pairs}}}"


# ---------------------------------------------------------------------------
# Histogram summary
# ---------------------------------------------------------------------------


def _summarise_histogram(values: list[float]) -> dict[str, Any]:
    """
    Compute summary statistics for a histogram bucket list.

    Parameters
    ----------
    values:
        Raw observed float values.

    Returns
    -------
    dict
        Keys: ``count`` (int), ``sum``, ``min``, ``max``, ``mean``
        (all float or None when empty).
    """
    if not values:
        return {
            "count": 0,
            "sum": None,
            "min": None,
            "max": None,
            "mean": None,
        }
    total = sum(values)
    count = len(values)
    return {
        "count": count,
        "sum": round(total, 9),
        "min": round(min(values), 9),
        "max": round(max(values), 9),
        "mean": round(total / count, 9),
    }


# ---------------------------------------------------------------------------
# MetricsCollector
# ---------------------------------------------------------------------------


class MetricsCollector:
    """
    In-memory metrics collector.

    Tracks three metric types:

    - **Counter** — monotonically increasing integer.  Call ``increment()``.
    - **Gauge** — current floating-point reading.  Call ``gauge()``.
    - **Histogram** — list of observed float values.  Call ``observe()``
      or use the ``timer()`` context manager.

    All methods are thread-safe via an internal ``threading.Lock``.

    For test isolation, create a fresh ``MetricsCollector()`` instance per
    test rather than using the module-level ``metrics`` singleton.

    Parameters
    ----------
    None — construct directly.  Use the module-level ``metrics`` singleton
    in application code.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # defaultdict(int) so unseen counter keys start at 0
        self._counters: dict[str, int] = defaultdict(int)
        self._gauges: dict[str, float] = {}
        # defaultdict(list) so unseen histogram keys start with an empty list
        self._histograms: dict[str, list[float]] = defaultdict(list)

        # Pre-register all known metric names so the snapshot always contains
        # a complete schema, even before any trading activity has occurred.
        self._register_defaults()

    # ------------------------------------------------------------------
    # Public mutation API
    # ------------------------------------------------------------------

    def increment(
        self,
        name: str,
        value: int = 1,
        labels: dict[str, str] | None = None,
    ) -> None:
        """
        Increment a counter metric by ``value``.

        Parameters
        ----------
        name:
            Counter metric name, e.g. ``"bars_processed_total"``.
        value:
            Amount to add.  Must be >= 1.
        labels:
            Optional label dict for cardinality breakdowns, e.g.
            ``{"strategy": "ma_crossover"}``.

        Raises
        ------
        ValueError
            If ``value < 1``.
        """
        if value < 1:
            raise ValueError(
                f"Counter increment value must be >= 1, got {value!r}"
            )
        key = _label_key(name, labels)
        with self._lock:
            self._counters[key] += value

    def gauge(
        self,
        name: str,
        value: float,
        labels: dict[str, str] | None = None,
    ) -> None:
        """
        Set a gauge metric to ``value``.

        The gauge is overwritten on each call — it represents the current
        reading of a continuously-varying quantity.

        Parameters
        ----------
        name:
            Gauge metric name, e.g. ``"portfolio_equity"``.
        value:
            Current measurement (any finite float).
        labels:
            Optional label dict.
        """
        key = _label_key(name, labels)
        with self._lock:
            self._gauges[key] = float(value)

    def observe(
        self,
        name: str,
        value: float,
        labels: dict[str, str] | None = None,
    ) -> None:
        """
        Record an observation into a histogram bucket.

        All raw values are stored; summary statistics (count, sum, min, max,
        mean) are computed on read in ``get_all()``.

        Sprint 2 note: for high-frequency histograms, replace the raw list
        with a fixed-size circular buffer or pre-bucketed implementation to
        cap memory usage.

        Parameters
        ----------
        name:
            Histogram metric name, e.g. ``"bar_processing_duration_seconds"``.
        value:
            Observed measurement (e.g. elapsed seconds).
        labels:
            Optional label dict.
        """
        key = _label_key(name, labels)
        with self._lock:
            self._histograms[key].append(float(value))

    # ------------------------------------------------------------------
    # Timing helper
    # ------------------------------------------------------------------

    @contextmanager
    def timer(
        self,
        name: str,
        labels: dict[str, str] | None = None,
    ) -> Generator[None, None, None]:
        """
        Context manager that automatically observes wall-clock duration.

        Measures the elapsed time from ``__enter__`` to ``__exit__`` (in
        seconds) and records it via ``observe()``.  The observation is
        recorded even if the body raises an exception.

        Parameters
        ----------
        name:
            Histogram metric name to record the duration into.
        labels:
            Optional label dict.

        Examples
        --------
        ::

            with metrics.timer("bar_processing_duration_seconds"):
                await _process_bar(current_bars, history)
        """
        start = time.monotonic()
        try:
            yield
        finally:
            self.observe(name, time.monotonic() - start, labels)

    # ------------------------------------------------------------------
    # Snapshot
    # ------------------------------------------------------------------

    def get_all(self) -> dict[str, Any]:
        """
        Return a serialisable snapshot of all metrics.

        Acquires the lock once and copies all internal state, so the
        returned dict is safe to mutate without affecting the collector.

        Returns
        -------
        dict[str, Any]
            Structure::

                {
                    "counters": {"metric_key": int, ...},
                    "gauges": {"metric_key": float, ...},
                    "histograms": {
                        "metric_key": {
                            "count": int,
                            "sum": float | None,
                            "min": float | None,
                            "max": float | None,
                            "mean": float | None,
                        },
                        ...
                    },
                }
        """
        with self._lock:
            counters_snapshot = dict(self._counters)
            gauges_snapshot = dict(self._gauges)
            histograms_snapshot = {
                k: _summarise_histogram(list(v))
                for k, v in self._histograms.items()
            }

        return {
            "counters": counters_snapshot,
            "gauges": gauges_snapshot,
            "histograms": histograms_snapshot,
        }

    # ------------------------------------------------------------------
    # Test utility
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """
        Clear all recorded metrics and restore default zero values.

        Intended for test isolation only — do not call in production code.
        """
        with self._lock:
            self._counters.clear()
            self._gauges.clear()
            self._histograms.clear()
            self._register_defaults()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _register_defaults(self) -> None:
        """
        Pre-populate zero/empty values for all known trading metrics.

        Called during ``__init__`` and ``reset()``.  Ensures the metrics
        snapshot always contains a complete, predictable schema — the
        ``/api/v1/metrics`` endpoint returns all fields before any trading
        activity has occurred.
        """
        # Counters: all start at 0
        _default_counters = (
            "bars_processed_total",
            "signals_generated_total",
            "orders_submitted_total",
            "fills_executed_total",
        )
        for counter_name in _default_counters:
            self._counters.setdefault(counter_name, 0)

        # Gauges: all start at 0.0
        _default_gauges = (
            "portfolio_equity",
            "portfolio_drawdown_pct",
            "active_positions",
        )
        for gauge_name in _default_gauges:
            self._gauges.setdefault(gauge_name, 0.0)

        # Histograms: start as empty lists
        self._histograms.setdefault("bar_processing_duration_seconds", [])


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

metrics: MetricsCollector = MetricsCollector()
"""
Process-wide metrics singleton.

This is the primary object used throughout the application::

    from common.metrics import metrics

    # In the bar processing loop:
    metrics.increment("bars_processed_total")
    with metrics.timer("bar_processing_duration_seconds"):
        await _process_bar(...)

    # After portfolio update:
    metrics.gauge("portfolio_equity", float(portfolio.equity))
    metrics.gauge("portfolio_drawdown_pct", float(portfolio.max_drawdown_pct))
    metrics.gauge("active_positions", len(portfolio.positions))

    # On fill:
    metrics.increment("fills_executed_total")

    # With labels:
    metrics.increment("signals_generated_total", labels={"strategy": strategy_id})
    metrics.increment("orders_submitted_total", labels={"side": order.side.value})
"""
