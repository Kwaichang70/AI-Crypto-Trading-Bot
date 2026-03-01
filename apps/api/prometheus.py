"""
apps/api/prometheus.py
-----------------------
Prometheus-compatible metrics scrape endpoint for the AI Crypto Trading Bot.

Bridges the existing ``MetricsCollector`` (in-memory counters, gauges,
histograms) to Prometheus text exposition format via ``prometheus-client``.

Architecture
------------
Uses a **pull-on-scrape** pattern with a custom ``Collector``. On every
``GET /metrics`` scrape request, ``TradingBotCollector.collect()`` reads the
current ``MetricsCollector`` state and yields Prometheus metric families.
No background sync thread is needed — metrics are always fresh.

A custom ``CollectorRegistry`` is used instead of the global default to
avoid conflicts with test-suite collectors and other libraries that
register on the global registry.

The endpoint is always public (no auth, no rate limiting) since monitoring
probes need unfettered access.

Metric naming
-------------
Internal names from ``MetricsCollector`` are mapped to Prometheus-convention
names with ``trading_`` prefix and appropriate suffixes:

  bars_processed_total        →  trading_bars_processed_total       (Counter)
  signals_generated_total     →  trading_signals_generated_total    (Counter)
  orders_submitted_total      →  trading_orders_submitted_total     (Counter)
  fills_executed_total        →  trading_fills_executed_total       (Counter)
  portfolio_equity            →  trading_portfolio_equity           (Gauge)
  portfolio_drawdown_pct      →  trading_drawdown_pct               (Gauge)
  active_positions            →  trading_active_positions           (Gauge)
  bar_processing_duration_seconds → trading_bar_processing_seconds  (Summary)

Additionally:
  app_uptime_seconds          →  app_uptime_seconds                (Gauge, synthetic)

References
----------
- prometheus-client: https://github.com/prometheus/client_python
- Text exposition format: https://prometheus.io/docs/instrumenting/exposition_formats/
"""

from __future__ import annotations

import re
import time
from typing import Any, Iterator

import structlog
from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Summary,
    generate_latest,
)
from prometheus_client.core import (
    CounterMetricFamily,
    GaugeMetricFamily,
    SummaryMetricFamily,
)
from prometheus_client.registry import Collector

from api.config import get_settings
from common.metrics import metrics as metrics_collector

__all__ = ["setup_prometheus", "PROMETHEUS_REGISTRY"]

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Custom registry — isolated from global default
# ---------------------------------------------------------------------------

PROMETHEUS_REGISTRY = CollectorRegistry(auto_describe=True)

# Content type for Prometheus text exposition format
CONTENT_TYPE_LATEST = "text/plain; version=0.0.4; charset=utf-8"


# ---------------------------------------------------------------------------
# Label key parsing
# ---------------------------------------------------------------------------

_LABEL_RE = re.compile(r"^([^{]+)(?:\{([^}]+)\})?$")
_PAIR_RE = re.compile(r'(\w+)="([^"]*)"')


def _parse_label_key(key: str) -> tuple[str, dict[str, str]]:
    """
    Parse a MetricsCollector storage key into (name, labels).

    The MetricsCollector encodes labels like:
        ``orders_submitted_total{side="buy"}``

    This function extracts:
        ``("orders_submitted_total", {"side": "buy"})``

    Parameters
    ----------
    key:
        Encoded metric key from MetricsCollector.

    Returns
    -------
    tuple[str, dict[str, str]]
        (metric_name, label_dict)
    """
    match = _LABEL_RE.match(key)
    if not match:
        return key, {}

    name = match.group(1)
    label_str = match.group(2)

    if not label_str:
        return name, {}

    labels = dict(_PAIR_RE.findall(label_str))
    return name, labels


# ---------------------------------------------------------------------------
# Metric name mapping
# ---------------------------------------------------------------------------

_COUNTER_MAP: dict[str, str] = {
    "bars_processed_total": "trading_bars_processed_total",
    "signals_generated_total": "trading_signals_generated_total",
    "orders_submitted_total": "trading_orders_submitted_total",
    "fills_executed_total": "trading_fills_executed_total",
    "http_requests_total": "http_requests_total",
}

_GAUGE_MAP: dict[str, str] = {
    "portfolio_equity": "trading_portfolio_equity",
    "portfolio_drawdown_pct": "trading_drawdown_pct",
    "active_positions": "trading_active_positions",
}

_SUMMARY_MAP: dict[str, str] = {
    "bar_processing_duration_seconds": "trading_bar_processing_seconds",
    "http_request_duration_seconds": "http_request_duration_seconds",
}


# ---------------------------------------------------------------------------
# Custom collector — pull-on-scrape bridge
# ---------------------------------------------------------------------------

class TradingBotCollector(Collector):
    """
    Prometheus collector that bridges MetricsCollector to Prometheus format.

    On each scrape, reads the MetricsCollector snapshot and yields metric
    families. This is a stateless pull-on-scrape design — no background
    sync is needed.

    Parameters
    ----------
    app_start_time:
        Monotonic timestamp for uptime calculation.
    """

    def __init__(self, app_start_time: float) -> None:
        self._app_start_time = app_start_time

    def describe(self) -> list[Any]:
        """Return empty list — defers to collect() at registration time."""
        return []

    def collect(self) -> Iterator[Any]:
        """
        Yield Prometheus metric families from MetricsCollector state.

        Reads counters, gauges, and histograms from the in-memory
        MetricsCollector and converts them to Prometheus exposition format.
        """
        snapshot = metrics_collector.get_all()

        # --- Counters ---
        yield from self._collect_counters(snapshot.get("counters", {}))

        # --- Gauges ---
        yield from self._collect_gauges(snapshot.get("gauges", {}))

        # --- Histograms (exposed as Summaries) ---
        yield from self._collect_histograms(snapshot.get("histograms", {}))

        # --- Synthetic: app_uptime_seconds ---
        uptime = GaugeMetricFamily(
            "app_uptime_seconds",
            "Application uptime in seconds since startup",
        )
        uptime.add_metric([], time.monotonic() - self._app_start_time)
        yield uptime

    def _collect_counters(self, counters: dict[str, int]) -> Iterator[Any]:
        """Convert MetricsCollector counters to Prometheus counter families."""
        # Group by base metric name (may have different label sets)
        grouped: dict[str, list[tuple[dict[str, str], float]]] = {}
        for key, value in counters.items():
            name, labels = _parse_label_key(key)
            prom_name = _COUNTER_MAP.get(name, f"trading_{name}")
            if prom_name not in grouped:
                grouped[prom_name] = []
            grouped[prom_name].append((labels, float(value)))

        for prom_name, samples in grouped.items():
            # Determine label names from first sample
            label_names = sorted(samples[0][0].keys()) if samples[0][0] else []
            family = CounterMetricFamily(
                prom_name,
                f"Counter: {prom_name}",
                labels=label_names,
            )
            for labels, sample_value in samples:
                label_values = [labels.get(k, "") for k in label_names]
                family.add_metric(label_values, sample_value)
            yield family

    def _collect_gauges(self, gauges: dict[str, float]) -> Iterator[Any]:
        """Convert MetricsCollector gauges to Prometheus gauge families."""
        grouped: dict[str, list[tuple[dict[str, str], float]]] = {}
        for key, value in gauges.items():
            name, labels = _parse_label_key(key)
            prom_name = _GAUGE_MAP.get(name, f"trading_{name}")
            if prom_name not in grouped:
                grouped[prom_name] = []
            grouped[prom_name].append((labels, float(value)))

        for prom_name, samples in grouped.items():
            label_names = sorted(samples[0][0].keys()) if samples[0][0] else []
            family = GaugeMetricFamily(
                prom_name,
                f"Gauge: {prom_name}",
                labels=label_names,
            )
            for labels, value in samples:
                label_values = [labels.get(k, "") for k in label_names]
                family.add_metric(label_values, value)
            yield family

    def _collect_histograms(self, histograms: dict[str, dict[str, Any]]) -> Iterator[Any]:
        """Convert MetricsCollector histogram summaries to Prometheus summary families."""
        for key, summary_data in histograms.items():
            name, labels = _parse_label_key(key)
            prom_name = _SUMMARY_MAP.get(name, f"trading_{name}")

            count = summary_data.get("count", 0)
            total = summary_data.get("sum") or 0.0

            label_names = sorted(labels.keys()) if labels else []
            family = SummaryMetricFamily(
                prom_name,
                f"Summary: {prom_name}",
                labels=label_names,
            )
            label_values = [labels.get(k, "") for k in label_names]
            family.add_metric(label_values, count_value=count, sum_value=total)
            yield family


# ---------------------------------------------------------------------------
# Application setup
# ---------------------------------------------------------------------------

_PROMETHEUS_CONFIGURED: bool = False


def setup_prometheus(app: FastAPI) -> None:
    """
    Configure the Prometheus /metrics scrape endpoint.

    This function should be called from ``create_app()`` in ``main.py``.
    When ``prometheus_enabled=False`` in settings, this function is a no-op.

    Idempotent: the collector is registered on the custom registry only once,
    even if ``create_app()`` is called multiple times (e.g. in test fixtures).
    (CR-S2P2-002)

    Parameters
    ----------
    app:
        The FastAPI application instance.
    """
    global _PROMETHEUS_CONFIGURED

    settings = get_settings()

    if not settings.prometheus_enabled:
        logger.info("prometheus.disabled")
        return

    # Register the custom collector only once (CR-S2P2-002)
    if not _PROMETHEUS_CONFIGURED:
        collector = TradingBotCollector(
            app_start_time=time.monotonic(),
        )
        PROMETHEUS_REGISTRY.register(collector)
        _PROMETHEUS_CONFIGURED = True

    # Mount the /metrics endpoint — always public, no auth
    @app.get(
        "/metrics",
        include_in_schema=False,
        tags=["observability"],
    )
    async def prometheus_metrics() -> PlainTextResponse:
        """Prometheus text exposition format scrape endpoint."""
        output = generate_latest(PROMETHEUS_REGISTRY)
        return PlainTextResponse(
            content=output,
            media_type=CONTENT_TYPE_LATEST,
        )

    logger.info(
        "prometheus.configured",
        endpoint="/metrics",
    )
