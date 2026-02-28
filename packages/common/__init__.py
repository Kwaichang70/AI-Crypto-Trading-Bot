"""
packages/common
---------------
Shared domain types, value objects, configuration, logging, and metrics
used across all packages and applications in the trading bot.

Public API
----------
From ``common.types``:
    OrderSide, OrderType, OrderStatus, SignalDirection, RunMode, TimeFrame,
    AssetClass, LogLevel

From ``common.models``:
    OHLCVBar

From ``common.config``:
    BaseAppConfig, generate_run_id, configure_structlog

From ``common.logging``:
    configure_logging, get_logger, set_request_id, clear_request_id

From ``common.metrics``:
    MetricsCollector, metrics
"""

from common.config import BaseAppConfig, configure_structlog, generate_run_id
from common.logging import (
    clear_request_id,
    configure_logging,
    get_logger,
    set_request_id,
)
from common.metrics import MetricsCollector, metrics
from common.models import OHLCVBar
from common.types import (
    AssetClass,
    LogLevel,
    OrderSide,
    OrderStatus,
    OrderType,
    RunMode,
    SignalDirection,
    TimeFrame,
)

__all__ = [
    # types
    "OrderSide",
    "OrderType",
    "OrderStatus",
    "SignalDirection",
    "RunMode",
    "TimeFrame",
    "AssetClass",
    "LogLevel",
    # models
    "OHLCVBar",
    # config
    "BaseAppConfig",
    "generate_run_id",
    "configure_structlog",
    # logging
    "configure_logging",
    "get_logger",
    "set_request_id",
    "clear_request_id",
    # metrics
    "MetricsCollector",
    "metrics",
]
