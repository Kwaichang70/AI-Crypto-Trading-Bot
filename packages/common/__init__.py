"""
packages/common
---------------
Shared domain types, value objects, configuration, logging, and metrics
used across all packages and applications in the trading bot.

Public API
----------
From ``common.types``:
    OrderSide, OrderType, OrderStatus, SignalDirection, RunMode, TimeFrame,
    AssetClass, LogLevel, QuoteCurrency

From ``common.models``:
    OHLCVBar

From ``common.config``:
    BaseAppConfig, generate_run_id, configure_structlog

From ``common.logging``:
    configure_logging, get_logger, set_request_id, clear_request_id

From ``common.redaction``:
    redact_text, register_secret, register_secrets_from_env, RedactingRenderer

From ``common.metrics``:
    MetricsCollector, metrics

From ``common.events``:
    BaseEvent, EventPriority, BarEvent, SignalEvent, OrderEvent, FillEvent,
    RiskEvent, EventBus
"""

from common.config import BaseAppConfig, configure_structlog, generate_run_id
from common.logging import (
    clear_request_id,
    configure_logging,
    get_logger,
    set_request_id,
)
from common.metrics import MetricsCollector, metrics
from common.redaction import (
    RedactingRenderer,
    redact_text,
    register_secret,
    register_secrets_from_env,
)
from common.models import OHLCVBar
from common.types import (
    AssetClass,
    LogLevel,
    OrderSide,
    OrderStatus,
    OrderType,
    QuoteCurrency,
    RunMode,
    SignalDirection,
    TimeFrame,
)
from common.events import (
    BaseEvent,
    EventPriority,
    BarEvent,
    SignalEvent,
    OrderEvent,
    FillEvent,
    RiskEvent,
    EventBus,
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
    "QuoteCurrency",
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
    # redaction (CF-36)
    "RedactingRenderer",
    "redact_text",
    "register_secret",
    "register_secrets_from_env",
    # metrics
    "MetricsCollector",
    "metrics",
    # events
    "BaseEvent",
    "EventPriority",
    "BarEvent",
    "SignalEvent",
    "OrderEvent",
    "FillEvent",
    "RiskEvent",
    "EventBus",
]
