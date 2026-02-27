"""
packages/trading/engines/__init__.py
--------------------------------------
Concrete execution engine implementations.

- PaperExecutionEngine: Simulated fills for backtesting and paper trading.
- LiveExecutionEngine:  Real order placement via CCXT exchange connectors.
"""

from __future__ import annotations

from trading.engines.live import LiveExecutionEngine
from trading.engines.paper import PaperExecutionEngine

__all__ = [
    "PaperExecutionEngine",
    "LiveExecutionEngine",
]
