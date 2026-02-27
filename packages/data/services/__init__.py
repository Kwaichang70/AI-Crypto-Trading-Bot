"""
packages/data/services/__init__.py
------------------------------------
Public surface of the data.services sub-package.

Import ``CCXTMarketDataService`` from here rather than directly from the
module so that internal refactors do not break external callers.
"""

from __future__ import annotations

from data.services.ccxt_market_data import CCXTMarketDataService

__all__ = ["CCXTMarketDataService"]
