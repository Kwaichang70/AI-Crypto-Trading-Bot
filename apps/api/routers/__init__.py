"""
apps/api/routers/__init__.py
-----------------------------
Router package for the AI Crypto Trading Bot API.

Each module in this package provides a FastAPI ``APIRouter`` that is
registered on the application via ``_register_routes()`` in ``main.py``.

Modules
-------
runs        -- Trading run lifecycle (start, list, get, stop)
orders      -- Order and fill queries scoped to a run
portfolio   -- Portfolio summary, equity curve, trades, positions
strategies  -- Strategy discovery (static listing)
"""

from api.routers import orders, portfolio, runs, strategies

__all__ = ["runs", "orders", "portfolio", "strategies"]
