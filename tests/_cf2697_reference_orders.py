"""CF-26-97 P3: the frozen reference for ``BaseExecutionEngine.get_open_orders``.

``reference_get_open_orders`` is the body of ``BaseExecutionEngine.get_open_orders`` at base commit
c1b45f5f6b0e07e74b6aa263ec4635d549969f36 (the docstring dropped, nothing else changed),
extracted with ``ast`` from ``git show c1b45f5:packages/trading/execution.py``.  ``self`` is any
object with an ``_orders`` dict.  It lives in its own module so that the P2 plugin, which imports
``tests._cf2697_reference`` before the workspace paths exist, keeps working.  Never edit it to
follow the production code.
"""

from __future__ import annotations

from typing import Any

from common.types import OrderStatus
from trading.models import Order


def reference_get_open_orders(self: Any) -> list[Order]:
    # P3: the body of BaseExecutionEngine.get_open_orders at c1b45f5 (docstring dropped), extracted
    # with ast from `git show c1b45f5:packages/trading/execution.py`; `self` has an `_orders` dict.
    terminal = {
        OrderStatus.FILLED,
        OrderStatus.CANCELED,
        OrderStatus.REJECTED,
        OrderStatus.EXPIRED,
    }
    return [o for o in self._orders.values() if o.status not in terminal]
