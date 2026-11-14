"""CF-26-97 P2: the frozen reference for ``PortfolioAccounting.get_max_drawdown``.

``reference_get_max_drawdown`` is the body of ``PortfolioAccounting.get_max_drawdown`` at base
commit c1b45f5f6b0e07e74b6aa263ec4635d549969f36 (the docstring dropped, nothing else changed),
extracted with ``ast`` from ``git show c1b45f5:packages/trading/portfolio.py``.  ``self`` is any
object with an ``_equity_curve`` list.  ``reference_max_dd_decimal`` is the same statements
returning the Decimal before ``float()``.  Never edit either to follow the production code.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any


def reference_get_max_drawdown(self: Any) -> float:
    if len(self._equity_curve) < 2:
        return 0.0

    peak = Decimal("0")
    max_dd = Decimal("0")

    for _, equity in self._equity_curve:
        if equity > peak:
            peak = equity
        if peak > Decimal("0"):
            dd = (peak - equity) / peak
            if dd > max_dd:
                max_dd = dd

    return float(max_dd)


def reference_max_dd_decimal(self: Any) -> Decimal:
    if len(self._equity_curve) < 2:
        return Decimal("0")

    peak = Decimal("0")
    max_dd = Decimal("0")

    for _, equity in self._equity_curve:
        if equity > peak:
            peak = equity
        if peak > Decimal("0"):
            dd = (peak - equity) / peak
            if dd > max_dd:
                max_dd = dd

    return max_dd
