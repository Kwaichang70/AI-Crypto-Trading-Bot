"""Add expected_price and slippage_bps_realized to fills (Sprint 42 QT-009)

Revision ID: 009
Revises: 008
Create Date: 2026-05-17 00:00:00.000000 UTC

Description
-----------
Sprint 42 QT-009 — execution-quality telemetry.  Adds two nullable columns
to the ``fills`` table so the engine can persist what it intended to pay /
receive next to what the exchange actually filled at.

Columns
-------
    expected_price          Numeric(20, 8) NULL
    slippage_bps_realized   Numeric(10, 4) NULL

Both columns are nullable so historical fills written before this sprint
remain valid; new rows from the paper engine (and later the live engine)
will populate them at fill-creation time.

No downgrade-time data conversion is required — dropping the columns is
safe because they hold strictly additional information; PnL, fees, and
position state are unaffected.
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# ---------------------------------------------------------------------------
# Revision metadata
# ---------------------------------------------------------------------------
revision: str = "009"
down_revision: Union[str, None] = "008"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "fills",
        sa.Column(
            "expected_price",
            sa.Numeric(precision=20, scale=8),
            nullable=True,
            comment=(
                "Pre-execution price expectation (market: last price before "
                "slippage; limit: order.price).  NULL on historical fills."
            ),
        ),
    )
    op.add_column(
        "fills",
        sa.Column(
            "slippage_bps_realized",
            sa.Numeric(precision=10, scale=4),
            nullable=True,
            comment=(
                "Realised slippage |price-expected|/expected*10000 in basis "
                "points.  NULL when no expected_price was captured."
            ),
        ),
    )


def downgrade() -> None:
    op.drop_column("fills", "slippage_bps_realized")
    op.drop_column("fills", "expected_price")
