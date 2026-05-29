"""Add walk_forward_oos_skill_score column to model_versions (Sprint 50 Cycle 5)

Revision ID: 016
Revises: 015
Create Date: 2026-05-29 00:00:00.000000 UTC

Description
-----------
Adds a nullable Numeric(10,6) column ``walk_forward_oos_skill_score`` to the
``model_versions`` table.  NULL for pre-Cycle-5 models; the activation
gate treats NULL as pass-with-warning.

Stores the DEFLATED MEDIAN OOS directional z-score skill proxy (2*acc-1)*sqrt(n)
across walk-forward folds (v2 schema).

NOTE: This is a directional z-score skill proxy, NOT a trading Sharpe --
see Cycle 5 quant review and reports/sprint50-cycle5-quant-backlog.md.
The metric is magnitude-blind: does not account for fees, slippage, or sizing.
A real per-fold BacktestRunner OOS gate (Cycle 6+) will replace this proxy.

Raw per-fold values are stored in the existing ``extra`` JSONB column under
key "walk_forward" -- no additional column is needed for fold metadata.
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "016"
down_revision: Union[str, None] = "015"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "model_versions",
        sa.Column(
            "walk_forward_oos_skill_score",
            sa.Numeric(precision=10, scale=6),
            nullable=True,
            comment=(
                "DEFLATED MEDIAN OOS directional z-score skill proxy (2*acc-1)*sqrt(n) "
                "across walk-forward folds (v2 schema). NOT a trading Sharpe -- "
                "magnitude-blind; does not account for fees/slippage/sizing. "
                "NULL for pre-Sprint-50-Cycle-5 models. "
                "Raw per-fold values in model_versions.extra['walk_forward']."
            ),
        ),
    )


def downgrade() -> None:
    op.drop_column("model_versions", "walk_forward_oos_skill_score")
