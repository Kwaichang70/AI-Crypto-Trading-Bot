"""${message}

Revision ID: ${up_revision}
Revises: ${down_revision | comma,n}
Create Date: ${create_date}

Description
-----------
<Add a clear human-readable description of what this migration does and WHY.
Include any data migration steps and their rollback strategy.>

Rollback strategy
-----------------
<Describe how to safely roll back this migration if it fails in production.
Note any irreversible operations (e.g., DROP COLUMN with data loss).>

Performance considerations
--------------------------
<Note any table scans, locks, or long-running operations this migration
may trigger in production (e.g., adding an index CONCURRENTLY).>
"""

from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
${imports if imports else ""}

# revision identifiers, used by Alembic.
revision: str = ${repr(up_revision)}
down_revision: Union[str, None] = ${repr(down_revision)}
branch_labels: Union[str, Sequence[str], None] = ${repr(branch_labels)}
depends_on: Union[str, Sequence[str], None] = ${repr(depends_on)}


def upgrade() -> None:
    """Apply this migration."""
    ${upgrades if upgrades else "pass"}


def downgrade() -> None:
    """Revert this migration."""
    ${downgrades if downgrades else "pass"}
