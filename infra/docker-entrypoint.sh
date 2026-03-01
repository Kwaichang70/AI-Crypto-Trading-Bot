#!/bin/bash
# =============================================================================
# docker-entrypoint.sh — Container startup script for the API service
#
# Responsibilities:
#   1. Run Alembic database migrations to head before the server starts.
#   2. exec the CMD (uvicorn) so that PID 1 is the application process.
#
# Design notes:
#   - `set -euo pipefail` ensures any migration failure aborts startup.
#   - Migrations are idempotent: `upgrade head` on an already-migrated DB
#     is a no-op. Safe on every container restart and rolling update.
#   - This script runs as appuser (UID 1001), not root.
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Step 1: Run database migrations
# ---------------------------------------------------------------------------
echo "[entrypoint] Running Alembic migrations..."
cd /app/infra/alembic
python -m alembic -c alembic.ini upgrade head
echo "[entrypoint] Migrations complete."

# ---------------------------------------------------------------------------
# Step 2: Start the application
# ---------------------------------------------------------------------------
echo "[entrypoint] Starting application: $*"
exec "$@"
