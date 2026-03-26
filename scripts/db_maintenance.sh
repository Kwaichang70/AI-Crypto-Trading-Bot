#!/usr/bin/env bash
# Database maintenance script for the AI Crypto Trading Bot.
# Run via cron: 0 3 * * 0  /opt/trading-bot/scripts/db_maintenance.sh >> /var/log/trading-bot-maintenance.log 2>&1

set -euo pipefail

COMPOSE_DIR="/opt/trading-bot/infra"
ENV_FILE="/opt/trading-bot/.env"
LOG_PREFIX="[$(date -u +%Y-%m-%dT%H:%M:%SZ)]"

echo "$LOG_PREFIX Starting database maintenance..."

# 1. VACUUM ANALYZE — reclaim dead rows and update query planner statistics
echo "$LOG_PREFIX Running VACUUM ANALYZE..."
docker exec crypto-trading-bot-postgres-1 psql -U trading -d trading_bot -c "VACUUM ANALYZE;" 2>&1
echo "$LOG_PREFIX VACUUM ANALYZE complete."

# 2. Archive old stopped/error runs (older than 90 days) — set status to 'archived'
echo "$LOG_PREFIX Archiving old runs (>90 days, stopped/error)..."
docker exec crypto-trading-bot-postgres-1 psql -U trading -d trading_bot -c "
  UPDATE runs
  SET status = 'archived'
  WHERE status IN ('stopped', 'error')
    AND created_at < NOW() - INTERVAL '90 days'
    AND status != 'archived';
" 2>&1
echo "$LOG_PREFIX Old runs archived."

# 3. Delete orphaned equity snapshots (no parent run)
echo "$LOG_PREFIX Cleaning orphaned equity snapshots..."
docker exec crypto-trading-bot-postgres-1 psql -U trading -d trading_bot -c "
  DELETE FROM equity_snapshots
  WHERE run_id NOT IN (SELECT id FROM runs);
" 2>&1
echo "$LOG_PREFIX Orphaned snapshots cleaned."

# 4. Delete orphaned skipped trades (no parent run)
echo "$LOG_PREFIX Cleaning orphaned skipped trades..."
docker exec crypto-trading-bot-postgres-1 psql -U trading -d trading_bot -c "
  DELETE FROM skipped_trades
  WHERE run_id NOT IN (SELECT id FROM runs);
" 2>&1
echo "$LOG_PREFIX Orphaned skipped trades cleaned."

# 5. Report table sizes
echo "$LOG_PREFIX Table sizes:"
docker exec crypto-trading-bot-postgres-1 psql -U trading -d trading_bot -c "
  SELECT
    schemaname || '.' || tablename AS table,
    pg_size_pretty(pg_total_relation_size(schemaname || '.' || tablename)) AS total_size,
    n_live_tup AS live_rows
  FROM pg_stat_user_tables
  ORDER BY pg_total_relation_size(schemaname || '.' || tablename) DESC;
" 2>&1

echo "$LOG_PREFIX Database maintenance complete."
