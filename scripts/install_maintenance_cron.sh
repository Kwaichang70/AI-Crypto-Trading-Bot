#!/usr/bin/env bash
# Install weekly database maintenance cron job (Sundays at 3:00 UTC)
set -euo pipefail

CRON_ENTRY="0 3 * * 0 /opt/trading-bot/scripts/db_maintenance.sh >> /var/log/trading-bot-maintenance.log 2>&1"

# Check if already installed
if crontab -l 2>/dev/null | grep -q "db_maintenance.sh"; then
    echo "Cron job already installed."
    exit 0
fi

# Add to crontab
(crontab -l 2>/dev/null; echo "$CRON_ENTRY") | crontab -
echo "Installed weekly maintenance cron: $CRON_ENTRY"
