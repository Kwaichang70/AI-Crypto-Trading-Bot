#!/usr/bin/env bash
# infra/caddy-config.sh - substitute {{TAILSCALE_FQDN}} in Caddyfile
#
# Usage: bash infra/caddy-config.sh
# Run this on the Hetzner server AFTER tailscaled is installed + running
# (STEP 1b in CADDY-RUNBOOK.md). Reads `tailscale status` to get the
# server's Tailscale FQDN and writes it into the Caddyfile.
#
# Idempotent: running twice is safe (second run is a no-op).

set -euo pipefail

CADDYFILE_OUT="${CADDYFILE_OUT:-/opt/trading-bot/infra/Caddyfile}"

if [[ ! -f "${CADDYFILE_OUT}" ]]; then
    echo "ERROR: Caddyfile not found at ${CADDYFILE_OUT}"
    exit 1
fi

# CR-011-V2: Idempotency guard - exit cleanly if placeholder already substituted
if ! grep -q '{{TAILSCALE_FQDN}}' "${CADDYFILE_OUT}"; then
    echo "INFO: {{TAILSCALE_FQDN}} placeholder not found - Caddyfile already substituted."
    echo "      Current FQDN in file:"
    grep -m1 'ts\.net' "${CADDYFILE_OUT}" || echo "  (none found - inspect file manually)"
    exit 0
fi

# Verify tailscaled is running
if ! command -v tailscale &> /dev/null; then
    echo "ERROR: tailscale binary not found. Install per CADDY-RUNBOOK.md STEP 1b."
    exit 1
fi

# Extract FQDN via tailscale status --json (preferred) with python3 fallback
FQDN=""
if command -v python3 &> /dev/null; then
    FQDN=$(tailscale status --json 2>/dev/null | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    print(d.get('Self', {}).get('DNSName', '').rstrip('.'))
except Exception:
    pass
" || true)
fi

if [[ -z "${FQDN}" ]]; then
    echo "ERROR: Could not auto-detect Tailscale FQDN."
    echo "Manual substitution:"
    echo "  1. Run: tailscale status"
    echo "  2. Note your server's FQDN (e.g. server.tail12345.ts.net)"
    echo "  3. Run: sed -i 's/{{TAILSCALE_FQDN}}/your.fqdn.ts.net/g' ${CADDYFILE_OUT}"
    exit 1
fi

echo "Detected Tailscale FQDN: ${FQDN}"
sed -i "s/{{TAILSCALE_FQDN}}/${FQDN}/g" "${CADDYFILE_OUT}"
echo "Substitution complete. Caddyfile now references ${FQDN}."
