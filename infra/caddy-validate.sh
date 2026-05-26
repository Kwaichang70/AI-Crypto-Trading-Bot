#!/usr/bin/env bash
# =============================================================================
# caddy-validate.sh — Post-deploy smoke test for Caddy reverse proxy
#
# Usage: bash infra/caddy-validate.sh <tailscale-fqdn>
#
# Prerequisites:
#   - Caddy stack is running (docker compose ps shows caddy healthy)
#   - Operator is connected to Tailscale
#   - Hetzner firewall has closed ports 3000/8000 from 0.0.0.0/0
#   - Tailscale HTTPS is enabled (cert available via built-in *.ts.net handling)
# =============================================================================
set -euo pipefail

TAILSCALE_FQDN=${1:?usage: $0 <tailscale-fqdn>  (e.g. server.tail12345.ts.net)}
PUBLIC_IP="167.235.51.90"

echo "=== Sprint 50 Cycle 1 — Caddy smoke test ==="
echo "Tailscale FQDN : ${TAILSCALE_FQDN}"
echo "Public IP      : ${PUBLIC_IP}"
echo ""

# --- Test 1: HTTPS UI reachable via Tailscale FQDN (browser-trusted cert) ---
echo -n "[1] UI proxy (https://${TAILSCALE_FQDN}/)... "
if curl -fs --max-time 10 "https://${TAILSCALE_FQDN}/" | grep -qi "trading\|html\|next"; then
    echo "OK"
else
    echo "FAIL — page body missing expected content"
fi

# --- Test 2: API proxy reachable via Tailscale FQDN ---
echo -n "[2] API proxy (https://${TAILSCALE_FQDN}/api/v1/health)... "
HTTP_CODE=$(curl -s --max-time 10 -o /dev/null -w "%{http_code}" \
    "https://${TAILSCALE_FQDN}/api/v1/health")
if [[ "${HTTP_CODE}" == "200" || "${HTTP_CODE}" == "401" ]]; then
    echo "OK (HTTP ${HTTP_CODE} — 401 expected when REQUIRE_API_AUTH=true)"
else
    echo "FAIL (HTTP ${HTTP_CODE})"
fi

# --- Test 3: Grafana reachable via Tailscale FQDN on port 3001 ---
echo -n "[3] Grafana proxy (https://${TAILSCALE_FQDN}:3001/)... "
HTTP_CODE=$(curl -s --max-time 10 -o /dev/null -w "%{http_code}" \
    "https://${TAILSCALE_FQDN}:3001/")
if [[ "${HTTP_CODE}" == "200" || "${HTTP_CODE}" == "302" ]]; then
    echo "OK (HTTP ${HTTP_CODE})"
else
    echo "FAIL (HTTP ${HTTP_CODE})"
fi

# --- Test 4: Port 3000 closed on public IP ---
echo -n "[4] Port 3000 PUBLIC closed... "
if ! curl -sf --max-time 5 "http://${PUBLIC_IP}:3000/" > /dev/null 2>&1; then
    echo "OK (unreachable from public internet)"
else
    echo "FAIL — port 3000 still reachable! Close in Hetzner firewall."
fi

# --- Test 5: Port 8000 closed on public IP ---
echo -n "[5] Port 8000 PUBLIC closed... "
if ! curl -sf --max-time 5 "http://${PUBLIC_IP}:8000/health" > /dev/null 2>&1; then
    echo "OK (unreachable from public internet)"
else
    echo "FAIL — port 8000 still reachable! Close in Hetzner firewall."
fi

echo ""
echo "=== Automated tests complete ==="
echo ""
echo "--- Test 6 (MANUAL — do not automate) ---"
echo ""
echo "Test 6 verifies that port 3001 is NOT reachable from the public internet."
echo "This test CANNOT be run from an operator machine on Tailscale because"
echo "Tailscale-NATed connections will appear to succeed even if the Hetzner"
echo "firewall is correctly blocking non-Tailscale traffic."
echo ""
echo "To run Test 6 correctly:"
echo "  1. Use a machine NOT connected to your Tailscale network."
echo "     Examples: mobile phone on cellular data, a cloud shell, a friend's machine."
echo "  2. From that machine, run:"
echo "       curl --max-time 5 https://${PUBLIC_IP}:3001/"
echo "  3. Expected result: connection timeout or refused (not a TLS handshake)."
echo "  4. If you get a TLS response, the Hetzner firewall rule for port 3001"
echo "     is missing or misconfigured — fix before considering deploy complete."
echo ""
echo "=== Smoke test complete ==="
