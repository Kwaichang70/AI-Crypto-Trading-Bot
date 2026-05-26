# Caddy Reverse Proxy — Operator Runbook

Sprint 50 Cycle 1 — Tailscale HTTPS certificate via built-in *.ts.net automatic cert handling.

Browser-trusted Let's Encrypt certificate issued for the server's Tailscale FQDN.
No manual CA install required. Access is private — only reachable via Tailscale VPN.

---

> [!CAUTION]
> **STEP 0 (Hetzner firewall lockdown) is a BLOCKING PREREQUISITE.**
> Deploying Caddy without first locking down the Hetzner firewall exposes the
> application over the public internet without meaningful transport security —
> WORSE than the current HTTP-only state, not better.
> **Do NOT run `docker compose up` until STEP 0 is confirmed complete.**

---

## STEP 0 — Hetzner Cloud firewall lockdown (BLOCKING PREREQUISITE)

This step MUST be completed and verified BEFORE rebuilding the Docker stack.

1. Log into https://console.hetzner.cloud and select your server project.
2. Navigate to Networking → Firewalls → select the firewall attached to 167.235.51.90.
3. Apply the following inbound rules (replace or add — delete any existing rules
   that allow ports 3000, 8000, or 3001 from 0.0.0.0/0):

   | Port / Protocol | Source CIDR              | Purpose                          |
   |-----------------|--------------------------|----------------------------------|
   | 22 / TCP        | 100.64.0.0/10            | SSH via Tailscale                |
   | 22 / TCP        | `<your-home-IP>/32`      | SSH fallback (optional)          |
   | 443 / TCP       | 100.64.0.0/10            | HTTPS — Tailscale only           |
   | 3001 / TCP      | 100.64.0.0/10            | Grafana HTTPS — Tailscale only   |

4. DELETE any existing rules that expose ports 3000, 8000, or 3001 to 0.0.0.0/0.
   Docker publishes host ports regardless of what the application listens on —
   the Hetzner firewall is the outer perimeter that drops non-Tailscale packets.

5. Verify the firewall change is saved and shows "Applied" in the console.
   Hetzner firewall changes propagate within ~30 seconds.

6. CONFIRM from a machine NOT on your Tailscale network:
   ```
   curl --max-time 5 http://167.235.51.90:8000/health
   ```
   This must time out or be refused. If it returns a response, do not proceed.

> [!NOTE]
> **Locked out of SSH?** If you accidentally block port 22 before setting up
> Tailscale, use the Hetzner Cloud Console VNC rescue mode to recover:
> https://console.hetzner.cloud → Server → Console (VNC web terminal).
> This gives you root access without SSH. From there, fix /etc/default/tailscaled
> or re-run `tailscale up` as needed.

---

## STEP 1 — Install Tailscale on the Hetzner server

SSH into the server (current public-IP SSH still works — port 22 is open):

```bash
ssh root@167.235.51.90
```

### STEP 1a — Enable Tailscale HTTPS in tailnet admin console (one-time)

On your Windows machine, open a browser and go to:
https://login.tailscale.com/admin/dns

Scroll to "HTTPS Certificates" and click **Enable**. This allows nodes in your
tailnet to obtain Let's Encrypt certificates for their Tailscale FQDNs via
the Tailscale CA integration. This is a one-time setting per tailnet.

### STEP 1b — Install tailscaled on the Hetzner server

```bash
curl -fsSL https://tailscale.com/install.sh | sh
tailscale up
```

Follow the authentication URL printed in the terminal to authorize the server
in your Tailscale account. After authorization, note the FQDN assigned to the
server. Run:

```bash
tailscale status
```

Look for a line like:
```
100.x.y.z   server-name        yourname@  linux   -
```

The Tailscale FQDN is: `server-name.tail12345.ts.net`
(visible in the tailnet admin console under Machines, or via `tailscale status --json`)

### STEP 1c — Allow Caddy to request certificates from tailscaled

Caddy calls the tailscaled Unix socket to obtain the TLS certificate. By default
tailscaled restricts this to root. Set the permitted UID:

```bash
# Find the UID that Caddy will run as inside its container.
# caddy:2-alpine runs as user 'caddy' (UID 1000 inside the container).
# The container mounts the host socket, so tailscaled checks the calling UID
# against TS_PERMIT_CERT_UID. Set it to 'caddy' (the host user if present,
# or the numeric UID the container process uses).

echo 'TS_PERMIT_CERT_UID=caddy' | sudo tee -a /etc/default/tailscaled

# Restart tailscaled to pick up the new setting:
sudo systemctl restart tailscaled

# Verify tailscaled is running:
sudo systemctl status tailscaled
```

> [!NOTE]
> If there is no `caddy` system user on the host, use the numeric UID that the
> Caddy container process runs as. For `caddy:2-alpine`, the container's caddy
> process UID is typically 1000. In that case:
> `echo 'TS_PERMIT_CERT_UID=1000' | sudo tee -a /etc/default/tailscaled`
> Confirm by running: `docker run --rm caddy:2-alpine id caddy`

---

## STEP 2 — Generate the Caddyfile with the real Tailscale FQDN

The `infra/Caddyfile` ships with a `{{TAILSCALE_FQDN}}` placeholder.
Substitute it before deploying. Two options:

**Option A — Automated (recommended):**

After syncing the code to the server:
```bash
cd /opt/trading-bot
bash infra/caddy-config.sh
```

This queries `tailscale status --json` and writes the real FQDN into `infra/Caddyfile`.

**Option B — Manual:**

```bash
# Replace <your-fqdn> with the actual FQDN from Step 1b
FQDN="server-name.tail12345.ts.net"
sed -i "s/{{TAILSCALE_FQDN}}/${FQDN}/g" /opt/trading-bot/infra/Caddyfile

# Verify no placeholders remain:
grep -n 'TAILSCALE_FQDN' /opt/trading-bot/infra/Caddyfile
# Expected: no output (all replaced)
```

---

## STEP 3 — Deploy the updated Docker stack

On the Hetzner server (SSH via Tailscale IP, or still via public IP before firewall goes live):

```bash
# Sync code from local machine (current deploy method):
# Run this from your Windows machine in WSL or Git Bash:
rsync -avz --exclude '.env' --exclude '.git' \
  /c/Users/DannydeLacombe/.claude/projects/AI\ Crypto\ Trading\ Bot/ \
  root@<tailscale-ip>:/opt/trading-bot/

# Then on the server — substitute FQDN first (STEP 2), then:
cd /opt/trading-bot/infra
bash ../infra/caddy-config.sh   # Only needed if Caddyfile not already updated

# Rebuild and start the stack including the new caddy service:
docker compose --env-file /opt/trading-bot/.env up -d --build api ui caddy

# Caddy starts after api, ui, and grafana are all healthy (depends_on).
# Tail Caddy logs to watch certificate fetch:
docker compose logs -f caddy
```

Watch for lines like:
```
obtained certificate   {"domain": "server-name.tail12345.ts.net"}
```

If you see `failed to get certificate: tailscale: permission denied`, revisit STEP 1c.

---

## STEP 4 — Update .env on the production server

Add or update in `/opt/trading-bot/.env`:

```bash
# Required for correct per-IP rate limiting behind Caddy
TRUSTED_PROXY_COUNT=1

# CORS: allow the Tailscale FQDN (and optionally localhost for dev)
ALLOWED_ORIGINS=["https://server-name.tail12345.ts.net"]

# Browser-side API URL baked into the Next.js bundle at build time.
# After changing this, rebuild the ui image: docker compose up -d --build ui
NEXT_PUBLIC_API_URL=https://server-name.tail12345.ts.net
```

Then rebuild the UI image to bake in the new `NEXT_PUBLIC_API_URL`:

```bash
docker compose --env-file /opt/trading-bot/.env up -d --build ui
```

---

## STEP 5 — Verify HTTPS works via Tailscale

From your Windows machine (PowerShell, with Tailscale running):

```powershell
# No -k flag needed — certificate is browser-trusted via Tailscale LE
curl.exe https://server-name.tail12345.ts.net/api/v1/health
```

Expected: JSON response (HTTP 200 or 401 if REQUIRE_API_AUTH=true).

```powershell
# Verify Grafana
curl.exe https://server-name.tail12345.ts.net:3001/
```

Expected: HTTP 200 or 302 (Grafana login redirect).

Or run the automated smoke test:

```bash
bash infra/caddy-validate.sh server-name.tail12345.ts.net
```

---

## STEP 6 — Verify public port isolation (mandatory)

Run the automated tests first:

```bash
bash infra/caddy-validate.sh server-name.tail12345.ts.net
```

Tests 4 and 5 confirm that ports 3000 and 8000 are unreachable from the public internet.

**Test 6 is manual.** From a machine NOT on your Tailscale network (mobile on cellular,
a cloud shell, or a friend's computer):

```bash
curl --max-time 5 https://167.235.51.90:3001/
```

Expected: connection timeout or refused. If you get a TLS response, the Hetzner
firewall rule for port 3001 is missing — fix it in the Hetzner Cloud Console before
considering this step complete.

---

## Rollback procedure

If Caddy fails to start, the certificate cannot be obtained, or HTTPS is unreachable:

**Step 0 — Stop Caddy first (REQUIRED before restoring ports):**

```bash
docker compose stop caddy
```

Skipping this step causes "address already in use" errors when restoring ports
443 and 3001 to the other services, because Caddy still holds those port bindings.

**Step 1 — Restore public port bindings** by editing `docker-compose.yml`:

```bash
# Re-add to api service:
#   ports:
#     - "${API_PORT:-8000}:8000"

# Re-add to ui service:
#   ports:
#     - "${UI_PORT:-3000}:3000"

# Re-add to grafana service:
#   ports:
#     - "${GRAFANA_PORT:-3001}:3000"
```

**Step 2 — Restart services without Caddy:**

```bash
docker compose up -d api ui grafana
```

**Step 3 — Investigate Caddy logs:**

```bash
docker compose logs caddy
```

Common failure modes:
- `failed to get certificate: tailscale: permission denied`
  -> Revisit STEP 1c: ensure TS_PERMIT_CERT_UID is set correctly in /etc/default/tailscaled
- `certificate: no handler for {{TAILSCALE_FQDN}}` → FQDN placeholder not substituted (STEP 2)
- `tailscale: dial unix /var/run/tailscale/tailscaled.sock: no such file` → tailscaled not running on host; run `systemctl start tailscaled`
- Tailscale HTTPS not enabled in admin console → STEP 1a

**The Caddy data volume (`caddy_data`) persists across rollbacks.** On successful retry,
Caddy will reuse the stored private key and request a renewed cert if needed.
