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

## STEP 2a — Pre-deploy config validation (mandatory, WP1.7b S-08/S-09)

Run this **before** `docker compose up`, every time `.env`, `docker-compose.yml`
or the Caddyfile changes.

### S-08 — validate compose renders, without leaking secrets to logs

```bash
cd /opt/trading-bot/infra
docker compose --env-file /opt/trading-bot/.env config --quiet
echo "exit code: $?"   # 0 = renders cleanly (all required vars present); non-zero = see stderr
```

> [!CAUTION]
> **Always pass `--quiet` here, never run the plain `docker compose config`
> in this step.** `docker compose config` (no `--quiet`) prints the fully
> **interpolated** compose file to stdout — every `${ADMIN_API_KEY:?...}`,
> `${POSTGRES_PASSWORD:?...}`, `${NEXTAUTH_SECRET:?...}` etc. is replaced
> with its real secret value in the printed output. `--quiet` validates and
> exits non-zero on any missing/`:?` variable **without** printing the
> rendered config. Never pipe the plain (non-`--quiet`) form to a file, to
> CI logs, or to a terminal that gets captured/screen-shared — the
> resulting output is a plaintext secret dump.
>
> This validation step does not require the Docker daemon to be running
> (`docker compose config` only reads and interpolates the compose file);
> it can be run as a pure syntax/completeness check even before the stack
> is started.

### S-09 — confirm `ADMIN_API_KEY` and `INTERNAL_ADMIN_API_KEY` match, without printing either

`infra/docker-compose.yml` intentionally keeps these as two separate `.env`
variables (api's `ADMIN_API_KEY`, ui's `INTERNAL_ADMIN_API_KEY`) rather than
one variable reused via compose substitution. Nothing enforces that an
operator sets them to the same value. A mismatch **fails closed** — every
admin action (kill-switch, kill-switch clear, resume, entries-latch clear)
returns 403 — so this is an availability check, not a security gate, but it
should still be run after editing `.env` and before declaring a deploy done.

Compare their SHA-256 hashes instead of the raw values, so neither key is
ever printed to the terminal or a log:

```bash
cd /opt/trading-bot
HASH_A=$(grep -E '^ADMIN_API_KEY=' .env | cut -d= -f2- | sha256sum | cut -d' ' -f1)
HASH_B=$(grep -E '^INTERNAL_ADMIN_API_KEY=' .env | cut -d= -f2- | sha256sum | cut -d' ' -f1)
if [ "$HASH_A" = "$HASH_B" ]; then
  echo "OK: ADMIN_API_KEY and INTERNAL_ADMIN_API_KEY match (sha256 $HASH_A)"
else
  echo "MISMATCH: ADMIN_API_KEY (sha256 $HASH_A) != INTERNAL_ADMIN_API_KEY (sha256 $HASH_B)"
  echo "Every admin action will return 403 until these are set identically."
fi
```

Only the hashes are ever displayed, never `$ADMIN_API_KEY` or
`$INTERNAL_ADMIN_API_KEY` themselves. If you see `MISMATCH`, fix `.env` and
re-run both S-08 and this check before proceeding to STEP 3.

### S-10 — request-body size limit (WP1.3a defence in depth)

Security found repeated event-loop DoS via large request bodies on
`/api/v1/optimize` (`reports/vp2-wp1.3a/security-report-r4.md` and
`-r5.md`). `infra/Caddyfile` now caps every proxied `/api/*` handle block
(`/api/auth/*`, `/api/admin/*`, and the generic `/api/*` → `api:8000`
block) at **1 MB** via:

```
request_body {
    max_size 1MB
}
```

Note the block form is required — `request_body max_size 1MB` written
inline on one line is silently ignored by Caddy.

This is **defence in depth, not the primary control**: the FastAPI app
enforces its own independent 1 MiB ASGI-level body-size limit (413) on
every request, regardless of whether it arrives through Caddy or directly
against `api:8000` (the operator curl fallback described in the
`handle /api/admin/*` comment in `infra/Caddyfile`). Caddy rejecting the
body first just means the oversized request never reaches the application
process at all, saving the parse/validation cost that the security reports
measured pinning the event loop.

Caddy returns **413 Payload Too Large** when a body exceeds `max_size`.

> [!NOTE]
> Known Caddy upstream nuance (caddyserver/caddy#4558, #5652 — open as of
> Caddy 2.11.x): with `reverse_proxy`, if the upstream service starts
> reading the request body before Caddy finishes enforcing `max_size`, an
> oversized request can occasionally surface to the client as **502 Bad
> Gateway** instead of 413. Either code means the oversized body was
> rejected before the application processed it — a plain `200` is the only
> outcome that means this control is not working. The independent
> API-level limit is what actually guarantees rejection regardless of
> which status code Caddy itself returns.

**Verify after any Caddyfile or `.env` change that touches routing:**

```bash
# 2 MB of zero bytes, well over the 1 MB cap
head -c 2000000 /dev/zero > /tmp/oversized-body.bin

# Through Caddy, against the generic /api/* block:
curl -sS -o /dev/null -w '%{http_code}\n' \
  --data-binary @/tmp/oversized-body.bin \
  -H 'Content-Type: application/json' \
  https://server-name.tail12345.ts.net/api/v1/optimize

# Expected: 413 (occasionally 502, see the note above — either is a PASS).
# A 200, or the request hanging/timing out, is a FAIL: investigate before
# considering the deploy complete.

rm -f /tmp/oversized-body.bin
```

Repeat against `/api/auth/*` and `/api/admin/*` if you changed either of
those blocks specifically; all three should reject the same 2 MB body the
same way.

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

## Kill-switch / admin routing deploy conditions (WP1.7b, C21 + CF-B5)

These apply once WP1.7a (persisted kill-switch latch + stop-with-flatten,
see `reports/vp2-wp1.7/final-synthesis-1.7a.md`) and WP1.7b (this admin
routing + UI work) are both deployed together. **1.7a alone is NOT
deployable** -- see DC-1/DC-2 below.

### DC-1 / DC-2 -- ship 1.7a and 1.7b together; keep live trading off until then

Do not deploy WP1.7a to any environment where the UI is used unless WP1.7b
ships with it in the same release. The pre-1.7b UI reads the old
kill-switch response shape and cannot send the `flatten` decision, so an
operator stopping a live run from the UI would see a misleading result.
`ENABLE_LIVE_TRADING` must stay `false` in any environment running 1.7a
without 1.7b.

### DC-3 -- keep the API at a single worker

`infra/Dockerfile.api` runs uvicorn with `--workers 1`. Do **not** add a
`command:` override in `docker-compose.yml`, and do not raise worker count
in any deploy script. The in-process `_RUN_ENGINES` registry and the
kill-switch latch mirror (`apps/api/services/kill_switch.py`) both assume a
single process; running more than one worker would let each worker hold an
independent, inconsistent latch/engine state (tracked as CF-L7, multi-worker
safety, for a future work package).

### DC-4 -- migration 018 must run before the app starts

The entrypoint (`infra/docker-entrypoint.sh`) runs `alembic upgrade head`
before starting the API process, so this is normally automatic. If the app
starts against a database that has not yet run migration 018
(`018_kill_switch_latch_flatten`), `kill_switch.load()` fails and every run
comes up latched with reason `latch_state_unknown` (fail-closed, per I4).
That is safe -- no run can enter new positions -- but it blocks run
creation, promotion and resume until the process is restarted against a
healthy, migrated database. If you see `latch_state_unknown` immediately
after a deploy, check `alembic current` against the API container's
database before doing anything else.

### DC-5 -- rollback steps

If WP1.7a/1.7b need to be rolled back:

1. **Record state first.** Before touching anything, capture:
   - `GET /api/v1/emergency/kill-switch` (global latch state).
   - Every run currently holding a per-run latch
     (`runs.entries_latch_reason IS NOT NULL`).
   - Any open positions on runs you are about to affect.
2. **Resolve positions.** For any run holding an open position that the
   rollback would otherwise orphan, stop or flatten it first through the
   still-running (pre-rollback) API -- the downgrade below removes the
   columns that track `flatten_incomplete`, so do this before downgrading.
3. **Revert the application code** (revert the WP1.7a/1.7b commit(s)) and
   redeploy the prior image.
4. **Downgrade the database:** `alembic downgrade 017`. This drops the
   `kill_switch_state` table and the per-run `entries_latch_reason` /
   `entries_latched_at` columns -- any latch state and `flatten_incomplete`
   markers are permanently lost, which is why step 1 (record state first)
   and step 2 (resolve positions first) are mandatory and come before this
   step, not after.
5. Restart the API so it picks up the reverted code against the downgraded
   schema.

### DC-6 -- a kill-switch clear can be overridden by a pending press (S-R3-02)

A kill-switch **clear** issued while a kill-switch **press** is still
pending (submitted but not yet acknowledged) may be overridden by that
press: the latch is fail-closed, so if both operations are in flight the
final state is **latched**, not cleared.

After issuing a clear, always confirm with:

```
GET /api/v1/emergency/kill-switch
```

that `latched` is `false`. If it is still `true`, re-issue the clear.
Never assume a clear succeeded just because the request returned 200.

### DC-7 -- operator behaviour changes to know before using the kill switch

- The kill switch no longer stops runs. It only blocks **new entries**;
  existing positions' exits (stop-loss, take-profit, trailing stop) keep
  running normally.
- Creating a **paper** run while the switch is latched now returns 409
  `kill_switch_active` (previously only live runs were blocked) -- a
  latched new run sitting silently inert was judged worse than an explicit
  rejection.
- Both the global clear and the per-run entries-latch clear require the
  admin key (`X-Admin-Key`, injected server-side by the Next.js admin proxy
  routes -- the browser never holds it). Clearing a per-run latch on a
  **live** run additionally requires `X-Live-Confirm-Token`.
- Stopping a **live** run now requires an explicit `flatten` decision
  (`true`/`false`); omitting it returns 422 `flatten_decision_required`.
  An emergency stop with `flatten=true` can take up to about 35 s -- the UI
  timeout for stop and kill-switch actions must be at least 60 s (AC8).
- An incomplete flatten (partial fill, in-flight order, etc.) leaves the
  run **running**, not stopped, with a persisted `flatten_incomplete`
  per-run latch. The stop call returns 409 `flatten_incomplete` with the
  partial result; retry the stop, or stop with `flatten=false` to abandon
  the position tracking (it stays on the exchange, unmanaged).

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
