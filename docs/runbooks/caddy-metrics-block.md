# Caddy block for `/api/v1/metrics` (CF-97-34, option A)

**Audience:** the operator of the production stack (`/opt/trading-bot`).
**Source:** CF-97-34 / SE-2696-06. This is a separate file from `api-container-memory.md` on purpose: that runbook is about container memory and its content is pinned by its own tests; this one is about edge exposure and has a different trigger and reader.

## 1. What the block does

`GET /api/v1/metrics` is a JSON snapshot of the API's in-memory metrics. It needs no API key and is exempt from rate limiting. It carries per-run equity, drawdown, open positions, order, fill and signal counters and the safety-state gauges. Before this change `infra/Caddyfile` sent it to the API through `handle /api/*`.

The Caddyfile now has, before that generic block:

```
handle /api/v1/metrics* {
    respond 404
}
```

It answers 404 for `/api/v1/metrics`, the trailing-slash form and any path starting with `/api/v1/metrics`, without contacting the API. It does not change the application: the route still answers on `api:8000` inside the compose network, so anything that can reach the API container directly can still read it. The durable fix (an API key on the route, or removing it) is a later decision (CF-97-34 option B or C).

What it does **not** touch:

- Prometheus scrapes `api:8000/metrics` (the Prometheus text endpoint, a different route) directly on the backend network (`infra/prometheus.yml`). Monitoring is unaffected.
- `/metrics` at the site root is not routed to the API by Caddy: the catch-all block sends it to the UI. The text endpoint is therefore not published through Caddy.
- The Grafana site block on port 3001 is unchanged. It proxies everything to Grafana, which has its own login (`GRAFANA_PASSWORD`, required by the compose file since CF-97-34; see `grafana-admin-password.md`).

## 2. Verify: validate, reload, live check

**Not verified when this was written.** The Caddyfile syntax and the matcher behaviour (case, percent-encoding, duplicate slashes) were not run against Caddy: no `caddy` binary was available. The tests in this repository only check the text of the file. The live check in step 3 is the evidence that the block works, not the tests and not the reload.

Run the steps in this order (`DC` is defined in `deploy-fase1-minimumset.md` section 0; the deployed file already holds the substituted hostname).

**Step 1. Validate the Caddyfile** before reloading:

```bash
$DC exec -T caddy caddy validate --config /etc/caddy/Caddyfile --adapter caddyfile
```

**Step 2. Reload:**

```bash
$DC exec -T caddy caddy reload --config /etc/caddy/Caddyfile --adapter caddyfile
```

**Step 3. Live check from outside the host**, from a device that can reach port 443 (a tailnet device; also from a non-tailnet network if the firewall is meant to block it). First the plain request:

```bash
curl -s -o /dev/null -w '%{http_code}\n' https://<host>/api/v1/metrics
```

Then every spelling, including the percent-encoded ones. `--path-as-is` keeps curl from tidying the path, so the encoded and dot-segment forms reach Caddy as written:

```bash
for p in '/api/v1/metrics' '/api/v1/metrics/' '/api/v1/metrics/x' \
         '/api/v1/%6detrics' '/api/v1/%6Detrics' '/api/v1%2Fmetrics' '/api%2Fv1%2Fmetrics' \
         '/API/v1/metrics' '/api/v1/x/../metrics' '/api//v1/metrics' \
         '/api/v1/Metrics' '//api/v1/metrics'; do
  printf '%s ' "$p"; curl -s --path-as-is -o /dev/null -w '%{http_code}\n' "https://<host>$p"
done
```

Expected: `404` on every line. The case, double-slash and dot-segment lines may also return `404` from the API itself, which is fine. **Any `200` is a failure:** stop, and remove the route or apply the lasting fix (CF-97-34 option C) at once. Other `/api/v1/` paths must still work, for example `https://<host>/api/v1/health/background`.

If the live check still shows `200` after the reload, assume the stale bind-mount inode (section 3), recreate the caddy container with `$DC up -d --force-recreate caddy` (brief interruption of 443 and 3001), and run the live check again. The check, not the reload, is the evidence.

**Step 4. Check from inside the compose network** that monitoring still works. In the Prometheus UI (`127.0.0.1:9090` on the host) the target `trading-bot-api` must be `UP`, or:

```bash
curl -s 'http://127.0.0.1:9090/api/v1/query?query=up%7Bjob%3D%22trading-bot-api%22%7D'
```

Expected: value `1`. The query returns metric names only, no secrets.

## 3. Hand-applied interim form (cannot deploy yet)

The deployed `/opt/trading-bot/infra/Caddyfile` differs from the repository file: `infra/caddy-config.sh` replaced `{{TAILSCALE_FQDN}}` in place. So do not copy the repository file over it. Insert the block from section 1 by hand into the primary site block, **above** the line `handle /api/* {` (Caddy sorts single-path `handle` blocks by path length and runs the catch-all last, so the position does not change which block answers; the repository file keeps the blocks in this order for readability). Then run steps 1 to 4 of section 2, in that order.

If the reload does not take effect, the file was probably replaced instead of edited in place, and a single-file bind mount keeps pointing at the old file; recreate the container as described in step 3. This behaviour is expected from how Docker bind mounts work and was not tested here.

The next normal deploy of the repository Caddyfile carries the same block, so the hand edit is then redundant. Until the change is committed and deployed, repeat the hand edit after every deploy that overwrites the file, and record this in the deploy checklist.

## 4. What this does not fix

- Anyone who can reach the API container directly (another container on the backend network) can still read the route.
- If port 443 of the host is reachable from outside Tailscale, the rest of the stack is exposed too. The Hetzner firewall rule that `infra/Caddyfile` requires (443 and 3001 only from `100.64.0.0/10`) is a host fact this repository cannot show; confirm it.

## 5. Second edge block: `/_next/image*` (CF-97-40-A, 37-C2)

**Why here:** this runbook already owns the validate, reload and live-check steps for 404 blocks in the primary site block of `infra/Caddyfile`, and its section 2 is the procedure to follow. The deploy runbook is long and release-specific, and `api-key-auth-switch.md` is about the key switch.

The Next.js Image Optimizer (`/_next/image`) is reachable without a session because the UI middleware matcher excludes `_next/static` and `_next/image`, and the dashboard does not use it. The Caddyfile now has, above the catch-all:

```
handle /_next/image* {
    respond 404
}
```

It matches only paths starting with `/_next/image`. `/_next/static/*` (the JavaScript and CSS the pages need) and every other `/_next/` path still reach the UI.

Position: Caddy sorts `handle` blocks that have a single path by the length of that path and runs the catch-all `handle {` (no matcher) last, so the position inside the primary site block does not change which block answers. The repository file keeps every edge block above the catch-all for readability. If you insert the block by hand into the deployed file (the section 3 procedure), put it in the primary site block directly below the `/api/v1/metrics*` block.

Do section 2 step 1 (validate) and step 2 (reload) as written, and record the Caddy version with the check results (the image tag floats):

```bash
$DC exec -T caddy caddy version
```

Then run this live check. Each line prints only a status code:

```bash
curl -s -o /dev/null -w '%{http_code}\n' 'https://<host>/_next/image?url=%2Fx&w=64&q=75'
curl -s -o /dev/null -w '%{http_code}\n' 'https://<host>/'
```

Expected: `404` for the first line; `400` there means the optimizer answered (the block missed). For the second line `200` or a redirect (`302`, `303`, `307`), **not** `404`: a `404` there means the block is too wide or the page is broken, so revert the Caddyfile and reload. Then open the dashboard in a browser and confirm that it loads styled (that proves `/_next/static/*` is still served).

Then every spelling. This step is mandatory. `--path-as-is` keeps curl from tidying the path, as in section 2 step 3. Each line prints the path, the status code and the number of body bytes, never the body:

```bash
for p in '/_next/image' '/_next/image/' '/_NEXT/IMAGE' '/_next/%69mage' '/_next%2Fimage' \
         '//_next/image' '/_next//image' '/_next/static/../image' '/_next/image%3Fx'; do
  printf '%s ' "$p"; curl -s --path-as-is -o /dev/null -w '%{http_code} %{size_download}\n' "https://<host>$p?url=%2Fx&w=64&q=75"
done
```

Expected: `404` on every line. Anything else is a failure: `400` means the optimizer answered (the block missed that spelling); `308` means Caddy did not match and Next redirected; `307` means Caddy did not match and the UI's sign-in redirect answered. On a failure in this loop keep the block (it still covers the other spellings), record the line and the Caddy version, and report it for review. A byte count of `0` shows that the `404` came from Caddy; a `404` with a larger byte count came from the UI: the optimizer was not reached, but Caddy did not match that spelling, so record and report that line too.

Optional control check (not tested): from the host, `$DC exec -T caddy wget -q -O /dev/null 'http://ui:3000/_next/image?url=%2Fx&w=64&q=75'; echo "rc=$?"` should fail with HTTP 400 and print a non-zero `rc`. That shows the optimizer still answers inside the compose network, so the loop above can see a miss.

The reviewers read the source of Caddy and of `next` 14.2.35: Next sends only the raw paths `/_next/image` and `/_next/image/` to the optimizer, and Caddy lowercases, decodes and cleans the path before matching, so it matches both however they are spelled. The loop above is the evidence on the deployed Caddy version.

Not verified when this was written: no `caddy` binary was available, so the syntax was not validated and the block was not run. `caddy validate` and the live checks above are the evidence. This block hides the route at the edge only; it does not change the `next` version (that is CF-97-44; CF-97-40-B is the `next-auth` bump), and anything that reaches `ui:3000` directly can still use the optimizer.
