# Grafana admin password (CF-97-34, SE-9734-04)

**Audience:** the operator of the production stack (`/opt/trading-bot`).
**Why a separate file:** it has its own trigger (a compose change that makes `GRAFANA_PASSWORD` required) and its own reader, and `caddy-metrics-block.md` is about one Caddy block and has pins of its own.

## 1. What changed, and what it does to your commands

`infra/docker-compose.yml` used to fall back to the password `admin` when `GRAFANA_PASSWORD` was unset. Grafana is published through Caddy on port 3001, and its admin account can read the same per-run trading series as the API (through the Prometheus data source) and can change alert contact points and notification policies. A known default password there is a High finding.

Now `GF_SECURITY_ADMIN_PASSWORD` is `${GRAFANA_PASSWORD:?GRAFANA_PASSWORD must be set}`: **Compose refuses to load the file while `GRAFANA_PASSWORD` is unset or empty.**

**This is a behaviour change for the whole project, not only for Grafana.** Compose interpolates the whole file before it looks at any service, so every `$DC` command (`up`, `logs`, `ps`, `exec`, `stop`, also for the `api` service alone) fails with `required variable GRAFANA_PASSWORD is missing a value` until the variable is set. This was checked by loading the file with dummy values (no daemon), with the variable set, unset, empty, and with a single service selected; it could not be scoped to the Grafana service. The compose file already works this way for `POSTGRES_PASSWORD`, `ADMIN_API_KEY`, `NEXTAUTH_SECRET` and the other required variables.

Therefore, **before deploying this compose file, add `GRAFANA_PASSWORD` to `/opt/trading-bot/.env`** (a strong random value; `.env.example` has the entry). The order matters:

1. Add `GRAFANA_PASSWORD` to `/opt/trading-bot/.env` with the stdin helper of `deploy-fase1-minimumset.md` section 0 (Helper 1; the value is not in shell history or an argv) **before** the new compose file reaches the host.
2. Right after updating the files, and **before** any `down`, `up` or `stop`, run `$DC config --quiet; echo "exit=$?"` (always with `--quiet`) and require exit code 0: it must print `exit=0` and nothing else.
3. Only then deploy.

If Compose refuses to load during an incident, the plain Docker CLI still works, because it does not interpolate the compose file: `docker ps`, `docker logs <container>` (piped to `grep`, as in the other runbooks) and `docker stop <container>`. Never use a general `docker inspect` of the api or grafana container.

## 2. The password only applies to a new volume

`GF_SECURITY_ADMIN_PASSWORD` is read only when the `grafana_data` volume is first initialised. An instance that was first started with the default keeps the password `admin` after this change, and also after `GRAFANA_PASSWORD` is added to `.env`. Setting the variable does not change an existing Grafana.

## 3. Interim check (run now, once)

From a device that can reach port 3001:

```bash
curl -s -o /dev/null -w '%{http_code}\n' -u admin:admin https://<host>:3001/api/user
```

Expected: `401`. Run it once, not in a loop (Grafana can lock an account after repeated failed logins; that behaviour was not checked here). `200` means the default password still works: reset it now (section 4) and run the check again. `admin` is the public default, not a secret of yours.

## 4. Resetting the password of an existing instance

Do not put the new password in a command argument: it would be visible in shell history and in the process list.

**Safe path, the Grafana UI:** sign in as `admin` with the current password, open the profile page, and change the password. Then run the check in section 3 again.

**If the current password is unknown, the CLI inside the container.** First look at what the pinned image supports (the compose file pins `grafana/grafana:11.6.0`); the help text prints no secret:

```bash
$DC exec -T grafana grafana cli admin reset-admin-password --help
```

Only if the help lists a `--password-from-stdin` flag, read the password without echo and pass it on stdin (`printf` is a shell builtin, so it does not appear in an argument list):

```bash
read -rs NEWPW; printf '%s' "$NEWPW" | $DC exec -T grafana grafana cli admin reset-admin-password --password-from-stdin; unset NEWPW
```

Whether Grafana 11.6.0 has that flag could not be checked when this was written (no container or network was available). If the help does not list it, use the UI path. Never use the form that puts the password as a trailing argument.

After any reset, run the check in section 3, and look at the Grafana contact points and notification policies: an admin can change both, and they route the alerts of `trading-safety` and `trading-infra`.

Also keep `GRAFANA_PASSWORD` in `.env` equal to the new password, with the same stdin helper, so a later volume re-creation does not bring back an old value.

## 5. Record the firewall rule

Ports 443 and 3001 must only be reachable from Tailscale (`100.64.0.0/10`); `infra/Caddyfile` lists this as prerequisite 1. This repository cannot show it. Record in the deploy checklist: the Hetzner firewall rule for 443 and for 3001, who confirmed it and when. From a device outside the tailnet, `https://<host>:3001/` must not answer (see `infra/CADDY-RUNBOOK.md` step 0 and `infra/caddy-validate.sh`).

## 6. Commands that print the password

The password is in the environment of the grafana container, and in the interpolated compose file. Never run `$DC config` without `--quiet`, `$DC exec grafana env`, `printenv`, or a general `docker inspect` of the container (the narrow `--format` forms of the other runbooks are fine).
