# API container memory: limit, no swap, and what to check after a restart

**Audience:** the operator of the production stack (`/opt/trading-bot`).
**Source:** CF-26-96, risk ruling RK-9703-50..52, arbiter task SY-97-135.

## 1. What is configured

In `infra/docker-compose.yml` the `api` service has:

- `deploy.resources.limits.memory: 768m` and a `256m` reservation (unchanged);
- `memswap_limit: 768m`, equal to the memory limit (new);
- `restart: unless-stopped` (unchanged);
- no `oom_kill_disable` (it must not be added: the container would hang instead of freeing memory).

In Docker, `memswap_limit` is memory plus swap. A value equal to the memory limit means **zero swap**.

## 2. Behaviour change (read this before deploying)

**With swap disabled, an API process that outgrows 768 MiB is killed (OOM, SIGKILL) instead of swapping.**

- Before: on a host with swap enabled, the container could use swap up to the size of the memory limit and slow down. The event loop that also runs live exits could stall without any error.
- After: the process is killed at once. Docker restarts it (`unless-stopped`). On boot the app runs `recover_orphaned_runs()`: **paper runs are rebuilt and restarted by themselves; live runs are set to `orphaned`, no task is started, and the bot does not protect an open live position until an operator resumes the run** (`POST /runs/{id}/resume`).
- On a host without swap nothing changes in practice: the limit already killed the process.
- This note does not change the 768m value. Raising it is a separate decision that depends on host RAM and other tenants.
- Applying the file change recreates the `api` container. That is a restart of the API process: do it in your usual deploy window, with the live-run procedure of `docs/runbooks/deploy-fase1-minimumset.md`.

The setting needs host support. On a host with cgroup v2 it works as written. On cgroup v1 without swap accounting, Docker prints "Your kernel does not support swap limit capabilities" and ignores `memswap_limit`; read the check in section 4.

## 3. Alert

`infra/grafana/provisioning/alerting/api-restart-alerts.yml` adds one Grafana rule, `trading-api-restarted` (critical): the API process restarted in the last 15 minutes (`app_uptime_seconds` dropped). It cannot tell an OOM kill from a crash or a planned deploy; a planned restart also fires it. No contact point is provisioned in the repo, so the rule only shows in the Grafana alert list until the operator adds a contact point (it needs a secret and does not belong in the repo).

Not covered by this change, and why: a memory-at-80% alert needs `process_resident_memory_bytes`, which `/metrics` does not expose (the app serves a custom registry without the process collector); and the `recovery.run_orphaned` log line has no path to Prometheus (no log shipper, no counter). Both need a code change or a new service and are proposed separately.

## 4. After any unexpected API restart

All commands run on the server as the deploy user, with `DC` as defined in `deploy-fase1-minimumset.md` section 0. Never run a general `docker inspect` on the api container: it prints secrets.

1. Was it an OOM kill? Read only these two fields:
   ```bash
   docker inspect --format '{{.State.OOMKilled}} {{.RestartCount}}' "$($DC ps -q api)"
   ```
   `true` means the last exit was an OOM kill. `OOMKilled` describes the last exit only.
2. Did boot recovery orphan live runs?
   ```bash
   $DC logs --since 30m api 2>&1 | grep -E 'recovery\.(run_orphaned|orphans_found|orphan_holding_unprotected|resume_stuck)'
   ```
3. Resume each orphaned live run promptly. Until it is resumed, watch every open live position through the exchange's own interface.
4. Check that the limits are in force (cgroup v2 path; on v1 read `memory.limit_in_bytes` and `memory.memsw.limit_in_bytes` in `/sys/fs/cgroup/memory/`):
   ```bash
   $DC exec -T api cat /sys/fs/cgroup/memory.max /sys/fs/cgroup/memory.swap.max
   ```
   Expected: `805306368` and `0`.
5. Record the host facts once (they are PRE-09 of the G-15 procedure): `MemTotal`, `MemAvailable`, `SwapTotal`, and whether `docker-compose.override.yml` exists next to `docker-compose.yml`. An override file can replace the values in this repo.

## 5. If restarts repeat

If restarts repeat and `OOMKilled` is true, find and stop the client first: the jobs that exhaust memory are backtests, optimisations and ML training. Latching the kill switch refuses only new paper and live runs (`POST /runs` with `mode: backtest` has no kill-switch check, see `runs.py` `create_run`); it does not stop a research job. Latch it as well if no new live run should start while the process keeps dying.

The kill switch covers no research job: backtest create and `POST /optimize` have no kill-switch check today. Until the process research bound (CF-97-32) or UQ-97-5 lands, the operator must stop any client that submits backtests or optimisations. Each further kill orphans the live runs again.
