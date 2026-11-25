#!/usr/bin/env node
/**
 * apps/ui/scripts/check-bff-traversal.mjs
 * -----------------------------------------
 * CF-97-37 (37-B, R6-2): real-build traversal and role check. Starts the
 * STANDALONE server of the current build (run `next build` first, or run
 * check-bff-key-leak.mjs, which builds) against a fake upstream on loopback,
 * sends raw request targets, and asserts for every case the status, the number
 * of upstream calls and, where the upstream was reached, EXACTLY the upstream
 * method and path. It also asserts that no admin path is ever reached.
 *
 *   node apps/ui/scripts/check-bff-traversal.mjs
 *
 * Next 14.2.35 collapses dot segments before routing; the route table judges
 * the path after that normalisation. This script pins the behaviour for the
 * measured Next version: after any `next` upgrade it fails until the constant
 * below is edited on purpose (a jest test compares it with the lockfile).
 * Secrets are dummy values passed through the child's environment only.
 * Output: case labels, statuses and counts; no secret value is printed.
 */

import { spawn } from "node:child_process";
import { randomBytes } from "node:crypto";
import fs from "node:fs";
import http from "node:http";
import net from "node:net";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { createRequire } from "node:module";

export const MEASURED_NEXT_VERSION = "14.2.35";

const UI_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const require = createRequire(import.meta.url);
// PR-HYG-01: random per run, passed to the server through its environment only.
const SECRET = randomBytes(32).toString("hex");
const KEY = randomBytes(32).toString("hex");
const UUID = "123e4567-e89b-42d3-a456-426614174000";

const freePort = () =>
  new Promise((resolve) => {
    const s = net.createServer();
    s.listen(0, "127.0.0.1", () => {
      const { port } = s.address();
      s.close(() => resolve(port));
    });
  });

/** [label, method, raw target, role, expected status, expected upstream method+path or null, extra request headers (optional)]. */
const CASES = [
  ["plain read", "GET", "/bff/api/v1/runs", "admin", 200, "GET /api/v1/runs"],
  ["%2e%2e collapsed by Next", "GET", "/bff/api/v1/runs/%2e%2e/emergency/kill-switch", "admin", 200, "GET /api/v1/emergency/kill-switch"],
  ["%2E%2E collapsed by Next", "GET", "/bff/api/v1/runs/%2E%2E/emergency/kill-switch", "admin", 200, "GET /api/v1/emergency/kill-switch"],
  [".%2e collapsed by Next", "GET", "/bff/api/v1/runs/.%2e/emergency/kill-switch", "admin", 200, "GET /api/v1/emergency/kill-switch"],
  ["raw ../ collapsed by Next", "GET", "/bff/api/v1/runs/../emergency/kill-switch", "admin", 200, "GET /api/v1/emergency/kill-switch"],
  ["dot collapsed to an unknown path", "GET", "/bff/api/v1/runs/%2e/x", "admin", 404, null],
  ["double-encoded dots", "GET", "/bff/api/v1/runs/%252e%252e/emergency/kill-switch", "admin", 400, null],
  ["encoded slash outside {symbol}", "GET", "/bff/api/v1/runs/a%2Fb", "admin", 400, null],
  ["encoded backslash", "GET", "/bff/api/v1/runs/a%5Cb", "admin", 400, null],
  ["malformed percent-encoding (Next's own 400)", "GET", "/bff/api/v1/runs/%c0%ae", "admin", 400, null],
  ["semicolon", "GET", "/bff/api/v1/runs;x", "admin", 400, null],
  ["trailing dot", "GET", "/bff/api/v1/emergency/kill-switch.", "admin", 404, null],
  ["encoded space", "GET", "/bff/api/v1/runs/%20", "admin", 400, null],
  ["plus", "GET", "/bff/api/v1/runs/a+b", "admin", 400, null],
  ["NUL", "GET", "/bff/api/v1/runs/%00", "admin", 400, null],
  ["non-ASCII dots", "GET", "/bff/api/v1/runs/%EF%BC%8E%EF%BC%8E", "admin", 400, null],
  ["double slash (Next redirects)", "GET", `/bff/api/v1/runs//${UUID}`, "admin", 308, null],
  ["trailing slash (Next redirects)", "GET", "/bff/api/v1/runs/", "admin", 308, null],
  ["non-UUID id", "GET", "/bff/api/v1/runs/not-a-uuid", "admin", 404, null],
  ["metrics", "GET", "/bff/api/v1/metrics", "admin", 404, null],
  ["kill-switch press", "POST", "/bff/api/v1/emergency/kill-switch", "admin", 404, null],
  ["kill-switch clear", "POST", "/bff/api/v1/emergency/kill-switch/clear", "admin", 404, null],
  ["run resume", "POST", `/bff/api/v1/runs/${UUID}/resume`, "admin", 404, null],
  ["latch clear", "POST", `/bff/api/v1/runs/${UUID}/entries-latch/clear`, "admin", 404, null],
  ["circuit-breaker reset", "POST", `/bff/api/v1/runs/${UUID}/circuit-breaker/reset`, "admin", 404, null],
  ["promotion-eligibility", "GET", `/bff/api/v1/runs/${UUID}/promotion-eligibility`, "admin", 404, null],
  ["admin path hidden behind dots", "POST", `/bff/api/v1/runs/%2e%2e/emergency/kill-switch/clear`, "admin", 404, null],
  ["symbol with encoded slash, api_key stripped", "POST", "/bff/api/v1/ml/retrain/BTC%2FUSDT?timeframe=1h&api_key=evil", "admin", 200, "POST /api/v1/ml/retrain/BTC%2FUSDT?timeframe=1h"],
  ["double-encoded symbol", "POST", "/bff/api/v1/ml/retrain/BTC%252FUSDT", "admin", 400, null],
  ["three-part symbol", "POST", "/bff/api/v1/ml/retrain/BTC/USDT/x", "admin", 404, null],
  ["api_key stripped on a read", "GET", "/bff/api/v1/runs?api_key=evil&limit=2", "admin", 200, "GET /api/v1/runs?limit=2"],
  ["viewer reads", "GET", "/bff/api/v1/runs", "viewer", 200, "GET /api/v1/runs"],
  ["viewer write refused", "POST", "/bff/api/v1/runs", "viewer", 403, null],
  ["viewer stop refused", "DELETE", `/bff/api/v1/runs/${UUID}`, "viewer", 403, null],
  ["viewer emergency-stop refused", "POST", `/bff/api/v1/runs/${UUID}/emergency-stop`, "viewer", 403, null],
  ["admin stop", "DELETE", `/bff/api/v1/runs/${UUID}?flatten=true`, "admin", 200, `DELETE /api/v1/runs/${UUID}?flatten=true`],
  ["unlisted address (valid JWT)", "GET", "/bff/api/v1/runs", "outsider", 401, null],
  ["no cookie", "GET", "/bff/api/v1/runs", "none", 401, null],
  // GHSA-xmf8-cvqr-rfgj: next-auth <= 4.24.14 answered a malformed Bearer value with 500.
  ["malformed Bearer on /bff/*", "GET", "/bff/api/v1/runs", "none", 401, null, { authorization: "Bearer %E0%A4%A" }],
  ["malformed Bearer on a page", "GET", "/runs", "none", 307, null, { authorization: "Bearer %E0%A4%A" }],
];

// Anything the upstream receives that matches one of these is a failure,
// whatever the case expected: the BFF must never reach an admin path.
const ADMIN_PATH = /(\/emergency\/kill-switch\/clear|\/resume|\/entries-latch|\/circuit-breaker|\/promotion-eligibility|\/metrics|\/health\/background)(\?|$)/;
const isAdminReach = (method, target) => ADMIN_PATH.test(target) || (method !== "GET" && /\/emergency\/kill-switch(\?|$)/.test(target));

async function main() {
  const installed = require(path.join(UI_ROOT, "node_modules/next/package.json")).version;
  if (installed !== MEASURED_NEXT_VERSION) {
    console.log(`next ${installed} differs from MEASURED_NEXT_VERSION ${MEASURED_NEXT_VERSION}: re-measure, then edit the constant on purpose`);
    return 1;
  }
  const server = path.join(UI_ROOT, ".next/standalone/server.js");
  if (!fs.existsSync(server)) {
    console.log("no standalone build: run `next build` first");
    return 2;
  }
  const { encode } = require(path.join(UI_ROOT, "node_modules/next-auth/jwt"));
  const cookies = {};
  for (const [role, email] of [["admin", "admin@example.com"], ["viewer", "viewer@example.com"], ["outsider", "outsider@example.com"]]) {
    const claim = role === "viewer" ? "viewer" : role === "admin" ? "admin" : "viewer";
    cookies[role] = `__Secure-next-auth.session-token=${await encode({ token: { email, name: role, role: claim, sub: role }, secret: SECRET })}`;
  }

  const seen = [];
  const upstream = http.createServer((req, res) => {
    seen.push({ method: req.method, target: req.url, keyOk: req.headers["x-api-key"] === KEY, cookie: "cookie" in req.headers });
    res.writeHead(200, { "content-type": "application/json" });
    res.end('{"ok":true}');
  });
  const upPort = await new Promise((r) => upstream.listen(0, "127.0.0.1", () => r(upstream.address().port)));
  const port = await freePort();
  const child = spawn(process.execPath, [server], {
    stdio: "ignore",
    env: {
      PATH: process.env.PATH,
      PORT: String(port),
      HOSTNAME: "127.0.0.1",
      NEXTAUTH_SECRET: SECRET,
      NEXTAUTH_URL: `https://localhost:${port}`,
      GOOGLE_OAUTH_CLIENT_ID: "dummy",
      GOOGLE_OAUTH_CLIENT_SECRET: "dummy",
      ADMIN_EMAILS: "admin@example.com",
      VIEWER_EMAILS: "viewer@example.com",
      API_KEY: KEY,
      INTERNAL_API_URL: `http://127.0.0.1:${upPort}`,
    },
  });

  const send = (method, target, cookie, extra = {}) =>
    new Promise((resolve) => {
      const headers = { "content-type": "application/json", "sec-fetch-site": "same-origin", ...extra };
      if (cookie) headers.cookie = cookie;
      const req = http.request({ host: "127.0.0.1", port, method, path: target, headers }, (res) => {
        res.resume();
        res.on("end", () => resolve(res.statusCode));
      });
      req.on("error", () => resolve(0));
      if (method !== "GET" && method !== "DELETE") req.write("{}");
      req.end();
    });

  let failures = 0;
  let adminReached = 0;
  try {
    for (let i = 0; i < 100 && (await send("GET", "/api/auth/session")) === 0; i += 1) await new Promise((r) => setTimeout(r, 300));
    for (const [label, method, target, role, status, expected, extra] of CASES) {
      const before = seen.length;
      const got = await send(method, target, cookies[role], extra);
      await new Promise((r) => setTimeout(r, 60));
      const calls = seen.slice(before);
      const wantCalls = expected ? 1 : 0;
      const reached = calls.map((c) => `${c.method} ${c.target}`);
      const bad = [];
      if (got !== status) bad.push(`status ${got} (want ${status})`);
      if (calls.length !== wantCalls) bad.push(`upstream calls ${calls.length} (want ${wantCalls})`);
      if (expected && reached[0] !== expected) bad.push(`upstream saw ${reached[0]} (want ${expected})`);
      if (calls.some((c) => !c.keyOk || c.cookie)) bad.push("upstream saw a wrong key or a cookie");
      if (calls.some((c) => isAdminReach(c.method, c.target))) {
        adminReached += 1;
        bad.push("ADMIN PATH REACHED");
      }
      if (bad.length) failures += 1;
      console.log(`${bad.length ? "FAIL" : "ok  "} ${label}: ${got}, upstream_calls=${calls.length}${bad.length ? " <- " + bad.join("; ") : ""}`);
    }
  } finally {
    child.kill("SIGTERM");
    upstream.close();
  }
  console.log(`cases: ${CASES.length}, failures: ${failures}, admin paths reached: ${adminReached}, next ${installed}`);
  return failures || adminReached ? 1 : 0;
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  main().then((code) => process.exit(code));
}
