#!/usr/bin/env node
/**
 * apps/ui/scripts/check-bff-key-leak.mjs
 * ----------------------------------------
 * CF-97-37 (37-B, DR-9737-25): build-output leak scan for server-only secrets.
 *
 * Runs `next build` with RANDOM (per run), alphanumeric sentinel values for the server
 * secrets (passed through the child's environment, never argv), then scans the
 * build output for each sentinel in four forms: raw, URL-encoded, JSON-escaped
 * and base64 (three alignments). It also fails on any *.body, *.html or *.rsc
 * file under .next/server/app/bff (a prerendered BFF route would bake data).
 *
 *   node apps/ui/scripts/check-bff-key-leak.mjs
 *
 * Takes no arguments; any argument exits 2. Never type real secrets on a command
 * line. Checking a production image for real secrets is a separate procedure
 * (CF-97-38). The build child gets an explicit allow-listed environment
 * (`buildEnv`), not the parent's. `selfTest()` plants and scans for its own
 * random value (never one of the run's sentinels or any supplied value) and runs
 * before the build.
 *
 * Output: file paths, counts and names only; a sentinel or any other value is never printed.
 * Exit 0 = clean, 1 = a hit or a vacuous scan, 2 = build/usage error.
 * NEXTAUTH_SECRET is baked into the Edge middleware by design (CF-97-38): it
 * is counted and reported but does not fail the scan.
 */

import { spawnSync } from "node:child_process";
import { randomBytes } from "node:crypto";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";

const UI_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

// PR-HYG-01: the sentinels are random per run (hex, so alphanumeric) and are never
// committed. The same objects are what the build receives and what the scan searches.
const randomValue = () => randomBytes(24).toString("hex");
export const SENTINELS = {
  API_KEY: randomValue(),
  INTERNAL_ADMIN_API_KEY: randomValue(),
  GOOGLE_OAUTH_CLIENT_SECRET: randomValue(),
};
const BAKED_BY_DESIGN = { NEXTAUTH_SECRET: randomValue() };

/** Minimum file counts so a scan of an empty or missing tree cannot pass. */
export const MIN_FILES = { ".next/static": 5, ".next/server": 20, ".next/standalone": 50 };
const ROOTS = [
  ".next/static",
  ".next/server",
  ".next/standalone",
  ".next/cache",
  ".next/required-server-files.json",
  ".next/trace",
  "public",
];

export function forms(value) {
  // base64 of the value as it would appear at byte offset 0, 1 or 2 (mod 3) of a
  // stream: only the complete 4-character groups made of value bytes are kept.
  const b64 = [0, 1, 2].map((offset) => {
    const encoded = Buffer.from("x".repeat(offset) + value).toString("base64");
    return encoded.slice(Math.ceil(offset / 3) * 4, Math.floor((offset + value.length) / 3) * 4);
  });
  return [value, encodeURIComponent(value), JSON.stringify(value).slice(1, -1), ...b64];
}

function* walk(target) {
  let stat;
  try {
    stat = fs.lstatSync(target);
  } catch {
    return;
  }
  if (stat.isSymbolicLink()) return;
  if (stat.isFile()) {
    yield target;
    return;
  }
  for (const entry of fs.readdirSync(target)) yield* walk(path.join(target, entry));
}

/** Returns { files, hits: [{file, name}] } for the given absolute roots. */
export function scan(roots, sentinels) {
  const needles = Object.entries(sentinels).flatMap(([name, value]) =>
    forms(value).map((form) => ({ name, form: Buffer.from(form) })),
  );
  const hits = [];
  let files = 0;
  for (const root of roots) {
    for (const file of walk(root)) {
      files += 1;
      const data = fs.readFileSync(file);
      for (const { name, form } of needles) {
        if (data.includes(form)) hits.push({ file, name });
      }
    }
  }
  return { files, hits };
}

/** The scan must find a planted sentinel in every form, and nothing in a clean tree. */
export function selfTest() {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "bff-leak-selftest-"));
  const clean = path.join(dir, "clean");
  const planted = path.join(dir, "planted");
  fs.mkdirSync(clean);
  fs.mkdirSync(path.join(planted, ".hidden"), { recursive: true });
  fs.writeFileSync(path.join(clean, "a.js"), "console.log('no secrets here');");
  const own = { SELFTEST: randomValue() };
  const value = own.SELFTEST;
  const plantedForms = forms(value);
  plantedForms.forEach((form, i) =>
    fs.writeFileSync(path.join(planted, i % 2 ? ".hidden" : ".", `p${i}.txt`), `xx${form}xx`),
  );
  const cleanResult = scan([clean], own);
  const plantedResult = scan([planted], own);
  fs.rmSync(dir, { recursive: true, force: true });
  return cleanResult.hits.length === 0 && plantedResult.hits.length >= plantedForms.length;
}

/** Parent variables the build may see besides the sentinels: nothing else leaks into it. */
export const ENV_ALLOW = [
  "PATH", "HOME", "TMPDIR", "LANG", "NODE_ENV",
  "HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY", "http_proxy", "https_proxy", "no_proxy",
  "NODE_EXTRA_CA_CERTS", "SSL_CERT_FILE",
];

/** The environment of the `next build` child: an allow-list of the parent plus the sentinels. */
export function buildEnv(parent, sentinels = SENTINELS, baked = BAKED_BY_DESIGN) {
  const env = {};
  for (const name of ENV_ALLOW) if (parent[name] !== undefined) env[name] = parent[name];
  return {
    ...env,
    ...sentinels,
    ...baked,
    NODE_ENV: "production",
    NEXT_TELEMETRY_DISABLED: "1",
    NEXTAUTH_URL: "https://localhost:3100",
    GOOGLE_OAUTH_CLIENT_ID: "dummy-client-id",
    ADMIN_EMAILS: "admin@example.invalid",
  };
}

function main() {
  if (process.argv.length > 2) {
    console.log("usage: check-bff-key-leak.mjs (takes no arguments)");
    return 2;
  }
  if (!selfTest()) {
    console.log("self-test FAILED: the scanner did not find a planted sentinel");
    return 1;
  }
  console.log("self-test ok");
  const build = spawnSync("npx", ["next", "build"], {
    cwd: UI_ROOT,
    env: buildEnv(process.env),
    stdio: ["ignore", "ignore", "inherit"],
  });
  if (build.status !== 0) {
    console.log(`build failed (exit ${build.status})`);
    return 2;
  }
  let failed = false;
  for (const [tree, min] of Object.entries(MIN_FILES)) {
    const { files } = scan([path.join(UI_ROOT, tree)], {});
    console.log(`${tree}: ${files} files (minimum ${min})`);
    if (files < min) failed = true;
  }
  const roots = ROOTS.map((r) => path.join(UI_ROOT, r));
  const { files, hits } = scan(roots, SENTINELS);
  console.log(`scanned ${files} files`);
  const seen = new Set(hits.map(({ file, name }) => `HIT ${name} in ${path.relative(UI_ROOT, file)}`));
  for (const line of seen) console.log(line);
  const baked = scan(roots, BAKED_BY_DESIGN).hits.length;
  console.log(`NEXTAUTH_SECRET baked-by-design occurrences (not a failure): ${baked}`);
  const bffApp = path.join(UI_ROOT, ".next/server/app/bff");
  let stray = 0;
  for (const file of walk(bffApp)) {
    if (/\.(body|html|rsc)$/.test(file)) {
      stray += 1;
      console.log(`PRERENDERED ${path.relative(UI_ROOT, file)}`);
    }
  }
  console.log(`hits: ${hits.length}, prerendered bff files: ${stray}`);
  return failed || hits.length > 0 || stray > 0 ? 1 : 0;
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  process.exit(main());
}
