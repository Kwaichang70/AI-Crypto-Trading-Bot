#!/usr/bin/env node
/**
 * apps/ui/scripts/check-bff-key-leak.mjs
 * ----------------------------------------
 * CF-97-37 (37-B, DR-9737-25): build-output leak scan for server-only secrets.
 *
 * Runs `next build` with FIXED, alphanumeric sentinel values for the server
 * secrets (passed through the child's environment, never argv), then scans the
 * build output for each sentinel in four forms: raw, URL-encoded, JSON-escaped
 * and base64 (three alignments). It also fails on any *.body, *.html or *.rsc
 * file under .next/server/app/bff (a prerendered BFF route would bake data).
 *
 *   node apps/ui/scripts/check-bff-key-leak.mjs [--no-build]
 *
 * Output: file paths and counts only; sentinel values are never printed.
 * Exit 0 = clean, 1 = a hit or a vacuous scan, 2 = build/usage error.
 * NEXTAUTH_SECRET is baked into the Edge middleware by design (CF-97-38): it
 * is counted and reported but does not fail the scan.
 */

import { spawnSync } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";

const UI_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

export const SENTINELS = {
  API_KEY: "SentinelApiKeyZ9q4Xv7Lm2Wk8Tn5Bd3Hc6Rj1Fa0Ue",
  INTERNAL_ADMIN_API_KEY: "SentinelAdminKeyP8s3Gy6Nw1Qz5Dm9Vb2Lx7Kc4Ta0Jh",
  GOOGLE_OAUTH_CLIENT_SECRET: "SentinelOauthSecretE7r2Ht5Yu9Io3Pl6Ka1Sd4Fg8Zx0Cv",
};
const BAKED_BY_DESIGN = { NEXTAUTH_SECRET: "SentinelNextauthSecretM3n8Bv5Cx1Zq7Lk4Jh9Gf2Ds6Aw0Tr" };

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
  const value = SENTINELS.API_KEY;
  const plantedForms = forms(value);
  plantedForms.forEach((form, i) =>
    fs.writeFileSync(path.join(planted, i % 2 ? ".hidden" : ".", `p${i}.txt`), `xx${form}xx`),
  );
  const cleanResult = scan([clean], SENTINELS);
  const plantedResult = scan([planted], SENTINELS);
  fs.rmSync(dir, { recursive: true, force: true });
  return cleanResult.hits.length === 0 && plantedResult.hits.length >= plantedForms.length;
}

function main() {
  if (!selfTest()) {
    console.log("self-test FAILED: the scanner did not find a planted sentinel");
    return 1;
  }
  console.log("self-test ok");
  if (!process.argv.includes("--no-build")) {
    const env = {
      ...process.env,
      ...SENTINELS,
      ...BAKED_BY_DESIGN,
      NEXT_TELEMETRY_DISABLED: "1",
      NEXTAUTH_URL: "https://localhost:3100",
      GOOGLE_OAUTH_CLIENT_ID: "dummy-client-id",
      ADMIN_EMAILS: "admin@example.invalid",
    };
    const build = spawnSync("npx", ["next", "build"], { cwd: UI_ROOT, env, stdio: ["ignore", "ignore", "inherit"] });
    if (build.status !== 0) {
      console.log(`build failed (exit ${build.status})`);
      return 2;
    }
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
