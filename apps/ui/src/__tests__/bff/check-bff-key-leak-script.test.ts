/** @jest-environment node */
/**
 * CF-97-39b r2: check-bff-key-leak.mjs takes no arguments, and the `next build`
 * child gets an allow-listed environment, never the parent's. Values are
 * generated at run time and are never printed or asserted as literals.
 */
import { spawnSync } from "child_process";
import { randomBytes } from "crypto";
import fs from "fs";
import os from "os";
import path from "path";

const SCRIPT = path.resolve(__dirname, "../../../scripts/check-bff-key-leak.mjs");

// A fake `npx` first on PATH: if the argument check ever regresses, the script reaches
// `spawnSync("npx", ...)`, which runs this stub (it records a marker and exits 9) instead
// of a real `next build`.
let stubDir: string;
beforeAll(() => {
  stubDir = fs.mkdtempSync(path.join(os.tmpdir(), "leak-fake-npx-"));
  fs.writeFileSync(path.join(stubDir, "npx"), `#!/bin/sh\ntouch "${path.join(stubDir, "npx-was-run")}"\nexit 9\n`, { mode: 0o755 });
});
afterAll(() => fs.rmSync(stubDir, { recursive: true, force: true }));

describe("arguments", () => {
  it.each([["--no-build"], ["--bogus"], ["x", "y"]])("%s exits 2 with the fixed usage line and starts nothing", (...args) => {
    const res = spawnSync(process.execPath, [SCRIPT, ...args], {
      encoding: "utf8",
      timeout: 20_000,
      env: { ...process.env, PATH: `${stubDir}${path.delimiter}${process.env.PATH ?? ""}` },
    });
    expect(res.status).toBe(2);
    expect(res.stdout).toBe("usage: check-bff-key-leak.mjs (takes no arguments)\n");
    expect(fs.existsSync(path.join(stubDir, "npx-was-run"))).toBe(false);
  });

  it("the flag is gone from the script", () => {
    expect(fs.readFileSync(SCRIPT, "utf8")).not.toContain("no-build");
  });
});

describe("the build environment is an allow-list (buildEnv)", () => {
  // Runs buildEnv in a child so the ESM script is imported untouched; prints booleans only.
  function probe(parent: Record<string, string>) {
    const code = `
      const m = await import(process.env.LEAK_SCRIPT);
      const e = m.buildEnv(process.env);
      const planted = process.env.PLANTED_NAMES.split(",");
      console.log(JSON.stringify({
        leaked: planted.filter((n) => n in e && e[n] === process.env[n]),
        sentinelsOk: Object.entries(m.SENTINELS).every(([k, v]) => e[k] === v),
        keys: Object.keys(e).sort(),
        allow: m.ENV_ALLOW,
      }));`;
    const res = spawnSync(process.execPath, ["--input-type=module", "-e", code], {
      encoding: "utf8",
      env: { ...parent, LEAK_SCRIPT: SCRIPT, PLANTED_NAMES: Object.keys(parent).join(",") } as unknown as NodeJS.ProcessEnv,
    });
    expect(res.status).toBe(0);
    return JSON.parse(res.stdout) as { leaked: string[]; sentinelsOk: boolean; keys: string[]; allow: string[] };
  }

  it("a planted EXTRA_SECRET, and parent copies of the sentinel names, do not reach the build", () => {
    const parent = {
      PATH: process.env.PATH ?? "",
      EXTRA_SECRET: randomBytes(16).toString("hex"),
      API_KEY: randomBytes(16).toString("hex"),
      INTERNAL_ADMIN_API_KEY: randomBytes(16).toString("hex"),
      NEXTAUTH_SECRET: randomBytes(16).toString("hex"),
      DATABASE_URL: randomBytes(16).toString("hex"),
    };
    const out = probe(parent);
    expect(out.leaked).toEqual(["PATH"]); // PATH is allow-listed; nothing else equals the parent's value
    expect(out.keys).not.toContain("EXTRA_SECRET");
    expect(out.keys).not.toContain("DATABASE_URL");
    expect(out.sentinelsOk).toBe(true);
  });

  it("ENV_ALLOW is exactly the reviewed list (CR-9739B-05)", () => {
    expect(probe({ PATH: process.env.PATH ?? "" }).allow).toEqual([
      "PATH", "HOME", "TMPDIR", "LANG", "NODE_ENV",
      "HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY", "http_proxy", "https_proxy", "no_proxy",
      "NODE_EXTRA_CA_CERTS", "SSL_CERT_FILE",
    ]);
  });

  it("the proxy and CA variables pass when set, and only the allow-list plus the fixed names exist", () => {
    const parent = {
      PATH: process.env.PATH ?? "",
      HTTPS_PROXY: "http://proxy.invalid:3128",
      https_proxy: "http://proxy.invalid:3128",
      NO_PROXY: "localhost",
      NODE_EXTRA_CA_CERTS: "/etc/ca.pem",
      SSL_CERT_FILE: "/etc/ca.pem",
      EXTRA_SECRET: randomBytes(16).toString("hex"),
    };
    const out = probe(parent);
    for (const name of ["HTTPS_PROXY", "https_proxy", "NO_PROXY", "NODE_EXTRA_CA_CERTS", "SSL_CERT_FILE"]) {
      expect(out.keys).toContain(name);
    }
    const fixed = [
      "API_KEY", "INTERNAL_ADMIN_API_KEY", "GOOGLE_OAUTH_CLIENT_SECRET", "NEXTAUTH_SECRET",
      "NODE_ENV", "NEXT_TELEMETRY_DISABLED", "NEXTAUTH_URL", "GOOGLE_OAUTH_CLIENT_ID", "ADMIN_EMAILS",
    ];
    expect(out.keys.filter((k) => !out.allow.includes(k) && !fixed.includes(k))).toEqual([]);
  });
});

describe("main() order", () => {
  it("checks arguments, then self-tests, then spawns the build with buildEnv(process.env)", () => {
    const text = fs.readFileSync(SCRIPT, "utf8");
    const main = text.slice(text.indexOf("function main()"));
    const at = (needle: string) => main.indexOf(needle);
    expect(at("process.argv.length > 2")).toBeGreaterThanOrEqual(0);
    expect(at("process.argv.length > 2")).toBeLessThan(at("selfTest()"));
    expect(at("selfTest()")).toBeLessThan(at("spawnSync("));
    expect(main).toContain("env: buildEnv(process.env)");
    expect(main).not.toMatch(/\.\.\.process\.env/);
  });
});

describe("selfTest() (SE-9739B-06)", () => {
  it("uses its own random value, not the run's sentinels, and still passes", () => {
    const text = fs.readFileSync(SCRIPT, "utf8");
    const body = text.slice(text.indexOf("export function selfTest()"), text.indexOf("/** Parent variables"));
    expect(body).toContain("randomValue()");
    expect(body).not.toContain("SENTINELS");
    const res = spawnSync(
      process.execPath,
      ["--input-type=module", "-e", "const m = await import(process.env.LEAK_SCRIPT); console.log(m.selfTest());"],
      { encoding: "utf8", env: { ...process.env, LEAK_SCRIPT: SCRIPT } },
    );
    expect(res.stdout.trim()).toBe("true");
  });
});
