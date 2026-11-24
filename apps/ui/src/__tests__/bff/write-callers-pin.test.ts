/**
 * CF-97-43 (CR-9743-03): static pin so a new caller of a write function cannot
 * appear without being classified. A viewer's write is refused by the BFF; the
 * dashboard must not offer it (hidden or disabled "admin only" via <AdminOnly>).
 *
 * Every non-test file that calls an exported write function of lib/api.ts or
 * lib/api-server.ts, or that spells a literal non-GET `method`, must be listed
 * below with a gate file that mentions AdminOnly (UI) or isAdmin( (server
 * route). A new caller, a new unclassified write export, or a request method
 * that is not a string literal fails here until it is classified. The pin does
 * not prove gating: the per-control viewer tests do (viewer-write-controls).
 */
import fs from "fs";
import path from "path";

const SRC = path.resolve(__dirname, "../..");

const walk = (dir: string): string[] =>
  fs.readdirSync(dir, { withFileTypes: true }).flatMap((e) => {
    const p = path.join(dir, e.name);
    if (e.isDirectory()) return e.name === "__tests__" ? [] : walk(p);
    return /\.tsx?$/.test(e.name) ? [p] : [];
  });
const rel = (p: string) => path.relative(SRC, p).split(path.sep).join("/");
const stripComments = (t: string) => t.replace(/\/\*[\s\S]*?\*\//g, "").replace(/(?<![:\w"'`])\/\/[^\n]*/g, "");
const FILES = walk(SRC).map((p) => ({ file: rel(p), text: stripComments(fs.readFileSync(p, "utf8")) }));

type Src = { file: string; text: string };

// A literal non-GET method in any quote style.
const WRITE_METHOD = /\bmethod\s*:\s*["'`](POST|PUT|PATCH|DELETE)["'`]/i;

/** Write exports of the given api files (`export function` and `export const x = (async) (`). */
function deriveWrites(files: Src[], apiFiles = ["lib/api.ts", "lib/api-server.ts"]): string[] {
  return apiFiles.flatMap((f) => {
    const text = files.find((x) => x.file === f)?.text ?? "";
    return text
      .split(/^(?=export )/m)
      .map((b) => ({
        name:
          /^export (?:async )?function (\w+)/.exec(b)?.[1] ??
          /^export const (\w+)\s*(?::[^=]+)?=\s*(?:async\s*)?(?:function\b|\(|\w+\s*=>)/.exec(b)?.[1],
        body: b,
      }))
      .filter((b) => b.name && (WRITE_METHOD.test(b.body) || /\bapi(Post|Delete)\b/.test(b.body) || /^api(Post|Delete)$/.test(b.name)))
      .map((b) => b.name as string);
  });
}

/** Every `method` property (also the shorthand) whose value is not a plain string literal. */
function nonLiteralMethods(text: string): string[] {
  const found: string[] = [];
  for (const m of text.matchAll(/\bmethod\s*:\s*([^,}\n]+)/g)) {
    const value = (m[1] as string).trim();
    // `method: string` is a type annotation, not a request option.
    if (value !== "string" && !/^(["'`])[A-Za-z]+\1$/.test(value)) found.push(value);
  }
  // Shorthand `{ method }` in the options of a fetch-like call (JSX `{method}` and destructuring are not requests).
  for (const call of text.matchAll(/\b(?:\w*[fF]etch\w*|requestJson)\s*(?:<[^>(]*>)?\(/g)) {
    const window = text.slice(call.index, call.index + 400);
    for (const m of window.matchAll(/[{,]\s*method\s*(?:,|\})/g)) found.push(m[0].trim());
  }
  return found;
}

const derived = deriveWrites(FILES);

const NAMED = [
  "createRun", "promoteRun", "stopRun", "archiveRun", "runOptimization", "trainModel",
  "retrainModel", "activateModelVersion", "emergencyStop", "apiPost", "apiDelete",
];

/** caller file -> [write functions it calls, files that gate it]. */
const CALLERS: Record<string, [string[], string[]]> = {
  "app/models/page.tsx": [["activateModelVersion", "retrainModel", "trainModel"], ["app/models/page.tsx"]],
  "app/optimize/[id]/page.tsx": [["createRun"], ["app/optimize/result-columns.tsx"]],
  "app/optimize/page.tsx": [["createRun", "runOptimization"], ["app/optimize/page.tsx", "app/optimize/result-columns.tsx"]],
  "app/runs/[id]/page.tsx": [["archiveRun"], ["app/runs/[id]/page.tsx"]],
  "app/runs/new/page.tsx": [["createRun"], ["app/runs/new/page.tsx"]],
  "components/promote-run-dialog.tsx": [["promoteRun"], ["app/runs/[id]/page.tsx"]],
  "components/stop-run-dialog.tsx": [["stopRun"], ["app/runs/[id]/page.tsx"]],
};

/** Files with a literal non-GET method outside lib/api.ts -> gate file. */
const LITERAL_METHOD: Record<string, string> = {
  "app/api/admin/kill-switch/clear/route.ts": "app/api/admin/kill-switch/clear/route.ts",
  "app/api/admin/kill-switch/route.ts": "app/api/admin/kill-switch/route.ts",
  "app/api/admin/runs/[id]/entries-latch/clear/route.ts": "app/api/admin/runs/[id]/entries-latch/clear/route.ts",
  "app/api/admin/runs/[id]/resume/route.ts": "app/api/admin/runs/[id]/resume/route.ts",
  "components/kill-switch-button.tsx": "components/kill-switch-button.tsx",
  "components/kill-switch-panel.tsx": "components/kill-switch-panel.tsx",
  "components/resume-run-dialog.tsx": "app/runs/[id]/page.tsx",
  "components/stop-run-dialog.tsx": "components/stop-run-dialog.tsx",
};

/**
 * Files that mention `method` without issuing a request with a non-literal one:
 * the BFF proxy forwards the incoming request's method after the role check in
 * the route handler; the route table declares methods as data.
 */
const NON_LITERAL_ALLOWED = ["lib/bff-proxy.ts", "lib/bff-routes.ts"];

const callerSet = (files: Src[], fns: string[]) => {
  const found: Record<string, string[]> = {};
  for (const { file, text } of files) {
    if (file === "lib/api.ts" || file === "lib/api-server.ts") continue;
    const used = fns.filter((n) => new RegExp(`\\b${n}\\b`).test(text)).sort();
    if (used.length) found[file] = used;
  }
  return found;
};
const literalWriteFiles = (files: Src[]) =>
  files.filter((f) => f.file !== "lib/api.ts" && WRITE_METHOD.test(f.text)).map((f) => f.file).sort();
const nonLiteralFiles = (files: Src[]) =>
  files.filter((f) => !NON_LITERAL_ALLOWED.includes(f.file) && nonLiteralMethods(f.text).length > 0).map((f) => f.file).sort();

const gated = (f: string) => {
  const t = FILES.find((x) => x.file === f)?.text;
  expect(t).toBeDefined();
  expect(/AdminOnly|\bisAdmin\(/.test(t as string)).toBe(true);
};

describe("every caller of a write function is classified, with a gate file that mentions AdminOnly (CF-97-43)", () => {
  it("the derived write exports include every named one, and nothing unnamed", () => {
    expect([...derived].sort()).toEqual([...NAMED].sort());
  });

  it("the set of caller files and the functions each calls equals the classified list", () => {
    const expected = Object.fromEntries(Object.entries(CALLERS).map(([f, [fns]]) => [f, [...fns].sort()]));
    expect(callerSet(FILES, derived)).toEqual(expected);
  });

  it.each(Object.entries(CALLERS))("%s names a gate file that mentions AdminOnly", (_file, [, gates]) => {
    gates.forEach(gated);
  });

  it("the files with a literal non-GET method (outside lib/api.ts) equal the classified list", () => {
    expect(literalWriteFiles(FILES)).toEqual(Object.keys(LITERAL_METHOD).sort());
    Object.values(LITERAL_METHOD).forEach(gated);
  });

  it("no non-test file (outside the two listed) passes a request method that is not a string literal", () => {
    expect(nonLiteralFiles(FILES)).toEqual([]);
  });
});

// Planted offender text (not real files): each case below survived the r2 pin.
describe("the pin catches the styles the first version missed (CR-9743-08)", () => {
  const apiWith = (text: string): Src[] => [{ file: "lib/api.ts", text }];

  it("a write export with a single-quoted method is derived", () => {
    const files = apiWith(`export async function wipeAll() {\n  return apiFetch("/x", { method: 'DELETE' });\n}`);
    expect(deriveWrites(files)).toEqual(["wipeAll"]);
  });

  it("a backtick-quoted method is derived too", () => {
    const files = apiWith("export function wipeAll() { return apiFetch(\"/x\", { method: `PUT` }); }");
    expect(deriveWrites(files)).toEqual(["wipeAll"]);
  });

  it("`export const x = async (...) =>` and `export const x = (...) =>` write exports are derived; a GET one is not", () => {
    const files = apiWith(
      [
        `export const wipeAll = async (id: string) => apiFetch("/x/" + id, { method: "DELETE" });`,
        `export const poke = (id: string) => apiPost("/y/" + id);`,
        `export const readAll = async () => apiFetch("/z", { method: "GET" });`,
      ].join("\n"),
    );
    expect(deriveWrites(files)).toEqual(["wipeAll", "poke"]);
  });

  it("a raw fetch with a single-quoted non-GET method in a new file is a literal-method file", () => {
    const files: Src[] = [{ file: "components/new-thing.tsx", text: `await fetch("/x", { method: 'POST' });` }];
    expect(literalWriteFiles(files)).toEqual(["components/new-thing.tsx"]);
  });

  it("a raw fetch whose method is a variable, a property or a template fails the non-literal check", () => {
    const bad = (text: string) => nonLiteralFiles([{ file: "components/new-thing.tsx", text }]);
    expect(bad(`const m = pick(); await fetch("/x", { method: m });`)).toEqual(["components/new-thing.tsx"]);
    expect(bad(`await fetch("/x", { method: req.method });`)).toEqual(["components/new-thing.tsx"]);
    expect(bad("await fetch(\"/x\", { method: `${verb}` });")).toEqual(["components/new-thing.tsx"]);
    expect(bad(`const method = "POST"; await fetch("/x", { method });`)).toEqual(["components/new-thing.tsx"]);
  });

  it("literal methods, type annotations and destructured props are not flagged", () => {
    const ok = (text: string) => nonLiteralFiles([{ file: "components/new-thing.tsx", text }]);
    expect(ok(`await fetch("/x", { method: "GET" });`)).toEqual([]);
    expect(ok(`await fetch("/x", { method: 'POST' });`)).toEqual([]);
    expect(ok("await fetch(\"/x\", { method: `DELETE` });")).toEqual([]);
    expect(ok(`function Badge({ method }: { method: string }) { return method; }`)).toEqual([]);
  });

  it("a new write function called from an unlisted file is a new caller", () => {
    const files: Src[] = [{ file: "components/new-thing.tsx", text: `import { wipeAll } from "@/lib/api"; void wipeAll();` }];
    expect(callerSet(files, ["wipeAll"])).toEqual({ "components/new-thing.tsx": ["wipeAll"] });
  });
});
