/**
 * apps/ui/src/components/config-warnings-banner.tsx
 * ------------------------------------------------------
 * WP1.3a (CF-13a-1 item 3) success-path banners for the exit-config/
 * pyramiding validator (reports/vp2-wp1.3a/synthesis-spec.md §5, §9,
 * SY-13a-16/18):
 *
 *   - `ConfigWarningsBanner`: `configWarnings[]` from a 201 create (W1-W5,
 *     W7, W8). `no_downside_exit` (W8) is styled as a critical warning per
 *     CF-13a-1 item 3's "no downside exit, flatten recommended" framing.
 *   - `ProtectiveResumeBanner`: `exitConfigWaived`/`exitManagerMissing` from
 *     a protective resume. `exitManagerMissing=true` is ALWAYS critical
 *     ("no downside exit, flatten recommended"), independent of whether a
 *     waiver was also recorded.
 */

import type { ConfigWarning, ExitConfigWaived } from "@/lib/types";
import { describeConfigWarning, describeExitConfigIssue } from "@/lib/exit-config";

const CRITICAL_WARNING_CODES: ReadonlySet<string> = new Set(["no_downside_exit"]);

export function ConfigWarningsBanner({
  warnings,
}: {
  warnings: readonly ConfigWarning[] | undefined | null;
}) {
  if (!warnings || warnings.length === 0) return null;

  const critical = warnings.filter((w) => CRITICAL_WARNING_CODES.has(w.code));
  const rest = warnings.filter((w) => !CRITICAL_WARNING_CODES.has(w.code));

  return (
    <div data-testid="config-warnings-banner" className="space-y-2">
      {critical.length > 0 && (
        <div className="rounded-lg border border-red-300 bg-red-50 px-4 py-3 text-sm text-red-700 dark:border-red-800 dark:bg-red-900/20 dark:text-red-400">
          <p className="font-semibold">No downside exit — flatten recommended</p>
          <ul className="mt-1 list-inside list-disc space-y-0.5 text-xs">
            {critical.map((w, i) => (
              <li key={i}>{describeConfigWarning(w)}</li>
            ))}
          </ul>
        </div>
      )}
      {rest.length > 0 && (
        <div className="rounded-lg border border-amber-300 bg-amber-50 px-4 py-3 text-sm text-amber-700 dark:border-amber-800 dark:bg-amber-900/20 dark:text-amber-400">
          <p className="font-semibold">Warnings</p>
          <ul className="mt-1 list-inside list-disc space-y-0.5 text-xs">
            {rest.map((w, i) => (
              <li key={i}>{describeConfigWarning(w)}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

export function ProtectiveResumeBanner({
  exitConfigWaived,
  exitManagerMissing,
}: {
  exitConfigWaived: ExitConfigWaived | null | undefined;
  exitManagerMissing: boolean | null | undefined;
}) {
  if (!exitConfigWaived && !exitManagerMissing) return null;

  return (
    <div
      data-testid="protective-resume-banner"
      className={[
        "space-y-2 rounded-lg border px-4 py-3 text-sm",
        exitManagerMissing
          ? "border-red-300 bg-red-50 text-red-700 dark:border-red-800 dark:bg-red-900/20 dark:text-red-400"
          : "border-amber-300 bg-amber-50 text-amber-700 dark:border-amber-800 dark:bg-amber-900/20 dark:text-amber-400",
      ].join(" ")}
    >
      {exitManagerMissing && (
        <p className="font-semibold">No downside exit — flatten recommended</p>
      )}
      {exitConfigWaived && (
        <>
          <p className="font-semibold">
            {exitManagerMissing ? "Exit config waived (protective resume)" : "Exit config partially waived (protective resume)"}
          </p>
          {exitConfigWaived.errors.length > 0 && (
            <ul className="list-inside list-disc space-y-0.5 text-xs">
              {exitConfigWaived.errors.map((issue, i) => (
                <li key={i}>{describeExitConfigIssue(issue)}</li>
              ))}
            </ul>
          )}
        </>
      )}
    </div>
  );
}
