/**
 * apps/ui/src/components/exit-config-error-panel.tsx
 * ------------------------------------------------------
 * WP1.3a (CF-13a-1 item 2) shared renderer for the exit-config/pyramiding
 * validator's structured 422 body, reused on run create, promote-to-live,
 * resume (normal mode) and optimize (reports/vp2-wp1.3a/synthesis-spec.md
 * §5, §9, SY-13a-18).
 *
 * `ExitConfigErrorPanel` takes the raw `ApiError.detail` (whatever
 * `apiFetch` captured from the response body) and renders:
 *   - `errors[]` — one line per field/reason, human-readable
 *   - `hint` (present on `live_pyramiding_forbidden`, e.g. SY-13a-09's
 *     "pass allowPyramiding=false to run single-entry")
 *   - `requires_one_of` (present on `exit_manager_required`)
 *   - `warnings[]` — the body also carries these even on a 422 (§5)
 *   - a generic fallback (`fallbackMessage`) when the detail doesn't
 *     unwrap to one of the three known codes at all (a different 422/409
 *     shape, a plain-string 400/404/500, etc.)
 *
 * `variant="optimize"` renders the combination-scoped shape instead
 * (`combo_index`/`params` per issue, `total_invalid`).
 */

import type { ConfigWarning } from "@/lib/types";
import {
  describeConfigWarning,
  describeExitConfigIssue,
  describeOptimizeExitConfigIssue,
  unwrapExitConfigDetail,
  unwrapOptimizeExitConfigDetail,
} from "@/lib/exit-config";

const CODE_TITLES: Record<string, string> = {
  invalid_exit_config: "Invalid exit configuration",
  exit_manager_required: "This strategy requires a downside exit",
  live_pyramiding_forbidden: "Live pyramiding is forbidden",
};

interface ExitConfigErrorPanelProps {
  /** The raw `ApiError.detail` (or `ApiError.detail`-shaped value) from a failed request. */
  detail: unknown;
  /** Shown when `detail` does not unwrap to a known exit-config/pyramiding code. */
  fallbackMessage: string;
  /** `optimize` renders `errors[]` items with their `combo_index`/`params`. */
  variant?: "default" | "optimize";
}

export function ExitConfigErrorPanel({
  detail,
  fallbackMessage,
  variant = "default",
}: ExitConfigErrorPanelProps) {
  if (variant === "optimize") {
    const parsed = unwrapOptimizeExitConfigDetail(detail);
    if (!parsed) {
      return <GenericFallback message={fallbackMessage} />;
    }
    return (
      <div
        data-testid="exit-config-error-panel"
        className="space-y-2 rounded-lg border border-red-300 bg-red-50 px-4 py-3 text-sm text-red-600 dark:border-red-800 dark:bg-red-900/20 dark:text-red-400"
      >
        <p className="font-semibold">{CODE_TITLES[parsed.code] ?? parsed.code}</p>
        {typeof parsed.total_invalid === "number" && (
          <p className="text-xs">
            {parsed.total_invalid} combination{parsed.total_invalid === 1 ? "" : "s"} failed
            validation{parsed.errors.length < parsed.total_invalid ? ` (showing first ${parsed.errors.length})` : ""}.
          </p>
        )}
        {parsed.errors.length > 0 && (
          <ul className="list-inside list-disc space-y-1 text-xs">
            {parsed.errors.map((issue, i) => (
              <li key={i}>{describeOptimizeExitConfigIssue(issue)}</li>
            ))}
          </ul>
        )}
        {parsed.warnings && <WarningsList warnings={parsed.warnings} />}
      </div>
    );
  }

  const parsed = unwrapExitConfigDetail(detail);
  if (!parsed) {
    return <GenericFallback message={fallbackMessage} />;
  }

  return (
    <div
      data-testid="exit-config-error-panel"
      className="space-y-2 rounded-lg border border-red-300 bg-red-50 px-4 py-3 text-sm text-red-600 dark:border-red-800 dark:bg-red-900/20 dark:text-red-400"
    >
      <p className="font-semibold">{CODE_TITLES[parsed.code] ?? parsed.code}</p>

      {parsed.code === "exit_manager_required" && (
        <p className="text-xs">
          <span className="font-medium">{parsed.strategy}</span> needs one of:{" "}
          {parsed.requires_one_of.join(", ")}
        </p>
      )}

      {(parsed.code === "live_pyramiding_forbidden") && parsed.strategy && (
        <p className="text-xs">
          Strategy: <span className="font-medium">{parsed.strategy}</span>
        </p>
      )}

      {parsed.errors.length > 0 && (
        <ul className="list-inside list-disc space-y-1 text-xs">
          {parsed.errors.map((issue, i) => (
            <li key={i}>{describeExitConfigIssue(issue)}</li>
          ))}
        </ul>
      )}

      {parsed.hint && (
        <p className="text-xs italic text-red-500 dark:text-red-300">Hint: {parsed.hint}</p>
      )}

      {parsed.warnings && <WarningsList warnings={parsed.warnings} />}
    </div>
  );
}

function WarningsList({ warnings }: { warnings: readonly ConfigWarning[] }) {
  if (warnings.length === 0) return null;
  return (
    <div className="border-t border-red-200 pt-2 text-xs text-amber-700 dark:border-red-900 dark:text-amber-400">
      <p className="font-medium">Warnings</p>
      <ul className="list-inside list-disc space-y-0.5">
        {warnings.map((w, i) => (
          <li key={i}>{describeConfigWarning(w)}</li>
        ))}
      </ul>
    </div>
  );
}

function GenericFallback({ message }: { message: string }) {
  return (
    <div className="rounded-lg border border-red-300 bg-red-50 px-4 py-3 text-sm text-red-600 dark:border-red-800 dark:bg-red-900/20 dark:text-red-400">
      {message}
    </div>
  );
}
