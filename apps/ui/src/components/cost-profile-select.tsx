/**
 * WP2.1b (SY-21-03): per-run cost-profile selector.
 *   backtest / optimize -> any registry id
 *   paper               -> live-equivalent ids only
 *   live                -> no selector (live books the exchange's real fees)
 * The empty value means "platform default" and is OMITTED from the request.
 */

import {
  DEFAULT_COST_PROFILE_ID,
  costProfileOptionsForMode,
  describeCostProfileOption,
  resolveCostProfileForMode,
  type CostSelectorMode,
} from "@/lib/cost-profile";

interface CostProfileSelectProps {
  mode: CostSelectorMode;
  /** Selected registry id; "" = platform default. */
  value: string;
  onChange: (id: string) => void;
  id?: string;
}

export function CostProfileSelect({
  mode,
  value,
  onChange,
  id = "cost-profile-select",
}: CostProfileSelectProps) {
  if (mode === "live") {
    return (
      <p className="text-xs text-slate-500" data-testid="cost-profile-live-note">
        Live runs use the exchange&apos;s actual fees; no cost profile can be chosen.
        The platform default ({DEFAULT_COST_PROFILE_ID}) is recorded as an estimate only.
      </p>
    );
  }

  const options = costProfileOptionsForMode(mode);
  // A value that is not offered in this mode (e.g. a stress id after switching
  // backtest -> paper) falls back to the default instead of being sent.
  const effective = resolveCostProfileForMode(mode, value) ?? "";
  const selected = options.find((o) => o.id === effective);

  return (
    <div className="space-y-2">
      <label
        htmlFor={id}
        className="block text-sm font-medium text-slate-700 dark:text-slate-300"
      >
        Cost profile
      </label>
      <select
        id={id}
        value={effective}
        onChange={(e) => onChange(e.target.value)}
        className="w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm text-slate-900 focus:border-indigo-500 focus:outline-none focus:ring-1 focus:ring-indigo-500 dark:border-slate-700 dark:bg-slate-800 dark:text-slate-200"
      >
        <option value="">Platform default ({DEFAULT_COST_PROFILE_ID})</option>
        {options.map((o) => (
          <option key={o.id} value={o.id}>
            {describeCostProfileOption(o)}
          </option>
        ))}
      </select>
      {mode === "paper" && (
        <p className="text-xs text-slate-500">
          Paper runs only accept live-equivalent profiles.
        </p>
      )}
      {selected && selected.kind !== "live" && (
        <p className="text-xs text-amber-600 dark:text-amber-400" data-testid="cost-profile-hint">
          {selected.kind === "legacy"
            ? "Legacy cost model: reproduces pre-WP2.1 results and is hidden from the leaderboard by default."
            : `${selected.kind === "stress" ? "Stress" : "Diagnostic"} profile: a sensitivity run, hidden from the leaderboard by default.`}
        </p>
      )}
    </div>
  );
}
