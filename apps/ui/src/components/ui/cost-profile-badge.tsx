/**
 * WP2.1b (SY-21-10): cost-model badge.
 *   `legacy costs`        -> amber
 *   `stress`/`diagnostic` -> slate
 *   anything else         -> the profile id
 * Renders the SERVER flags only (`legacyCosts`, `costProfile`); a missing
 * value reads as legacy.
 */

import { costProfileLabel, type CostCarrier } from "@/lib/cost-profile";

const KIND_CLASSES = {
  legacy: "badge-warning",
  stress: "badge-neutral",
  diagnostic: "badge-neutral",
  profile:
    "inline-flex items-center rounded-full bg-indigo-50 px-2.5 py-0.5 font-mono text-xs font-medium text-indigo-700 dark:bg-indigo-900/30 dark:text-indigo-300",
} as const;

interface CostProfileBadgeProps {
  value: CostCarrier;
  /** Append the profile id after a `stress` / `diagnostic` label (which hides it). */
  showId?: boolean;
}

export function CostProfileBadge({ value, showId = false }: CostProfileBadgeProps) {
  const label = costProfileLabel(value);
  const id = value.costProfile?.id;
  const idHidden = label.kind === "stress" || label.kind === "diagnostic";
  return (
    <span
      className={KIND_CLASSES[label.kind]}
      title={label.tooltip}
      data-testid="cost-profile-badge"
      data-kind={label.kind}
    >
      {label.text}
      {showId && idHidden && id ? <span className="ml-1 font-mono">{id}</span> : null}
    </span>
  );
}
