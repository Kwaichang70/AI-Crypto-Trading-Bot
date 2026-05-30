/**
 * StrategyCard — presentational component for a single trading strategy.
 *
 * Extracted from `app/strategies/page.tsx` (Sprint 51 Cycle 2): the Next.js App
 * Router only permits framework-reserved exports (`default`, `metadata`,
 * `generateStaticParams`, …) from a `page.tsx`. The Cycle 2 jest test imports
 * `StrategyCard` directly, so it must live in a component file that may export
 * named symbols freely.
 *
 * Self-contained: the `TagPill` and `ParameterSchemaTable` helpers were used
 * exclusively by `StrategyCard`, so they moved here too rather than being
 * imported back from the page (which would create a page → component → page
 * dependency).
 */

import type { Strategy } from "@/lib/types";
import { StrategyStatusBadge } from "@/components/ui/strategy-status-badge";

// ---------------------------------------------------------------------------
// Strategy tag pill
// ---------------------------------------------------------------------------

function TagPill({ tag }: { tag: string }) {
  return (
    <span className="rounded-full bg-slate-100 px-2 py-0.5 text-xs text-slate-600 dark:bg-slate-800 dark:text-slate-400">
      {tag}
    </span>
  );
}

// ---------------------------------------------------------------------------
// Schema property table
// ---------------------------------------------------------------------------

function ParameterSchemaTable({ strategy }: { strategy: Strategy }) {
  const entries = Object.entries(strategy.parameterSchema.properties);
  if (entries.length === 0) return <p className="text-xs text-slate-500">No configurable parameters.</p>;

  return (
    <table className="mt-2 w-full text-xs">
      <thead>
        <tr className="border-b border-slate-200 text-left text-slate-500 dark:border-slate-800">
          <th className="py-1 pr-3 font-medium">Parameter</th>
          <th className="py-1 pr-3 font-medium">Type</th>
          <th className="py-1 pr-3 font-medium">Default</th>
          <th className="py-1 font-medium">Description</th>
        </tr>
      </thead>
      <tbody>
        {entries.map(([name, prop]) => (
          <tr key={name} className="border-b border-slate-100 dark:border-slate-800/40">
            <td className="py-1.5 pr-3 font-mono text-slate-700 dark:text-slate-300">{name}</td>
            <td className="py-1.5 pr-3 text-slate-500">{prop.type}</td>
            <td className="py-1.5 pr-3 font-mono text-slate-400">
              {prop.default !== undefined ? String(prop.default) : "—"}
            </td>
            <td className="py-1.5 text-slate-500">{prop.description ?? "—"}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

// ---------------------------------------------------------------------------
// Strategy card
// ---------------------------------------------------------------------------

export function StrategyCard({ strategy }: { strategy: Strategy }) {
  return (
    <div className="card space-y-3">
      <div className="flex items-start justify-between gap-2">
        <div>
          <h2 className="text-base font-semibold text-slate-900 dark:text-slate-100">
            {strategy.displayName}
          </h2>
          <p className="text-xs text-slate-500">
            {strategy.name} · v{strategy.version}
          </p>
        </div>
        <StrategyStatusBadge status={strategy.status} />
      </div>

      <p className="text-sm text-slate-500 dark:text-slate-400">{strategy.description}</p>

      {strategy.status === "demoted" && strategy.demotionReason && (
        <p className="text-xs text-amber-700 dark:text-amber-400">{strategy.demotionReason}</p>
      )}

      {strategy.status === "demoted" &&
        strategy.promotionRequirements &&
        strategy.promotionRequirements.length > 0 && (
          <div className="space-y-1">
            <p className="text-xs font-medium text-slate-500">To re-promote:</p>
            <div className="flex flex-wrap gap-1">
              {strategy.promotionRequirements.map((req) => (
                <TagPill key={req} tag={req} />
              ))}
            </div>
          </div>
        )}

      {strategy.tags.length > 0 && (
        <div className="flex flex-wrap gap-1">
          {strategy.tags.map((tag) => (
            <TagPill key={tag} tag={tag} />
          ))}
        </div>
      )}

      {/* Parameter schema */}
      <details className="group">
        <summary className="cursor-pointer list-none text-xs font-medium text-slate-500 hover:text-slate-700 dark:hover:text-slate-300">
          <span className="group-open:hidden">Show parameters</span>
          <span className="hidden group-open:inline">Hide parameters</span>
        </summary>
        <div className="mt-2 overflow-x-auto">
          <ParameterSchemaTable strategy={strategy} />
        </div>
      </details>
    </div>
  );
}
