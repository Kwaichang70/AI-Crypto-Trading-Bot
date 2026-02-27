/**
 * Strategies Page — Server Component.
 * Lists all available trading strategies with parameter schema detail.
 */

import type { Metadata } from "next";
import { fetchStrategies } from "@/lib/api";
import type { Strategy } from "@/lib/types";
import { Header } from "@/components/layout/header";

export const metadata: Metadata = { title: "Strategies" };

// ---------------------------------------------------------------------------
// Strategy tag pill
// ---------------------------------------------------------------------------

function TagPill({ tag }: { tag: string }) {
  return (
    <span className="rounded-full bg-slate-800 px-2 py-0.5 text-xs text-slate-400">
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
        <tr className="border-b border-slate-800 text-left text-slate-500">
          <th className="py-1 pr-3 font-medium">Parameter</th>
          <th className="py-1 pr-3 font-medium">Type</th>
          <th className="py-1 pr-3 font-medium">Default</th>
          <th className="py-1 font-medium">Description</th>
        </tr>
      </thead>
      <tbody>
        {entries.map(([name, prop]) => (
          <tr key={name} className="border-b border-slate-800/40">
            <td className="py-1.5 pr-3 font-mono text-slate-300">{name}</td>
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

function StrategyCard({ strategy }: { strategy: Strategy }) {
  return (
    <div className="card space-y-3">
      <div className="flex items-start justify-between gap-2">
        <div>
          <h2 className="text-base font-semibold text-slate-100">
            {strategy.displayName}
          </h2>
          <p className="text-xs text-slate-500">
            {strategy.name} · v{strategy.version}
          </p>
        </div>
      </div>

      <p className="text-sm text-slate-400">{strategy.description}</p>

      {strategy.tags.length > 0 && (
        <div className="flex flex-wrap gap-1">
          {strategy.tags.map((tag) => (
            <TagPill key={tag} tag={tag} />
          ))}
        </div>
      )}

      {/* Parameter schema */}
      <details className="group">
        <summary className="cursor-pointer list-none text-xs font-medium text-slate-500 hover:text-slate-300">
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

// ---------------------------------------------------------------------------
// Page
// ---------------------------------------------------------------------------

export default async function StrategiesPage() {
  const result = await fetchStrategies();

  if (!result.ok) {
    return (
      <div className="space-y-6">
        <Header title="Strategies" subtitle="Available trading strategies" />
        <div className="rounded-lg border border-red-800 bg-red-900/20 px-4 py-3 text-sm text-red-400">
          Could not load strategies: {result.error.message}
        </div>
      </div>
    );
  }

  const { strategies, total } = result.data;

  return (
    <div className="space-y-6">
      <Header
        title="Strategies"
        subtitle={`${total} available strategy${total !== 1 ? "ies" : "y"}`}
      />

      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
        {strategies.map((strategy) => (
          <StrategyCard key={strategy.name} strategy={strategy} />
        ))}
      </div>

      {strategies.length === 0 && (
        <p className="py-12 text-center text-slate-500">
          No strategies available. Ensure the API server is running.
        </p>
      )}
    </div>
  );
}
