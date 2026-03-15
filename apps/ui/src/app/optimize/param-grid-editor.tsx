"use client";

import type { JsonSchemaProperty } from "@/lib/types";

export interface ParamGridRow {
  id: string;
  paramName: string;
  valuesRaw: string; // comma-separated, e.g. "5, 10, 20"
}

interface ParamGridEditorProps {
  /** Available parameter names from the selected strategy's schema. */
  paramNames: string[];
  /** Map of paramName -> type from the schema properties. */
  paramTypes: Record<string, JsonSchemaProperty>;
  rows: ParamGridRow[];
  onChange: (rows: ParamGridRow[]) => void;
}

export function ParamGridEditor({
  paramNames,
  paramTypes,
  rows,
  onChange,
}: ParamGridEditorProps) {
  function addRow() {
    const used = new Set(rows.map((r) => r.paramName));
    const next = paramNames.find((n) => !used.has(n)) ?? paramNames[0] ?? "";
    onChange([...rows, { id: crypto.randomUUID(), paramName: next, valuesRaw: "" }]);
  }

  function removeRow(idx: number) {
    onChange(rows.filter((_, i) => i !== idx));
  }

  function updateRow(idx: number, patch: Partial<ParamGridRow>) {
    onChange(rows.map((r, i) => (i === idx ? { ...r, ...patch } : r)));
  }

  /** Estimated combination count for pre-flight warning. */
  const estimatedCombinations = rows.length === 0
    ? 0
    : rows.reduce((acc, r) => {
        const count = r.valuesRaw
          .split(",")
          .map((v) => v.trim())
          .filter(Boolean).length;
        return acc * Math.max(1, count);
      }, 1);

  return (
    <div className="space-y-3">
      {rows.length === 0 && (
        <p className="text-sm text-slate-500">
          No parameters added yet. Click &ldquo;Add Parameter&rdquo; to start.
        </p>
      )}

      {rows.map((row, idx) => {
        const schema = paramTypes[row.paramName];
        const typeHint = schema
          ? schema.type === "integer"
            ? "integers"
            : schema.type === "number"
              ? "decimals"
              : "values"
          : "values";

        return (
          <div key={row.id} className="flex items-start gap-2">
            {/* Parameter name selector */}
            <select
              value={row.paramName}
              onChange={(e) => updateRow(idx, { paramName: e.target.value })}
              className="w-44 shrink-0 rounded-lg border border-slate-700 bg-slate-800 px-3 py-2 text-sm text-slate-200 focus:border-indigo-500 focus:outline-none focus:ring-1 focus:ring-indigo-500"
            >
              {paramNames.map((name) => (
                <option key={name} value={name}>
                  {name}
                </option>
              ))}
            </select>

            {/* Values input */}
            <div className="flex-1">
              <input
                type="text"
                value={row.valuesRaw}
                placeholder={`Comma-separated ${typeHint}, e.g. 5, 10, 20`}
                onChange={(e) => updateRow(idx, { valuesRaw: e.target.value })}
                className="w-full rounded-lg border border-slate-700 bg-slate-800 px-3 py-2 text-sm text-slate-200 placeholder-slate-500 focus:border-indigo-500 focus:outline-none focus:ring-1 focus:ring-indigo-500"
              />
            </div>

            {/* Remove button */}
            <button
              type="button"
              onClick={() => removeRow(idx)}
              className="mt-0.5 rounded p-1.5 text-slate-500 hover:bg-slate-700 hover:text-red-400"
              title="Remove parameter"
            >
              <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>
        );
      })}

      <div className="flex items-center justify-between">
        <button
          type="button"
          onClick={addRow}
          disabled={paramNames.length === 0}
          className="flex items-center gap-1.5 rounded-lg border border-slate-700 px-3 py-1.5 text-sm text-slate-400 hover:border-indigo-500 hover:text-indigo-400 disabled:cursor-not-allowed disabled:opacity-40"
        >
          <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M12 4v16m8-8H4" />
          </svg>
          Add Parameter
        </button>

        {rows.length > 0 && (
          <span className={`text-xs ${estimatedCombinations > 500 ? "text-amber-400" : "text-slate-500"}`}>
            ~{estimatedCombinations.toLocaleString()} combination{estimatedCombinations !== 1 ? "s" : ""}
            {estimatedCombinations > 500 && " — consider reducing the grid"}
          </span>
        )}
      </div>
    </div>
  );
}
