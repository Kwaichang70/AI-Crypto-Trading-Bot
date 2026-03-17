/**
 * Reusable CSV export button with download icon.
 * Wraps exportToCsv() from lib/csv-export.
 */
"use client";

import { exportToCsv, type CsvColumn } from "@/lib/csv-export";

interface ExportCsvButtonProps<T> {
  filename: string;
  columns: CsvColumn<T>[];
  data: readonly T[];
  disabled?: boolean;
  /** "sm" for page-level headers, "xs" for inline tab buttons. */
  size?: "xs" | "sm";
  label?: string;
}

export function ExportCsvButton<T>({
  filename,
  columns,
  data,
  disabled,
  size = "xs",
  label = "Export CSV",
}: ExportCsvButtonProps<T>) {
  const isDisabled = disabled || data.length === 0;

  const sizeClasses =
    size === "sm"
      ? "px-3 py-2 text-sm text-slate-300"
      : "px-3 py-1.5 text-xs text-slate-400";

  const iconSize = size === "sm" ? "h-4 w-4" : "h-3.5 w-3.5";

  return (
    <button
      type="button"
      disabled={isDisabled}
      onClick={() => exportToCsv(filename, columns, data)}
      aria-label={`Export ${filename} as CSV`}
      className={`flex items-center gap-1.5 rounded-lg border border-slate-700 bg-slate-800 font-medium transition-colors hover:border-indigo-500 hover:bg-indigo-600/10 hover:text-indigo-300 disabled:cursor-not-allowed disabled:opacity-40 ${sizeClasses}`}
    >
      <svg
        className={iconSize}
        fill="none"
        viewBox="0 0 24 24"
        stroke="currentColor"
        strokeWidth={2}
      >
        <path
          strokeLinecap="round"
          strokeLinejoin="round"
          d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"
        />
      </svg>
      {label}
    </button>
  );
}
