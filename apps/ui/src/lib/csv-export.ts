/**
 * apps/ui/src/lib/csv-export.ts
 * --------------------------------
 * Shared CSV export utility.
 *
 * Usage:
 *   exportToCsv("runs.csv", COLUMNS, runs);
 *
 * Features:
 *   - RFC 4180-compliant value escaping (quotes, commas, tabs, embedded newlines)
 *   - UTF-8 BOM prefix for Excel compatibility
 *   - Browser download via Blob + createObjectURL
 */

export interface CsvColumn<T> {
  header: string;
  value: (row: T) => string | number | boolean | null;
}

/**
 * Escape a single CSV cell value per RFC 4180:
 *   - Wrap in double quotes if the value contains a comma, double quote,
 *     tab, carriage return, or newline.
 *   - Escape embedded double quotes by doubling them ("").
 *   - Null/undefined values become the empty string.
 */
function escapeCell(raw: string | number | boolean | null | undefined): string {
  if (raw === null || raw === undefined) return "";
  const str = String(raw);

  // Defense against CSV formula injection: cells starting with =, +, -, or @
  // are treated as formulas by Excel/Google Sheets. Prefix with a single quote
  // to force text interpretation.
  const first = str.charAt(0);
  if (first === "=" || first === "+" || first === "-" || first === "@") {
    return `"'${str.replace(/"/g, '""')}"`;
  }

  if (
    str.includes(",") ||
    str.includes('"') ||
    str.includes("\n") ||
    str.includes("\r") ||
    str.includes("\t")
  ) {
    return `"${str.replace(/"/g, '""')}"`;
  }
  return str;
}

/**
 * Build a CSV string from a column spec and data array.
 * Includes a UTF-8 BOM (\uFEFF) so Excel opens the file with correct encoding.
 */
function buildCsv<T>(columns: CsvColumn<T>[], data: readonly T[]): string {
  const BOM = "\uFEFF";
  const header = columns.map((c) => escapeCell(c.header)).join(",");
  const rows = data.map((row) =>
    columns.map((c) => escapeCell(c.value(row))).join(","),
  );
  return BOM + [header, ...rows].join("\r\n");
}

/**
 * Generate a CSV from `data` using the provided `columns` spec and trigger a
 * browser file download for `filename`.
 *
 * @param filename  The suggested filename (e.g. "runs.csv").
 * @param columns   Column definitions describing header text and value accessor.
 * @param data      The rows to serialise.
 */
export function exportToCsv<T>(
  filename: string,
  columns: CsvColumn<T>[],
  data: readonly T[],
): void {
  if (typeof document === "undefined") return;

  const csv = buildCsv(columns, data);
  const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
  const url = URL.createObjectURL(blob);

  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  anchor.style.display = "none";
  document.body.appendChild(anchor);
  anchor.click();
  document.body.removeChild(anchor);

  // Delay revocation so Safari has time to initiate the download (CR-004).
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
