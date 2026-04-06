/**
 * apps/ui/src/__tests__/lib/csv-export.test.ts
 * -----------------------------------------------
 * Unit tests for the CSV export utility in apps/ui/src/lib/csv-export.ts.
 *
 * Public API:
 *   exportToCsv(filename, columns, data): void  — triggers browser download
 *   CsvColumn<T>: { header: string; value: (row: T) => string|number|boolean|null }
 *
 * escapeCell and buildCsv are private functions, so all tests exercise them
 * indirectly through exportToCsv. The Blob constructor is mocked to capture the
 * raw CSV string before it reaches the browser download machinery.
 */

import { exportToCsv, type CsvColumn } from "@/lib/csv-export";

// ---------------------------------------------------------------------------
// Test helper — captures the raw CSV string produced by exportToCsv
// ---------------------------------------------------------------------------

/**
 * Calls exportToCsv and returns the raw CSV string that was passed to the
 * Blob constructor. DOM download machinery (anchor click, revokeObjectURL) is
 * stubbed so no real browser download occurs.
 */
function captureExportedCsv<T>(
  columns: CsvColumn<T>[],
  data: readonly T[],
): string {
  let capturedCsv = "";

  // Replace global Blob with a lightweight mock that records the first part.
  const OriginalBlob = global.Blob;
  class CaptureBlobMock {
    constructor(parts: BlobPart[]) {
      capturedCsv = parts[0] as string;
    }
    slice() { return new CaptureBlobMock([]); }
    stream(): ReadableStream { return new ReadableStream(); }
    text(): Promise<string> { return Promise.resolve(capturedCsv); }
    arrayBuffer(): Promise<ArrayBuffer> { return Promise.resolve(new ArrayBuffer(0)); }
    get size(): number { return capturedCsv.length; }
    get type(): string { return "text/csv;charset=utf-8;"; }
  }
  global.Blob = CaptureBlobMock as unknown as typeof Blob;

  // Stub URL methods so jsdom doesn't throw on createObjectURL.
  const origCreate = URL.createObjectURL;
  const origRevoke = URL.revokeObjectURL;
  URL.createObjectURL = (_blob: Blob) => "blob:mock";
  URL.revokeObjectURL = (_url: string) => undefined;

  // Stub the anchor element so anchor.click() doesn't throw in jsdom.
  const anchor = {
    href: "",
    download: "",
    style: { display: "" },
    click: jest.fn(),
  };
  const origCreateElement = document.createElement.bind(document);
  jest.spyOn(document, "createElement").mockImplementationOnce((tag: string) => {
    if (tag === "a") return anchor as unknown as HTMLElement;
    return origCreateElement(tag);
  });
  jest.spyOn(document.body, "appendChild").mockImplementationOnce(
    () => anchor as unknown as Node,
  );
  jest.spyOn(document.body, "removeChild").mockImplementationOnce(
    () => anchor as unknown as Node,
  );

  exportToCsv("test.csv", columns, data);

  // Restore everything before returning.
  global.Blob = OriginalBlob;
  URL.createObjectURL = origCreate;
  URL.revokeObjectURL = origRevoke;
  jest.restoreAllMocks();

  return capturedCsv;
}

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

interface Item {
  name: string;
  amount: number;
  note: string | null;
}

const SIMPLE_COLUMNS: CsvColumn<Item>[] = [
  { header: "Name", value: (r) => r.name },
  { header: "Amount", value: (r) => r.amount },
  { header: "Note", value: (r) => r.note },
];

const SIMPLE_DATA: Item[] = [
  { name: "Alice", amount: 100, note: "first" },
  { name: "Bob", amount: 200, note: null },
];

// ---------------------------------------------------------------------------
// Tests: escapeCell behaviour (exercised via exportToCsv)
// ---------------------------------------------------------------------------

describe("csv-export — escapeCell", () => {
  it("produces an empty cell for a null value", () => {
    const cols: CsvColumn<{ v: null }>[] = [
      { header: "V", value: (r) => r.v },
    ];
    const csv = captureExportedCsv(cols, [{ v: null }]);
    // lines: BOM + "V\r\n" (header) + "" (data row with no content)
    const dataRow = csv.slice(1).split("\r\n")[1];
    expect(dataRow).toBe("");
  });

  it("passes through a plain string without wrapping in quotes", () => {
    const cols: CsvColumn<{ v: string }>[] = [
      { header: "V", value: (r) => r.v },
    ];
    const csv = captureExportedCsv(cols, [{ v: "hello" }]);
    const dataRow = csv.slice(1).split("\r\n")[1];
    expect(dataRow).toBe("hello");
  });

  it("wraps a value containing a comma in double quotes", () => {
    const cols: CsvColumn<{ v: string }>[] = [
      { header: "V", value: (r) => r.v },
    ];
    const csv = captureExportedCsv(cols, [{ v: "hello, world" }]);
    const dataRow = csv.slice(1).split("\r\n")[1];
    expect(dataRow).toBe('"hello, world"');
  });

  it("doubles embedded double-quotes per RFC 4180", () => {
    const cols: CsvColumn<{ v: string }>[] = [
      { header: "V", value: (r) => r.v },
    ];
    const csv = captureExportedCsv(cols, [{ v: 'say "hello"' }]);
    const dataRow = csv.slice(1).split("\r\n")[1];
    expect(dataRow).toBe('"say ""hello"""');
  });

  it("wraps a value containing an embedded newline in double quotes", () => {
    const cols: CsvColumn<{ v: string }>[] = [
      { header: "V", value: (r) => r.v },
    ];
    const csv = captureExportedCsv(cols, [{ v: "line1\nline2" }]);
    const dataRow = csv.slice(1).split("\r\n")[1];
    expect(dataRow).toBe('"line1\nline2"');
  });

  it("prefixes = with a single-quote inside double-quotes (formula injection guard)", () => {
    const cols: CsvColumn<{ v: string }>[] = [
      { header: "V", value: (r) => r.v },
    ];
    const csv = captureExportedCsv(cols, [{ v: "=SUM(A1:A10)" }]);
    const dataRow = csv.slice(1).split("\r\n")[1];
    expect(dataRow).toBe('"\'=SUM(A1:A10)"');
  });

  it("prefixes + with a single-quote inside double-quotes (formula injection guard)", () => {
    const cols: CsvColumn<{ v: string }>[] = [
      { header: "V", value: (r) => r.v },
    ];
    const csv = captureExportedCsv(cols, [{ v: "+cmd" }]);
    const dataRow = csv.slice(1).split("\r\n")[1];
    expect(dataRow).toBe('"\'+cmd"');
  });

  it("prefixes - with a single-quote inside double-quotes (formula injection guard)", () => {
    const cols: CsvColumn<{ v: string }>[] = [
      { header: "V", value: (r) => r.v },
    ];
    const csv = captureExportedCsv(cols, [{ v: "-cmd" }]);
    const dataRow = csv.slice(1).split("\r\n")[1];
    expect(dataRow).toBe("\"'-cmd\"");
  });

  it("prefixes @ with a single-quote inside double-quotes (formula injection guard)", () => {
    const cols: CsvColumn<{ v: string }>[] = [
      { header: "V", value: (r) => r.v },
    ];
    const csv = captureExportedCsv(cols, [{ v: "@SUM" }]);
    const dataRow = csv.slice(1).split("\r\n")[1];
    expect(dataRow).toBe('"\'@SUM"');
  });
});

// ---------------------------------------------------------------------------
// Tests: buildCsv structure (exercised via exportToCsv)
// ---------------------------------------------------------------------------

describe("csv-export — buildCsv", () => {
  it("includes a UTF-8 BOM (U+FEFF) as the very first character", () => {
    const csv = captureExportedCsv(SIMPLE_COLUMNS, SIMPLE_DATA);
    expect(csv.charCodeAt(0)).toBe(0xfeff);
  });

  it("generates the correct comma-separated header row from column definitions", () => {
    const csv = captureExportedCsv(SIMPLE_COLUMNS, SIMPLE_DATA);
    const lines = csv.slice(1).split("\r\n");
    expect(lines[0]).toBe("Name,Amount,Note");
  });

  it("generates the correct data row for the first record", () => {
    const csv = captureExportedCsv(SIMPLE_COLUMNS, SIMPLE_DATA);
    const lines = csv.slice(1).split("\r\n");
    expect(lines[1]).toBe("Alice,100,first");
  });

  it("renders an empty trailing cell when the column value is null", () => {
    const csv = captureExportedCsv(SIMPLE_COLUMNS, SIMPLE_DATA);
    const lines = csv.slice(1).split("\r\n");
    // Bob.note is null → empty cell at end of row
    expect(lines[2]).toBe("Bob,200,");
  });

  it("uses CRLF line endings between all rows", () => {
    const csv = captureExportedCsv(SIMPLE_COLUMNS, SIMPLE_DATA);
    const parts = csv.slice(1).split("\r\n");
    // header + 2 data rows = exactly 3 parts
    expect(parts).toHaveLength(3);
  });

  it("outputs only the header row when data is empty", () => {
    const csv = captureExportedCsv(SIMPLE_COLUMNS, []);
    const lines = csv.slice(1).split("\r\n");
    expect(lines).toHaveLength(1);
    expect(lines[0]).toBe("Name,Amount,Note");
  });

  it("serialises boolean true as the string 'true'", () => {
    const cols: CsvColumn<{ flag: boolean }>[] = [
      { header: "Flag", value: (r) => r.flag },
    ];
    const csv = captureExportedCsv(cols, [{ flag: true }]);
    const lines = csv.slice(1).split("\r\n");
    expect(lines[1]).toBe("true");
  });

  it("serialises boolean false as the string 'false'", () => {
    const cols: CsvColumn<{ flag: boolean }>[] = [
      { header: "Flag", value: (r) => r.flag },
    ];
    const csv = captureExportedCsv(cols, [{ flag: false }]);
    const lines = csv.slice(1).split("\r\n");
    expect(lines[1]).toBe("false");
  });
});
