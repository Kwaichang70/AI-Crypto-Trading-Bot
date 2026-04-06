/**
 * apps/ui/src/__tests__/components/data-table.test.tsx
 * -------------------------------------------------------
 * Unit tests for the DataTable<T> component.
 *
 * Tests cover: header rendering, row rendering, empty state, loading skeleton,
 * sort-on-click (asc → desc → new column resets), ARIA attributes, keyboard
 * navigation, custom className forwarding, and the sortValue accessor path.
 */

import React from "react";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { DataTable, type Column } from "@/components/ui/data-table";

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

interface Row {
  id: string;
  name: string;
  score: number;
}

const ROWS: Row[] = [
  { id: "a", name: "Alice", score: 30 },
  { id: "b", name: "Bob", score: 10 },
  { id: "c", name: "Carol", score: 20 },
];

const BASE_COLUMNS: Column<Row>[] = [
  {
    key: "name",
    header: "Name",
    render: (row) => row.name,
    sortable: true,
    sortValue: (row) => row.name,
  },
  {
    key: "score",
    header: "Score",
    render: (row) => String(row.score),
    sortable: true,
    sortValue: (row) => row.score,
  },
  {
    key: "static",
    header: "Static",
    render: (row) => row.id,
  },
];

function keyExtractor(row: Row) {
  return row.id;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe("DataTable", () => {
  // ---- Header rendering ----------------------------------------------------

  it("renders column headers from the columns prop", () => {
    render(
      <DataTable columns={BASE_COLUMNS} data={ROWS} keyExtractor={keyExtractor} />,
    );
    expect(screen.getByText("Name")).toBeInTheDocument();
    expect(screen.getByText("Score")).toBeInTheDocument();
    expect(screen.getByText("Static")).toBeInTheDocument();
  });

  // ---- Row rendering -------------------------------------------------------

  it("renders a row for each data entry", () => {
    render(
      <DataTable columns={BASE_COLUMNS} data={ROWS} keyExtractor={keyExtractor} />,
    );
    expect(screen.getByText("Alice")).toBeInTheDocument();
    expect(screen.getByText("Bob")).toBeInTheDocument();
    expect(screen.getByText("Carol")).toBeInTheDocument();
  });

  it("renders cell content via the render accessor", () => {
    render(
      <DataTable columns={BASE_COLUMNS} data={ROWS} keyExtractor={keyExtractor} />,
    );
    // Score column renders the number as a string
    expect(screen.getByText("10")).toBeInTheDocument();
    expect(screen.getByText("30")).toBeInTheDocument();
  });

  // ---- Empty state ---------------------------------------------------------

  it("shows the default empty message when data is empty", () => {
    render(
      <DataTable columns={BASE_COLUMNS} data={[]} keyExtractor={keyExtractor} />,
    );
    expect(screen.getByText("No data available.")).toBeInTheDocument();
  });

  it("shows a custom emptyMessage when provided and data is empty", () => {
    render(
      <DataTable
        columns={BASE_COLUMNS}
        data={[]}
        keyExtractor={keyExtractor}
        emptyMessage="Nothing here yet."
      />,
    );
    expect(screen.getByText("Nothing here yet.")).toBeInTheDocument();
  });

  // ---- Loading skeleton ----------------------------------------------------

  it("renders loading skeleton divs instead of the table when isLoading=true", () => {
    const { container } = render(
      <DataTable
        columns={BASE_COLUMNS}
        data={ROWS}
        keyExtractor={keyExtractor}
        isLoading
      />,
    );
    // Loading skeleton — no table element, just pulse divs
    expect(container.querySelector("table")).toBeNull();
    const skeletons = container.querySelectorAll(".animate-pulse");
    expect(skeletons.length).toBeGreaterThan(0);
  });

  it("does not render data rows while isLoading=true", () => {
    render(
      <DataTable
        columns={BASE_COLUMNS}
        data={ROWS}
        keyExtractor={keyExtractor}
        isLoading
      />,
    );
    expect(screen.queryByText("Alice")).toBeNull();
  });

  // ---- Sort: ascending on first click --------------------------------------

  it("sorts column ascending on first click", async () => {
    const user = userEvent.setup();
    render(
      <DataTable columns={BASE_COLUMNS} data={ROWS} keyExtractor={keyExtractor} />,
    );

    await user.click(screen.getByRole("button", { name: /score/i }));

    // After ascending sort by score: Bob(10), Carol(20), Alice(30)
    const cells = screen.getAllByRole("cell");
    const scoreTextCells = cells.filter((c) =>
      ["10", "20", "30"].includes(c.textContent ?? ""),
    );
    expect(scoreTextCells[0]).toHaveTextContent("10");
    expect(scoreTextCells[1]).toHaveTextContent("20");
    expect(scoreTextCells[2]).toHaveTextContent("30");
  });

  // ---- Sort: descending on second click ------------------------------------

  it("reverses sort direction on second click of the same column", async () => {
    const user = userEvent.setup();
    render(
      <DataTable columns={BASE_COLUMNS} data={ROWS} keyExtractor={keyExtractor} />,
    );

    const scoreHeader = screen.getByRole("button", { name: /score/i });
    await user.click(scoreHeader); // asc
    await user.click(scoreHeader); // desc

    // Descending: Alice(30), Carol(20), Bob(10)
    const cells = screen.getAllByRole("cell");
    const scoreTextCells = cells.filter((c) =>
      ["10", "20", "30"].includes(c.textContent ?? ""),
    );
    expect(scoreTextCells[0]).toHaveTextContent("30");
    expect(scoreTextCells[1]).toHaveTextContent("20");
    expect(scoreTextCells[2]).toHaveTextContent("10");
  });

  // ---- Sort: resets direction when clicking a different column -------------

  it("resets sort direction to ascending when clicking a different column", async () => {
    const user = userEvent.setup();
    render(
      <DataTable columns={BASE_COLUMNS} data={ROWS} keyExtractor={keyExtractor} />,
    );

    const scoreHeader = screen.getByRole("button", { name: /score/i });
    await user.click(scoreHeader); // score asc
    await user.click(scoreHeader); // score desc

    // Now click Name — should start ascending
    await user.click(screen.getByRole("button", { name: /name/i }));

    // Ascending by name: Alice, Bob, Carol
    const rows = screen.getAllByRole("row").slice(1); // skip header
    expect(rows[0]).toHaveTextContent("Alice");
    expect(rows[1]).toHaveTextContent("Bob");
    expect(rows[2]).toHaveTextContent("Carol");
  });

  // ---- ARIA: sortable columns expose role="button" and tabIndex -----------

  it("gives sortable column headers role=button and tabIndex=0", () => {
    render(
      <DataTable columns={BASE_COLUMNS} data={ROWS} keyExtractor={keyExtractor} />,
    );
    const nameHeader = screen.getByRole("button", { name: /name/i });
    expect(nameHeader).toHaveAttribute("tabindex", "0");
  });

  // ---- ARIA: non-sortable columns have no role attribute ------------------

  it("does not give non-sortable headers a role attribute", () => {
    render(
      <DataTable columns={BASE_COLUMNS} data={ROWS} keyExtractor={keyExtractor} />,
    );
    // "Static" header should not be a button
    const allButtons = screen.queryAllByRole("button");
    const staticButton = allButtons.find((b) =>
      b.textContent?.includes("Static"),
    );
    expect(staticButton).toBeUndefined();
  });

  // ---- ARIA: aria-sort reflects current sort state -------------------------

  it("sets aria-sort=none on sortable columns that are not active", () => {
    render(
      <DataTable columns={BASE_COLUMNS} data={ROWS} keyExtractor={keyExtractor} />,
    );
    const nameHeader = screen.getByRole("button", { name: /name/i });
    expect(nameHeader).toHaveAttribute("aria-sort", "none");
  });

  it("sets aria-sort=ascending after first click", async () => {
    const user = userEvent.setup();
    render(
      <DataTable columns={BASE_COLUMNS} data={ROWS} keyExtractor={keyExtractor} />,
    );
    const nameHeader = screen.getByRole("button", { name: /name/i });
    await user.click(nameHeader);
    expect(nameHeader).toHaveAttribute("aria-sort", "ascending");
  });

  it("sets aria-sort=descending after second click", async () => {
    const user = userEvent.setup();
    render(
      <DataTable columns={BASE_COLUMNS} data={ROWS} keyExtractor={keyExtractor} />,
    );
    const nameHeader = screen.getByRole("button", { name: /name/i });
    await user.click(nameHeader);
    await user.click(nameHeader);
    expect(nameHeader).toHaveAttribute("aria-sort", "descending");
  });

  // ---- Keyboard: Enter triggers sort --------------------------------------

  it("triggers sort on Enter keydown on a sortable header", async () => {
    const user = userEvent.setup();
    render(
      <DataTable columns={BASE_COLUMNS} data={ROWS} keyExtractor={keyExtractor} />,
    );
    const nameHeader = screen.getByRole("button", { name: /name/i });
    nameHeader.focus();
    await user.keyboard("{Enter}");
    expect(nameHeader).toHaveAttribute("aria-sort", "ascending");
  });

  // ---- Keyboard: Space triggers sort --------------------------------------

  it("triggers sort on Space keydown on a sortable header", async () => {
    const user = userEvent.setup();
    render(
      <DataTable columns={BASE_COLUMNS} data={ROWS} keyExtractor={keyExtractor} />,
    );
    const nameHeader = screen.getByRole("button", { name: /name/i });
    nameHeader.focus();
    await user.keyboard(" ");
    expect(nameHeader).toHaveAttribute("aria-sort", "ascending");
  });

  // ---- Custom className applied to column cells ---------------------------

  it("applies custom className to both header and data cells", () => {
    const colsWithClass: Column<Row>[] = [
      {
        key: "name",
        header: "Name",
        render: (row) => row.name,
        className: "text-right",
      },
    ];
    const { container } = render(
      <DataTable
        columns={colsWithClass}
        data={[{ id: "x", name: "Xavier", score: 5 }]}
        keyExtractor={keyExtractor}
      />,
    );
    const rightaligned = container.querySelectorAll(".text-right");
    // th + td should both carry the class
    expect(rightaligned.length).toBeGreaterThanOrEqual(2);
  });

  // ---- sortValue used for ordering ----------------------------------------

  it("uses sortValue accessor rather than render output for ordering", async () => {
    const user = userEvent.setup();
    // Render shows an emoji prefix, but sortValue returns raw number
    const numericCol: Column<Row>[] = [
      {
        key: "score",
        header: "Score",
        render: (row) => `★ ${row.score}`,
        sortable: true,
        sortValue: (row) => row.score,
      },
    ];
    render(
      <DataTable columns={numericCol} data={ROWS} keyExtractor={keyExtractor} />,
    );
    await user.click(screen.getByRole("button", { name: /score/i }));

    const rows = screen.getAllByRole("row").slice(1);
    // Ascending numeric: Bob(10), Carol(20), Alice(30)
    expect(rows[0]).toHaveTextContent("★ 10");
    expect(rows[1]).toHaveTextContent("★ 20");
    expect(rows[2]).toHaveTextContent("★ 30");
  });
});
