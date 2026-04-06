/**
 * apps/ui/src/__tests__/components/tabs.test.tsx
 * -------------------------------------------------
 * Unit tests for the Tabs component.
 *
 * TabsProps:
 *   tabs: Tab[]            — array of { id: string; label: string }
 *   children: (activeId: string) => React.ReactNode
 *   defaultTab?: string    — initial active tab id (defaults to tabs[0].id)
 *
 * The component renders:
 *   - A tablist with one role="tab" button per Tab entry
 *   - aria-selected=true on the active tab, false on others
 *   - A role="tabpanel" div whose content is produced by calling children(activeId)
 *   - Active tab carries the indigo border/text classes; inactive tabs carry
 *     transparent border classes
 */

import React from "react";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { Tabs, type Tab } from "@/components/ui/tabs";

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

const TABS: Tab[] = [
  { id: "overview", label: "Overview" },
  { id: "trades", label: "Trades" },
  { id: "fills", label: "Fills" },
];

function renderTabs(overrides: Partial<React.ComponentProps<typeof Tabs>> = {}) {
  return render(
    <Tabs tabs={TABS} {...overrides}>
      {(activeId) => <div data-testid="panel-content">{activeId}</div>}
    </Tabs>,
  );
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe("Tabs", () => {
  // ---- Renders all tab labels ---------------------------------------------

  it("renders a tab button for every entry in the tabs array", () => {
    renderTabs();
    expect(screen.getByRole("tab", { name: "Overview" })).toBeInTheDocument();
    expect(screen.getByRole("tab", { name: "Trades" })).toBeInTheDocument();
    expect(screen.getByRole("tab", { name: "Fills" })).toBeInTheDocument();
  });

  it("renders the tablist container with correct aria-label", () => {
    renderTabs();
    expect(
      screen.getByRole("tablist", { name: "Section tabs" }),
    ).toBeInTheDocument();
  });

  // ---- Default active tab -------------------------------------------------

  it("activates the first tab by default when defaultTab is omitted", () => {
    renderTabs();
    const overviewTab = screen.getByRole("tab", { name: "Overview" });
    expect(overviewTab).toHaveAttribute("aria-selected", "true");
  });

  it("activates the specified defaultTab when provided", () => {
    renderTabs({ defaultTab: "trades" });
    expect(screen.getByRole("tab", { name: "Trades" })).toHaveAttribute(
      "aria-selected",
      "true",
    );
    expect(screen.getByRole("tab", { name: "Overview" })).toHaveAttribute(
      "aria-selected",
      "false",
    );
  });

  // ---- Active tab styling -------------------------------------------------

  it("applies the active border/text class to the active tab", () => {
    renderTabs();
    const activeTab = screen.getByRole("tab", { name: "Overview" });
    // Active tab has border-indigo-500 class (from the component's conditional classes)
    expect(activeTab.className).toContain("border-indigo-500");
  });

  it("applies the inactive border-transparent class to non-active tabs", () => {
    renderTabs();
    const inactiveTab = screen.getByRole("tab", { name: "Trades" });
    expect(inactiveTab.className).toContain("border-transparent");
  });

  // ---- Clicking a tab calls onChange (internal state update) --------------

  it("sets the clicked tab as active on click", async () => {
    const user = userEvent.setup();
    renderTabs();

    await user.click(screen.getByRole("tab", { name: "Trades" }));

    expect(screen.getByRole("tab", { name: "Trades" })).toHaveAttribute(
      "aria-selected",
      "true",
    );
    expect(screen.getByRole("tab", { name: "Overview" })).toHaveAttribute(
      "aria-selected",
      "false",
    );
  });

  // ---- Tab panel content renders correct activeId ------------------------

  it("renders the first tab's id as panel content initially", () => {
    renderTabs();
    expect(screen.getByTestId("panel-content")).toHaveTextContent("overview");
  });

  it("updates panel content to the newly active tab id after click", async () => {
    const user = userEvent.setup();
    renderTabs();

    await user.click(screen.getByRole("tab", { name: "Fills" }));

    expect(screen.getByTestId("panel-content")).toHaveTextContent("fills");
  });

  // ---- Tabpanel role and aria-controls ------------------------------------

  it("renders a tabpanel element", () => {
    renderTabs();
    expect(screen.getByRole("tabpanel")).toBeInTheDocument();
  });

  it("tabpanel id matches the active tab's aria-controls value", () => {
    renderTabs();
    const activeTab = screen.getByRole("tab", { name: "Overview" });
    const panelId = activeTab.getAttribute("aria-controls");
    const panel = screen.getByRole("tabpanel");
    expect(panel).toHaveAttribute("id", panelId ?? "");
  });
});
