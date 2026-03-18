"use client";

import { useState } from "react";

export interface Tab {
  id: string;
  label: string;
}

interface TabsProps {
  tabs: Tab[];
  children: (activeId: string) => React.ReactNode;
  defaultTab?: string;
}

export function Tabs({ tabs, children, defaultTab }: TabsProps) {
  const [activeId, setActiveId] = useState(defaultTab ?? tabs[0]?.id ?? "");

  return (
    <div>
      <div
        className="flex border-b border-slate-200 dark:border-slate-800"
        role="tablist"
        aria-label="Section tabs"
      >
        {tabs.map((tab) => (
          <button
            key={tab.id}
            role="tab"
            aria-selected={activeId === tab.id}
            aria-controls={`tabpanel-${tab.id}`}
            onClick={() => setActiveId(tab.id)}
            className={[
              "px-4 py-2.5 text-sm font-medium transition-colors border-b-2 -mb-px",
              activeId === tab.id
                ? "border-indigo-500 text-indigo-600 dark:text-indigo-400"
                : "border-transparent text-slate-500 hover:text-slate-700 dark:text-slate-400 dark:hover:text-slate-200",
            ].join(" ")}
          >
            {tab.label}
          </button>
        ))}
      </div>

      <div
        role="tabpanel"
        id={`tabpanel-${activeId}`}
        className="pt-4"
      >
        {children(activeId)}
      </div>
    </div>
  );
}
