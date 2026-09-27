/**
 * apps/ui/src/components/resume-run-dialog.tsx
 * ------------------------------------------------
 * WP1.7b (S13) resume dialog for an 'orphaned' live run.
 *
 * `POST /api/v1/runs/{id}/resume` (apps/api/routers/runs.py) requires
 * X-Admin-Key PLUS the full 3-layer live-trading safety gate (env flag +
 * API keys + X-Live-Confirm-Token header) for BOTH mode=normal AND
 * mode=protective — `LiveTradingGate.check_gate` is called unconditionally
 * before either mode proceeds (runs.py:2463-2474). This intentionally
 * departs from reports/vp2-wp1.7/ui-design.md's original suggestion that
 * protective resumes skip the typed confirmation — that draft predates the
 * merged 1.7a backend and is superseded by it (flagged as a deviation in
 * this WP's producer report).
 *
 * Uses the admin proxy `/api/admin/runs/[id]/resume` (server-side
 * X-Admin-Key injection) and forwards the typed token unchanged as
 * `X-Live-Confirm-Token`.
 */

"use client";

import { useState } from "react";
import { adminFetch, RESUME_TIMEOUT_MS } from "@/lib/admin-fetch";
import type { Run } from "@/lib/types";
import { LiveConfirmDialog } from "@/components/live-confirm-dialog";

interface ResumeRunDialogProps {
  runId: string;
  onClose: () => void;
  onResumed: (run: Run) => void;
}

export function ResumeRunDialog({ runId, onClose, onResumed }: ResumeRunDialogProps) {
  const [mode, setMode] = useState<"normal" | "protective">("protective");
  const [tokenDialogOpen, setTokenDialogOpen] = useState(false);
  const [loading, setLoading] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  async function submit(liveConfirmToken: string) {
    setLoading(true);
    setErrorMessage(null);
    const result = await adminFetch<Run>(
      `/api/admin/runs/${runId}/resume?mode=${mode}`,
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "X-Live-Confirm-Token": liveConfirmToken,
        },
        body: JSON.stringify({}),
      },
      RESUME_TIMEOUT_MS,
    );
    setLoading(false);
    if (result.ok) {
      setTokenDialogOpen(false);
      onResumed(result.data);
      onClose();
    } else {
      setErrorMessage(result.error.message);
    }
  }

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm"
      onClick={(e) => {
        if (e.target === e.currentTarget && !loading) onClose();
      }}
    >
      <div
        role="dialog"
        aria-modal="true"
        aria-labelledby="resume-run-dialog-title"
        className="mx-4 w-full max-w-md rounded-xl border border-slate-700 bg-slate-900 p-6 shadow-2xl"
      >
        <h2 id="resume-run-dialog-title" className="text-base font-semibold text-slate-100">
          Resume Run {runId.slice(0, 8)}…
        </h2>
        <p className="mt-1 text-sm text-slate-400">
          Resumes this orphaned live run under its existing ID. Both modes
          require the live-trading confirmation token.
        </p>

        <fieldset className="mt-4 space-y-2">
          <label className="flex items-start gap-2 text-sm text-slate-300">
            <input
              type="radio"
              name="resume-mode"
              checked={mode === "protective"}
              onChange={() => setMode("protective")}
              className="mt-1"
            />
            <span>
              <span className="font-medium">Protective</span> — exempt from
              the global/per-run entries latch; always starts latched
              in-process either way (recommended default).
            </span>
          </label>
          <label className="flex items-start gap-2 text-sm text-slate-300">
            <input
              type="radio"
              name="resume-mode"
              checked={mode === "normal"}
              onChange={() => setMode("normal")}
              className="mt-1"
            />
            <span>
              <span className="font-medium">Normal</span> — rejected with 409
              if the global kill switch or this run&apos;s own entries latch
              is active.
            </span>
          </label>
        </fieldset>

        {errorMessage && (
          <div className="mt-3 rounded-lg border border-red-300 bg-red-50 px-3 py-2 text-xs text-red-600 dark:border-red-800 dark:bg-red-900/20 dark:text-red-400">
            {errorMessage}
          </div>
        )}

        <div className="mt-4 flex gap-3">
          <button
            type="button"
            onClick={onClose}
            disabled={loading}
            className="flex-1 rounded-lg border border-slate-600 bg-slate-800 px-4 py-2 text-sm font-medium text-slate-300 hover:bg-slate-700 disabled:opacity-50"
          >
            Cancel
          </button>
          <button
            type="button"
            onClick={() => setTokenDialogOpen(true)}
            disabled={loading}
            className="flex-1 rounded-lg bg-indigo-600 px-4 py-2 text-sm font-medium text-white hover:bg-indigo-500 disabled:opacity-50"
          >
            Resume ({mode})
          </button>
        </div>
      </div>

      <LiveConfirmDialog
        open={tokenDialogOpen}
        title={`Confirm resume (${mode})`}
        description="Type the live-trading confirmation token to resume this run."
        confirmLabel="Resume"
        loading={loading}
        onCancel={() => setTokenDialogOpen(false)}
        onConfirm={(token) => submit(token)}
      />
    </div>
  );
}
