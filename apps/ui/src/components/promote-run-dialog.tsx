/**
 * apps/ui/src/components/promote-run-dialog.tsx
 * ------------------------------------------------
 * WP1.3a (CF-13a-1 item 2) confirmation dialog for `POST
 * /api/v1/runs/{id}/promote-to-live` (apps/api/routers/runs.py
 * `promote_to_live`).
 *
 * Unlike `<ResumeRunDialog>`, promotion needs only the live-trading
 * confirmation token — no `X-Admin-Key` (the endpoint has no admin
 * dependency), so `promoteRun` (`@/lib/api`) is called directly rather than
 * through an `/api/admin/*` proxy.
 *
 * Renders the shared `<ExitConfigErrorPanel>` on a 422
 * (`invalid_exit_config` / `exit_manager_required` /
 * `live_pyramiding_forbidden`, synthesis-spec.md §5/§9/SY-13a-18) with a
 * generic fallback for every other error shape (404 source run not found,
 * 400 promotion-gate-not-met, 403 live-gate-failed).
 *
 * The in-dialog trigger button reads "Promote…" — deliberately distinct
 * from both the run page's own "Promote to Live" action button (which
 * remains mounted, unconditionally, behind this modal) and the nested
 * `<LiveConfirmDialog>`'s "Promote" submit button, so all three stay
 * independently addressable by accessible name once the token dialog is
 * open on top of this one.
 */

"use client";

import { useState } from "react";
import { promoteRun } from "@/lib/api";
import type { Run } from "@/lib/types";
import { LiveConfirmDialog } from "@/components/live-confirm-dialog";
import { ExitConfigErrorPanel } from "@/components/exit-config-error-panel";

interface PromoteRunDialogProps {
  sourceRunId: string;
  onClose: () => void;
  onPromoted: (newLiveRun: Run) => void;
}

export function PromoteRunDialog({ sourceRunId, onClose, onPromoted }: PromoteRunDialogProps) {
  const [tokenDialogOpen, setTokenDialogOpen] = useState(false);
  const [loading, setLoading] = useState(false);
  const [errorDetail, setErrorDetail] = useState<unknown>(undefined);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  async function submit(liveConfirmToken: string) {
    setLoading(true);
    setErrorMessage(null);
    setErrorDetail(undefined);
    const result = await promoteRun(sourceRunId, liveConfirmToken);
    setLoading(false);
    if (result.ok) {
      setTokenDialogOpen(false);
      onPromoted(result.data);
      onClose();
    } else {
      setErrorMessage(result.error.message);
      setErrorDetail(result.error.detail);
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
        aria-labelledby="promote-run-dialog-title"
        className="mx-4 w-full max-w-md rounded-xl border border-slate-700 bg-slate-900 p-6 shadow-2xl"
      >
        <h2 id="promote-run-dialog-title" className="text-base font-semibold text-slate-100">
          Promote to Live
        </h2>
        <p className="mt-1 text-sm text-slate-400">
          Creates a new LIVE run with this paper run&apos;s strategy configuration.
          Requires the live-trading confirmation token.
        </p>

        {errorMessage && (
          <div className="mt-3">
            <ExitConfigErrorPanel detail={errorDetail} fallbackMessage={errorMessage} />
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
            Promote…
          </button>
        </div>
      </div>

      <LiveConfirmDialog
        open={tokenDialogOpen}
        title="Confirm promotion to live"
        description="Type the live-trading confirmation token to promote this paper run."
        confirmLabel="Promote"
        loading={loading}
        onCancel={() => setTokenDialogOpen(false)}
        onConfirm={(token) => submit(token)}
      />
    </div>
  );
}
