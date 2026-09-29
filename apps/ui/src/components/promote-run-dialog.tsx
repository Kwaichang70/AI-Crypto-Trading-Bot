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
import {
  classifyIdempotencyError,
  getStructuredErrorCode,
  IDEMPOTENCY_CLIENT_ERROR_MESSAGE,
  IDEMPOTENCY_IN_PROGRESS_MESSAGE,
  IDEMPOTENCY_REUSED_MESSAGE,
  useIdempotencyKey,
  useSubmitLock,
} from "@/lib/idempotency";

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
  // WP7.0 (SY-70-21/C-13): a 409 idempotency_in_progress renders a neutral
  // notice, distinct from the generic ExitConfigErrorPanel fallback below.
  const [inProgressNotice, setInProgressNotice] = useState(false);
  // WP7.0 (SY-70-19): snapshot = sourceRunId — the same key is reused across
  // every retry of promoting THIS source run, including a live-timeout retry
  // that re-opens the token dialog (G-13: the token itself is never kept).
  const idempotency = useIdempotencyKey();
  const submitLock = useSubmitLock();

  async function submit(liveConfirmToken: string) {
    if (!submitLock.tryAcquire()) return;
    setLoading(true);
    setErrorMessage(null);
    setErrorDetail(undefined);
    setInProgressNotice(false);

    try {
      const idempotencyKey = idempotency.keyFor(sourceRunId);
      const result = await promoteRun(sourceRunId, { idempotencyKey, liveConfirmToken });

      if (result.ok) {
        idempotency.reset();
        setTokenDialogOpen(false);
        onPromoted(result.data);
        onClose();
        return;
      }

      // WP70-S-03 (round 2): close the token dialog on EVERY settled non-OK
      // branch, not just success/in-progress. `<LiveConfirmDialog>` only
      // clears its typed token on an `open` transition (mount effect), so
      // leaving `tokenDialogOpen` true after a failure both (a) keeps the
      // stale token sitting in the DOM after the attempt has settled (I9),
      // and (b) visually hides the error panel behind the still-open z-50
      // overlay. Closing it here, unconditionally, before the per-code
      // branches below, fixes both: the panel renders on the now-visible
      // underlying dialog, and clicking "Promote…" again reopens
      // `<LiveConfirmDialog>` fresh (its own `open`-transition effect wipes
      // the field) for a same-key (or freshly-reset-key) retry.
      setTokenDialogOpen(false);

      if (result.error.status === 0) {
        // SY-70-19: status 0 keeps the key — a same-key retry is safe.
        setErrorMessage(result.error.message);
        setErrorDetail(result.error.detail);
        return;
      }

      const outcome = classifyIdempotencyError(result.error.detail);
      if (outcome.kind === "in_progress") {
        setInProgressNotice(true);
        return;
      }
      if (outcome.kind === "reused") {
        idempotency.reset();
        setErrorMessage(IDEMPOTENCY_REUSED_MESSAGE);
        return;
      }
      if (outcome.kind === "client_error") {
        console.error("idempotency client error", getStructuredErrorCode(result.error.detail));
        idempotency.reset();
        setErrorMessage(IDEMPOTENCY_CLIENT_ERROR_MESSAGE);
        return;
      }

      // Every other code (403, WP1.3a 422, 404 source not found, ...) keeps
      // the key and falls through to the existing panel.
      setErrorMessage(result.error.message);
      setErrorDetail(result.error.detail);
    } finally {
      setLoading(false);
      submitLock.release();
    }
  }

  // SY-70-22: reopens the token dialog for a fresh, retyped token, reusing
  // the same key (G-13 — never a retained token).
  function retry() {
    setInProgressNotice(false);
    setTokenDialogOpen(true);
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

        {inProgressNotice && (
          <div
            role="status"
            aria-live="polite"
            className="mt-3 space-y-2 rounded-lg border border-amber-700/60 bg-amber-900/20 px-4 py-3 text-sm text-amber-400"
          >
            <p>{IDEMPOTENCY_IN_PROGRESS_MESSAGE}</p>
            <button
              type="button"
              onClick={retry}
              className="rounded-lg border border-amber-600 bg-transparent px-3 py-1.5 text-xs font-medium text-amber-400 hover:bg-amber-900/40"
            >
              Check again
            </button>
          </div>
        )}

        {errorMessage && !inProgressNotice && (
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
