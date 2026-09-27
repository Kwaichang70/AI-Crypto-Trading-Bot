/**
 * apps/ui/src/components/kill-switch-button.tsx
 * -----------------------------------------------
 * Global kill-switch PRESS control for the operator dashboard (relocated
 * from the homepage into the sidebar's "System" section — see
 * `apps/ui/src/components/layout/sidebar.tsx` — so it is global, not
 * homepage-only).
 *
 * Visible only to users with role "admin" (wrapped in <AdminOnly>).
 * Clicking the red button opens a confirmation modal that requires:
 *   1. Optional incident reason text (max 500 chars).
 *   2. An optional "also flatten every newly-latched running engine" choice.
 *   3. A hard-typed confirmation string: "EMERGENCY STOP" (14 chars, case-sensitive).
 *
 * On submit, calls POST /api/admin/kill-switch (the Next.js route handler
 * that proxies to FastAPI and keeps INTERNAL_ADMIN_API_KEY server-side).
 *
 * WP1.7a/1.7b (CF-B1): the response is the WP1.7a camelCase
 * `KillSwitchPressResponse` shape — this LATCHES every running engine
 * (blocks new BUYs; exits keep running, D3). It does NOT stop runs and
 * there is no `runsStopped`/`tasksCancelled` count to report any more.
 *
 * AC8: the request timeout is >= 60s (KILL_SWITCH_TIMEOUT_MS) — an optional
 * flatten pass can run every newly-latched engine's flatten() in parallel,
 * each bounded at ~30s + a 5s outer margin server-side.
 *
 * Accessibility:
 *   - Modal uses role="dialog" + aria-modal="true" + aria-labelledby.
 *   - Focus moves to the confirmation input on open; Escape closes.
 *   - Confirm button is disabled until confirmation string matches exactly.
 *   - Loading state disables both action buttons and shows a spinner.
 */

"use client";

import { useState, useEffect, useRef, useCallback } from "react";
import { AdminOnly } from "@/components/admin-only";
import { useToast } from "@/components/ui/toast";
import { FlattenResultView } from "@/components/flatten-result-view";
import { adminFetch, KILL_SWITCH_TIMEOUT_MS } from "@/lib/admin-fetch";
import type { KillSwitchPressResponse } from "@/lib/types";

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const CONFIRMATION_PHRASE = "EMERGENCY STOP" as const;

// ---------------------------------------------------------------------------
// Spinner
// ---------------------------------------------------------------------------

function Spinner() {
  return (
    <svg
      className="h-4 w-4 animate-spin"
      viewBox="0 0 24 24"
      fill="none"
      aria-hidden="true"
    >
      <circle
        className="opacity-25"
        cx="12"
        cy="12"
        r="10"
        stroke="currentColor"
        strokeWidth="4"
      />
      <path
        className="opacity-75"
        fill="currentColor"
        d="M4 12a8 8 0 018-8v4a4 4 0 00-4 4H4z"
      />
    </svg>
  );
}

// ---------------------------------------------------------------------------
// Modal
// ---------------------------------------------------------------------------

interface ModalProps {
  onClose: () => void;
  onConfirm: (reason: string, flatten: boolean) => Promise<void>;
  loading: boolean;
  result: KillSwitchPressResponse | null;
}

function KillSwitchModal({ onClose, onConfirm, loading, result }: ModalProps) {
  const [reason, setReason] = useState("");
  const [flatten, setFlatten] = useState(false);
  const [confirmText, setConfirmText] = useState("");
  const firstFocusRef = useRef<HTMLTextAreaElement>(null);

  const confirmed = confirmText === CONFIRMATION_PHRASE;

  // Move focus to the textarea on mount; Escape closes the modal.
  useEffect(() => {
    firstFocusRef.current?.focus();

    function handleKeyDown(e: KeyboardEvent) {
      if (e.key === "Escape" && !loading) {
        onClose();
      }
    }
    document.addEventListener("keydown", handleKeyDown);
    return () => document.removeEventListener("keydown", handleKeyDown);
  }, [onClose, loading]);

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!confirmed || loading) return;
    void onConfirm(reason, flatten);
  }

  return (
    /* Overlay — FE-004: aria-hidden removed entirely (it was incorrect on a live modal overlay) */
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm"
      onClick={(e) => {
        // Close on backdrop click only when not loading
        if (e.target === e.currentTarget && !loading) onClose();
      }}
    >
      {/* Dialog */}
      <div
        role="dialog"
        aria-modal="true"
        aria-labelledby="kill-switch-dialog-title"
        className="mx-4 w-full max-w-md rounded-xl border border-slate-700 bg-slate-900 p-6 shadow-2xl"
      >
        {/* Header */}
        <div className="mb-4 flex items-start gap-3">
          {/* Warning icon */}
          <span className="mt-0.5 shrink-0 text-red-500" aria-hidden="true">
            <svg viewBox="0 0 20 20" fill="currentColor" className="h-6 w-6">
              <path
                fillRule="evenodd"
                d="M8.485 2.495c.673-1.167 2.357-1.167 3.03 0l6.28 10.875c.673 1.167-.17 2.625-1.516 2.625H3.72c-1.347 0-2.189-1.458-1.515-2.625L8.485 2.495zM10 5a.75.75 0 01.75.75v3.5a.75.75 0 01-1.5 0v-3.5A.75.75 0 0110 5zm0 9a1 1 0 100-2 1 1 0 000 2z"
                clipRule="evenodd"
              />
            </svg>
          </span>
          <div>
            <h2
              id="kill-switch-dialog-title"
              className="text-base font-semibold text-slate-100"
            >
              Global Kill Switch
            </h2>
            <p className="mt-1 text-sm text-slate-400">
              Latches every running paper/live engine — blocks new BUYs.
              Exits (stop-loss, brackets, trailing stops) keep running. No
              run is stopped and no task is cancelled.
            </p>
          </div>
        </div>

        {!result && (
          <form onSubmit={handleSubmit} className="space-y-4">
            {/* Optional reason */}
            <div>
              <label
                htmlFor="kill-switch-reason"
                className="mb-1.5 block text-xs font-medium text-slate-400"
              >
                Incident note{" "}
                <span className="font-normal text-slate-500">(optional)</span>
              </label>
              <textarea
                ref={firstFocusRef}
                id="kill-switch-reason"
                value={reason}
                onChange={(e) => setReason(e.target.value)}
                maxLength={500}
                placeholder="Incident note (optional)"
                disabled={loading}
                rows={3}
                className="w-full resize-none rounded-lg border border-slate-600 bg-slate-800 px-3 py-2 text-sm text-slate-200 placeholder-slate-500 focus:border-slate-400 focus:outline-none focus:ring-1 focus:ring-slate-400 disabled:opacity-50"
              />
              <p className="mt-1 text-right text-xs text-slate-600">
                {reason.length}/500
              </p>
            </div>

            {/* Optional flatten pass */}
            <label className="flex items-start gap-2 text-sm text-slate-300">
              <input
                type="checkbox"
                checked={flatten}
                onChange={(e) => setFlatten(e.target.checked)}
                disabled={loading}
                className="mt-0.5"
              />
              <span>
                Also flatten every newly-latched running engine (sells every
                held symbol through the capped exit path; can take up to
                ~30s per engine, run in parallel).
              </span>
            </label>

            {/* Hard confirmation */}
            <div>
              <label
                htmlFor="kill-switch-confirm"
                className="mb-1.5 block text-xs font-medium text-slate-400"
              >
                Type{" "}
                <span className="font-mono font-semibold text-red-400">
                  EMERGENCY STOP
                </span>{" "}
                to confirm
              </label>
              <input
                id="kill-switch-confirm"
                type="text"
                value={confirmText}
                onChange={(e) => setConfirmText(e.target.value)}
                disabled={loading}
                autoComplete="off"
                spellCheck={false}
                className="w-full rounded-lg border border-slate-600 bg-slate-800 px-3 py-2 font-mono text-sm text-slate-200 placeholder-slate-500 focus:border-red-500 focus:outline-none focus:ring-1 focus:ring-red-500 disabled:opacity-50"
                placeholder="EMERGENCY STOP"
              />
            </div>

            {/* Actions */}
            <div className="flex gap-3 pt-1">
              <button
                type="button"
                onClick={onClose}
                disabled={loading}
                className="flex-1 rounded-lg border border-slate-600 bg-slate-800 px-4 py-2 text-sm font-medium text-slate-300 transition-colors hover:bg-slate-700 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-slate-400 disabled:cursor-not-allowed disabled:opacity-50"
              >
                Cancel
              </button>
              <button
                type="submit"
                disabled={!confirmed || loading}
                className="flex flex-1 items-center justify-center gap-2 rounded-lg bg-red-600 px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-red-700 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-red-500 disabled:cursor-not-allowed disabled:opacity-50"
              >
                {loading && <Spinner />}
                Activate Kill Switch
              </button>
            </div>
          </form>
        )}

        {result && (
          <div className="space-y-3">
            <p className="text-sm text-slate-300">
              Latched {result.runsLatched.length} run
              {result.runsLatched.length !== 1 ? "s" : ""}
              {result.latchPersisted ? "" : " (WARNING: not persisted — in-memory only)"}.
            </p>
            {result.orphanedLiveRunIds.length > 0 && (
              <p className="text-xs text-amber-400">
                {result.orphanedLiveRunIds.length} orphaned live run
                {result.orphanedLiveRunIds.length !== 1 ? "s" : ""} found.
              </p>
            )}
            {result.resumingRunIds.length > 0 && (
              <p className="text-xs text-amber-400">
                {result.resumingRunIds.length} resuming run
                {result.resumingRunIds.length !== 1 ? "s" : ""} moved to
                orphaned.
              </p>
            )}
            {Object.entries(result.flattenResults).map(([runId, fr]) => (
              <div key={runId} className="space-y-1">
                <p className="font-mono text-xs text-slate-500">{runId}</p>
                <FlattenResultView result={fr} />
              </div>
            ))}
            {result.errors.length > 0 && (
              <div className="rounded-lg border border-red-300 bg-red-50 px-3 py-2 text-xs text-red-600 dark:border-red-800 dark:bg-red-900/20 dark:text-red-400">
                {result.errors.map((e) => (
                  <p key={e.runId}>
                    {e.runId.slice(0, 8)}…: {e.error}
                  </p>
                ))}
              </div>
            )}
            <button
              type="button"
              onClick={onClose}
              className="w-full rounded-lg bg-slate-700 px-4 py-2 text-sm font-medium text-white hover:bg-slate-600"
            >
              Done
            </button>
          </div>
        )}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main export
// ---------------------------------------------------------------------------

export function KillSwitchButton() {
  const [modalOpen, setModalOpen] = useState(false);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<KillSwitchPressResponse | null>(null);
  const { toast } = useToast();

  // FE-003: wrap onClose in useCallback so the modal's useEffect dependency
  // array gets a stable reference — avoids spurious effect re-runs.
  const handleClose = useCallback(() => {
    if (!loading) {
      setModalOpen(false);
      setResult(null);
    }
  }, [loading]);

  const handleConfirm = useCallback(
    async (reason: string, flatten: boolean) => {
      setLoading(true);
      try {
        const apiResult = await adminFetch<KillSwitchPressResponse>(
          "/api/admin/kill-switch",
          {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ reason: reason.trim() || undefined, flatten }),
          },
          KILL_SWITCH_TIMEOUT_MS,
        );

        if (apiResult.ok) {
          setResult(apiResult.data);
          const errCount = apiResult.data.errors.length;
          if (errCount > 0) {
            toast(
              `Kill switch latched ${apiResult.data.runsLatched.length} run(s) with ${errCount} error(s).`,
              "error",
            );
          } else {
            toast(
              `Kill switch latched ${apiResult.data.runsLatched.length} run(s). Entries blocked; exits keep running.`,
              "success",
            );
          }
        } else {
          toast(apiResult.error.message, "error");
        }
      } finally {
        setLoading(false);
      }
    },
    [toast],
  );

  return (
    <AdminOnly>
      <button
        type="button"
        onClick={() => setModalOpen(true)}
        className="rounded-lg bg-red-600 px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-red-700 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-red-500"
      >
        Global Kill Switch
      </button>

      {modalOpen && (
        <KillSwitchModal
          onClose={handleClose}
          onConfirm={handleConfirm}
          loading={loading}
          result={result}
        />
      )}
    </AdminOnly>
  );
}
