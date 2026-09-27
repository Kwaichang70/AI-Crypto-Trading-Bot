/**
 * apps/ui/src/components/kill-switch-panel.tsx
 * -------------------------------------------------
 * WP1.7b (S13) sidebar kill-switch section:
 *   - A latch-status badge, visible to EVERY signed-in user (the backend's
 *     `GET /emergency/kill-switch` only requires X-API-Key, not admin — so
 *     this calls it directly via `fetchKillSwitchStatus`, no admin proxy;
 *     see synthesis-spec.md SY-11 and this WP's producer report for why
 *     the "GET status" proxy in the assignment brief is intentionally not
 *     built as an admin-key proxy).
 *   - `latched`/`unlatched`/"Latch state unknown" (`source: "unknown"`,
 *     fail-closed after a boot that could not read the DB row) states.
 *   - The PRESS control (`KillSwitchButton`, admin-only).
 *   - A CLEAR action (admin-only): reason + admin key, reports
 *     `runsKeptLatched` so the operator knows which engines are still
 *     latched for an unrelated reason (e.g. their own `flatten_incomplete`).
 */

"use client";

import { useCallback, useEffect, useState } from "react";
import { fetchKillSwitchStatus } from "@/lib/api";
import { adminFetch, KILL_SWITCH_TIMEOUT_MS } from "@/lib/admin-fetch";
import type { KillSwitchClearResponse, KillSwitchStatus } from "@/lib/types";
import { AdminOnly } from "@/components/admin-only";
import { KillSwitchButton } from "@/components/kill-switch-button";
import { useToast } from "@/components/ui/toast";

const POLL_INTERVAL_MS = 15_000;

function StatusPill({
  status,
  unavailable,
}: {
  status: KillSwitchStatus | null;
  unavailable: boolean;
}) {
  // WP17b-S-10 (round 2): a failed GET must never look identical to
  // "still loading" forever -- the operator needs to know the latch state
  // is simply unknown right now (e.g. the API is unreachable), not assume
  // it's about to arrive.
  if (unavailable) {
    return (
      <span
        title="Could not reach the API to read the kill-switch status."
        className="inline-flex items-center rounded-full bg-slate-200 px-2.5 py-0.5 text-xs font-medium text-slate-600 dark:bg-slate-700 dark:text-slate-300"
      >
        Unavailable
      </span>
    );
  }

  if (!status) {
    return (
      <span className="inline-flex items-center rounded-full bg-slate-100 px-2.5 py-0.5 text-xs font-medium text-slate-500 dark:bg-slate-800 dark:text-slate-400">
        Loading…
      </span>
    );
  }

  if (status.source === "unknown") {
    return (
      <span
        title="The latch state could not be read at boot — fail-closed, treated as latched."
        className="inline-flex items-center rounded-full bg-amber-100 px-2.5 py-0.5 text-xs font-medium text-amber-700 dark:bg-amber-900/30 dark:text-amber-400"
      >
        Latch state unknown
      </span>
    );
  }

  if (status.latched) {
    return (
      <span
        title={status.reason ?? undefined}
        className="inline-flex items-center rounded-full bg-red-100 px-2.5 py-0.5 text-xs font-medium text-red-700 dark:bg-red-900/30 dark:text-red-400"
      >
        Entries halted
      </span>
    );
  }

  return (
    <span className="inline-flex items-center rounded-full bg-emerald-100 px-2.5 py-0.5 text-xs font-medium text-emerald-700 dark:bg-emerald-900/30 dark:text-emerald-400">
      Normal
    </span>
  );
}

function ClearModal({
  onClose,
  onCleared,
}: {
  onClose: () => void;
  onCleared: (result: KillSwitchClearResponse) => void;
}) {
  const [reason, setReason] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (reason.trim().length < 3) {
      setError("Reason must be at least 3 characters.");
      return;
    }
    setLoading(true);
    setError(null);
    const result = await adminFetch<KillSwitchClearResponse>(
      "/api/admin/kill-switch/clear",
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ reason: reason.trim() }),
      },
      KILL_SWITCH_TIMEOUT_MS,
    );
    setLoading(false);
    if (result.ok) {
      onCleared(result.data);
    } else {
      setError(result.error.message);
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
        aria-labelledby="kill-switch-clear-title"
        className="mx-4 w-full max-w-md rounded-xl border border-slate-700 bg-slate-900 p-6 shadow-2xl"
      >
        <h2 id="kill-switch-clear-title" className="text-base font-semibold text-slate-100">
          Clear Global Kill Switch
        </h2>
        <p className="mt-1 text-sm text-slate-400">
          Removes the global latch. Per-run latches (e.g. an unrelated
          `flatten_incomplete`) are left untouched.
        </p>
        <form onSubmit={(e) => void handleSubmit(e)} className="mt-4 space-y-3">
          <textarea
            value={reason}
            onChange={(e) => setReason(e.target.value)}
            placeholder="Reason (3-500 characters)"
            maxLength={500}
            rows={3}
            disabled={loading}
            className="w-full resize-none rounded-lg border border-slate-600 bg-slate-800 px-3 py-2 text-sm text-slate-200 placeholder-slate-500 disabled:opacity-50"
          />
          {error && <p className="text-xs text-red-400">{error}</p>}
          <div className="flex gap-3">
            <button
              type="button"
              onClick={onClose}
              disabled={loading}
              className="flex-1 rounded-lg border border-slate-600 bg-slate-800 px-4 py-2 text-sm font-medium text-slate-300 hover:bg-slate-700 disabled:opacity-50"
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={loading || reason.trim().length < 3}
              className="flex-1 rounded-lg bg-indigo-600 px-4 py-2 text-sm font-medium text-white hover:bg-indigo-500 disabled:cursor-not-allowed disabled:opacity-50"
            >
              {loading ? "Clearing…" : "Clear Latch"}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}

export function KillSwitchPanel() {
  const [status, setStatus] = useState<KillSwitchStatus | null>(null);
  const [unavailable, setUnavailable] = useState(false);
  const [clearOpen, setClearOpen] = useState(false);
  const { toast } = useToast();

  const poll = useCallback(() => {
    void fetchKillSwitchStatus().then((res) => {
      if (res.ok) {
        setStatus(res.data);
        setUnavailable(false);
      } else {
        setUnavailable(true);
      }
    });
  }, []);

  useEffect(() => {
    poll();
    const intervalId = setInterval(poll, POLL_INTERVAL_MS);
    return () => clearInterval(intervalId);
  }, [poll]);

  function handleCleared(result: KillSwitchClearResponse) {
    setClearOpen(false);
    poll();
    if (result.runsKeptLatched.length > 0) {
      toast(
        `Cleared. ${result.runsKeptLatched.length} run(s) remain latched for other reasons.`,
        "warning",
      );
    } else {
      toast(`Cleared. ${result.runsUnlatched.length} run(s) unlatched.`, "success");
    }
  }

  return (
    <div className="space-y-2 px-3 py-2">
      <div className="flex items-center justify-between">
        <span className="text-xs font-medium text-slate-500 dark:text-slate-400">
          Kill Switch
        </span>
        <StatusPill status={status} unavailable={unavailable} />
      </div>
      <div className="flex flex-col gap-2">
        <KillSwitchButton />
        <AdminOnly>
          <button
            type="button"
            onClick={() => setClearOpen(true)}
            className="rounded-lg border border-slate-600 bg-slate-800 px-4 py-2 text-sm font-medium text-slate-300 transition-colors hover:bg-slate-700"
          >
            Clear Kill Switch
          </button>
        </AdminOnly>
      </div>
      {clearOpen && <ClearModal onClose={() => setClearOpen(false)} onCleared={handleCleared} />}
    </div>
  );
}
