/**
 * apps/ui/src/components/live-confirm-dialog.tsx
 * --------------------------------------------------
 * Generic typed live-trading confirmation modal (S13).
 *
 * Used everywhere the backend's 3-layer live-trading safety gate
 * (env flag + API keys + `X-Live-Confirm-Token` header) requires a typed
 * token: starting a live run (`createRun`), resuming an orphaned live run
 * (`resume_run`, BOTH modes — the backend's `LiveTradingGate.check_gate`
 * call is unconditional, not just for mode=normal), and clearing a live
 * run's entries latch.
 *
 * Security invariants (WP1.7 synthesis-spec I11 / SY-10):
 *   - The token is held ONLY in this component's own local state. It is
 *     never lifted to a parent's state, never written to localStorage/
 *     sessionStorage, and is discarded (state reset) on close AND on
 *     unmount — reopening always starts from an empty field.
 *   - The token is handed to the caller's `onConfirm(token)` exactly once,
 *     to be sent as the `X-Live-Confirm-Token` HEADER of the actual
 *     request. This component never puts it in a request body itself.
 */

"use client";

import { useEffect, useRef, useState } from "react";

interface LiveConfirmDialogProps {
  open: boolean;
  title: string;
  description: React.ReactNode;
  confirmLabel?: string;
  loading: boolean;
  onCancel: () => void;
  onConfirm: (token: string) => void | Promise<void>;
}

export function LiveConfirmDialog({
  open,
  title,
  description,
  confirmLabel = "Confirm",
  loading,
  onCancel,
  onConfirm,
}: LiveConfirmDialogProps) {
  const [token, setToken] = useState("");
  const inputRef = useRef<HTMLInputElement>(null);

  // Reset (never persist) the token every time the dialog opens OR closes,
  // and on unmount — it must never survive past a single confirm attempt.
  useEffect(() => {
    if (open) {
      setToken("");
      inputRef.current?.focus();
    }
    return () => setToken("");
  }, [open]);

  useEffect(() => {
    function handleKeyDown(e: KeyboardEvent) {
      if (e.key === "Escape" && !loading) handleCancel();
    }
    if (open) document.addEventListener("keydown", handleKeyDown);
    return () => document.removeEventListener("keydown", handleKeyDown);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open, loading]);

  if (!open) return null;

  function handleCancel() {
    if (loading) return;
    setToken("");
    onCancel();
  }

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!token || loading) return;
    void onConfirm(token);
  }

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm"
      onClick={(e) => {
        if (e.target === e.currentTarget) handleCancel();
      }}
    >
      <div
        role="dialog"
        aria-modal="true"
        aria-labelledby="live-confirm-dialog-title"
        className="mx-4 w-full max-w-md rounded-xl border border-red-700/60 bg-slate-900 p-6 shadow-2xl"
      >
        <h2 id="live-confirm-dialog-title" className="text-base font-semibold text-slate-100">
          {title}
        </h2>
        <div className="mt-1 text-sm text-slate-400">{description}</div>

        <form onSubmit={handleSubmit} className="mt-4 space-y-4">
          <div>
            <label
              htmlFor="live-confirm-token"
              className="mb-1.5 block text-xs font-medium text-slate-400"
            >
              Live trading confirmation token
            </label>
            <input
              ref={inputRef}
              id="live-confirm-token"
              type="password"
              value={token}
              onChange={(e) => setToken(e.target.value)}
              disabled={loading}
              autoComplete="off"
              spellCheck={false}
              placeholder="LIVE_TRADING_CONFIRM_TOKEN"
              className="w-full rounded-lg border border-red-700/60 bg-slate-800 px-3 py-2 font-mono text-sm text-slate-200 placeholder-slate-500 focus:border-red-500 focus:outline-none focus:ring-1 focus:ring-red-500 disabled:opacity-50"
            />
            <p className="mt-1 text-xs text-slate-500">
              Never stored — sent once, as a request header, and discarded
              when this dialog closes.
            </p>
          </div>

          <div className="flex gap-3">
            <button
              type="button"
              onClick={handleCancel}
              disabled={loading}
              className="flex-1 rounded-lg border border-slate-600 bg-slate-800 px-4 py-2 text-sm font-medium text-slate-300 transition-colors hover:bg-slate-700 disabled:cursor-not-allowed disabled:opacity-50"
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={!token || loading}
              className="flex-1 rounded-lg bg-red-600 px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-red-700 disabled:cursor-not-allowed disabled:opacity-50"
            >
              {loading ? "Working…" : confirmLabel}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
