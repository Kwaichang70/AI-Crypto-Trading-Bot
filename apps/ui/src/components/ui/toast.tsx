"use client";

/**
 * apps/ui/src/components/ui/toast.tsx
 * ------------------------------------
 * Lightweight, zero-dependency toast notification system.
 *
 * Usage:
 *   1. Wrap the app in <ToastProvider> (already done in layout.tsx).
 *   2. In any client component: const { toast } = useToast();
 *      toast("Message", "success" | "error" | "info" | "warning");
 *
 * Behaviour:
 *   - Toasts stack upward from bottom-right corner, z-50.
 *   - Auto-dismiss: 4 s for success/info, 6 s for error/warning.
 *   - Manual dismiss via the X button.
 *   - Max 5 visible at once (oldest removed when a 6th is pushed).
 *   - Slide-in animation via the custom `animate-toast-in` Tailwind class
 *     (defined in tailwind.config.ts).
 */

import {
  createContext,
  useCallback,
  useContext,
  useState,
} from "react";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export type ToastType = "success" | "error" | "info" | "warning";

interface ToastItem {
  id: string;
  message: string;
  type: ToastType;
}

interface ToastContextValue {
  toast: (message: string, type?: ToastType) => void;
}

// ---------------------------------------------------------------------------
// Context
// ---------------------------------------------------------------------------

const ToastContext = createContext<ToastContextValue>({ toast: () => {} });

export function useToast(): ToastContextValue {
  return useContext(ToastContext);
}

// ---------------------------------------------------------------------------
// Visual config per type
// ---------------------------------------------------------------------------

const TYPE_CONFIG: Record<
  ToastType,
  {
    containerClass: string;
    iconPath: React.ReactNode;
    iconClass: string;
    dismissMs: number;
  }
> = {
  success: {
    containerClass:
      "border-emerald-500/40 bg-slate-900 dark:bg-slate-900",
    iconClass: "text-emerald-400",
    iconPath: (
      // Checkmark circle
      <svg
        viewBox="0 0 20 20"
        fill="currentColor"
        className="h-4 w-4 shrink-0"
        aria-hidden="true"
      >
        <path
          fillRule="evenodd"
          d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.857-9.809a.75.75 0 00-1.214-.882l-3.483 4.79-1.88-1.88a.75.75 0 10-1.06 1.061l2.5 2.5a.75.75 0 001.137-.089l4-5.5z"
          clipRule="evenodd"
        />
      </svg>
    ),
    dismissMs: 4000,
  },
  error: {
    containerClass:
      "border-red-500/40 bg-slate-900 dark:bg-slate-900",
    iconClass: "text-red-400",
    iconPath: (
      // X circle
      <svg
        viewBox="0 0 20 20"
        fill="currentColor"
        className="h-4 w-4 shrink-0"
        aria-hidden="true"
      >
        <path
          fillRule="evenodd"
          d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.28 7.22a.75.75 0 00-1.06 1.06L8.94 10l-1.72 1.72a.75.75 0 101.06 1.06L10 11.06l1.72 1.72a.75.75 0 101.06-1.06L11.06 10l1.72-1.72a.75.75 0 00-1.06-1.06L10 8.94 8.28 7.22z"
          clipRule="evenodd"
        />
      </svg>
    ),
    dismissMs: 6000,
  },
  info: {
    containerClass:
      "border-indigo-500/40 bg-slate-900 dark:bg-slate-900",
    iconClass: "text-indigo-400",
    iconPath: (
      // Info circle
      <svg
        viewBox="0 0 20 20"
        fill="currentColor"
        className="h-4 w-4 shrink-0"
        aria-hidden="true"
      >
        <path
          fillRule="evenodd"
          d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a.75.75 0 000 1.5h.253a.25.25 0 01.244.304l-.459 2.066A1.75 1.75 0 0010.747 15H11a.75.75 0 000-1.5h-.253a.25.25 0 01-.244-.304l.459-2.066A1.75 1.75 0 009.253 9H9z"
          clipRule="evenodd"
        />
      </svg>
    ),
    dismissMs: 4000,
  },
  warning: {
    containerClass:
      "border-amber-500/40 bg-slate-900 dark:bg-slate-900",
    iconClass: "text-amber-400",
    iconPath: (
      // Warning triangle (exclamation)
      <svg
        viewBox="0 0 20 20"
        fill="currentColor"
        className="h-4 w-4 shrink-0"
        aria-hidden="true"
      >
        <path
          fillRule="evenodd"
          d="M8.485 2.495c.673-1.167 2.357-1.167 3.03 0l6.28 10.875c.673 1.167-.17 2.625-1.516 2.625H3.72c-1.347 0-2.189-1.458-1.515-2.625L8.485 2.495zM10 5a.75.75 0 01.75.75v3.5a.75.75 0 01-1.5 0v-3.5A.75.75 0 0110 5zm0 9a1 1 0 100-2 1 1 0 000 2z"
          clipRule="evenodd"
        />
      </svg>
    ),
    dismissMs: 6000,
  },
};

// ---------------------------------------------------------------------------
// ToastItem component
// ---------------------------------------------------------------------------

function ToastItemView({
  toast: t,
  onDismiss,
}: {
  toast: ToastItem;
  onDismiss: () => void;
}) {
  const cfg = TYPE_CONFIG[t.type];

  return (
    <div
      role="alert"
      aria-live="assertive"
      className={[
        "flex w-80 max-w-xs items-start gap-3 rounded-lg border px-3.5 py-3 shadow-lg",
        "animate-toast-in",
        cfg.containerClass,
      ].join(" ")}
    >
      {/* Icon */}
      <span className={cfg.iconClass}>{cfg.iconPath}</span>

      {/* Message */}
      <p className="min-w-0 flex-1 break-words text-sm text-slate-200">
        {t.message}
      </p>

      {/* Dismiss */}
      <button
        type="button"
        onClick={onDismiss}
        aria-label="Dismiss notification"
        className="ml-auto shrink-0 rounded p-0.5 text-slate-500 transition-colors hover:text-slate-300 focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-slate-400"
      >
        <svg
          viewBox="0 0 16 16"
          fill="currentColor"
          className="h-3.5 w-3.5"
          aria-hidden="true"
        >
          <path d="M3.72 3.72a.75.75 0 011.06 0L8 6.94l3.22-3.22a.75.75 0 111.06 1.06L9.06 8l3.22 3.22a.75.75 0 11-1.06 1.06L8 9.06l-3.22 3.22a.75.75 0 01-1.06-1.06L6.94 8 3.72 4.78a.75.75 0 010-1.06z" />
        </svg>
      </button>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Provider
// ---------------------------------------------------------------------------

export function ToastProvider({ children }: { children: React.ReactNode }) {
  const [toasts, setToasts] = useState<ToastItem[]>([]);

  const toast = useCallback((message: string, type: ToastType = "info") => {
    const id =
      Date.now().toString(36) + Math.random().toString(36).slice(2, 5);

    // Keep at most 5 toasts; slice off the oldest when a 6th arrives.
    setToasts((prev) => [...prev.slice(-4), { id, message, type }]);

    const { dismissMs } = TYPE_CONFIG[type];
    setTimeout(() => {
      setToasts((prev) => prev.filter((t) => t.id !== id));
    }, dismissMs);
  }, []);

  const dismiss = useCallback((id: string) => {
    setToasts((prev) => prev.filter((t) => t.id !== id));
  }, []);

  return (
    <ToastContext.Provider value={{ toast }}>
      {children}

      {/* Fixed toast container — bottom-right, stacks upward */}
      <div
        aria-label="Notifications"
        className="fixed bottom-4 right-4 z-50 flex flex-col-reverse gap-2"
      >
        {toasts.map((t) => (
          <ToastItemView
            key={t.id}
            toast={t}
            onDismiss={() => dismiss(t.id)}
          />
        ))}
      </div>
    </ToastContext.Provider>
  );
}
