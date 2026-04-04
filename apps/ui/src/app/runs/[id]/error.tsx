"use client";

// Route-level error boundary for the run detail page.
// Next.js App Router picks this up automatically for any unhandled error
// thrown inside app/runs/[id]/page.tsx or its children.
// Unlike app/error.tsx this does NOT need to render <html>/<body> — the
// root layout is preserved, so the sidebar and header remain visible.
// See: https://nextjs.org/docs/app/api-reference/file-conventions/error

import Link from "next/link";

interface RunDetailErrorProps {
  error: Error & { digest?: string };
  reset: () => void;
}

export default function RunDetailError({ error, reset }: RunDetailErrorProps) {
  const isDev = process.env.NODE_ENV === "development";

  return (
    <div className="flex min-h-[60vh] flex-col items-center justify-center px-4">
      <div className="w-full max-w-lg rounded-xl border border-slate-200 bg-white p-8 shadow-sm dark:border-slate-800 dark:bg-slate-900">

        {/* Icon */}
        <div className="mb-5 flex h-10 w-10 items-center justify-center rounded-full bg-red-100 dark:bg-red-900/30">
          <span className="text-base font-bold text-red-600 dark:text-red-400" aria-hidden="true">
            !
          </span>
        </div>

        {/* Heading */}
        <h1 className="text-base font-semibold text-slate-900 dark:text-slate-100">
          Failed to load run details
        </h1>
        <p className="mt-2 text-sm text-slate-500 dark:text-slate-400">
          The run detail page could not be rendered. This may be a transient API
          error — try again or return to the runs list.
        </p>

        {/* Digest */}
        {error.digest && (
          <p className="mt-3 font-mono text-xs text-slate-400 dark:text-slate-500">
            Error ID: {error.digest}
          </p>
        )}

        {/* Developer detail */}
        {isDev && (
          <pre className="mt-4 overflow-x-auto rounded-lg bg-slate-100 p-3 text-xs text-slate-700 dark:bg-slate-800 dark:text-slate-300">
            {error.message}
          </pre>
        )}

        {/* Actions */}
        <div className="mt-6 flex flex-wrap gap-3">
          <button
            type="button"
            onClick={reset}
            className="rounded-lg bg-indigo-600 px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-indigo-700 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500 focus-visible:ring-offset-2 dark:focus-visible:ring-offset-slate-900"
          >
            Try again
          </button>
          <Link
            href="/runs"
            className="rounded-lg border border-slate-200 bg-white px-4 py-2 text-sm font-medium text-slate-700 transition-colors hover:bg-slate-50 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500 focus-visible:ring-offset-2 dark:border-slate-700 dark:bg-slate-800 dark:text-slate-200 dark:hover:bg-slate-700 dark:focus-visible:ring-offset-slate-900"
          >
            Back to runs
          </Link>
        </div>
      </div>
    </div>
  );
}
