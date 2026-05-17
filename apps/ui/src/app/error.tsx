"use client";

// Next.js App Router global error boundary.
// This file is automatically used by Next.js for any unhandled error that
// propagates out of the root layout. It must be a Client Component.
// See: https://nextjs.org/docs/app/api-reference/file-conventions/error

interface ErrorPageProps {
  error: Error & { digest?: string };
  reset: () => void;
}

export default function GlobalError({ error, reset }: ErrorPageProps) {
  const isDev = process.env.NODE_ENV === "development";

  return (
    <html lang="en" className="dark">
      <body className="min-h-screen bg-slate-50 font-sans antialiased dark:bg-slate-950">
        <div className="flex min-h-screen flex-col items-center justify-center px-4">
          <div className="w-full max-w-md rounded-xl border border-slate-200 bg-white p-8 shadow-sm dark:border-slate-800 dark:bg-slate-900">

            {/* Icon */}
            <div className="mb-5 flex h-12 w-12 items-center justify-center rounded-full bg-red-100 dark:bg-red-900/30">
              <span className="text-xl font-bold text-red-600 dark:text-red-400" aria-hidden="true">
                X
              </span>
            </div>

            {/* Heading */}
            <h1 className="text-lg font-semibold text-slate-900 dark:text-slate-100">
              Application error
            </h1>
            <p className="mt-2 text-sm text-slate-500 dark:text-slate-400">
              An unexpected error prevented the dashboard from loading. You can
              try again or reload the page.
            </p>

            {/* Digest — a stable, non-sensitive server error ID */}
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
            <div className="mt-6 flex gap-3">
              <button
                type="button"
                onClick={reset}
                className="rounded-lg bg-indigo-600 px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-indigo-700 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500 focus-visible:ring-offset-2 dark:focus-visible:ring-offset-slate-900"
              >
                Try again
              </button>
              <button
                type="button"
                onClick={() => window.location.reload()}
                className="rounded-lg border border-slate-200 bg-white px-4 py-2 text-sm font-medium text-slate-700 transition-colors hover:bg-slate-50 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500 focus-visible:ring-offset-2 dark:border-slate-700 dark:bg-slate-800 dark:text-slate-200 dark:hover:bg-slate-700 dark:focus-visible:ring-offset-slate-900"
              >
                Reload page
              </button>
            </div>
          </div>
        </div>
      </body>
    </html>
  );
}
