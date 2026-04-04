"use client";

export default function GlobalError({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  const isDev = process.env.NODE_ENV === "development";

  return (
    <html lang="en">
      <body className="flex min-h-screen items-center justify-center bg-slate-50 dark:bg-slate-950">
        <div className="mx-auto max-w-md rounded-lg border border-slate-200 bg-white p-8 shadow-sm dark:border-slate-800 dark:bg-slate-900">
          <h1 className="text-xl font-bold text-slate-900 dark:text-slate-100">
            Application Error
          </h1>
          <p className="mt-3 text-sm text-slate-600 dark:text-slate-400">
            An unexpected error occurred. Please try refreshing the page.
          </p>
          {error.digest && (
            <p className="mt-2 text-xs text-slate-500 dark:text-slate-500">
              Error ID: {error.digest}
            </p>
          )}
          {isDev && (
            <pre className="mt-3 max-h-32 overflow-auto rounded bg-slate-100 p-3 text-xs text-slate-800 dark:bg-slate-800 dark:text-slate-200">
              {error.message}
            </pre>
          )}
          <div className="mt-6 flex gap-3">
            <button
              onClick={reset}
              className="rounded-md bg-indigo-600 px-4 py-2 text-sm font-medium text-white hover:bg-indigo-700"
            >
              Try again
            </button>
            <button
              onClick={() => window.location.reload()}
              className="rounded-md border border-slate-300 px-4 py-2 text-sm font-medium text-slate-700 hover:bg-slate-50 dark:border-slate-600 dark:text-slate-300 dark:hover:bg-slate-800"
            >
              Reload page
            </button>
          </div>
        </div>
      </body>
    </html>
  );
}
