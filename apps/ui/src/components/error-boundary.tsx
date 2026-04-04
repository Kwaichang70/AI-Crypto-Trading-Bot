"use client";

import { Component, type ReactNode } from "react";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface Props {
  children: ReactNode;
  /** Optional custom fallback UI. When omitted the default error card is shown. */
  fallback?: ReactNode;
}

interface State {
  hasError: boolean;
  error: Error | null;
}

// ---------------------------------------------------------------------------
// ErrorBoundary — reusable class component (React requirement for error boundaries)
// ---------------------------------------------------------------------------

export class ErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, info: React.ErrorInfo): void {
    // In production this is where you would forward to an observability sink
    // (e.g. Sentry, Datadog RUM). For now we log to the console so the
    // developer can see the component stack without crashing the page.
    console.error("[ErrorBoundary] Caught unhandled error:", error, info.componentStack);
  }

  private handleReset = (): void => {
    this.setState({ hasError: false, error: null });
  };

  render(): ReactNode {
    if (!this.state.hasError) {
      return this.props.children;
    }

    if (this.props.fallback) {
      return this.props.fallback;
    }

    const isDev = process.env.NODE_ENV === "development";

    return (
      <div className="card flex flex-col gap-4 p-6">
        {/* Header row */}
        <div className="flex items-start gap-3">
          <div className="flex h-8 w-8 flex-shrink-0 items-center justify-center rounded-full bg-red-100 dark:bg-red-900/30">
            <span
              className="h-4 w-4 text-sm font-bold text-red-600 dark:text-red-400"
              aria-hidden="true"
            >
              !
            </span>
          </div>
          <div className="min-w-0">
            <h2 className="text-sm font-semibold text-slate-900 dark:text-slate-100">
              Something went wrong
            </h2>
            <p className="mt-0.5 text-xs text-slate-500 dark:text-slate-400">
              This section encountered an unexpected error and could not be displayed.
            </p>
          </div>
        </div>

        {/* Developer detail — only shown in development */}
        {isDev && this.state.error && (
          <pre className="overflow-x-auto rounded-lg bg-slate-100 p-3 text-xs text-slate-700 dark:bg-slate-800 dark:text-slate-300">
            {this.state.error.message}
          </pre>
        )}

        {/* Action */}
        <button
          type="button"
          onClick={this.handleReset}
          className="self-start rounded-lg border border-slate-200 bg-white px-3 py-1.5 text-xs font-medium text-slate-700 transition-colors hover:bg-slate-50 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500 focus-visible:ring-offset-2 dark:border-slate-700 dark:bg-slate-800 dark:text-slate-200 dark:hover:bg-slate-700 dark:focus-visible:ring-offset-slate-900"
        >
          Try again
        </button>
      </div>
    );
  }
}
