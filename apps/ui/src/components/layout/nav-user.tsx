/**
 * apps/ui/src/components/layout/nav-user.tsx
 * --------------------------------------------
 * Auth-aware user chip shown in the top navigation bar.
 *
 * - Authenticated: displays "{email} ({role})" + "Sign out" button.
 * - Unauthenticated: displays "Sign in with Google" button.
 *   (In practice middleware redirects before this state is reached;
 *    the fallback is shown during the brief sign-in flow redirect.)
 *
 * This is a Client Component because it uses useSession() which relies
 * on React Context provided by SessionProvider in layout.tsx.
 */

"use client";

import { useSession, signIn, signOut } from "next-auth/react";

export function NavUser() {
  const { data: session, status } = useSession();

  if (status === "loading") {
    return (
      <div className="h-7 w-32 animate-pulse rounded-md bg-slate-200 dark:bg-slate-800" />
    );
  }

  if (!session) {
    return (
      <button
        type="button"
        onClick={() => signIn("google")}
        className="inline-flex items-center gap-1.5 rounded-md bg-indigo-600 px-3 py-1.5 text-xs font-medium text-white hover:bg-indigo-700 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-indigo-600"
      >
        Sign in with Google
      </button>
    );
  }

  // CR-004: Defensive destructure — session.user fields may be undefined
  // on the first render tick before the session callback has propagated.
  const email = session.user?.email ?? "unknown";
  const role = session.user?.role ?? "viewer";

  return (
    <div className="flex items-center gap-2">
      <span className="hidden text-xs text-slate-500 dark:text-slate-400 sm:block">
        {email}{" "}
        <span
          className={
            role === "admin"
              ? "font-semibold text-indigo-500 dark:text-indigo-400"
              : "font-medium text-slate-400 dark:text-slate-500"
          }
        >
          ({role})
        </span>
      </span>
      <button
        type="button"
        onClick={() => signOut({ callbackUrl: "/api/auth/signin" })}
        className="inline-flex items-center gap-1.5 rounded-md border border-slate-200 px-2.5 py-1 text-xs font-medium text-slate-600 hover:bg-slate-100 dark:border-slate-700 dark:text-slate-300 dark:hover:bg-slate-800"
      >
        Sign out
      </button>
    </div>
  );
}
