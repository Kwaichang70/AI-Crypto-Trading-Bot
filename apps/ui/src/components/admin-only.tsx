/**
 * apps/ui/src/components/admin-only.tsx
 * ---------------------------------------
 * Thin gate component that renders children only when the signed-in user
 * has role "admin".  Used in future cycles to protect:
 *   - Kill-switch button (cycle 3)
 *   - Promotion gate controls (cycle 5)
 *
 * This is a Client Component so it can use useSession().
 * For Server Component gating, use isAdmin() from @/lib/auth instead.
 *
 * Usage:
 *   <AdminOnly fallback={<p>Admin only.</p>}>
 *     <DangerousButton />
 *   </AdminOnly>
 */

"use client";

import { useSession } from "next-auth/react";
import type { ReactNode } from "react";

interface AdminOnlyProps {
  children: ReactNode;
  /** Optional content shown to non-admin authenticated users. Defaults to null. */
  fallback?: ReactNode;
}

/** Fallback for controls that stay visible to viewers: a disabled "admin only" button. */
export function AdminOnlyNotice() {
  return (
    <button type="button" disabled className="cursor-not-allowed rounded-lg border border-slate-300 px-4 py-2 text-sm text-slate-500 opacity-60 dark:border-slate-700">
      admin only
    </button>
  );
}

export function AdminOnly({ children, fallback = null }: AdminOnlyProps) {
  const { data: session, status } = useSession();

  if (status === "loading") return null;

  if (session?.user?.role === "admin") {
    return <>{children}</>;
  }

  return <>{fallback}</>;
}
