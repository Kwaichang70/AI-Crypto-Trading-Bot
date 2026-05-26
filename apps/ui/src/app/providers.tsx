/**
 * apps/ui/src/app/providers.tsx
 * ------------------------------
 * Client-boundary wrapper for NextAuth SessionProvider.
 *
 * layout.tsx is a Server Component. SessionProvider requires "use client"
 * because it uses React Context. This thin wrapper satisfies both constraints
 * without forcing the entire layout to become a client component.
 */

"use client";

import { SessionProvider } from "next-auth/react";
import type { ReactNode } from "react";

export function Providers({ children }: { children: ReactNode }) {
  return <SessionProvider>{children}</SessionProvider>;
}
