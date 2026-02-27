import type { Metadata, Viewport } from "next";
import { Inter } from "next/font/google";
import "./globals.css";

// -------------------------------------------------------------------------
// Font
// -------------------------------------------------------------------------
const inter = Inter({
  subsets: ["latin"],
  // Expose as a CSS variable so Tailwind's fontFamily config can reference it
  variable: "--font-inter",
  // Only include the weights we actually use to minimise bundle size
  weight: ["400", "500", "600", "700"],
  display: "swap",
});

// -------------------------------------------------------------------------
// Metadata — shared across all routes
// -------------------------------------------------------------------------
export const metadata: Metadata = {
  title: {
    default: "Trading Bot Dashboard",
    // Child pages use: export const metadata = { title: "Page Title" }
    // and Next.js will render "Page Title | Trading Bot Dashboard"
    template: "%s | Trading Bot Dashboard",
  },
  description:
    "AI Crypto Trading Bot — backtest, paper trade, and monitor live positions from one dashboard.",
  // Prevent search engines from indexing an internal trading tool
  robots: { index: false, follow: false },
};

export const viewport: Viewport = {
  themeColor: [
    { media: "(prefers-color-scheme: light)", color: "#f8fafc" },
    { media: "(prefers-color-scheme: dark)", color: "#020617" },
  ],
};

// -------------------------------------------------------------------------
// Root Layout
// -------------------------------------------------------------------------
export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    /*
      `dark` class is statically applied here; a future ThemeProvider
      component can toggle it via client-side JS once the preference is
      known. Starting with `dark` ensures the dark colour-scheme is the
      default — appropriate for a trading dashboard used in low-light.
    */
    <html lang="en" className={`${inter.variable} dark`} suppressHydrationWarning>
      <body className="min-h-screen bg-slate-950 font-sans text-slate-100 antialiased">
        {/*
          Main application shell.
          The nav, sidebar, and content area will be composed here as
          dedicated layout components in subsequent implementation sprints.
        */}
        <div className="flex min-h-screen flex-col">
          {/* Top navigation bar — placeholder for the nav component */}
          <header className="sticky top-0 z-40 border-b border-slate-800 bg-slate-950/80 backdrop-blur-sm">
            <div className="mx-auto flex h-14 max-w-screen-2xl items-center justify-between px-4 sm:px-6 lg:px-8">
              <span className="text-sm font-semibold tracking-tight text-slate-100">
                Trading Bot
              </span>
              {/* Navigation links will be composed here */}
            </div>
          </header>

          {/* Page content */}
          <main className="mx-auto w-full max-w-screen-2xl flex-1 px-4 py-6 sm:px-6 lg:px-8">
            {children}
          </main>
        </div>
      </body>
    </html>
  );
}
