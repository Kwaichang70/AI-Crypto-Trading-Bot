import type { Metadata, Viewport } from "next";
import { Inter } from "next/font/google";
import { Sidebar } from "@/components/layout/sidebar";
import "./globals.css";

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-inter",
  weight: ["400", "500", "600", "700"],
  display: "swap",
});

export const metadata: Metadata = {
  title: {
    default: "Trading Bot Dashboard",
    template: "%s | Trading Bot Dashboard",
  },
  description:
    "AI Crypto Trading Bot — backtest, paper trade, and monitor live positions from one dashboard.",
  robots: { index: false, follow: false },
};

export const viewport: Viewport = {
  themeColor: [
    { media: "(prefers-color-scheme: light)", color: "#f8fafc" },
    { media: "(prefers-color-scheme: dark)", color: "#020617" },
  ],
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html
      lang="en"
      className={`${inter.variable} dark`}
      suppressHydrationWarning
    >
      <body className="min-h-screen bg-slate-950 font-sans text-slate-100 antialiased">
        {/* Top navigation bar */}
        <header className="sticky top-0 z-40 border-b border-slate-800 bg-slate-950/80 backdrop-blur-sm">
          <div className="mx-auto flex h-14 max-w-screen-2xl items-center justify-between px-4 sm:px-6 lg:px-8">
            <span className="text-sm font-semibold tracking-tight text-slate-100">
              Trading Bot
            </span>
            <nav className="flex items-center gap-4 lg:hidden" aria-label="Mobile navigation">
              {/* Mobile nav links — mirror sidebar items */}
              <a href="/" className="text-xs text-slate-400 hover:text-slate-200">Dashboard</a>
              <a href="/runs" className="text-xs text-slate-400 hover:text-slate-200">Runs</a>
              <a href="/runs/new" className="text-xs text-slate-400 hover:text-slate-200">New Run</a>
              <a href="/strategies" className="text-xs text-slate-400 hover:text-slate-200">Strategies</a>
              <a href="/models" className="text-xs text-slate-400 hover:text-slate-200">Models</a>
            </nav>
          </div>
        </header>

        {/* Two-column shell: sidebar + main */}
        <div className="mx-auto flex max-w-screen-2xl gap-8 px-4 sm:px-6 lg:px-8">
          <Sidebar />
          <main className="min-w-0 flex-1 py-6">
            {children}
          </main>
        </div>
      </body>
    </html>
  );
}
