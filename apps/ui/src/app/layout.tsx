import type { Metadata, Viewport } from "next";
import { Inter } from "next/font/google";
import { Sidebar } from "@/components/layout/sidebar";
import { MobileNav } from "@/components/layout/mobile-nav";
import { ThemeProvider } from "@/components/theme-provider";
import { ThemeToggle } from "@/components/ui/theme-toggle";
import { ToastProvider } from "@/components/ui/toast";
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
      <body className="min-h-screen font-sans antialiased">
        <ThemeProvider>
          <ToastProvider>
            {/* Top navigation bar */}
            <header className="sticky top-0 z-40 border-b border-slate-200 bg-white/80 backdrop-blur-sm dark:border-slate-800 dark:bg-slate-950/80">
              <div className="mx-auto flex h-14 max-w-screen-2xl items-center justify-between px-4 sm:px-6 lg:px-8">
                <span className="text-sm font-semibold tracking-tight text-slate-900 dark:text-slate-100">
                  Trading Bot
                </span>
                <div className="flex items-center gap-2">
                  <MobileNav />
                  <ThemeToggle />
                </div>
              </div>
            </header>

            {/* Two-column shell: sidebar + main */}
            <div className="mx-auto flex max-w-screen-2xl gap-8 px-4 sm:px-6 lg:px-8">
              <Sidebar />
              <main className="min-w-0 flex-1 py-6">
                {children}
              </main>
            </div>
          </ToastProvider>
        </ThemeProvider>
      </body>
    </html>
  );
}
