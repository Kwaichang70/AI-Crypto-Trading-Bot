import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./src/pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  darkMode: "class",
  theme: {
    extend: {
      colors: {
        // Trading-specific semantic tokens
        profit: {
          DEFAULT: "#22c55e", // green-500
          muted: "#16a34a",   // green-600
        },
        loss: {
          DEFAULT: "#ef4444", // red-500
          muted: "#dc2626",   // red-600
        },
        neutral: {
          trading: "#94a3b8", // slate-400
        },
      },
      fontFamily: {
        sans: ["var(--font-inter)", "system-ui", "sans-serif"],
        mono: ["var(--font-mono)", "ui-monospace", "monospace"],
      },
      keyframes: {
        "toast-in": {
          "0%": { opacity: "0", transform: "translateX(100%)" },
          "100%": { opacity: "1", transform: "translateX(0)" },
        },
      },
      animation: {
        "toast-in": "toast-in 0.25s ease-out forwards",
      },
    },
  },
  plugins: [],
};

export default config;
