/** @type {import('next').NextConfig} */
const nextConfig = {
  // Required for Docker multi-stage build (Dockerfile.ui copies .next/standalone)
  output: "standalone",

  // Strict React mode for catching subtle bugs early
  reactStrictMode: true,

  // Rewrites proxy /api/v1/* to the FastAPI backend, keeping the browser
  // origin on the UI host and avoiding CORS issues in development.
  //
  // CRITICAL: source MUST be /api/v1/:path* (not /api/:path*) so that
  // /api/auth/* (NextAuth file-based routes added in Sprint 50 cycle 2)
  // are NOT intercepted by this rewrite. /api/auth/* must reach Next.js's
  // App Router file-based handler at /app/api/auth/[...nextauth]/route.ts.
  //
  // In production, Caddy reverse-proxy already routes /api/v1/* directly
  // to api:8000, so this Next.js rewrite is only used for local `next dev`
  // and as a fallback for server-side fetches from Next.js itself.
  async rewrites() {
    const apiBase = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";
    return [
      {
        source: "/api/v1/:path*",
        destination: `${apiBase}/api/v1/:path*`,
      },
    ];
  },
};

export default nextConfig;
