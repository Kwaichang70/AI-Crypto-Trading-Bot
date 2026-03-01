/** @type {import('next').NextConfig} */
const nextConfig = {
  // Required for Docker multi-stage build (Dockerfile.ui copies .next/standalone)
  output: "standalone",

  // Strict React mode for catching subtle bugs early
  reactStrictMode: true,

  // Rewrites proxy /api/* to the FastAPI backend, keeping the browser
  // origin on the UI host and avoiding CORS issues in development.
  // In production, the reverse-proxy (nginx / Docker Compose) handles this.
  async rewrites() {
    const apiBase = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";
    return [
      {
        source: "/api/:path*",
        destination: `${apiBase}/:path*`,
      },
    ];
  },
};

export default nextConfig;
