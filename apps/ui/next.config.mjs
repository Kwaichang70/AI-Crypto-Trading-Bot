/** @type {import('next').NextConfig} */
const nextConfig = {
  // Required for Docker multi-stage build (Dockerfile.ui copies .next/standalone)
  output: "standalone",

  // Strict React mode for catching subtle bugs early
  reactStrictMode: true,

  // CF-97-37 (37-B): NO rewrites and NO `env:` entry. The browser reaches the API
  // only through the BFF route handler (src/app/bff/api/v1/[...path]); the old
  // /api/v1/:path* rewrite was a keyless proxy and is deleted. Server secrets
  // (API_KEY, INTERNAL_API_URL, INTERNAL_ADMIN_API_KEY) are runtime environment
  // only: never listed here, never a Dockerfile ARG (pinned by tests).
};

export default nextConfig;
