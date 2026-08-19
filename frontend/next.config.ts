import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Screenshots are served by the FastAPI backend off disk. Proxying them
  // through Next keeps every URL same-origin, so the <img> tags in the browser
  // preview need no CORS handling and no absolute backend URL baked in.
  async rewrites() {
    const backend = process.env.BACKEND_URL ?? "http://127.0.0.1:8000";
    return [
      { source: "/api/:path*", destination: `${backend}/api/:path*` },
      { source: "/screenshots/:path*", destination: `${backend}/screenshots/:path*` },
    ];
  },
};

export default nextConfig;
