import type { NextConfig } from "next";

// Static export: the whole UI is embedded into the `spraypaint` binary via
// rust-embed and served by `spraypaint serve` from the user's own machine.
// There is no Node server at runtime, so no SSR, no image optimiser, no
// route handlers — every page must be prerenderable to plain files.
const nextConfig: NextConfig = {
  output: "export",
  distDir: "out",
  images: { unoptimized: true },
  trailingSlash: true,
};

export default nextConfig;
