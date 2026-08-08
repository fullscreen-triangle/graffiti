import type { Metadata } from "next";
// KaTeX is gone: the only math in this UI was in the old ReportTab, which
// described a `.grf` execution the binary never performs. Its stylesheet dragged
// 1.08 MB of webfonts into an export that ships inside the executable.
import "./globals.css";

export const metadata: Metadata = {
  title: "spraypaint — semantic causal propagation",
  description: "Full-text search ranked by BM25 within scenes, allocated by water-filling",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark">
      <body className="antialiased">{children}</body>
    </html>
  );
}
