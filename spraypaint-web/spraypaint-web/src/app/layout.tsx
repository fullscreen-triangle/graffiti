import type { Metadata } from "next";
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
