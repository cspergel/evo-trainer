import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Evolve-Trader Dashboard",
  description: "System visibility for the evolutionary trading system",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="bg-gray-950 text-gray-100 min-h-screen">
        <nav className="bg-gray-900 border-b border-gray-800 px-6 py-3">
          <div className="flex items-center justify-between">
            <h1 className="text-lg font-bold text-white">
              Evolve-Trader
            </h1>
            <span className="text-xs text-gray-500">Phase 5 — Read-Only Dashboard</span>
          </div>
        </nav>
        <main className="p-6">{children}</main>
      </body>
    </html>
  );
}
