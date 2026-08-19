import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Browser Agent",
  description: "An LLM agent that operates a real web browser.",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
