import type { Metadata } from "next";
import { Poppins, Space_Mono } from "next/font/google";
import "./globals.css";
import Sidebar from "@/components/Sidebar";
import Copilot from "@/components/Copilot";

const poppins = Poppins({
  variable: "--font-poppins",
  subsets: ["latin"],
  weight: ["300", "400", "500", "600", "700", "800"],
});

const spaceMono = Space_Mono({
  variable: "--font-space-mono",
  subsets: ["latin"],
  weight: ["400", "700"],
});

export const metadata: Metadata = {
  title: "CURA AI | Advanced Drug Discovery",
  description: "AI-powered platform for molecule analysis and drug discovery.",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body className={`${poppins.variable} ${spaceMono.variable} antialiased`}
        style={{ background: "var(--bg)", color: "var(--text-dark)", minHeight: "100vh" }}
      >
        <div className="flex min-h-screen">
          <Sidebar />
          <main className="flex-1 min-h-screen overflow-y-auto px-8 py-10">
            {children}
          </main>
          <Copilot />
        </div>
      </body>
    </html>
  );
}
