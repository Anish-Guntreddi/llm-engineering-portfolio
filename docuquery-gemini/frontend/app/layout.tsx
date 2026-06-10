import type { Metadata } from "next";
import "./globals.css";
import Header from "@/components/Header";
import { ToastProvider } from "@/components/Toast";

export const metadata: Metadata = {
  title: "DocuQuery-Gemini",
  description: "Ask questions over your documents with traceable citations.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>
        <ToastProvider>
          <div className="flex min-h-screen flex-col">
            <Header />
            <main className="flex-1">{children}</main>
          </div>
        </ToastProvider>
      </body>
    </html>
  );
}
