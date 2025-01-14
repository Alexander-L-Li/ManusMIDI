import localFont from "next/font/local";
import "./globals.css";

export const metadata = {
  title: "Manus MIDI"
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body className="font-sans">
        {children}
      </body>
    </html>
  );
}
