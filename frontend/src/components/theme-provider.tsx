import * as React from "react";
import { ThemeProvider as NextThemesProvider } from "next-themes";

export function ThemeProvider({ children, ...props }: any) {
  return (
    <NextThemesProvider
      attribute="class"
      defaultTheme="dark"   // 👈 sets dark as default
      enableSystem={false}  // 👈 ignores system light/dark preference
      {...props}
    >
      {children}
    </NextThemesProvider>
  );
}
