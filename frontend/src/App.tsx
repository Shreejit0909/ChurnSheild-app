import { useState } from "react";
import { Toaster } from "@/components/ui/toaster";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { ThemeProvider } from "@/components/theme-provider";
import { SidebarProvider } from "@/components/ui/sidebar";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route, useLocation } from "react-router-dom";
import { SplashScreen } from "@/components/splash-screen";
import { AppSidebar } from "@/components/app-sidebar";
import { DashboardHeader } from "@/components/dashboard-header";
import GetStarted from "./pages/GetStarted";
import Dashboard from "./pages/Dashboard";
import Predictions from "./pages/Predictions";
import BatchPredictions from "./pages/BatchPredictions";
import Logs from "./pages/Logs";
import Analytics from "./pages/Analytics";
import Profile from "./pages/Profile";   // ✅ fixed
import NotFound from "./pages/NotFound";
import Retention from "./pages/Retention";

const queryClient = new QueryClient();

const AppContent = () => {
  const location = useLocation();
  const isGetStartedPage = location.pathname === "/";

  if (isGetStartedPage) {
    return (
      <Routes>
        <Route path="/" element={<GetStarted />} />
        <Route path="/dashboard" element={<Dashboard />} />
        <Route path="/predictions" element={<Predictions />} />
        <Route path="/batch" element={<BatchPredictions />} />
        <Route path="/logs" element={<Logs />} />
        <Route path="/retention" element={<Retention />} /> 
        <Route path="/analytics" element={<Analytics />} />
        <Route path="/profile" element={<Profile />} />   {/* ✅ fixed */}
        <Route path="*" element={<NotFound />} />
      </Routes>
    );
  }

  return (
    <SidebarProvider>
      <div className="min-h-screen flex w-full bg-gradient-subtle">
        <AppSidebar />
        <div className="flex-1 flex flex-col">
          <DashboardHeader />
          <main className="flex-1">
            <Routes>
              <Route path="/" element={<GetStarted />} />
              <Route path="/dashboard" element={<Dashboard />} />
              <Route path="/predictions" element={<Predictions />} />
              <Route path="/batch" element={<BatchPredictions />} />
              <Route path="/logs" element={<Logs />} />
              <Route path="/retention" element={<Retention />} />
              <Route path="/analytics" element={<Analytics />} />
              <Route path="/profile" element={<Profile />} /> {/* ✅ fixed */}
              <Route path="*" element={<NotFound />} />
            </Routes>
          </main>
        </div>
      </div>
    </SidebarProvider>
  );
};

const App = () => {
  const [showSplash, setShowSplash] = useState(true);
  const handleSplashComplete = () => setShowSplash(false);

  if (showSplash) return <SplashScreen onComplete={handleSplashComplete} />;

  return (
    <QueryClientProvider client={queryClient}>
      <ThemeProvider attribute="class" defaultTheme="light" enableSystem>
        <TooltipProvider>
          <BrowserRouter>
            <AppContent />
          </BrowserRouter>
          <Toaster />
          <Sonner />
        </TooltipProvider>
      </ThemeProvider>
    </QueryClientProvider>
  );
};

export default App;
