import { NavLink, useLocation } from "react-router-dom";
import { 
  BarChart3, 
  Brain, 
  FileText, 
  Home, 
  Settings, 
  Upload,
  TrendingUp,
  Megaphone,
  User
} from "lucide-react";
import {
  Sidebar,
  SidebarContent,
  SidebarGroup,
  SidebarGroupContent,
  SidebarGroupLabel,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
  useSidebar,
} from "@/components/ui/sidebar";
import logo from "@/assets/Logo.png";

const navigationItems = [
  { title: "Dashboard", url: "/dashboard", icon: Home },
  { title: "Predictions", url: "/predictions", icon: Brain },
  { title: "Batch Predictions", url: "/batch", icon: Upload },
  { title: "Logs", url: "/logs", icon: FileText },
  { title: "Retention", url: "/retention", icon: Megaphone },  // ✅ fixed: was href → url
  { title: "Profile", url: "/profile", icon: User },
  

];

export function AppSidebar() {
  const { state } = useSidebar();
  const location = useLocation();
  const collapsed = state === "collapsed";

  const getNavClassName = (isActive: boolean) =>
    `flex items-center gap-3 px-3 py-2 rounded-lg transition-colors duration-200 ${
      isActive
        ? "bg-brand-primary text-white shadow-brand"
        : "text-muted-foreground hover:text-foreground hover:bg-accent"
    }`;

  return (
    <Sidebar className={collapsed ? "w-16" : "w-64"} collapsible="icon">
      <SidebarContent className="bg-gradient-card border-r shadow-medium">
        {/* Logo */}
        <div className="p-6 border-b">
          <div className="flex items-center gap-3">
            <img src={logo} alt="ChurnShield AI" className="w-8 h-8" />
            {!collapsed && (
              <div>
                <h2 className="text-lg font-bold bg-gradient-primary bg-clip-text text-transparent">
                  ChurnShield AI
                </h2>
                <p className="text-xs text-muted-foreground">Churn Prediction & Retention</p>
              </div>
            )}
          </div>
        </div>

        <SidebarGroup>
          <SidebarGroupLabel>Navigation</SidebarGroupLabel>
          <SidebarGroupContent>
            <SidebarMenu>
              {navigationItems.map((item) => {
                const isActive = location.pathname === item.url;
                return (
                  <SidebarMenuItem key={item.title}>
                    <SidebarMenuButton asChild>
                      <NavLink
                        to={item.url}
                        className={getNavClassName(isActive)}
                      >
                        <item.icon className="w-5 h-5" />
                        {!collapsed && (
                          <span className="font-medium">{item.title}</span>
                        )}
                      </NavLink>
                    </SidebarMenuButton>
                  </SidebarMenuItem>
                );
              })}
            </SidebarMenu>
          </SidebarGroupContent>
        </SidebarGroup>
      </SidebarContent>
    </Sidebar>
  );
}
