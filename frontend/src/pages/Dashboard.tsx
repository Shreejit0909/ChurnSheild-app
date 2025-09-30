import { useEffect, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import {
  AreaChart, Area, PieChart, Pie, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis
} from "recharts";
import { Users, AlertTriangle, DollarSign, Zap } from "lucide-react";
import { motion } from "framer-motion";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";

import { API } from "@/config";

const COLORS = ["#ef4444", "#facc15", "#22c55e"]; // High, Medium, Low

export default function Dashboard() {
  const [summary, setSummary] = useState<any | null>(null);
  const [churnTrend, setChurnTrend] = useState<any[]>([]);
  const [recent, setRecent] = useState<any[]>([]);
  const [topCustomers, setTopCustomers] = useState<any[]>([]);

  useEffect(() => {
    fetchSummary();
    fetchTrend();
    fetchRecent();
    fetchTop();
  }, []);

  async function fetchSummary() {
    const res = await fetch(`${API}/dashboard/summary`);
    setSummary(await res.json());
  }

  async function fetchTrend() {
    const res = await fetch(`${API}/dashboard/trend`);
    setChurnTrend(await res.json());
  }

  async function fetchRecent() {
    const res = await fetch(`${API}/dashboard/recent`);
    setRecent(await res.json());
  }

  async function fetchTop() {
    const res = await fetch(`${API}/dashboard/top`);
    setTopCustomers(await res.json());
  }

  const riskData = summary
    ? [
        { label: "High", value: summary.risk_counts?.high || 0 },
        { label: "Medium", value: summary.risk_counts?.medium || 0 },
        { label: "Low", value: summary.risk_counts?.low || 0 },
      ]
    : [];

  return (
    <div className="flex-1 space-y-8 p-8 relative">
      {/* Hero */}
      <div className="grid grid-cols-2 gap-6 items-center">
        <div>
          <h1 className="text-4xl font-extrabold tracking-tight">
            AI-Powered{" "}
            <motion.span
              className="bg-gradient-to-r from-pink-500 via-purple-500 to-indigo-500 bg-clip-text text-transparent"
              animate={{ backgroundPosition: ["0% 50%", "100% 50%"] }}
              transition={{ repeat: Infinity, duration: 4, ease: "linear" }}
            >
              Churn Prediction
            </motion.span>{" "}
            at Your Fingertips
          </h1>
          <p className="mt-3 text-lg text-muted-foreground">
            Predict. Prevent. Retain. All in one dashboard.
          </p>
        </div>
        <div className="rounded-xl bg-gradient-to-r from-indigo-500/20 to-pink-500/20 p-6 glass">
          <img src="src/assets/churn.png" className="w-full" />
        </div>
      </div>

      {/* KPI Metrics */}
      <div className="grid grid-cols-4 gap-4">
        <MetricCard icon={<Users />} title="Customers Analyzed" value={summary?.customers || 0} suffix="+" />
        <MetricCard icon={<AlertTriangle />} title="High Risk Customers" value={`${summary?.high_risk_pct || 0}%`} />
        <MetricCard icon={<DollarSign />} title="Revenue at Risk" value={`$${summary?.revenue_at_risk || 0}`} />
        <MetricCard icon={<Zap />} title="Actions (Pending/Taken)" value={`${summary?.actions_pending || 0} / ${summary?.actions_taken || 0}`} />
      </div>

      {/* Risk Overview */}
      <div className="grid grid-cols-3 gap-6">
        <Card>
          <CardHeader><CardTitle>Risk Split</CardTitle></CardHeader>
          <CardContent className="h-80 flex items-center justify-center">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={riskData}
                  dataKey="value"
                  nameKey="label"
                  innerRadius={70}
                  outerRadius={120} // bigger radius
                  paddingAngle={3}
                  label={({ name, value }) => `${name}: ${value}`}
                >
                  <Cell fill="#ef4444" /> {/* High */}
                  <Cell fill="#facc15" /> {/* Medium */}
                  <Cell fill="#22c55e" /> {/* Low */}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        <Card className="col-span-2">
          <CardHeader><CardTitle>Churn Trend (6 months)</CardTitle></CardHeader>
          <CardContent className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={churnTrend}>
                <defs>
                  <linearGradient id="colorChurn" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#ef4444" stopOpacity={0.8}/>
                    <stop offset="95%" stopColor="#ef4444" stopOpacity={0}/>
                  </linearGradient>
                </defs>
                <XAxis dataKey="month" />
                <YAxis />
                <Tooltip />
                <Area type="monotone" dataKey="churn_rate" stroke="#ef4444" fillOpacity={1} fill="url(#colorChurn)" />
              </AreaChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </div>

      {/* Top Customers Preview */}
      <Card>
        <CardHeader><CardTitle>Top Customers at Risk</CardTitle></CardHeader>
        <CardContent>
          <table className="w-full text-sm">
            <thead>
              <tr className="text-left">
                <th>ID</th>
                <th>Customer</th>
                <th>Probability</th>
                <th>Risk</th>
                <th>Revenue</th>
                <th>Action</th>
              </tr>
            </thead>
            <tbody>
              {topCustomers.slice(0,5).map((c, i) => (
                <tr key={i} className="border-t">
                  <td>{c.id}</td>
                  <td>{c.name}</td>
                  <td>
                    <div className="w-32 bg-gray-200 rounded h-2">
                      <div
                        className={`h-2 rounded ${c.risk==="high"?"bg-red-500":c.risk==="medium"?"bg-yellow-500":"bg-green-500"}`}
                        style={{ width: `${c.churn_probability*100}%` }}
                      />
                    </div>
                  </td>
                  <td><Badge variant={c.risk==="high"?"destructive":c.risk==="medium"?"secondary":"default"}>{c.risk}</Badge></td>
                  <td>${c.revenue}</td>
                  <td>
                    {/* ✅ Popup for Actions */}
                    <Dialog>
                      <DialogTrigger asChild>
                        <Button size="sm">View</Button>
                      </DialogTrigger>
                      <DialogContent>
                        <DialogHeader>
                          <DialogTitle>Actions for {c.name || c.id}</DialogTitle>
                        </DialogHeader>
                        <div className="space-y-3">
                          <p className="text-sm text-muted-foreground">
                            Reason: {c.reason}
                          </p>
                          {c.actions && c.actions.length > 0 ? (
                            c.actions.map((a: string, idx: number) => (
                              <p key={idx} className="text-sm">✅ {a}</p>
                            ))
                          ) : (
                            <p className="text-muted-foreground text-sm">
                              No actions taken yet.
                            </p>
                          )}
                          {/* Show action status */}
                          <p className="text-xs text-muted-foreground">
                            Status: {c.action_status || "pending"}
                          </p>
                        </div>
                      </DialogContent>
                    </Dialog>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>  
        </CardContent>
      </Card>

      {/* Recent Activity */}
      <Card>
        <CardHeader><CardTitle>Recent Activity</CardTitle></CardHeader>
        <CardContent>
          <div className="space-y-3">
            {recent.map((r, i) => (
              <div key={i} className="flex items-center gap-3">
                <Badge>{r.type}</Badge>
                <p>{r.message}</p>
                <span className="text-xs text-muted-foreground">{new Date(r.time).toLocaleString()}</span>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      <footer className="text-center py-6 text-sm text-muted-foreground border-t mt-8">
        © {new Date().getFullYear()} RetainAI — <span className="italic">“Predict. Prevent. Retain.”</span>
      </footer>

    </div>
  );
}

function MetricCard({ icon, title, value, suffix }: any) {
  return (
    <Card className="glass">
      <CardContent className="flex items-center gap-4 p-6">
        <div className="p-3 rounded-full bg-gradient-to-r from-indigo-500 to-pink-500 text-white">
          {icon}
        </div>
        <div>
          <p className="text-sm text-muted-foreground">{title}</p>
          <h3 className="text-2xl font-bold">{value}{suffix}</h3>
        </div>
      </CardContent>
    </Card>
  );
}
