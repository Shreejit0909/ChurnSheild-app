import { useQuery } from "@tanstack/react-query";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  LineChart,
  Line,
  PieChart,
  Pie,
  Cell,
  Legend,
} from "recharts";
import { TrendingUp, BarChart3 } from "lucide-react";
import { Skeleton } from "@/components/ui/skeleton";

type SegmentStat = { segment: string; churnRate: number };
type RegionStat = { name: string; value: number; color: string };
type TrendStat = { month: string; churnRate: number };
type RiskDistribution = { risk: string; count: number };
type PlanStat = { plan: string; churnRate: number };
type SizeStat = { size: string; churnRate: number };

type AnalyticsResponse = {
  churn_by_segment: SegmentStat[];
  churn_by_region: RegionStat[];
  churn_trend: TrendStat[];
  risk_distribution?: RiskDistribution[];
  churn_by_plan?: PlanStat[];
  churn_by_size?: SizeStat[];
  summary?: {
    total_predictions: number;
    high_risk_pct: number;
    avg_churn_prob: number;
    top_region?: string;
  };
};

import { API } from "@/config";
const API_URL = `${API}/analytics`;


export default function Analytics() {
  const { data, isLoading } = useQuery<AnalyticsResponse>({
    queryKey: ["analytics"],
    queryFn: async () => {
      const res = await fetch(API_URL);
      if (!res.ok) throw new Error("Failed to fetch analytics");
      return res.json();
    },
    refetchInterval: 10000,
  });

  const ChartSkeleton = () => (
    <div className="h-80 flex items-center justify-center">
      <Skeleton className="w-full h-full rounded-xl" />
    </div>
  );

  const riskColors = {
    high: "#ef4444",
    medium: "#facc15",
    low: "#22c55e",
  };

  return (
    <div className="flex-1 space-y-6 p-6">
      {/* Header */}
      <div className="flex items-center gap-3">
        <BarChart3 className="h-8 w-8 text-brand-primary" />
        <div>
          <h1 className="text-3xl font-bold">Analytics Dashboard</h1>
          <p className="text-muted-foreground">
            Deep insights into churn patterns and model performance
          </p>
        </div>
      </div>

      {/* Summary Cards */}
      {data?.summary && (
        <div className="grid gap-6 md:grid-cols-4">
          <Card className="shadow-soft bg-gradient-card">
            <CardHeader><CardTitle>Total Predictions</CardTitle></CardHeader>
            <CardContent className="text-3xl font-bold">
              {data.summary.total_predictions}
            </CardContent>
          </Card>

          <Card className="shadow-soft bg-gradient-card">
            <CardHeader><CardTitle>High Risk %</CardTitle></CardHeader>
            <CardContent className="text-3xl font-bold text-destructive">
              {data.summary.high_risk_pct.toFixed(1)}%
            </CardContent>
          </Card>

          <Card className="shadow-soft bg-gradient-card">
            <CardHeader><CardTitle>Avg Churn Probability</CardTitle></CardHeader>
            <CardContent className="text-3xl font-bold text-brand-primary">
              {(data.summary.avg_churn_prob * 100).toFixed(1)}%
            </CardContent>
          </Card>

          <Card className="shadow-soft bg-gradient-card">
            <CardHeader><CardTitle>Top Region</CardTitle></CardHeader>
            <CardContent className="text-2xl font-semibold">
              {data.summary.top_region || "N/A"}
            </CardContent>
          </Card>
        </div>
      )}

      {/* Risk Distribution */}
      <Card className="bg-gradient-card shadow-soft">
        <CardHeader><CardTitle>Risk Distribution</CardTitle></CardHeader>
        <CardContent>
          {isLoading ? (
            <ChartSkeleton />
          ) : data?.risk_distribution?.length ? (
            <ResponsiveContainer width="100%" height={300}>
              <PieChart>
                <Pie
                  data={data.risk_distribution}
                  dataKey="count"
                  nameKey="risk"
                  cx="50%"
                  cy="50%"
                  outerRadius={80}
                  label
                >
                  {data.risk_distribution.map((entry, i) => (
                    <Cell key={i} fill={riskColors[entry.risk] || "#8884d8"} />
                  ))}
                </Pie>
                <Tooltip />
                <Legend />
              </PieChart>
            </ResponsiveContainer>
          ) : (
            <p className="text-muted-foreground">No risk distribution data.</p>
          )}
        </CardContent>
      </Card>

      {/* Region Distribution */}
      <Card className="bg-gradient-card shadow-soft">
        <CardHeader><CardTitle>Customer Distribution by Region</CardTitle></CardHeader>
        <CardContent>
          {isLoading ? (
            <ChartSkeleton />
          ) : data?.churn_by_region?.length ? (
            <ResponsiveContainer width="100%" height={300}>
              <PieChart>
                <Pie
                  data={data.churn_by_region}
                  dataKey="value"
                  nameKey="name"
                  cx="50%"
                  cy="50%"
                  outerRadius={80}
                  label
                >
                  {data.churn_by_region.map((entry, i) => (
                    <Cell key={i} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip />
                <Legend />
              </PieChart>
            </ResponsiveContainer>
          ) : (
            <p className="text-muted-foreground">No region distribution data.</p>
          )}
        </CardContent>
      </Card>

      {/* Churn Trend */}
      <Card className="bg-gradient-card shadow-soft">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <TrendingUp className="h-5 w-5" />
            Churn Rate Trend
          </CardTitle>
        </CardHeader>
        <CardContent>
          {isLoading ? (
            <ChartSkeleton />
          ) : data?.churn_trend?.length ? (
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={data.churn_trend}>
                <CartesianGrid strokeDasharray="3 3" stroke="#ccc" />
                <XAxis dataKey="month" />
                <YAxis />
                <Tooltip formatter={(v) => [`${Number(v).toFixed(1)}%`, "Churn Rate"]} />
                <Line type="monotone" dataKey="churnRate" stroke="#3b82f6" strokeWidth={3} dot />
              </LineChart>
            </ResponsiveContainer>
          ) : (
            <p className="text-muted-foreground">No churn trend data.</p>
          )}
        </CardContent>
      </Card>

      {/* Churn by Plan & Size (side by side) */}
      <div className="grid gap-6 md:grid-cols-2">
        <Card className="bg-gradient-card shadow-soft">
          <CardHeader><CardTitle>Churn by Product/Plan</CardTitle></CardHeader>
          <CardContent>
            {isLoading ? (
              <ChartSkeleton />
            ) : data?.churn_by_plan?.length ? (
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={data.churn_by_plan}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="plan" />
                  <YAxis />
                  <Tooltip formatter={(v) => [`${Number(v).toFixed(1)}%`, "Churn Rate"]} />
                  <Bar dataKey="churnRate" fill="#6366f1" />
                </BarChart>
              </ResponsiveContainer>
            ) : (
              <p className="text-muted-foreground">No plan churn data.</p>
            )}
          </CardContent>
        </Card>

        <Card className="bg-gradient-card shadow-soft">
          <CardHeader><CardTitle>Churn by Customer Size</CardTitle></CardHeader>
          <CardContent>
            {isLoading ? (
              <ChartSkeleton />
            ) : data?.churn_by_size?.length ? (
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={data.churn_by_size}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="size" />
                  <YAxis />
                  <Tooltip formatter={(v) => [`${Number(v).toFixed(1)}%`, "Churn Rate"]} />
                  <Bar dataKey="churnRate" fill="#10b981" />
                </BarChart>
              </ResponsiveContainer>
            ) : (
              <p className="text-muted-foreground">No size churn data.</p>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
