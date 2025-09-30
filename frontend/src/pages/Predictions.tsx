// src/pages/Predictions.tsx
import { useState } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Badge } from "@/components/ui/badge";
import { Brain, TrendingUp } from "lucide-react";
import {
  PieChart,
  Pie,
  Cell,
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  Tooltip,
  CartesianGrid,
  YAxis,
} from "recharts";

type FormState = {
  // Usage & Engagement
  monthly_usage: string;
  avg_session_duration: string;
  days_since_last_login: string;
  features_used: string;
  usage_trend: string; // numeric or categorical

  // Support & Satisfaction
  support_tickets_90d: string;
  escalated_tickets: string;
  avg_resolution_time: string;
  support_satisfaction: string;

  // Billing & Contract
  payment_failures_90d: string;
  contract_type: string;
  days_to_renewal: string;
  monthly_revenue: string;

  // Profile
  segment: string;
  company_size: string;
  region: string;
};

type TopReason = { feature: string; importance?: number; contribution?: number };
type PredictionResponse = {
  customer_id?: string;
  churn_probability: number;
  risk_label: string;
  confidence?: number;
  top_reasons?: TopReason[];
  warnings?: string[];
  timestamp?: string;
};

const DEFAULT_FORM: FormState = {
  monthly_usage: "",
  avg_session_duration: "",
  days_since_last_login: "",
  features_used: "",
  usage_trend: "1.0",

  support_tickets_90d: "",
  escalated_tickets: "",
  avg_resolution_time: "",
  support_satisfaction: "",

  payment_failures_90d: "",
  contract_type: "",
  days_to_renewal: "",
  monthly_revenue: "",

  segment: "",
  company_size: "",
  region: "",
};

export default function Predictions() {
  const queryClient = useQueryClient();

  const [activeTab, setActiveTab] = useState<"usage" | "support" | "billing" | "profile">("usage");
  const [form, setForm] = useState<FormState>(DEFAULT_FORM);
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  const [clientErrors, setClientErrors] = useState<string[]>([]);

  // ------------------- Mutation -------------------
  const API_URL = import.meta.env.VITE_API_URL;
  const mutation = useMutation<PredictionResponse, Error, any>({
    mutationFn: async (payload: any) => {
      const res = await fetch(`${API_URL}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!res.ok) {
        const text = await res.text();
        throw new Error(text || "Prediction failed");
      }
      return res.json();
    },
    onSuccess: (data) => {
      setPrediction(data);
      setClientErrors([]);
      queryClient.invalidateQueries({ queryKey: ["logs"] });
      queryClient.invalidateQueries({ queryKey: ["analytics"] });
    },
    onError: (err) => {
      setPrediction(null);
      setClientErrors([err.message]);
    },
  });

  // ------------------- Validation -------------------
  const validate = (): string[] => {
    const errs: string[] = [];

    // required basics
    if (!form.contract_type) errs.push("Contract type is required");
    if (!form.segment) errs.push("Customer segment is required");
    if (!form.region) errs.push("Region is required");

    // numeric checks (allow empty for optional - backend will default)
    const nonNegFields = [
      ["monthly_usage", "Monthly usage"],
      ["avg_session_duration", "Avg session duration"],
      ["days_since_last_login", "Days since last login"],
      ["features_used", "Features used"],
      ["support_tickets_90d", "Support tickets"],
      ["escalated_tickets", "Escalated tickets"],
      ["avg_resolution_time", "Avg resolution time"],
      ["payment_failures_90d", "Payment failures"],
      ["days_to_renewal", "Days to renewal"],
      ["monthly_revenue", "Monthly revenue"],
      ["company_size", "Company size"],
    ] as [keyof FormState, string][];

    for (const [k, label] of nonNegFields) {
      const v = (form[k] ?? "").toString().trim();
      if (v !== "") {
        const n = Number(v);
        if (Number.isNaN(n) || n < 0) errs.push(`${label} must be a number ≥ 0`);
      }
    }

    // support satisfaction range
    if (form.support_satisfaction) {
      const s = Number(form.support_satisfaction);
      if (Number.isNaN(s) || s < 1 || s > 5) errs.push("Support satisfaction must be 1–5");
    }

    return errs;
  };

  // ------------------- Helpers -------------------
  const setField = (k: keyof FormState, v: string) => setForm({ ...form, [k]: v });

  const getRiskColor = (risk: string) => {
    switch ((risk || "").toLowerCase()) {
      case "high":
        return "destructive";
      case "medium":
        return "secondary";
      case "low":
        return "default";
      default:
        return "default";
    }
  };

  const normalizePayloadFeatures = () => {
    // Build features object with typed values where possible.
    return {
      monthly_usage: form.monthly_usage === "" ? null : Number(form.monthly_usage),
      avg_session_duration: form.avg_session_duration === "" ? null : Number(form.avg_session_duration),
      days_since_last_login: form.days_since_last_login === "" ? null : Number(form.days_since_last_login),
      features_used: form.features_used === "" ? null : Number(form.features_used),
      usage_trend: parseFloat(form.usage_trend) || null,

      support_tickets_90d: form.support_tickets_90d === "" ? null : Number(form.support_tickets_90d),
      escalated_tickets: form.escalated_tickets === "" ? null : Number(form.escalated_tickets),
      avg_resolution_time: form.avg_resolution_time === "" ? null : Number(form.avg_resolution_time),
      support_satisfaction: form.support_satisfaction === "" ? null : Number(form.support_satisfaction),

      payment_failures_90d: form.payment_failures_90d === "" ? null : Number(form.payment_failures_90d),
      contract_type: form.contract_type || null,
      days_to_renewal: form.days_to_renewal === "" ? null : Number(form.days_to_renewal),
      monthly_revenue: form.monthly_revenue === "" ? null : Number(form.monthly_revenue),

      segment: form.segment || null,
      company_size: form.company_size === "" ? null : Number(form.company_size),
      region: form.region || null,
    };
  };

  // normalize top_reasons to {feature, importance}
  const reasonsToBars = (top_reasons?: TopReason[]) => {
    if (!top_reasons || top_reasons.length === 0) return [];
    // some backends may return 'contribution' or 'importance'
    const arr = top_reasons.map((r) => ({
      feature: r.feature,
      value: (r.importance ?? r.contribution ?? 0) * 1, // use as-is (backend likely gives small floats)
    }));
    // convert to percent-like numbers for bar chart (if tiny values, scale)
    const maxVal = Math.max(...arr.map((a) => Math.abs(a.value)), 1e-6);
    // scale so largest bar near 100
    const scale = maxVal < 1 ? 100 : 1;
    return arr.map((a) => ({ feature: a.feature, value: Math.abs(a.value) * scale }));
  };

  // ------------------- Submit -------------------
  const onSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    setPrediction(null);
    const errs = validate();
    if (errs.length) {
      setClientErrors(errs);
      return;
    }

    const payload = {
      customer_id: `UI_USER_${Date.now()}`,
      features: normalizePayloadFeatures(),
    };

    mutation.mutate(payload);
  };

  // ------------------- Render -------------------
  const donutData = prediction
    ? [
        { name: "churn", value: prediction.churn_probability },
        { name: "stay", value: 1 - prediction.churn_probability },
      ].map((d) => ({ ...d, value: Math.round(d.value * 1000) / 10 })) // one decimal percent scale
    : [{ name: "empty", value: 1 }];

  const donutColors = prediction
    ? prediction.churn_probability >= 0.5
      ? ["#ef4444", "#f3f4f6"]
      : prediction.churn_probability >= 0.25
      ? ["#f59e0b", "#f3f4f6"]
      : ["#10b981", "#f3f4f6"]
    : ["#94a3b8", "#f3f4f6"];

  const topBars = reasonsToBars(prediction?.top_reasons);

  return (
    <div className="flex-1 space-y-6 p-6">
      <div className="flex items-center gap-3">
        <Brain className="h-8 w-8 text-brand-primary" />
        <div>
          <h1 className="text-3xl font-bold">Customer Churn Prediction</h1>
          <p className="text-muted-foreground">
            Fill grouped customer fields to get an instant churn score
          </p>
        </div>
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        {/* Left: Tabs + Form */}
        <Card className="bg-gradient-card shadow-soft">
          <CardHeader>
            <CardTitle>Customer Features</CardTitle>
          </CardHeader>
          <CardContent>
            {/* Tabs */}
            <div className="flex gap-2 mb-4">
              <button
                className={`px-3 py-1 rounded ${activeTab === "usage" ? "bg-brand-primary text-white" : "bg-muted"}`}
                onClick={() => setActiveTab("usage")}
                type="button"
              >
                Usage & Engagement
              </button>
              <button
                className={`px-3 py-1 rounded ${activeTab === "support" ? "bg-brand-primary text-white" : "bg-muted"}`}
                onClick={() => setActiveTab("support")}
                type="button"
              >
                Support
              </button>
              <button
                className={`px-3 py-1 rounded ${activeTab === "billing" ? "bg-brand-primary text-white" : "bg-muted"}`}
                onClick={() => setActiveTab("billing")}
                type="button"
              >
                Billing
              </button>
              <button
                className={`px-3 py-1 rounded ${activeTab === "profile" ? "bg-brand-primary text-white" : "bg-muted"}`}
                onClick={() => setActiveTab("profile")}
                type="button"
              >
                Profile
              </button>
            </div>

            <form onSubmit={onSubmit} className="space-y-4">
              {/* Usage Tab */}
              {activeTab === "usage" && (
                <div className="grid gap-4 md:grid-cols-2">
                  <div className="space-y-2">
                    <Label htmlFor="monthly_usage">Monthly Usage (hours)</Label>
                    <Input id="monthly_usage" type="number" step="any" value={form.monthly_usage}
                      onChange={(e) => setField("monthly_usage", e.target.value)} />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="avg_session_duration">Avg Session Duration (mins)</Label>
                    <Input id="avg_session_duration" type="number" step="any" value={form.avg_session_duration}
                      onChange={(e) => setField("avg_session_duration", e.target.value)} />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="days_since_last_login">Days Since Last Login</Label>
                    <Input id="days_since_last_login" type="number" value={form.days_since_last_login}
                      onChange={(e) => setField("days_since_last_login", e.target.value)} />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="features_used">Features Used (count)</Label>
                    <Input id="features_used" type="number" value={form.features_used}
                      onChange={(e) => setField("features_used", e.target.value)} />
                  </div>
                  <div className="col-span-full space-y-2">
                    <Label htmlFor="usage_trend">Usage Trend (0.5 down — 1 stable — 1.5 up)</Label>
                    <Input id="usage_trend" type="number" step="0.1" value={form.usage_trend}
                      onChange={(e) => setField("usage_trend", e.target.value)} />
                  </div>
                </div>
              )}

              {/* Support Tab */}
              {activeTab === "support" && (
                <div className="grid gap-4 md:grid-cols-2">
                  <div className="space-y-2">
                    <Label htmlFor="support_tickets_90d">Support Tickets (90d)</Label>
                    <Input id="support_tickets_90d" type="number" value={form.support_tickets_90d}
                      onChange={(e) => setField("support_tickets_90d", e.target.value)} />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="escalated_tickets">Escalated Tickets</Label>
                    <Input id="escalated_tickets" type="number" value={form.escalated_tickets}
                      onChange={(e) => setField("escalated_tickets", e.target.value)} />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="avg_resolution_time">Avg Resolution Time (hours)</Label>
                    <Input id="avg_resolution_time" type="number" step="any" value={form.avg_resolution_time}
                      onChange={(e) => setField("avg_resolution_time", e.target.value)} />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="support_satisfaction">Support Satisfaction (1-5)</Label>
                    <Input id="support_satisfaction" type="number" step="0.1" value={form.support_satisfaction}
                      onChange={(e) => setField("support_satisfaction", e.target.value)} />
                  </div>
                </div>
              )}

              {/* Billing Tab */}
              {activeTab === "billing" && (
                <div className="grid gap-4 md:grid-cols-2">
                  <div className="space-y-2">
                    <Label htmlFor="payment_failures_90d">Payment Failures (90d)</Label>
                    <Input id="payment_failures_90d" type="number" value={form.payment_failures_90d}
                      onChange={(e) => setField("payment_failures_90d", e.target.value)} />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="contract_type">Contract Type</Label>
                    <Select value={form.contract_type} onValueChange={(v) => setField("contract_type", v)}>
                      <SelectTrigger>
                        <SelectValue placeholder="Select contract" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="monthly">Monthly</SelectItem>
                        <SelectItem value="quarterly">Quarterly</SelectItem>
                        <SelectItem value="annual">Annual</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="days_to_renewal">Days to Renewal</Label>
                    <Input id="days_to_renewal" type="number" value={form.days_to_renewal}
                      onChange={(e) => setField("days_to_renewal", e.target.value)} />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="monthly_revenue">Monthly Revenue ($)</Label>
                    <Input id="monthly_revenue" type="number" step="any" value={form.monthly_revenue}
                      onChange={(e) => setField("monthly_revenue", e.target.value)} />
                  </div>
                </div>
              )}

              {/* Profile Tab */}
              {activeTab === "profile" && (
                <div className="grid gap-4 md:grid-cols-2">
                  <div className="space-y-2">
                    <Label htmlFor="segment">Segment</Label>
                    <Select value={form.segment} onValueChange={(v) => setField("segment", v)}>
                      <SelectTrigger>
                        <SelectValue placeholder="Select segment" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="enterprise">Enterprise</SelectItem>
                        <SelectItem value="mid_market">Mid-Market</SelectItem>
                        <SelectItem value="small_business">Small Business</SelectItem>
                        <SelectItem value="startup">Startup</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="company_size">Company Size (employees)</Label>
                    <Input id="company_size" type="number" value={form.company_size}
                      onChange={(e) => setField("company_size", e.target.value)} />
                  </div>
                  <div className="col-span-full space-y-2">
                    <Label htmlFor="region">Region</Label>
                    <Select value={form.region} onValueChange={(v) => setField("region", v)}>
                      <SelectTrigger>
                        <SelectValue placeholder="Select region" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="North America">North America</SelectItem>
                        <SelectItem value="Europe">Europe</SelectItem>
                        <SelectItem value="Asia Pacific">Asia Pacific</SelectItem>
                        <SelectItem value="Latin America">Latin America</SelectItem>
                        <SelectItem value="Other">Other</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                </div>
              )}

              {/* Client errors */}
              {clientErrors.length > 0 && (
                <div className="text-sm text-destructive space-y-1">
                  {clientErrors.map((err, i) => (
                    <div key={i}>• {err}</div>
                  ))}
                </div>
              )}

              <div className="pt-4">
                <Button type="submit" disabled={mutation.isPending} className="w-full bg-gradient-primary">
                  {mutation.isPending ? "Predicting..." : "Predict Churn Risk"}
                </Button>
              </div>
            </form>
          </CardContent>
        </Card>

        {/* Right: Results */}
        <Card className="bg-gradient-card shadow-soft">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <TrendingUp className="h-5 w-5" />
              Prediction Results
            </CardTitle>
          </CardHeader>
          <CardContent>
            {mutation.isPending ? (
              <div className="text-center py-12 text-muted-foreground">Predicting...</div>
            ) : prediction ? (
              <div className="space-y-6">
                <div className="flex flex-col items-center">
                  <div style={{ width: 160, height: 160 }}>
                    <ResponsiveContainer width="100%" height="100%">
                      <PieChart>
                        <Pie
                          data={donutData}
                          dataKey="value"
                          startAngle={90}
                          endAngle={-270}
                          innerRadius={48}
                          outerRadius={70}
                          paddingAngle={2}
                        >
                          {donutData.map((entry, idx) => (
                            <Cell key={idx} fill={donutColors[idx % donutColors.length]} />
                          ))}
                        </Pie>
                      </PieChart>
                    </ResponsiveContainer>
                  </div>

                  <div className="mt-2 text-4xl font-bold">
                    {(prediction.churn_probability * 100).toFixed(1)}%
                  </div>
                  <div className="text-lg text-muted-foreground">Churn Probability</div>
                  <Badge variant={getRiskColor(prediction.risk_label)} className="text-sm px-4 py-1 mt-2">
                    {prediction.risk_label.toUpperCase()} Risk
                  </Badge>

                  {typeof prediction.confidence !== "undefined" && (
                    <div className="text-xs text-muted-foreground mt-1">
                      Confidence: {(prediction.confidence * 100).toFixed(1)}%
                    </div>
                  )}
                </div>

                {/* Top reasons bar chart */}
                {topBars.length > 0 && (
                  <div>
                    <h4 className="font-semibold mb-2">Top Risk Factors</h4>
                    <div style={{ width: "100%", height: 160 }}>
                      <ResponsiveContainer width="100%" height="100%">
                        <BarChart data={topBars} layout="vertical" margin={{ left: 20 }}>
                          <CartesianGrid strokeDasharray="3 3" />
                          <XAxis type="number" />
                          <YAxis dataKey="feature" type="category" width={160} />
                          <Tooltip formatter={(value: any) =>
                            typeof value === "number" ? `${Number(value).toFixed(1)}` : value
                          } />
                          <Bar dataKey="value" radius={[4, 4, 4, 4]} fill="hsl(var(--brand-primary))" />
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                  </div>
                )}

                {/* Warnings & details */}
                {prediction.warnings && prediction.warnings.length > 0 && (
                  <div className="text-xs text-muted-foreground">
                    <details>
                      <summary className="cursor-pointer">View Warnings ({prediction.warnings.length})</summary>
                      <ul className="mt-2 list-disc pl-5">
                        {prediction.warnings.map((w, i) => <li key={i}>{w}</li>)}
                      </ul>
                    </details>
                  </div>
                )}

                {/* Full reason list fallback */}
                {prediction.top_reasons && prediction.top_reasons.length > 0 && (
                  <div className="text-sm">
                    <h5 className="font-medium mb-1">Reason summary</h5>
                    <ul className="list-disc pl-5 text-sm">
                      {prediction.top_reasons.map((r, i) => (
                        <li key={i}>
                          <strong>{r.feature}</strong>
                          {" "}
                          {typeof r.importance !== "undefined"
                            ? `(${(r.importance * 100).toFixed(1)}%)`
                            : typeof r.contribution !== "undefined"
                              ? `(${r.contribution.toFixed(3)})`
                              : ""}
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            ) : (
              <div className="text-center py-12 text-muted-foreground">
                <Brain className="h-12 w-12 mx-auto mb-4 opacity-50" />
                <p>Fill the form and click <strong>Predict Churn Risk</strong> to get results</p>
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
