import { useState, useRef } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Upload, FileSpreadsheet, Download } from "lucide-react";
import Papa from "papaparse";

import { API } from "@/config";

const API_URL = API; // then append endpoints when calling fetch


export default function BatchPredictions() {
  const [results, setResults] = useState<any[]>([]);
  const [insights, setInsights] = useState<any | null>(null);
  const [batchId, setBatchId] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [statusMsg, setStatusMsg] = useState<string>("");

  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setLoading(true);
    setStatusMsg("⏳ Processing file…");

    Papa.parse(file, {
      header: true,
      skipEmptyLines: true,
      complete: async (parsed) => {
        try {
          const rows = parsed.data;

          const response = await fetch(`${API_URL}/batch_predict`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ rows }),
          });

          if (!response.ok) {
            const errText = await response.text();
            throw new Error(`API request failed: ${errText}`);
          }

          const data = await response.json();

          setBatchId(data.batch_id);

          setResults(
            data.results.map((r: any, idx: number) => ({
              id: r.customer_id || rows[idx]?.customer_id || `Row-${idx + 1}`,
              churnProbability: r.churn_probability,
              riskLevel: r.risk_label,
            }))
          );

          setInsights(data.insights);
          setStatusMsg(`✅ Processed ${data.insights.total} customers`);
        } catch (err: any) {
          console.error("Batch prediction error:", err);
          setStatusMsg(`❌ Error: ${err.message}`);
        } finally {
          setLoading(false);
          if (fileInputRef.current) {
            fileInputRef.current.value = ""; // reset input
          }
        }
      },
    });
  };

  const handleDownload = () => {
    if (batchId) {
      window.open(`${API_URL}/batch/${batchId}/download`, "_blank");
    }
  };

  const getRiskBadgeVariant = (risk: string) => {
    switch (risk.toLowerCase()) {
      case "high": return "destructive";
      case "medium": return "secondary";
      case "low": return "default";
      default: return "default";
    }
  };

  return (
    <div className="flex-1 space-y-6 p-6">
      <div className="flex items-center gap-3">
        <Upload className="h-8 w-8 text-brand-primary" />
        <div>
          <h1 className="text-3xl font-bold">Batch Predictions</h1>
          <p className="text-muted-foreground">
            Upload CSV files to predict churn for multiple customers
          </p>
        </div>
      </div>

      {/* Upload Section */}
      <Card className="bg-gradient-card shadow-soft">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <FileSpreadsheet className="h-5 w-5" />
            Upload Customer Data
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="border-2 border-dashed border-border rounded-lg p-8 text-center relative">
            <Upload className="h-12 w-12 mx-auto mb-4 text-muted-foreground" />
            <div className="space-y-2">
              <p className="text-lg font-medium">Drop your CSV file here</p>
              <p className="text-sm text-muted-foreground">
                or click to browse (Max file size: 10MB)
              </p>
            </div>
            <input
              ref={fileInputRef}
              type="file"
              accept=".csv"
              onChange={handleFileUpload}
              className="absolute inset-0 opacity-0 cursor-pointer"
            />
            <Button className="mt-4 bg-gradient-primary" disabled={loading}>
              {loading ? "Processing..." : "Choose File"}
            </Button>
          </div>
          {statusMsg && (
            <p className="mt-4 text-sm font-medium text-center text-muted-foreground">
              {statusMsg}
            </p>
          )}
        </CardContent>
      </Card>

      {/* Insights */}
      {insights && (
        <Card className="bg-gradient-card shadow-soft">
          <CardHeader className="flex flex-row items-center justify-between">
            <CardTitle>Batch Insights</CardTitle>
            <Button
              variant="outline"
              className="flex items-center gap-2"
              onClick={handleDownload}
              disabled={!batchId}
            >
              <Download className="h-4 w-4" />
              Download CSV
            </Button>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-3 gap-4 text-center mb-6">
              <div>
                <p className="text-sm text-muted-foreground">Total Customers</p>
                <p className="text-2xl font-bold">{insights.total}</p>
              </div>
              <div>
                <p className="text-sm text-muted-foreground">High Risk %</p>
                <p className="text-2xl font-bold text-destructive">
                  {insights.high_risk_pct.toFixed(1)}%
                </p>
              </div>
              <div>
                <p className="text-sm text-muted-foreground">Avg Churn Prob</p>
                <p className="text-2xl font-bold text-brand-primary">
                  {(insights.avg_churn_prob * 100).toFixed(1)}%
                </p>
              </div>
            </div>

            <h3 className="text-lg font-semibold mb-2">Top Churn Customers</h3>
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Customer ID</TableHead>
                  <TableHead>Churn Probability</TableHead>
                  <TableHead>Risk Level</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {insights.top_customers.map((c: any, idx: number) => (
                  <TableRow key={idx}>
                    <TableCell>{c.customer_id || `Row-${c.index}`}</TableCell>
                    <TableCell>{(c.churn_probability * 100).toFixed(1)}%</TableCell>
                    <TableCell>
                      <Badge variant={getRiskBadgeVariant(c.risk_label)}>
                        {c.risk_label.toUpperCase()}
                      </Badge>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </CardContent>
        </Card>
      )}

      {/* All Results */}
      {results.length > 0 && (
        <Card className="bg-gradient-card shadow-soft">
          <CardHeader>
            <CardTitle>All Predictions</CardTitle>
          </CardHeader>
          <CardContent>
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Customer ID</TableHead>
                  <TableHead>Churn Probability</TableHead>
                  <TableHead>Risk Level</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {results.map((r, idx) => (
                  <TableRow key={idx}>
                    <TableCell className="font-medium">{r.id}</TableCell>
                    <TableCell>{(r.churnProbability * 100).toFixed(1)}%</TableCell>
                    <TableCell>
                      <Badge variant={getRiskBadgeVariant(r.riskLevel)}>
                        {r.riskLevel.toUpperCase()}
                      </Badge>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
