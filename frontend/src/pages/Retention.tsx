import { useEffect, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import {
  Table, TableBody, TableCell, TableHead, TableHeader, TableRow,
} from "@/components/ui/table";
import { Megaphone, RefreshCw, Copy, Search } from "lucide-react";

import { API } from "@/config";


type LogRow = {
  id: number;
  customer_id?: string;
  churn_probability: number;
  risk_label?: string;
  features?: Record<string, any>;
  action?: string[] | string;
  action_status?: string;
  created_at?: string;
  batch_id?: string;
  contacted_at?: string;
};

export default function Retention() {
  const [summary, setSummary] = useState<any | null>(null);

  const [topCustomers, setTopCustomers] = useState<LogRow[]>([]);
  const [takenActions, setTakenActions] = useState<LogRow[]>([]);

  const [page, setPage] = useState(1);
  const [takenPage, setTakenPage] = useState(1);

  const [selected, setSelected] = useState<LogRow | null>(null);
  const [suggestion, setSuggestion] = useState<any | null>(null);
  const [takenDetail, setTakenDetail] = useState<LogRow | null>(null);

  const [filterMode, setFilterMode] = useState<"today" | "month" | "all" | "range">("today");
  const [startDate, setStartDate] = useState<string>("");
  const [endDate, setEndDate] = useState<string>("");

  const [takenFilter, setTakenFilter] = useState<"today" | "month" | "all" | "range">("all");
  const [takenStart, setTakenStart] = useState<string>("");
  const [takenEnd, setTakenEnd] = useState<string>("");

  const [searchTop, setSearchTop] = useState("");
  const [searchTaken, setSearchTaken] = useState("");

  useEffect(() => {
    fetchSummary();
    fetchTopCustomers();
    fetchTakenActions();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [page, filterMode, startDate, endDate, takenPage, takenFilter, takenStart, takenEnd]);

  async function fetchSummary() {
    const res = await fetch(`${API}/retention/summary`);
    setSummary(await res.json());
  }

  async function fetchTopCustomers() {
    const qs: string[] = [];
    if (filterMode === "today") qs.push("today=true");
    if (filterMode === "month") qs.push("month=true");
    if (filterMode === "range" && startDate) qs.push(`start_date=${startDate}`);
    if (filterMode === "range" && endDate) qs.push(`end_date=${endDate}`);
    const offset = (page - 1) * 10;
    qs.push("limit=10", `offset=${offset}`, "status=pending");
    const res = await fetch(`${API}/retention/top?${qs.join("&")}`);
    const json = await res.json();
    setTopCustomers(json.results || []);
  }

  async function fetchTakenActions() {
    const qs: string[] = [];
    if (takenFilter === "today") qs.push("today=true");
    if (takenFilter === "month") qs.push("month=true");
    if (takenFilter === "range" && takenStart) qs.push(`start_date=${takenStart}`);
    if (takenFilter === "range" && takenEnd) qs.push(`end_date=${takenEnd}`);
    const offset = (takenPage - 1) * 10;
    qs.push("limit=10", `offset=${offset}`, "status=taken");
    const res = await fetch(`${API}/retention/top?${qs.join("&")}`);
    const json = await res.json();
    setTakenActions(json.results || []);
  }

  async function askAgent(log: LogRow) {
    setSelected(log);
    setSuggestion(null);
    const res = await fetch(`${API}/agent/suggest`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ log_id: log.id, use_llm: true }),
    });
    setSuggestion(await res.json());
  }

  async function applyAction(actionCode: string) {
    if (!selected) return;
    await fetch(`${API}/agent/apply_action`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ log_id: selected.id, action: actionCode }),
    });
    setSuggestion((prev: any) => {
      const next = { ...prev };
      next.recommended_actions = next.recommended_actions.map((a: any) =>
        a.code === actionCode ? { ...a, applied: true } : a
      );
      return next;
    });
    fetchSummary();
  }

  async function markDone() {
    if (!selected) return;
    await fetch(`${API}/agent/apply_action`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ log_id: selected.id, action: "suggestion_done" }),
    });
    fetchTopCustomers();
    fetchTakenActions();
    fetchSummary();
    setSelected(null);
    setSuggestion(null);
  }

  const formatPct = (p?: number) => (p == null ? "-" : `${(p * 100).toFixed(1)}%`);
  const formatUSD = (v?: number) => (v == null ? "—" : `$${Number(v).toLocaleString()}`);
  const getBadge = (risk?: string) =>
    risk === "high" ? "destructive" : risk === "medium" ? "secondary" : "default";

  return (
    <div className="flex-1 space-y-6 p-6">
      <div className="flex items-center gap-3">
        <Megaphone className="h-8 w-8 text-brand-primary" />
        <div>
          <h1 className="text-3xl font-bold">Retention Intelligence</h1>
          <p className="text-muted-foreground">AI-assisted retention actions</p>
        </div>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-4 gap-4">
        <StatCard title="High Risk Customers" value={summary?.risk_counts?.high || 0} />
        <StatCard title="Revenue at Risk (USD)" value={formatUSD(summary?.revenue_at_risk_usd)} />
        <StatCard title="Actions Pending" value={summary?.actions_pending || 0} />
        <StatCard title="Actions Taken" value={summary?.actions_applied || 0} />
      </div>

      {/* Top Customers */}
      <CustomerTable
        title="Top Customers"
        data={topCustomers}
        filterMode={filterMode}
        setFilterMode={setFilterMode}
        startDate={startDate}
        setStartDate={setStartDate}
        endDate={endDate}
        setEndDate={setEndDate}
        page={page}
        setPage={setPage}
        search={searchTop}
        setSearch={setSearchTop}
        onSuggest={askAgent}
      />

      {/* Suggestion Modal */}
      {selected && suggestion && (
        <Modal onClose={() => setSelected(null)}>
          <h3 className="text-xl font-semibold mb-4">Suggestions for {selected.customer_id}</h3>
          <div className="grid grid-cols-2 gap-6">
            <div>
              <h4 className="font-semibold">Rule-based</h4>
              {suggestion.reasons?.map((r: any, i: number) => <p key={i}>• {r.text}</p>)}
              <h4 className="font-semibold mt-3">Actions</h4>
              {suggestion.recommended_actions?.map((a: any) => (
                <div key={a.code} className="flex items-center gap-2">
                  <input type="checkbox" checked={a.applied} onChange={() => applyAction(a.code)} />
                  <span className={a.applied ? "text-green-600" : ""}>{a.text}</span>
                </div>
              ))}
            </div>
            <div>
              <h4 className="font-semibold">LLM Suggestion</h4>
              {suggestion.llm_enhanced_message
                ? <pre className="whitespace-pre-wrap">{suggestion.llm_enhanced_message}</pre>
                : <p className="text-muted-foreground">⚠️ LLM not configured</p>}
              {suggestion.llm_enhanced_message && (
                <Button size="sm" onClick={() => navigator.clipboard.writeText(suggestion.llm_enhanced_message)}>
                  Copy <Copy className="w-4 h-4" />
                </Button>
              )}
            </div>
          </div>
          <div className="mt-4 flex justify-end gap-2">
            <Button onClick={markDone}>Done</Button>
            <Button variant="outline" onClick={() => setSelected(null)}>Close</Button>
          </div>
        </Modal>
      )}

      {/* Actions Taken */}
      <CustomerTable
        title="Actions Taken"
        data={takenActions}
        filterMode={takenFilter}
        setFilterMode={setTakenFilter}
        startDate={takenStart}
        setStartDate={setTakenStart}
        endDate={takenEnd}
        setEndDate={setTakenEnd}
        page={takenPage}
        setPage={setTakenPage}
        search={searchTaken}
        setSearch={setSearchTaken}
        onRowClick={(row) => setTakenDetail(row)}
      />

      {/* Taken Detail Modal */}
      {takenDetail && (
        <Modal onClose={() => setTakenDetail(null)}>
          <h3 className="text-xl font-semibold mb-4">Actions for {takenDetail.customer_id}</h3>
          <p><strong>Risk:</strong> {takenDetail.risk_label}</p>
<p><strong>Churn predicted:</strong> {takenDetail.created_at ? new Date(takenDetail.created_at).toLocaleString() : "-"}</p>
<p><strong>Action taken:</strong> {takenDetail.contacted_at ? new Date(takenDetail.contacted_at).toLocaleString() : "-"}</p>
<p><strong>Applied:</strong> {Array.isArray(takenDetail.action) ? takenDetail.action.join(", ") : takenDetail.action}</p>
          <div className="mt-4 flex justify-end"><Button onClick={() => setTakenDetail(null)}>Close</Button></div>
        </Modal>
      )}
    </div>
  );
}

function StatCard({ title, value }: { title: string; value: any }) {
  return (
    <Card>
      <CardHeader><CardTitle>{title}</CardTitle></CardHeader>
      <CardContent><div className="text-2xl font-bold">{value}</div></CardContent>
    </Card>
  );
}

function CustomerTable({ title, data, filterMode, setFilterMode, startDate, setStartDate, endDate, setEndDate, page, setPage, search, setSearch, onSuggest, onRowClick }: any) {
  const formatPct = (p?: number) => (p == null ? "-" : `${(p * 100).toFixed(1)}%`);
  const getBadge = (risk?: string) =>
    risk === "high" ? "destructive" : risk === "medium" ? "secondary" : "default";

const filtered = search
  ? data.filter((row: any) => {
      const idStr = row.id?.toString();   // <-- use log.id here
      return idStr?.includes(search);
    })
  : data;




  return (
    <Card>
      <CardHeader className="flex justify-between items-center">
        <CardTitle>{title}</CardTitle>
        <div className="flex gap-2 items-center">
          <Search className="w-4 h-4 text-muted-foreground" />
          <input
            type="text"
            placeholder="Search ID"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="border p-1 rounded"
          />
        </div>
      </CardHeader>
      <CardContent>
        <div className="flex gap-2 mb-2">
          {["today", "month", "all", "range"].map((f) => (
            <Button
              key={f}
              size="sm"
              variant={filterMode === f ? "default" : "outline"}
              onClick={() => setFilterMode(f)}
            >
              {f}
            </Button>
          ))}
          {filterMode === "range" && (
            <>
              <input type="date" value={startDate} onChange={(e) => setStartDate(e.target.value)} />
              <input type="date" value={endDate} onChange={(e) => setEndDate(e.target.value)} />
            </>
          )}
        </div>
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>ID</TableHead>
              <TableHead>Customer</TableHead>
              <TableHead>Prob</TableHead>
              <TableHead>Risk</TableHead>
              <TableHead>Actions</TableHead>
              {onSuggest && <TableHead>Suggest</TableHead>}
            </TableRow>
          </TableHeader>
          <TableBody>
            {filtered.map((row: any) => (
              <TableRow key={row.id} onClick={() => onRowClick && onRowClick(row)} className={onRowClick ? "cursor-pointer hover:bg-muted" : ""}>
                <TableCell>{row.id}</TableCell>
                <TableCell>{row.customer_id}</TableCell>
                <TableCell>{formatPct(row.churn_probability)}</TableCell>
                <TableCell><Badge variant={getBadge(row.risk_label)}>{row.risk_label}</Badge></TableCell>
                <TableCell>{Array.isArray(row.action) ? row.action.join(", ") : row.action}</TableCell>
                {onSuggest && (
                  <TableCell>
                    <Button size="sm" onClick={(e) => { e.stopPropagation(); onSuggest(row); }}>Suggest</Button>
                  </TableCell>
                )}
              </TableRow>
            ))}
          </TableBody>
        </Table>
        <div className="flex justify-between mt-2">
          <Button size="sm" onClick={() => setPage((p: number) => Math.max(1, p - 1))}>Previous</Button>
          <span>Page {page}</span>
          <Button size="sm" onClick={() => setPage((p: number) => p + 1)}>Next</Button>
        </div>
      </CardContent>
    </Card>
  );
}

function Modal({ children, onClose }: { children: any; onClose: () => void }) {
  return (
    <div className="fixed inset-0 bg-black/50 flex justify-center items-start p-6 z-50">
      <div className="bg-background p-6 rounded shadow-lg w-full max-w-4xl relative">
        {children}
        <button onClick={onClose} className="absolute top-2 right-2 text-muted-foreground">✖</button>
      </div>
    </div>
  );
}
