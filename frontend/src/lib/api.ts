import axios from "axios";

// Backend API base URL (FastAPI)
const API_BASE = "http://127.0.0.1:8000"; // change when deployed

export const api = axios.create({
  baseURL: API_BASE,
  headers: { "Content-Type": "application/json" },
});

// ---- Endpoints ----
export const predictCustomer = (features: any, customer_id?: string) =>
  api.post("/predict", { features, customer_id });

export const batchPredict = (rows: any[]) =>
  api.post("/batch_predict", { rows });

export const explainCustomer = (features: any, customer_id?: string) =>
  api.post("/explain", { features, customer_id });

export const getMetrics = () => api.get("/metrics");

export const getFeatureImportance = () => api.get("/feature-importance");

export const getLogs = () => api.get("/logs");

export const healthCheck = () => api.get("/health");
