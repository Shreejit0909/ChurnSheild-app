# backend/app/main.py
from fastapi import FastAPI, HTTPException, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import joblib
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, date
from sqlalchemy.orm import Session
from sqlalchemy import func, case
import uuid
import io
import os
import requests
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from datetime import datetime, timedelta
import pytz
from dotenv import load_dotenv


# Load .env file from backend folder
load_dotenv()

IST = pytz.timezone("Asia/Kolkata")

# DB imports
from app.db import Base, engine, SessionLocal, PredictionLog
os.environ.setdefault(
    "GEMINI_ENDPOINT",
    f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={os.getenv('GEMINI_API_KEY','')}"
)
# Optional scheduler import
try:
    from fastapi_utils.tasks import repeat_every
    _HAS_REPEAT = True
except Exception:
    _HAS_REPEAT = False

# ---------- Paths ----------
BASE_APP = Path(__file__).resolve().parent  # backend/app
MODELS_DIR = BASE_APP / "models"
ARTIFACTS_DIR = BASE_APP / "artifacts"

PIPELINE_PATH = MODELS_DIR / "pipeline.joblib"
FEATURES_PATH = ARTIFACTS_DIR / "feature_columns.json"
GLOBAL_IMPORTANCE_PATH = ARTIFACTS_DIR / "global_feature_importance.json"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"
SHAP_EXPLAINER_PATH = ARTIFACTS_DIR / "shap_explainer.joblib"
SHAP_BACKGROUND_PATH = ARTIFACTS_DIR / "shap_background.csv"

# ---------- FastAPI ----------
app = FastAPI(title="Churn Prediction API", version="0.9")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "*"  # restrict in prod
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# DB init
Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ---------- Globals ----------
pipeline = None
feature_config = {"numeric": [], "categorical": [], "feature_names": []}
global_importance = []
metrics = {}
shap_explainer = None

# ---------- Startup ----------
@app.on_event("startup")
def startup_load():
    global pipeline, feature_config, global_importance, metrics, shap_explainer

    if PIPELINE_PATH.exists():
        try:
            pipeline = joblib.load(PIPELINE_PATH)
            app.state.model_loaded = True
        except Exception as e:
            app.state.model_loaded = False
            app.state.startup_error = f"Failed to load model: {e}"
    else:
        app.state.model_loaded = False
        app.state.startup_error = f"Model not found at {PIPELINE_PATH}"

    if FEATURES_PATH.exists():
        with open(FEATURES_PATH, "r") as f:
            feature_config.update(json.load(f))

    if GLOBAL_IMPORTANCE_PATH.exists():
        with open(GLOBAL_IMPORTANCE_PATH, "r") as f:
            global_importance.extend(json.load(f))

    if METRICS_PATH.exists():
        with open(METRICS_PATH, "r") as f:
            try:
                metrics.update(json.load(f))
            except Exception:
                metrics.clear()

    if SHAP_EXPLAINER_PATH.exists():
        try:
            shap_explainer = joblib.load(SHAP_EXPLAINER_PATH)
        except Exception:
            shap_explainer = None

# ---------- Request models ----------
class PredictRequest(BaseModel):
    features: Dict[str, Any]
    customer_id: Optional[str] = None

class BatchPredictRequest(BaseModel):
    rows: List[Dict[str, Any]]

class AgentSuggestRequest(BaseModel):
    log_id: Optional[int] = None
    features: Optional[Dict[str, Any]] = None
    customer_id: Optional[str] = None
    use_llm: Optional[bool] = False

class AgentApplyRequest(BaseModel):
    log_id: int
    action: str
    note: Optional[str] = None

# ---------- Helpers ----------
def _build_df_from_features(feature_dict: Dict[str, Any]) -> pd.DataFrame:
    expected = feature_config.get("numeric", []) + feature_config.get("categorical", [])
    row = feature_dict.copy() if feature_dict else {}
    for col in expected:
        if col not in row:
            row[col] = np.nan
    return pd.DataFrame([row], columns=expected)

def _format_top_reasons(global_imp: List[Dict[str, Any]], top_k: int = 5):
    return global_imp[:top_k] if global_imp else []

def _parse_actions_field(action_field) -> List[str]:
    if not action_field:
        return []
    if isinstance(action_field, list):
        return action_field
    try:
        return json.loads(action_field)
    except Exception:
        # legacy single-string
        return [action_field] if action_field else []

def _save_log(db: Session, customer_id: str, probability: float, risk: str, features: Dict[str, Any],
              batch_id: Optional[str] = None, source: Optional[str] = "ui", action: Optional[str] = None):
    log = PredictionLog(
        customer_id=customer_id or "unknown",
        churn_probability=probability,
        risk_label=risk,
        features=json.dumps(features),
        created_at=datetime.now(IST),
        batch_id=batch_id,
        source=source,
        action=action if action is None else json.dumps(action) if not isinstance(action, str) else json.dumps([action]) if isinstance(action, str) else json.dumps(action),
        action_status="pending"
    )
    db.add(log)
    db.commit()
    db.refresh(log)
    print(f"✅ Saved log -> id={log.id}, customer={log.customer_id}, prob={log.churn_probability}, risk={log.risk_label}, batch={log.batch_id}")
    return log

def _validate_and_normalize_input(features: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    warnings: List[str] = []
    normalized = (features or {}).copy()
    defaults = {
        "monthly_usage": 10.0,
        "usage_trend": 1.0,
        "features_used": 3,
        "days_since_last_login": 7,
        "avg_session_duration": 30.0,
        "support_tickets_90d": 0,
        "avg_resolution_time": 12.0,
        "support_satisfaction": 4.0,
        "escalated_tickets": 0,
        "monthly_revenue": 100.0,
        "payment_failures_90d": 0,
        "days_to_renewal": 90,
        "company_size": 50,
        "revenue_per_employee": 2.0,
        "usage_intensity": 1.0,
        "manual_risk_score": 0.1
    }
    for k, default in defaults.items():
        v = normalized.get(k, None)
        if v is None:
            normalized[k] = default
            warnings.append(f"{k} missing — defaulted to {default}")
        else:
            try:
                normalized[k] = float(v)
            except Exception:
                normalized[k] = default
                warnings.append(f"{k} invalid — defaulted to {default}")
    for cat in feature_config.get("categorical", []):
        if cat in normalized and normalized[cat] is not None:
            try:
                normalized[cat] = str(normalized[cat])
            except Exception:
                normalized[cat] = "missing"
                warnings.append(f"{cat} coerced to string 'missing'")
    return normalized, warnings

def _compute_confidence(proba: float) -> float:
    base_auc = float(metrics.get("roc_auc", 0.8)) if metrics else 0.8
    decisiveness = abs(proba - 0.5) * 2  # 0..1
    conf = base_auc * (0.5 + 0.5 * decisiveness)
    return float(min(max(conf, 0.0), 0.99))

# ---------------- Gemini/LLM helper ----------------
def generate_with_gemini(prompt: str) -> Optional[str]:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return None

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"maxOutputTokens": 256}
    }

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        if "candidates" in data and len(data["candidates"]) > 0:
            cand = data["candidates"][0]
            if "content" in cand and "parts" in cand["content"]:
                parts = cand["content"]["parts"]
                if parts and "text" in parts[0]:
                    return parts[0]["text"]
        return None
    except Exception as e:
        print("Gemini call failed:", e, resp.text if 'resp' in locals() else "")
        return None


# ---------- Endpoints ----------
@app.get("/health")
def health():
    if getattr(app.state, "startup_error", None):
        return {"status": "degraded", "error": app.state.startup_error}
    return {"status": "ok", "model_loaded": app.state.model_loaded, "timestamp": datetime.now(IST).isoformat()}

@app.post("/predict")
def predict(req: PredictRequest, db: Session = Depends(get_db)):
    if not pipeline:
        raise HTTPException(status_code=503, detail="Model not available. Train model first.")
    normalized_features, warnings = _validate_and_normalize_input(req.features or {})
    df = _build_df_from_features(normalized_features)
    try:
        proba = float(pipeline.predict_proba(df)[:, 1][0])
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error during prediction: {e}")
    risk_label = "high" if proba >= 0.5 else "medium" if proba >= 0.25 else "low"
    top_reasons = []
    try:
        if shap_explainer is not None:
            shap_vals = shap_explainer.shap_values(df)
            arr = np.array(shap_vals)
            if arr.ndim == 3: arr = arr[0]
            if arr.ndim == 2:
                row_vals = arr[0]
                names = feature_config.get("numeric", []) + feature_config.get("categorical", [])
                pairs = [{"feature": names[i], "contribution": float(row_vals[i])} for i in range(len(names))]
                top_reasons = sorted(pairs, key=lambda x: abs(x["contribution"]), reverse=True)[:5]
    except Exception:
        top_reasons = []
    if not top_reasons:
        top_reasons = _format_top_reasons(global_importance, top_k=5)
    confidence = _compute_confidence(proba)
    log = _save_log(db, req.customer_id or "unknown", float(proba), risk_label, normalized_features, source="ui")
    return {
        "customer_id": req.customer_id,
        "churn_probability": float(round(proba, 4)),
        "risk_label": risk_label,
        "confidence": float(round(confidence, 3)),
        "top_reasons": top_reasons,
        "log_id": log.id,
        "warnings": [],
        "timestamp": datetime.now(IST).isoformat()
    }

@app.post("/batch_predict")
def batch_predict(req: BatchPredictRequest, db: Session = Depends(get_db)):
    if not pipeline:
        raise HTTPException(status_code=503, detail="Model not available. Train model first.")
    if not req.rows:
        raise HTTPException(status_code=400, detail="No rows provided.")
    batch_id = str(uuid.uuid4())
    expected = feature_config.get("numeric", []) + feature_config.get("categorical", [])
    df = pd.DataFrame(req.rows)
    for col in expected:
        if col not in df.columns:
            df[col] = np.nan
    df = df[expected]
    try:
        probas = pipeline.predict_proba(df)[:, 1].tolist()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error during batch prediction: {e}")
    results = []
    for i, p in enumerate(probas):
        risk = "high" if p >= 0.5 else "medium" if p >= 0.25 else "low"
        cust_id = req.rows[i].get("customer_id") or f"batch_{i}"
        _save_log(db, cust_id, float(p), risk, req.rows[i], batch_id=batch_id, source="batch")
        results.append({
            "index": i,
            "customer_id": cust_id,
            "churn_probability": float(round(p, 4)),
            "risk_label": risk,
            **req.rows[i]
        })
    top_customers = sorted(results, key=lambda x: x["churn_probability"], reverse=True)[:10]
    avg_prob = float(np.mean(probas)) if probas else 0
    high_risk_pct = (sum(1 for r in results if r["risk_label"] == "high") / len(results)) * 100 if results else 0
    return {
        "batch_id": batch_id,
        "results": results,
        "insights": {
            "total": len(results),
            "avg_churn_prob": avg_prob,
            "high_risk_pct": high_risk_pct,
            "top_customers": top_customers
        },
        "timestamp": datetime.now(IST).isoformat()
    }

# ---------- Agent endpoints ----------
def _rule_based_suggestions(features: Dict[str, Any], churn_probability: float, risk_label: str):
    reasons = []
    actions = []
    def _num(k, default=None):
        try:
            v = features.get(k, None) if features else None
            return float(v) if v is not None else default
        except Exception:
            return default
    support_satisfaction = _num("support_satisfaction", None)
    payment_failures = _num("payment_failures_90d", 0)
    usage_trend = _num("usage_trend", None)
    days_last_login = _num("days_since_last_login", None)
    escalated = _num("escalated_tickets", 0)
    manual_risk = _num("manual_risk_score", 0)
    if support_satisfaction is not None and support_satisfaction <= 3:
        reasons.append({"code": "low_support", "text": f"Low support satisfaction ({support_satisfaction}/5)", "severity": "high"})
        actions.append({"code": "support_call", "text": "Assign senior support manager + priority ticket", "priority": 1})
    if payment_failures and payment_failures >= 1:
        reasons.append({"code": "payment_issues", "text": f"{int(payment_failures)} payment failure(s) in last 90 days", "severity": "high" if payment_failures >= 2 else "medium"})
        actions.append({"code": "payment_contact", "text": "Offer payment assistance / retry + email reminders", "priority": 2})
    if usage_trend is not None and usage_trend < 0.95:
        reasons.append({"code": "declining_usage", "text": f"Usage trending down ({usage_trend})", "severity": "medium"})
        actions.append({"code": "engagement", "text": "Offer targeted onboarding / product tour", "priority": 3})
    if days_last_login is not None and days_last_login > 30:
        reasons.append({"code": "inactivity", "text": f"User inactive for {int(days_last_login)} days", "severity": "medium"})
        actions.append({"code": "reengage", "text": "Send re-engagement email + in-app message", "priority": 4})
    if escalated and escalated > 0:
        reasons.append({"code": "escalated_tickets", "text": f"{int(escalated)} escalated support tickets", "severity": "high"})
        actions.append({"code": "case_manager", "text": "Assign case manager and escalate to retention team", "priority": 1})
    if manual_risk and manual_risk > 5:
        reasons.append({"code": "manual_risk", "text": f"Manual risk score high ({manual_risk})", "severity": "high"})
        actions.append({"code": "sales_outreach", "text": "Direct sales outreach + custom offer", "priority": 1})
    if churn_probability >= 0.75:
        actions.insert(0, {"code": "immediate_discount", "text": "Offer immediate time-limited discount (e.g. 10-20%)", "priority": 0})
    seen = set()
    dedup_actions = []
    for a in sorted(actions, key=lambda x: x["priority"]):
        if a["code"] not in seen:
            dedup_actions.append(a)
            seen.add(a["code"])
    return {"reasons": reasons, "recommended_actions": dedup_actions}

@app.post("/agent/suggest")
def agent_suggest(req: AgentSuggestRequest, db: Session = Depends(get_db)):
    features = req.features
    churn_prob = 0.0
    risk_label = "unknown"
    customer_id = req.customer_id
    existing_actions: List[str] = []
    action_status = "pending"
    if req.log_id:
        log = db.query(PredictionLog).filter(PredictionLog.id == req.log_id).first()
        if not log:
            raise HTTPException(status_code=404, detail="log_id not found")
        try:
            features = json.loads(log.features or "{}")
        except Exception:
            features = {}
        churn_prob = float(log.churn_probability or 0.0)
        risk_label = log.risk_label or risk_label
        customer_id = customer_id or log.customer_id
        existing_actions = _parse_actions_field(log.action)
        action_status = log.action_status or "pending"
    if features is None:
        features = {}
    rb = _rule_based_suggestions(features, churn_prob, risk_label)
    generated_message = None
    if req.use_llm:
        prompt = (
            f"Customer {customer_id}\n"
            f"Churn probability: {churn_prob:.2f}, risk: {risk_label}\n"
            f"Key features: {json.dumps(features)}\n"
            f"Detected reasons: {rb['reasons']}\n"
            f"Suggested actions: {rb['recommended_actions']}\n\n"
            "Write a short (2-3 sentence) retention note for sales/CS with exact steps and a 1-line personalized message to send."
        )
        generated_message = generate_with_gemini(prompt)
    # attach flags to recommended actions to mark ones already applied
    for a in rb["recommended_actions"]:
        a["applied"] = a["code"] in existing_actions
    return {
        "customer_id": customer_id,
        "churn_probability": churn_prob,
        "risk_label": risk_label,
        "reasons": rb["reasons"],
        "recommended_actions": rb["recommended_actions"],
        "existing_actions": existing_actions,
        "action_status": action_status,
        "llm_enhanced_message": generated_message,
        "timestamp": datetime.now(IST).isoformat()
    }

@app.post("/agent/apply_action")
def agent_apply_action(req: AgentApplyRequest, db: Session = Depends(get_db)):
    log = db.query(PredictionLog).filter(PredictionLog.id == req.log_id).first()
    if not log:
        raise HTTPException(status_code=404, detail="log_id not found")
    existing = _parse_actions_field(log.action)
    # note: ensure it's a list and we store JSON string
    if req.action not in existing:
        existing.append(req.action)
    log.action = json.dumps(existing)
    # when an action is applied we mark action_status as applied
    log.action_status = "applied"
    log.contacted_at = datetime.now(IST)
    db.add(log)
    db.commit()
    db.refresh(log)
    return {"ok": True, "log_id": log.id, "actions": existing, "action_status": log.action_status, "contacted_at": log.contacted_at.isoformat()}

# ---------- Retention endpoints ----------
@app.get("/retention/summary")
def retention_summary(
    batch_id: Optional[str] = Query(None),
    start_date: Optional[str] = Query(None),
    end_date: Optional[str] = Query(None),
    db: Session = Depends(get_db)
):
    query = db.query(PredictionLog)
    if batch_id:
        query = query.filter(PredictionLog.batch_id == batch_id)
    if start_date:
        try:
            sd = datetime.fromisoformat(start_date)
            query = query.filter(PredictionLog.created_at >= sd)
        except Exception:
            pass
    if end_date:
        try:
            ed = datetime.fromisoformat(end_date) + timedelta(days=1)
            query = query.filter(PredictionLog.created_at < ed)
        except Exception:
            pass
    total = query.count() or 0
    # risk counts
    risk_counts_rows = query.with_entities(PredictionLog.risk_label, func.count().label("count")).group_by(PredictionLog.risk_label).all()
    risk_counts = {row.risk_label: row.count for row in risk_counts_rows}
    # revenue at risk (USD) only considering high risk
    high_logs = query.filter(PredictionLog.risk_label == "high").all()
    revenue_at_risk = 0.0
    for log in high_logs:
        try:
            feat = json.loads(log.features or "{}")
            monthly_rev = float(feat.get("monthly_revenue") or 0)
            revenue_at_risk += monthly_rev * float(log.churn_probability or 0.0)
        except Exception:
            continue
    actions_pending = query.filter(PredictionLog.action_status == "pending").count() or 0
    actions_applied = query.filter(PredictionLog.action_status == "applied").count() or 0
    actions_done = query.filter(PredictionLog.action_status == "done").count() or 0
    return {
        "total_predictions": total,
        "risk_counts": risk_counts,
        "revenue_at_risk_usd": revenue_at_risk,
        "actions_pending": actions_pending,
        "actions_applied": actions_applied,
        "actions_done": actions_done,
        "batch_id": batch_id or "all"
    }

@app.get("/retention/top")
def retention_top(
    limit: int = 10,
    offset: int = 0,
    today: Optional[bool] = Query(False),
    month: Optional[bool] = Query(False),
    start_date: Optional[str] = Query(None),
    end_date: Optional[str] = Query(None),
    status: str = Query("pending"),  # pending | taken | all
    db: Session = Depends(get_db)
):
    query = db.query(PredictionLog)

    # status filters
    if status == "pending":
        query = query.filter(PredictionLog.action_status == "pending")
    elif status == "taken":
        query = query.filter(PredictionLog.action_status != "pending")

    # date filters → use created_at for pending, contacted_at for taken
    if today:
        today_iso = datetime.now(IST).date().isoformat()
        if status == "taken":
            query = query.filter(func.date(PredictionLog.contacted_at) == today_iso)
        else:
            query = query.filter(func.date(PredictionLog.created_at) == today_iso)

    if month:
        ym = datetime.now(IST).strftime("%Y-%m")
        if status == "taken":
            query = query.filter(func.strftime("%Y-%m", PredictionLog.contacted_at) == ym)
        else:
            query = query.filter(func.strftime("%Y-%m", PredictionLog.created_at) == ym)

    if start_date:
        try:
            sd = datetime.fromisoformat(start_date)
            if status == "taken":
                query = query.filter(PredictionLog.contacted_at >= sd)
            else:
                query = query.filter(PredictionLog.created_at >= sd)
        except Exception:
            pass

    if end_date:
        try:
            ed = datetime.fromisoformat(end_date) + timedelta(days=1)
            if status == "taken":
                query = query.filter(PredictionLog.contacted_at < ed)
            else:
                query = query.filter(PredictionLog.created_at < ed)
        except Exception:
            pass

    # pagination
    total = query.count()
    rows = query.order_by(PredictionLog.churn_probability.desc()).offset(offset).limit(limit).all()

    # results
    results = []
    for r in rows:
        try:
            feats = json.loads(r.features or "{}")
        except Exception:
            feats = {}
        action_list = _parse_actions_field(r.action)
        results.append({
            "id": r.id,
            "customer_id": r.customer_id,
            "churn_probability": r.churn_probability,
            "risk_label": r.risk_label,
            "features": feats,
            "batch_id": r.batch_id,
            "action": action_list,
            "action_status": r.action_status,
            "created_at": r.created_at.isoformat(),
            "contacted_at": r.contacted_at.isoformat() if r.contacted_at else None
        })

    return {"results": results, "total": total}

# ---------- Download Batch CSV ----------
@app.get("/batch/{batch_id}/download")
def download_batch(batch_id: str, db: Session = Depends(get_db)):
    logs = db.query(PredictionLog).filter(PredictionLog.batch_id == batch_id).all()
    if not logs:
        raise HTTPException(status_code=404, detail="Batch not found")
    df = pd.DataFrame([
        {
            "customer_id": log.customer_id,
            "churn_probability": log.churn_probability,
            "risk_label": log.risk_label,
            **(json.loads(log.features) if log.features else {})
        }
        for log in logs
    ])
    output = io.StringIO()
    df.to_csv(output, index=False)
    output.seek(0)
    return StreamingResponse(output, media_type="text/csv", headers={
        "Content-Disposition": f"attachment; filename=batch_{batch_id}.csv"
    })

# ---------- Logs endpoint (keeps compatibility for frontend) ----------
@app.get("/logs")
def get_logs(
    risk: str = Query("all"),
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    page: int = 1,
    page_size: int = 20,
    db: Session = Depends(get_db),
):
    query = db.query(PredictionLog).order_by(PredictionLog.created_at.desc())
    if risk != "all":
        query = query.filter(PredictionLog.risk_label.ilike(risk))
    if start_date:
        try:
            sd = datetime.fromisoformat(start_date)
            query = query.filter(PredictionLog.created_at >= sd)
        except Exception:
            pass
    if end_date:
        try:
            ed = datetime.fromisoformat(end_date) + timedelta(days=1)
            query = query.filter(PredictionLog.created_at < ed)
        except Exception:
            pass
    total = query.count()
    logs = query.offset((page - 1) * page_size).limit(page_size).all()
    return {
        "page": page,
        "page_size": page_size,
        "total": total,
        "results": [
            {
                "id": log.id,
                "customer_id": log.customer_id,
                "timestamp": log.created_at.isoformat(),
                "churn_probability": log.churn_probability,
                "risk_label": log.risk_label,
                "batch_id": log.batch_id,
                "action": _parse_actions_field(log.action),
                "action_status": log.action_status,
                "contacted_at": log.contacted_at.isoformat() if log.contacted_at else None,
                "source": log.source,
                "features": json.loads(log.features) if log.features else {},
                "model_version": metrics.get("trained_at", "unknown"),
            }
            for log in logs
        ]
    }

# ---------- Analytics (kept same but using model above) ----------
@app.get("/analytics")
def analytics(
    batch_id: Optional[str] = Query(None),  # filter by batch_id if provided
    db: Session = Depends(get_db)
):
    query = db.query(PredictionLog)
    if batch_id:
        query = query.filter(PredictionLog.batch_id == batch_id)
    total = query.count() or 0
    avg_prob = query.with_entities(func.avg(PredictionLog.churn_probability)).scalar() or 0
    high_count = query.filter(PredictionLog.risk_label == "high").count() or 0
    high_pct = (high_count / total * 100) if total else 0
    risk_data = (query.with_entities(PredictionLog.risk_label, func.count().label("count")).group_by(PredictionLog.risk_label).all())
    risk_distribution = [{"risk": row.risk_label, "count": row.count} for row in risk_data]
    churn_by_segment = [{"segment": row.risk_label, "churnRate": (row.count / total * 100) if total else 0} for row in risk_data]
    region_data = (query.with_entities(func.json_extract(PredictionLog.features, "$.region").label("region"), func.count().label("count")).group_by("region").all())
    colors = ["hsl(var(--brand-primary))","hsl(var(--brand-secondary))","hsl(var(--info))","hsl(var(--muted-foreground))"]
    churn_by_region = [{"name": (row.region or "Unknown"), "value": row.count, "color": colors[i % len(colors)]} for i, row in enumerate(region_data)]
    top_region = max(churn_by_region, key=lambda x: x["value"])["name"] if churn_by_region else None
    trend_data = (query.with_entities(func.strftime("%Y-%m", PredictionLog.created_at).label("month"), func.count().label("total"), func.coalesce(func.sum(case((PredictionLog.risk_label == "high", 1), else_=0)), 0).label("churns")).group_by("month").all())
    churn_trend = [{"month": datetime.strptime(row.month, "%Y-%m").strftime("%b %Y"), "churnRate": (row.churns / row.total) * 100 if row.total else 0} for row in trend_data]
    plan_data = (query.with_entities(func.json_extract(PredictionLog.features, "$.plan").label("plan"), func.count().label("total"), func.coalesce(func.sum(case((PredictionLog.risk_label == "high", 1), else_=0)), 0).label("churns")).group_by("plan").all())
    churn_by_plan = [{"plan": (row.plan or "Unknown"), "churnRate": (row.churns / row.total) * 100 if row.total else 0} for row in plan_data]
    size_data = (query.with_entities(func.json_extract(PredictionLog.features, "$.company_size").label("company_size"), func.count().label("total"), func.coalesce(func.sum(case((PredictionLog.risk_label == "high", 1), else_=0)), 0).label("churns")).all())
    def categorize_size(n):
        if not n:
            return "Unknown"
        try:
            n = int(n)
            if n < 50:
                return "SMB"
            elif n < 500:
                return "Mid-Market"
            else:
                return "Enterprise"
        except:
            return "Unknown"
    size_buckets = {}
    for row in size_data:
        bucket = categorize_size(row.company_size)
        if bucket not in size_buckets:
            size_buckets[bucket] = {"total": 0, "churns": 0}
        size_buckets[bucket]["total"] += row.total or 0
        size_buckets[bucket]["churns"] += row.churns or 0
    churn_by_size = [{"size": k, "churnRate": (v["churns"] / v["total"]) * 100 if v["total"] else 0} for k, v in size_buckets.items()]
    return {
        "risk_distribution": risk_distribution,
        "churn_by_segment": churn_by_segment,
        "churn_by_region": churn_by_region,
        "churn_trend": churn_trend,
        "churn_by_plan": churn_by_plan,
        "churn_by_size": churn_by_size,
        "summary": {
            "total_predictions": total,
            "high_risk_pct": high_pct,
            "avg_churn_prob": avg_prob,
            "top_region": top_region,
            "batch_id": batch_id or "all"
        }
    }


# ---------- Dashboard endpoints ----------
@app.get("/dashboard/summary")
def dashboard_summary(db: Session = Depends(get_db)):
    total = db.query(PredictionLog).count() or 0
    high_count = db.query(PredictionLog).filter(PredictionLog.risk_label == "high").count() or 0
    medium_count = db.query(PredictionLog).filter(PredictionLog.risk_label == "medium").count() or 0
    low_count = db.query(PredictionLog).filter(PredictionLog.risk_label == "low").count() or 0

    high_pct = (high_count / total * 100) if total else 0

    # revenue at risk (USD) only for high risk customers
    high_logs = db.query(PredictionLog).filter(PredictionLog.risk_label == "high").all()
    revenue_at_risk = 0.0
    for log in high_logs:
        try:
            feat = json.loads(log.features or "{}")
            monthly_rev = float(feat.get("monthly_revenue") or 0)
            revenue_at_risk += monthly_rev * float(log.churn_probability or 0.0)
        except Exception:
            continue

    # actions dynamic
    actions_pending = db.query(PredictionLog).filter(PredictionLog.action_status == "pending").count() or 0
    actions_taken = db.query(PredictionLog).filter(PredictionLog.action_status != "pending").count() or 0

    return {
        "customers": total,
        "high_risk_pct": round(high_pct, 1),
        "revenue_at_risk": round(revenue_at_risk, 2),
        "actions_pending": actions_pending,
        "actions_taken": actions_taken,
        "risk_counts": {
            "high": high_count,
            "medium": medium_count,
            "low": low_count
        }
    }

@app.get("/dashboard/trend")
def dashboard_trend(db: Session = Depends(get_db)):
    # monthly churn trend
    rows = (
        db.query(
            func.strftime("%Y-%m", PredictionLog.created_at).label("month"),
            func.count().label("total"),
            func.coalesce(func.sum(case((PredictionLog.risk_label == "high", 1), else_=0)), 0).label("churns"),
        )
        .group_by("month").all()
    )
    trend = []
    for r in rows:
        churn_rate = (r.churns / r.total) if r.total else 0
        trend.append({"month": r.month, "churn_rate": round(churn_rate, 3)})
    return trend


@app.get("/dashboard/recent")
def dashboard_recent(db: Session = Depends(get_db)):
    # Pull both predictions (created_at) and actions (contacted_at)
    pred_logs = (
        db.query(PredictionLog)
        .order_by(PredictionLog.created_at.desc())
        .limit(10).all()
    )
    act_logs = (
        db.query(PredictionLog)
        .filter(PredictionLog.contacted_at != None)
        .order_by(PredictionLog.contacted_at.desc())
        .limit(10).all()
    )

    events = []

    # Predictions
    for log in pred_logs:
        events.append({
            "type": "Prediction",
            "message": f"Predicted {log.customer_id} at {round(log.churn_probability*100,1)}% ({log.risk_label})",
            "time": log.created_at.isoformat(),
        })

    # Actions
    for log in act_logs:
        actions = []
        try:
            actions = json.loads(log.action) if log.action else []
        except Exception:
            actions = [log.action] if log.action else []
        events.append({
            "type": "Action",
            "message": f"Applied {', '.join(actions) or 'retention step'} for {log.customer_id}",
            "time": log.contacted_at.isoformat(),
        })

    # Merge + sort by time desc
    events = sorted(events, key=lambda x: x["time"], reverse=True)[:15]

    return events



@app.get("/dashboard/top")
def dashboard_top(db: Session = Depends(get_db)):
    logs = (
        db.query(PredictionLog)
        .order_by(PredictionLog.churn_probability.desc())
        .limit(10)
        .all()
    )

    result = []
    for log in logs:
        features = json.loads(log.features or "{}")

        # ✅ Parse stored actions
        actions = []
        try:
            actions = json.loads(log.action) if log.action else []
        except Exception:
            actions = [log.action] if log.action else []

        result.append({
            "id": log.customer_id,
            "name": log.customer_id,
            "churn_probability": float(log.churn_probability or 0.0),
            "risk": log.risk_label,
            "revenue": features.get("monthly_revenue", 0),
            "reason": "High churn probability" if log.risk_label == "high" else "Moderate churn probability" if log.risk_label == "medium" else "Low churn probability",
            "actions": actions,                           # ✅ comes from DB
            "action_status": log.action_status or "none"  # ✅ status (pending/applied/done)
        })

    return result


# Optional scheduled job code remains if required...



@app.get("/test-gemini")
def test_gemini():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="❌ GEMINI_API_KEY not set")

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [{
            "parts": [{"text": "Hello Gemini, give me one idea to retain a SaaS customer."}]
        }]
    }

    try:
        res = requests.post(url, headers=headers, json=payload)
        print("👉 Gemini raw response:", res.text)  # DEBUG
        res.raise_for_status()
        data = res.json()
        answer = data["candidates"][0]["content"]["parts"][0]["text"]
        return {"status": "ok", "answer": answer}
    except Exception as e:
        print("❌ Gemini error:", e)  # DEBUG
        raise HTTPException(status_code=500, detail=f"Gemini error: {str(e)}")

# ---------- Optional automation (safe) ----------
if _HAS_REPEAT:
    @app.on_event("startup")
    @repeat_every(seconds=86400)  # once per day
    def scheduled_predictions():
        """Automatically run daily predictions from latest CSV in data/raw"""
        if not pipeline:
            return
        db = SessionLocal()
        try:
            data_dir = BASE_APP.parent / "data" / "raw"
            csv_files = sorted(data_dir.glob("customer_data_*.csv"))
            if not csv_files:
                return
            data_path = csv_files[-1]
            df = pd.read_csv(data_path)
            expected = feature_config.get("numeric", []) + feature_config.get("categorical", [])
            df = df[expected]
            probas = pipeline.predict_proba(df)[:, 1]
            for i, p in enumerate(probas):
                risk = "high" if p >= 0.5 else "medium" if p >= 0.25 else "low"
                _save_log(db, f"auto_{i}", float(p), risk, df.iloc[i].to_dict())
            print(f"✅ Automated prediction job saved {len(probas)} logs")
        finally:
            db.close()
else:
    @app.post("/run_scheduled_predictions")
    def run_scheduled_predictions_manual(db: Session = Depends(get_db)):
        """If fastapi_utils isn't installed, call this endpoint to run the job manually."""
        if not pipeline:
            raise HTTPException(status_code=503, detail="Model not available.")
        data_dir = BASE_APP.parent / "data" / "raw"
        csv_files = sorted(Path(data_dir).glob("customer_data_*.csv"))
        if not csv_files:
            return {"saved": 0, "note": "no csv found"}
        data_path = csv_files[-1]
        df = pd.read_csv(data_path)
        expected = feature_config.get("numeric", []) + feature_config.get("categorical", [])
        df = df[expected]
        probas = pipeline.predict_proba(df)[:, 1]
        for i, p in enumerate(probas):
            risk = "high" if p >= 0.5 else "medium" if p >= 0.25 else "low"
            _save_log(db, f"auto_{i}", float(p), risk, df.iloc[i].to_dict())
        return {"saved": int(len(probas))}


