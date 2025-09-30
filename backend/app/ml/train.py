"""
Training script for churn prediction.
- Loads latest synthetic dataset
- Preprocesses (numeric + categorical features)
- Hyperparameter tuning with RandomizedSearchCV
- Trains best XGBoost model
- Saves model, metrics, feature importances, SHAP explainer
"""

import json
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from xgboost import XGBClassifier
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
import shap   # NEW

# ---------- Paths ----------
BASE = Path(__file__).resolve().parents[2]  # backend/
DATA_DIR = BASE / "data" / "raw"
MODELS_DIR = BASE / "app" / "models"
ARTIFACTS_DIR = BASE / "app" / "artifacts"

MODEL_FILENAME = "pipeline.joblib"
FEATURES_FILENAME = "feature_columns.json"
GLOBAL_IMPORTANCE = "global_feature_importance.json"
METRICS_FILE = "metrics.json"
SHAP_EXPLAINER = "shap_explainer.joblib"
SHAP_BACKGROUND = "shap_background.csv"

for d in (MODELS_DIR, ARTIFACTS_DIR):
    d.mkdir(parents=True, exist_ok=True)


def find_latest_csv(directory: Path):
    files = sorted(directory.glob("customer_data_*.csv*"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        raise FileNotFoundError(f"No customer_data_*.csv found in {directory}")
    return files[0]


def load_data(filepath: Path) -> pd.DataFrame:
    return pd.read_csv(filepath, parse_dates=["signup_date"])


def build_pipeline(numeric_features, categorical_features):
    """Preprocessing + model pipeline"""
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse=False))
    ])
    preprocessor = ColumnTransformer([
        ("num", num_pipeline, numeric_features),
        ("cat", cat_pipeline, categorical_features)
    ])

    clf = XGBClassifier(
        random_state=42,
        eval_metric="logloss",
        use_label_encoder=False,
        n_jobs=-1
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("clf", clf)
    ])
    return pipeline


def main():
    # 1) Load latest dataset
    data_path = find_latest_csv(DATA_DIR)
    print(f"📂 Loading data from: {data_path}")
    df = load_data(data_path)

    target_col = "is_churned"
    drop_cols = ["customer_id", "signup_date", "churn_probability"]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns] + [target_col])
    y = df[target_col].astype(int)

    # Define features
    numeric_features = [
        "tenure_days", "monthly_usage", "usage_trend", "features_used",
        "days_since_last_login", "avg_session_duration", "support_tickets_90d",
        "avg_resolution_time", "support_satisfaction", "escalated_tickets",
        "monthly_revenue", "payment_failures_90d", "days_to_renewal",
        "company_size", "revenue_per_employee", "usage_intensity", "manual_risk_score"
    ]
    categorical_features = [
        "segment", "contract_type", "industry", "region", "lifecycle_stage"
    ]

    numeric_features = [c for c in numeric_features if c in X.columns]
    categorical_features = [c for c in categorical_features if c in X.columns]
    X = X[numeric_features + categorical_features]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Build pipeline
    pipeline = build_pipeline(numeric_features, categorical_features)

    # Hyperparameter search (XGBoost)
    param_dist = {
        "clf__n_estimators": [100, 200, 300],
        "clf__max_depth": [3, 5, 7, 9],
        "clf__learning_rate": [0.01, 0.05, 0.1, 0.2],
        "clf__subsample": [0.6, 0.8, 1.0],
        "clf__colsample_bytree": [0.6, 0.8, 1.0]
    }

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_dist,
        n_iter=20,
        scoring="roc_auc",
        cv=cv,
        verbose=1,
        random_state=42,
        n_jobs=-1
    )
    search.fit(X_train, y_train)

    best_model = search.best_estimator_
    print(f"🏆 Best Params: {search.best_params_}")

    # Evaluate
    y_proba = best_model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    auc = roc_auc_score(y_test, y_proba)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred).tolist()
    report = classification_report(y_test, y_pred, output_dict=True)

    metrics = {
        "roc_auc": float(auc),
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1_score": float(f1),
        "confusion_matrix": cm,
        "classification_report": report,
        "best_params": search.best_params_,
        "trained_at": datetime.now().isoformat()
    }
    (ARTIFACTS_DIR / METRICS_FILE).write_text(json.dumps(metrics, indent=2))
    print(f"📊 Metrics saved: {ARTIFACTS_DIR / METRICS_FILE}")

    # Feature importances
    preproc = best_model.named_steps["preprocessor"]
    num_names = numeric_features
    cat_names = []
    if categorical_features:
        ohe = preproc.named_transformers_["cat"].named_steps["onehot"]
        cat_names = ohe.get_feature_names_out(categorical_features).tolist()
    feature_names = num_names + cat_names

    model = best_model.named_steps["clf"]
    importances = model.feature_importances_

    importance_pairs = sorted(
        zip(feature_names, importances.tolist()), key=lambda x: x[1], reverse=True
    )
    importance_dict = [{"feature": f, "importance": v} for f, v in importance_pairs]
    (ARTIFACTS_DIR / GLOBAL_IMPORTANCE).write_text(json.dumps(importance_dict, indent=2))
    print(f"📊 Global importances saved: {ARTIFACTS_DIR / GLOBAL_IMPORTANCE}")

    # Save model + features
    joblib.dump(best_model, MODELS_DIR / MODEL_FILENAME)
    (ARTIFACTS_DIR / FEATURES_FILENAME).write_text(json.dumps({
        "numeric": numeric_features,
        "categorical": categorical_features,
        "feature_names": feature_names
    }, indent=2))
    print(f"✅ Pipeline saved: {MODELS_DIR / MODEL_FILENAME}")

    # --------- SHAP Explainer ---------
    print("⚡ Building SHAP explainer (may take a while)...")
    X_bg = X_train.sample(100, random_state=42)
    explainer = shap.TreeExplainer(model)
    joblib.dump(explainer, ARTIFACTS_DIR / SHAP_EXPLAINER)
    X_bg.to_csv(ARTIFACTS_DIR / SHAP_BACKGROUND, index=False)
    print(f"🧠 SHAP explainer + background saved to {ARTIFACTS_DIR}")

    # Final summary
    print("🎯 Training complete.")
    print("ROC AUC:", auc, "| Accuracy:", acc, "| Precision:", prec, "| Recall:", rec, "| F1:", f1)


if __name__ == "__main__":
    main()
