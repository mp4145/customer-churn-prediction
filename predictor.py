from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd

FEATURE_COLUMNS = [
    "SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges",
    "total_addons_count", "is_high_risk_fiber", "monthly_to_total_ratio",
    "gender_encoded", "partner_encoded", "dependents_encoded",
    "phoneservice_encoded", "paperlessbilling_encoded", "multiplelines_encoded",
    "internetservice_encoded", "onlinesecurity_encoded", "onlinebackup_encoded",
    "deviceprotection_encoded", "techsupport_encoded", "streamingtv_encoded",
    "streamingmovies_encoded", "contract_encoded", "paymentmethod_encoded",
    "customertenurecategory_encoded",
]

REQUIRED_INPUTS = {
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod", "MonthlyCharges", "TotalCharges",
}


def _encode_yes(value: Any) -> int:
    return int(str(value).strip().lower() == "yes")


def _map_value(value: Any, mapping: dict[str, int], default: int) -> int:
    return mapping.get(str(value), default)


def build_features(customer: dict[str, Any]) -> pd.DataFrame:
    missing = sorted(REQUIRED_INPUTS - customer.keys())
    if missing:
        raise ValueError(f"Missing customer fields: {', '.join(missing)}")

    row = dict(customer)
    total_charges = row["TotalCharges"]
    row["TotalCharges"] = 0.0 if str(total_charges).strip().lower() in {"", "na", "nan", "null", "none"} else float(total_charges)
    row["MonthlyCharges"] = float(row["MonthlyCharges"])
    row["tenure"] = int(row["tenure"])

    tenure_category = "New" if row["tenure"] <= 12 else "Intermediate" if row["tenure"] <= 48 else "Loyal"
    addons = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]

    features = {
        "SeniorCitizen": int(row["SeniorCitizen"]),
        "tenure": row["tenure"],
        "MonthlyCharges": row["MonthlyCharges"],
        "TotalCharges": row["TotalCharges"],
        "total_addons_count": sum(row.get(service) == "Yes" for service in addons),
        "is_high_risk_fiber": int(row["InternetService"] == "Fiber optic" and row["MonthlyCharges"] > 70.0),
        "monthly_to_total_ratio": row["MonthlyCharges"] / (row["TotalCharges"] + 1.0),
        "gender_encoded": int(row["gender"] == "Female"),
        "partner_encoded": _encode_yes(row["Partner"]),
        "dependents_encoded": _encode_yes(row["Dependents"]),
        "phoneservice_encoded": _encode_yes(row["PhoneService"]),
        "paperlessbilling_encoded": _encode_yes(row["PaperlessBilling"]),
        "multiplelines_encoded": _map_value(row["MultipleLines"], {"No": 0, "Yes": 1, "No phone service": 2}, 2),
        "internetservice_encoded": _map_value(row["InternetService"], {"DSL": 0, "Fiber optic": 1, "No": 2}, 2),
        "onlinesecurity_encoded": _map_value(row["OnlineSecurity"], {"No": 0, "Yes": 1, "No internet service": 2}, 2),
        "onlinebackup_encoded": _map_value(row["OnlineBackup"], {"No": 0, "Yes": 1, "No internet service": 2}, 2),
        "deviceprotection_encoded": _map_value(row["DeviceProtection"], {"No": 0, "Yes": 1, "No internet service": 2}, 2),
        "techsupport_encoded": _map_value(row["TechSupport"], {"No": 0, "Yes": 1, "No internet service": 2}, 2),
        "streamingtv_encoded": _map_value(row["StreamingTV"], {"No": 0, "Yes": 1, "No internet service": 2}, 2),
        "streamingmovies_encoded": _map_value(row["StreamingMovies"], {"No": 0, "Yes": 1, "No internet service": 2}, 2),
        "contract_encoded": _map_value(row["Contract"], {"Month-to-month": 0, "One year": 1, "Two year": 2}, 2),
        "paymentmethod_encoded": _map_value(row["PaymentMethod"], {"Bank transfer (automatic)": 0, "Credit card": 1, "Electronic check": 2, "Mailed check": 3}, 3),
        "customertenurecategory_encoded": _map_value(tenure_category, {"New": 0, "Intermediate": 1, "Loyal": 2}, 2),
    }
    return pd.DataFrame([[features[column] for column in FEATURE_COLUMNS]], columns=FEATURE_COLUMNS)


def load_bundle(path: str | Path = "artifacts/churn_model.joblib") -> dict[str, Any]:
    return joblib.load(path)


def predict(customer: dict[str, Any], bundle: dict[str, Any]) -> dict[str, Any]:
    features = build_features(customer)
    scaled = bundle["scaler"].transform(features)
    probability = float(bundle["model"].predict_proba(scaled)[0, 1])
    threshold = float(bundle.get("threshold", 0.5))
    return {
        "will_churn": probability >= threshold,
        "churn_probability": round(probability, 4),
        "threshold": threshold,
        "model_version": bundle.get("model_version", "unknown"),
    }
