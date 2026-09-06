from pathlib import Path

import joblib
import numpy as np
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from predictor import build_features

DATASET = "sanjaycusat/telco-customer-churn3"
ARTIFACT = Path("artifacts/churn_model.joblib")


def main() -> None:
    stream = load_dataset(DATASET, streaming=True, split="train")
    rows = []
    labels = []
    for row in stream:
        rows.append(build_features(row).iloc[0].tolist())
        labels.append(int(row["Churn"] == "Yes"))

    features = np.asarray(rows, dtype=float)
    target = np.asarray(labels, dtype=int)
    x_train, _, y_train, _ = train_test_split(
        features, target, test_size=0.2, random_state=42, stratify=target
    )

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    imbalance_ratio = float((y_train == 0).sum() / (y_train == 1).sum())
    model = XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.05,
        scale_pos_weight=imbalance_ratio,
        random_state=42,
        eval_metric="logloss",
    )
    model.fit(x_train, y_train)

    ARTIFACT.parent.mkdir(exist_ok=True)
    joblib.dump({
        "model": model,
        "scaler": scaler,
        "threshold": 0.5,
        "model_version": "balanced-xgb-2026-09-05",
    }, ARTIFACT)
    print(f"Saved {ARTIFACT} ({len(rows)} rows, {features.shape[1]} features)")


if __name__ == "__main__":
    main()
