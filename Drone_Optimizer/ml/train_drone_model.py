# train_drone_model.py
# Train a quick LogisticRegression model and save as joblib (dict with keys 'model' and 'columns').
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import joblib
import json
import argparse
import os

def train(csv_path="drone_deliveries.csv", model_out="drone_success_model.joblib", columns_out="columns.json"):
    df = pd.read_csv(csv_path)
    # Keep numeric features only (match the existing model)
    feature_cols = ["payload_kg","distance_km","battery_pct","wind_speed_mps","rain_mm","num_stops"]
    X = df[feature_cols]
    y = df["success"].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000))
    ])

    pipe.fit(X_train, y_train)
    probs = pipe.predict_proba(X_test)[:,1]
    preds = pipe.predict(X_test)
    print("\\nMetrics on test set:")
    print(classification_report(y_test, preds, digits=3))
    print("ROC AUC:", roc_auc_score(y_test, probs))

    # Save model as a dict for compatibility
    os.makedirs(os.path.dirname(model_out) or ".", exist_ok=True)
    joblib.dump({"model": pipe, "columns": feature_cols}, model_out)
    with open(columns_out, "w") as f:
        json.dump(feature_cols, f, indent=2)
    print("Saved model to", model_out)
    print("Saved columns to", columns_out)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="drone_deliveries.csv")
    p.add_argument("--model_out", default="drone_success_model.joblib")
    p.add_argument("--columns_out", default="columns.json")
    args = p.parse_args()
    train(csv_path=args.data, model_out=args.model_out, columns_out=args.columns_out)