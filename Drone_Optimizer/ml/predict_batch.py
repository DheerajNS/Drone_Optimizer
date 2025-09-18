#!/usr/bin/env python3
# predict_batch.py
# Usage: python predict_batch.py --model drone_success_model.joblib --input sample.csv --output preds.csv

import argparse
import joblib
import pandas as pd
import sys
import os
import json
from typing import List

def load_model(model_path):
    data = joblib.load(model_path)
    if isinstance(data, dict) and "model" in data:
        return data["model"], data.get("columns", None)
    return data, None

def prepare_inputs(df: pd.DataFrame, model_columns: List[str]):
    # Keep only model_columns. If some are missing, try to coerce if possible.
    missing = [c for c in model_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required model columns in input: {missing}")
    return df[model_columns].astype(float)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="drone_success_model.joblib")
    p.add_argument("--input", required=True)
    p.add_argument("--output", default="predictions.csv")
    args = p.parse_args()

    if not os.path.exists(args.model):
        print("Model file not found:", args.model, file=sys.stderr)
        sys.exit(2)

    model, cols = load_model(args.model)
    if cols is None:
        raise RuntimeError("Model did not include 'columns' metadata. Re-train with columns saved.")

    # read input (csv or json)
    if args.input.lower().endswith(".csv"):
        df = pd.read_csv(args.input)
    elif args.input.lower().endswith(".json"):
        df = pd.read_json(args.input)
    else:
        print("Unsupported input file - use .csv or .json", file=sys.stderr)
        sys.exit(2)

    # keep original for output
    original = df.copy()
    X = prepare_inputs(df, cols)
    probs = model.predict_proba(X)[:,1]
    preds = (probs >= 0.5).astype(int)

    original["success_prob"] = probs
    original["prediction"] = preds
    original.to_csv(args.output, index=False)
    print(f"Wrote predictions to {args.output}")

if __name__ == "__main__":
    main()